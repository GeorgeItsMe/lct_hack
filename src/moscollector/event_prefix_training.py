"""Replay exactly one original training epoch with additional validation checks.

The checks select a prefix state but never affect optimizer updates or how far
the epoch runs. Original completed studies and their checkpoints are untouched.
"""

import math
import time

import numpy as np
import torch

from moscollector.event_sequence_model import EventCountNetwork
from moscollector.event_sequence_training import array_digest, forward_batch, validation_loss
from moscollector.goal90_research import read
from moscollector.neural_count_research import TRAINING, cpu_state, save_torch
from moscollector.prepare import sha256, write_json


def check_interval(original_rows, batch_size):
    if original_rows <= 0 or batch_size <= 0:
        raise ValueError("Positive original row count and batch size required")
    return max(1, math.ceil(math.ceil(original_rows / batch_size) / 4))


def fit_prefix(directory, data, source, config, provenance, device, every_steps, settings=None):
    settings = dict(TRAINING if settings is None else settings)
    if not isinstance(every_steps, int) or every_steps <= 0:
        raise ValueError("Positive integer check interval required")
    for part in ("train", "validation"):
        row = data[part]
        if (
            not len(row["target"])
            or row["weight"].shape != row["target"].shape
            or not np.isfinite(row["target"]).all()
            or not np.isfinite(row["weight"]).all()
            or np.any(row["target"] < 0)
            or np.any(row["weight"] <= 0)
        ):
            raise ValueError("Invalid prefix training weights/targets")
    signature = {
        **provenance,
        "config": config,
        "settings": settings,
        "every_steps": every_steps,
        "device": device,
        "torch": str(torch.__version__),
        "epochs_replayed": 1,
        "data_hashes": {p: {k: array_digest(v) for k, v in row.items()} for p, row in data.items()},
        "event_source_hashes": {k: array_digest(v) for k, v in source.items()},
    }
    directory.mkdir(parents=True, exist_ok=True)
    weights, meta_path, last = directory / "model.pt", directory / "fit.json", directory / "last.pt"
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"])
    model = EventCountNetwork(**config).to(device)
    if meta_path.exists():
        meta = read(meta_path)
        if (
            meta["signature"] != signature
            or meta["model_sha256"] != sha256(weights)
            or meta["last_sha256"] != sha256(last)
        ):
            raise ValueError("Changed completed prefix inputs or weights")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        return model, meta
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Changed prefix checkpoint inputs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print("RESUME PREFIX", directory, state["step"], flush=True)
    else:
        initial = validation_loss(model, data["validation"], source, device)
        state = {
            "signature": signature,
            "step": 0,
            "next_start": 0,
            "training_loss_sum": 0.0,
            "best_step": 0,
            "best_loss": initial,
            "best_state": cpu_state(model),
            "history": [{"step": 0, "validation_loss": initial, "best_step": 0}],
        }
    train = data["train"]
    order = np.random.default_rng(settings["seed"] + 1).permutation(len(train["target"]))
    mean_weight = float(train["weight"].mean())
    started = time.perf_counter()
    print("START PREFIX", directory, "rows", len(order), "check_every", every_steps, flush=True)
    for start in range(state["next_start"], len(order), settings["batch_size"]):
        model.train()
        ids = order[start : start + settings["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        raw = forward_batch(model, train, ids, source, device)
        target = torch.from_numpy(train["target"][ids]).to(device=device, dtype=torch.float32)
        weight = torch.from_numpy(train["weight"][ids]).to(device=device, dtype=torch.float32)
        values = torch.exp(raw) - target * raw
        loss = (values * weight).mean() / mean_weight
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite prefix loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        state["step"] += 1
        state["next_start"] = start + len(ids)
        state["training_loss_sum"] += float((values.detach() * weight).sum().cpu())
        if state["step"] % every_steps == 0 or state["next_start"] == len(order):
            val = validation_loss(model, data["validation"], source, device)
            if val < state["best_loss"] - settings["min_delta"]:
                state.update(best_loss=val, best_step=state["step"], best_state=cpu_state(model))
            row = {"step": state["step"], "validation_loss": val, "best_step": state["best_step"]}
            state["history"].append(row)
            state.update(model=cpu_state(model), optimizer=optimizer.state_dict())
            save_torch(last, state)
            write_json(
                directory / "progress.json",
                {
                    **row,
                    "rows_done": state["next_start"],
                    "rows_total": len(order),
                    "elapsed_this_process_seconds": time.perf_counter() - started,
                },
            )
            print("CHECK PREFIX", directory, row, flush=True)
    if state["step"] != math.ceil(len(order) / settings["batch_size"]):
        raise ValueError("Prefix replay did not finish exactly one epoch")
    model.load_state_dict(state["best_state"])
    save_torch(weights, state["best_state"])
    meta = {
        "signature": signature,
        "best_step": state["best_step"],
        "best_validation_loss": state["best_loss"],
        "completed_steps": state["step"],
        "completed_rows": state["next_start"],
        "first_epoch_training_loss": state["training_loss_sum"] / float(train["weight"].sum()),
        "first_epoch_validation_loss": state["history"][-1]["validation_loss"],
        "history": state["history"],
        "model_sha256": sha256(weights),
        "last_sha256": sha256(last),
    }
    write_json(meta_path, meta)
    return model, meta
