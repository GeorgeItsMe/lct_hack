"""Atomic, resumable telemetry pretraining and count fine-tuning stages."""

import time

import numpy as np
import torch

from moscollector.goal90_research import read
from moscollector.neural_count_research import TRAINING, cpu_state, forward_batch, save_torch, validation_loss
from moscollector.prepare import sha256, write_json
from moscollector.self_supervised_model import pretext_batch, pretext_targets

PRETRAIN = {**TRAINING, "epochs": 8, "patience": 2, "seed": 43}


def validation(model, data, history_values, device, stage, indices):
    if stage == "count":
        return validation_loss(model, data, history_values, device)
    model.eval()
    loss_sum, cells = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(data["numeric"]), 1024):
            ids = np.arange(start, min(start + 1024, len(data["numeric"])))
            _, valid = pretext_targets(data, ids, indices)
            if not valid.any():
                continue
            loss, n = pretext_batch(model, data, ids, history_values, device, indices)
            loss_sum += float(loss.cpu()) * n
            cells += n
    if not cells:
        raise ValueError("No pretext validation cells")
    return loss_sum / cells


def fit_stage(directory, model, data, history_values, device, stage, signature, indices=None):
    if stage not in ("pretext", "count"):
        raise ValueError("Unknown training stage")
    config = PRETRAIN if stage == "pretext" else TRAINING
    signature = {
        **signature,
        "stage": stage,
        "training": config,
        "device": device,
        "torch": str(torch.__version__),
        "target_indices": indices,
    }
    directory.mkdir(parents=True, exist_ok=True)
    meta_path, weights, checkpoint = directory / "fit.json", directory / "model.pt", directory / "last.pt"
    model = model.to(device)
    if meta_path.exists():
        meta = read(meta_path)
        if meta["signature"] != signature or sha256(weights) != meta["model_sha256"]:
            raise ValueError("Self-supervised stage changed")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        return model, meta
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Self-supervised checkpoint configuration changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print("RESUME selfsup", directory, state["epoch"], flush=True)
    else:
        loss = validation(model, data["validation"], history_values, device, stage, indices)
        state = {
            "signature": signature,
            "epoch": 0,
            "best_epoch": 0,
            "best_loss": loss,
            "best_state": cpu_state(model),
            "bad_epochs": 0,
            "history": [{"epoch": 0, "validation_loss": loss}],
        }
    start_time = time.perf_counter()
    print("START selfsup", directory, stage, device, flush=True)
    for epoch in range(state["epoch"] + 1, config["epochs"] + 1):
        if state["bad_epochs"] >= config["patience"]:
            break
        model.train()
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(data["train"]["numeric"]))
        total, support = 0.0, 0
        for start in range(0, len(order), config["batch_size"]):
            ids = order[start : start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            if stage == "pretext":
                _, valid = pretext_targets(data["train"], ids, indices)
                if not valid.any():
                    continue
                loss, n = pretext_batch(model, data["train"], ids, history_values, device, indices)
            else:
                raw = forward_batch(model, data["train"], ids, history_values, device)
                target = torch.from_numpy(data["train"]["target"][ids]).to(device)
                loss = torch.nn.functional.poisson_nll_loss(raw, target, log_input=True, full=False)
                n = len(ids)
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite self-supervised training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach().cpu()) * n
            support += n
        if not support:
            raise ValueError("No supported self-supervised training samples")
        val = validation(model, data["validation"], history_values, device, stage, indices)
        if val < state["best_loss"] - config["min_delta"]:
            state.update(best_loss=val, best_epoch=epoch, best_state=cpu_state(model), bad_epochs=0)
        else:
            state["bad_epochs"] += 1
        state.update(epoch=epoch, model=cpu_state(model), optimizer=optimizer.state_dict())
        row = {
            "epoch": epoch,
            "training_loss": total / support,
            "validation_loss": val,
            "best_epoch": state["best_epoch"],
            "elapsed_this_process_seconds": time.perf_counter() - start_time,
        }
        state["history"].append(row)
        save_torch(checkpoint, state)
        write_json(directory / "progress.json", {**row, "checkpoint_sha256": sha256(checkpoint)})
        print("EPOCH selfsup", directory, row, flush=True)
    model.load_state_dict(state["best_state"])
    save_torch(weights, state["best_state"])
    meta = {
        "signature": signature,
        "stage": stage,
        "best_epoch": state["best_epoch"],
        "best_validation_loss": state["best_loss"],
        "completed_epochs": state["epoch"],
        "history": state["history"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_sha256": sha256(weights),
        "weights_file": str(weights),
    }
    write_json(meta_path, meta)
    print("DONE selfsup", directory, "best", state["best_epoch"], flush=True)
    return model, meta
