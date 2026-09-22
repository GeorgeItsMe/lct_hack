"""Weighted count fitting and resumable checkpoints for raw-onset networks."""

import hashlib
import time

import numpy as np
import torch

from moscollector.event_sequence_data import event_batch
from moscollector.event_sequence_model import EventCountNetwork
from moscollector.goal90_research import read
from moscollector.neural_count_research import TRAINING, cpu_state, save_torch
from moscollector.prepare import sha256, write_json


def array_digest(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{value.dtype}:{value.shape}".encode())
    h.update(value)
    return h.hexdigest()


def forward_batch(model, data, ids, source, device):
    numeric = torch.from_numpy(data["numeric"][ids]).to(device)
    categorical = torch.from_numpy(data["categorical"][ids]).to(device)
    if not model.use_history:
        return model(numeric, categorical)
    categories, values, lengths = event_batch(
        source["encoded"], source["times"], data["bounds"][ids], data["query_times"][ids]
    )
    return model(
        numeric,
        categorical,
        torch.from_numpy(categories).to(device),
        torch.from_numpy(values).to(device),
        torch.from_numpy(lengths).to(device),
    )


def predict(model, data, source, device, batch_size=1024):
    model.eval()
    output = np.empty(len(data["numeric"]), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(output), batch_size):
            ids = np.arange(start, min(start + batch_size, len(output)))
            output[ids] = forward_batch(model, data, ids, source, device).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite event-network predictions")
    return output


def validation_loss(model, data, source, device):
    raw = predict(model, data, source, device)
    return float(np.average(np.exp(raw) - data["target"] * raw, weights=data["weight"]))


def fit_network(directory, data, source, config, provenance, device, settings=None):
    settings = dict(TRAINING if settings is None else settings)
    directory.mkdir(parents=True, exist_ok=True)
    for part in ("train", "validation"):
        rows = data[part]
        if (
            not len(rows["target"])
            or rows["weight"].shape != rows["target"].shape
            or not np.isfinite(rows["weight"]).all()
            or np.any(rows["weight"] <= 0)
            or not np.isfinite(rows["target"]).all()
            or np.any(rows["target"] < 0)
        ):
            raise ValueError("Invalid weighted event-network targets")
    signature = {
        **provenance,
        "config": config,
        "settings": settings,
        "device": device,
        "torch": str(torch.__version__),
        "data_hashes": {
            part: {name: array_digest(value) for name, value in rows.items()} for part, rows in data.items()
        },
        "event_source_hashes": {name: array_digest(value) for name, value in source.items()},
    }
    weights, metadata, checkpoint = directory / "model.pt", directory / "fit.json", directory / "last.pt"
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"])
    model = EventCountNetwork(**config).to(device)
    if metadata.exists():
        meta = read(metadata)
        if meta["signature"] != signature or sha256(weights) != meta["model_sha256"]:
            raise ValueError("Event-network fit inputs or weights changed")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        return model, meta
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Event-network checkpoint belongs to different inputs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print("RESUME event network", directory, "epoch", state["epoch"], flush=True)
    else:
        loss = validation_loss(model, data["validation"], source, device)
        state = {
            "signature": signature,
            "epoch": 0,
            "best_epoch": 0,
            "best_loss": loss,
            "best_state": cpu_state(model),
            "bad_epochs": 0,
            "history": [{"epoch": 0, "validation_loss": loss}],
        }
    started = time.perf_counter()
    train = data["train"]
    weight_mean = float(train["weight"].mean())
    print("START event network", directory, "device", device, "rows", len(train["target"]), flush=True)
    for epoch in range(state["epoch"] + 1, settings["epochs"] + 1):
        if state["bad_epochs"] >= settings["patience"]:
            break
        model.train()
        order = np.random.default_rng(settings["seed"] + epoch).permutation(len(train["target"]))
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            ids = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            raw = forward_batch(model, train, ids, source, device)
            target = torch.from_numpy(train["target"][ids]).to(device=device, dtype=torch.float32)
            weight = torch.from_numpy(train["weight"][ids]).to(device=device, dtype=torch.float32)
            values = torch.exp(raw) - target * raw
            # Fixed global mean, rather than random minibatch weight sum, keeps
            # this a stochastic gradient of the full weighted objective.
            loss = (values * weight).mean() / weight_mean
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite weighted event-network loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            total += float((values.detach() * weight).sum().cpu())
            if start and start // settings["batch_size"] % 500 == 0:
                print(
                    "BATCH event network",
                    directory,
                    "epoch",
                    epoch,
                    "rows",
                    start + len(ids),
                    "/",
                    len(order),
                    flush=True,
                )
        val = validation_loss(model, data["validation"], source, device)
        if val < state["best_loss"] - settings["min_delta"]:
            state.update(best_loss=val, best_epoch=epoch, best_state=cpu_state(model), bad_epochs=0)
        else:
            state["bad_epochs"] += 1
        state.update(epoch=epoch, model=cpu_state(model), optimizer=optimizer.state_dict())
        row = {
            "epoch": epoch,
            "training_loss": total / float(train["weight"].sum()),
            "validation_loss": val,
            "best_epoch": state["best_epoch"],
            "elapsed_this_process_seconds": time.perf_counter() - started,
        }
        state["history"].append(row)
        save_torch(checkpoint, state)
        write_json(directory / "progress.json", {**row, "checkpoint_sha256": sha256(checkpoint)})
        print("EPOCH event network", directory, row, flush=True)
    model.load_state_dict(state["best_state"])
    save_torch(weights, state["best_state"])
    meta = {
        "signature": signature,
        "config": config,
        "best_epoch": state["best_epoch"],
        "best_validation_loss": state["best_loss"],
        "completed_epochs": state["epoch"],
        "history": state["history"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_sha256": sha256(weights),
        "weights_file": str(weights),
        "device": device,
    }
    write_json(metadata, meta)
    print("DONE event network", directory, "best", meta["best_epoch"], flush=True)
    return model, meta
