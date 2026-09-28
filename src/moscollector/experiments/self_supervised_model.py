"""Optional past-history forecasting objective and restricted encoder transfer."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from moscollector.experiments.neural_sequence_data import history_batch

STATIC = {
    "hour",
    "day_of_week",
    "month",
    "weekend",
    "channel_count",
    "temperature_channels",
    "smoke_channels",
    "pump_channels",
    "water_channels",
}


def telemetry_indices(codec):
    columns = codec["numeric"]
    if any(c.startswith("target_") or c in ("eligible", "as_of") for c in columns):
        raise ValueError("Forbidden pretext feature")
    result = [i for i, name in enumerate(columns) if name not in STATIC and not name.startswith("past_")]
    if not result:
        raise ValueError("No dynamic telemetry targets")
    return result


class HistoryPredictor(nn.Module):
    """Current observations are targets, not model inputs; no incident head."""

    def __init__(self, count_model, target_dim):
        super().__init__()
        if not count_model.use_history or target_dim <= 0:
            raise ValueError("Pretext requires a history model and telemetry targets")
        self.history_projection = deepcopy(count_model.history_projection)
        self.gru = deepcopy(count_model.gru)
        self.decoder = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, target_dim))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, history, lengths):
        output, _ = self.gru(self.history_projection(history))
        state = output[torch.arange(len(output), device=output.device), (lengths - 1).clamp(min=0)]
        state = state * (lengths > 0).unsqueeze(1)
        return self.decoder(state)


def transfer_history(pretext, count_model):
    before = {
        name: value.detach().cpu().clone()
        for name, value in count_model.state_dict().items()
        if not name.startswith(("history_projection.", "gru."))
    }
    count_model.history_projection.load_state_dict(pretext.history_projection.state_dict())
    count_model.gru.load_state_dict(pretext.gru.state_dict())
    for name, value in count_model.state_dict().items():
        if name in before:
            torch.testing.assert_close(value.detach().cpu(), before[name], rtol=0, atol=0)


def pretext_targets(data, ids, target_indices):
    numeric = data["numeric"][ids]
    width = numeric.shape[1] // 2
    if (
        numeric.shape[1] != 2 * width
        or not target_indices
        or max(target_indices) >= width
        or min(target_indices) < 0
    ):
        raise ValueError("Invalid pretext target columns")
    target = numeric[:, target_indices]
    valid = numeric[:, np.asarray(target_indices) + width] == 0
    valid &= (data["positions"][ids] >= 0).any(axis=1)[:, None]
    return target, valid


def masked_huber(prediction, target, valid):
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("Mismatched pretext tensors")
    count = int(valid.sum().item())
    if not count:
        raise ValueError("No observed telemetry targets")
    # Mask first, so even a caller-provided NaN at an unobserved location cannot
    # contaminate the loss. Actual encoded targets are already finite.
    p, y = torch.where(valid, prediction, 0), torch.where(valid, target, 0)
    if not torch.isfinite(p).all().item() or not torch.isfinite(y).all().item():
        raise ValueError("Nonfinite observed pretext value")
    # MPS SmoothL1 reduction requires contiguous operands for column-selected
    # NumPy targets; their torch views can otherwise retain Fortran strides.
    return torch.nn.functional.smooth_l1_loss(
        p.contiguous(), y.contiguous(), beta=1, reduction="sum"
    ) / count, count


def pretext_batch(model, data, ids, history_values, device, target_indices):
    tokens, lengths = history_batch(history_values, data["positions"][ids], data["ages"][ids])
    target, valid = pretext_targets(data, ids, target_indices)
    predicted = model(torch.from_numpy(tokens).to(device), torch.from_numpy(lengths).to(device))
    return masked_huber(predicted, torch.from_numpy(target).to(device), torch.from_numpy(valid).to(device))


def pretext_baselines(data, history_values, target_indices):
    """Same valid target cells as the network; missing last values use train mean."""
    total = {"zero_training_mean": 0.0, "last_snapshot_with_mean_for_missing": 0.0}
    cells = 0
    for start in range(0, len(data["numeric"]), 2048):
        ids = np.arange(start, min(start + 2048, len(data["numeric"])))
        target, valid = pretext_targets(data, ids, target_indices)
        lengths = (data["positions"][ids] >= 0).sum(axis=1)
        last = data["positions"][ids, np.maximum(lengths - 1, 0)]
        previous = history_values[np.maximum(last, 0)][:, target_indices]
        for name, forecast in (
            ("zero_training_mean", np.zeros_like(target)),
            ("last_snapshot_with_mean_for_missing", previous),
        ):
            error = np.abs(forecast[valid].astype(float) - target[valid])
            total[name] += float(np.where(error < 1, 0.5 * error**2, error - 0.5).sum())
        cells += int(valid.sum())
    if not cells:
        raise ValueError("No pretext validation support")
    return {"observed_cells": cells, **{name: value / cells for name, value in total.items()}}
