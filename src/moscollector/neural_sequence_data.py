"""Causal same-object history and training-only transforms for neural counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.train import CATEGORICAL

HISTORY_STEPS = 16
HISTORY_HOURS = 48


def history_positions(history, current, steps=HISTORY_STEPS, hours=HISTORY_HOURS):
    """Oldest-first observations in [t-hours,t), right padded by -1.

    Uses only object IDs and timestamps, never eligibility/future-label columns.
    Return positions into the supplied history, preserving current row order.
    """
    if steps <= 0 or hours <= 0 or history.duplicated(["object_id", "as_of"]).any():
        raise ValueError("Invalid history grid")
    if history.as_of.isna().any() or current.as_of.isna().any():
        raise ValueError("Missing timestamps")
    indices = np.full((len(current), steps), -1, dtype=np.int32)
    ages = np.zeros((len(current), steps), dtype=np.float32)
    hist = {
        obj: rows.sort_values("as_of") for obj, rows in history.reset_index(drop=True).groupby("object_id")
    }
    for obj, rows in current.reset_index(drop=True).groupby("object_id"):
        if obj not in hist:
            continue
        h = hist[obj]
        times = h.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        stop = np.searchsorted(times, at, side="left")
        begin = np.maximum(np.searchsorted(times, at - int(hours * HOUR_NS), side="left"), stop - steps)
        positions = begin[:, None] + np.arange(steps)
        valid = positions < stop[:, None]
        safe = np.minimum(positions, len(h) - 1)
        ids = np.where(valid, h.index.to_numpy()[safe], -1)
        age = np.where(valid, (at[:, None] - times[safe]) / HOUR_NS, 0)
        if np.any(age[valid] <= 0) or np.any(age[valid] > hours):
            raise ValueError("Noncausal history")
        indices[rows.index], ages[rows.index] = ids, age
    return indices, ages


def numeric_values(frame, columns):
    raw = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(raw)
    clean = np.where(valid, raw, 0)
    transformed = np.sign(clean) * np.log1p(np.abs(clean))
    return transformed, valid


def fit_codec(training, columns):
    if not len(training) or any(c in ("as_of", "eligible") or c.startswith("target_") for c in columns):
        raise ValueError("Empty training data or forbidden model input")
    numeric = [c for c in columns if c not in CATEGORICAL]
    values, valid = numeric_values(training, numeric)
    count = valid.sum(axis=0)
    mean = values.sum(axis=0, dtype=np.float64) / np.maximum(count, 1)
    variance = np.where(valid, (values - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1
    vocabulary = {
        col: {
            value: i + 1
            for i, value in enumerate(sorted(training[col].fillna("unknown").astype(str).unique()))
        }
        for col in CATEGORICAL
    }
    return {
        "columns": list(columns),
        "numeric": numeric,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "vocabulary": vocabulary,
        "transform": "signed_log1p_then_training_mean_std_clip8_plus_missing_indicators",
        "training_rows": len(training),
    }


def encode(frame, codec):
    values, valid = numeric_values(frame, codec["numeric"])
    scaled = np.clip((values - np.asarray(codec["mean"])) / np.asarray(codec["scale"]), -8, 8)
    scaled[~valid] = 0
    numeric = np.concatenate([scaled, ~valid], axis=1).astype(np.float32)
    categorical = np.column_stack(
        [
            frame[col]
            .fillna("unknown")
            .astype(str)
            .map(codec["vocabulary"][col])
            .fillna(0)
            .to_numpy(dtype=np.int64)
            for col in CATEGORICAL
        ]
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite encoded features")
    return numeric, categorical


def history_batch(values, indices, ages):
    """Materialize only one minibatch; padded values never masquerade as reports."""
    valid = indices >= 0
    if np.any(indices < -1) or np.any(indices >= len(values)) or indices.shape != ages.shape:
        raise ValueError("Invalid history references")
    if not len(values):
        if valid.any():
            raise ValueError("References into empty history")
        tokens = np.zeros((*indices.shape, values.shape[1]), dtype=np.float32)
    else:
        tokens = values[np.maximum(indices, 0)].copy()
        tokens[~valid] = 0
    age_feature = (np.log1p(ages) / np.log1p(HISTORY_HOURS)).astype(np.float32)
    return np.concatenate([tokens, age_feature[..., None]], axis=2), valid.sum(axis=1).astype(np.int64)
