"""Strictly-past, signal-balanced raw-onset sequences for optional neural research."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.experiments.channel_novelty_features import SIGNALS
from moscollector.experiments.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

HISTORY_HOURS = 48
PER_SIGNAL = 16
STEPS = PER_SIGNAL * len(SIGNALS)
FOLDER = PROCESSED / "event-sequences-v37"
KEYS = ["object_id", "as_of"]


def prepare_events(onsets, catalog):
    columns = ["object_id", "channel_id", "signal", "ts"]
    frame = onsets[columns].copy()
    if frame.isna().any().any() or frame.duplicated(["channel_id", "signal", "ts"]).any():
        raise ValueError("Invalid canonical onset history")
    if catalog.channel_id.duplicated().any():
        raise ValueError("Duplicate catalog channel")
    frame = frame.merge(
        catalog[["channel_id", "object_id", "sensor_type"]].rename(columns={"object_id": "catalog_object"}),
        on="channel_id",
        how="left",
        validate="many_to_one",
    )
    if frame.catalog_object.isna().any() or not frame.object_id.eq(frame.catalog_object).all():
        raise ValueError("Onset ownership disagrees with catalog")
    if not frame.signal.isin(SIGNALS).all():
        raise ValueError("Unknown canonical signal family")
    frame["sensor_type"] = frame.sensor_type.fillna("__UNKNOWN__").astype(str)
    frame["signal_code"] = frame.signal.map({name: i + 1 for i, name in enumerate(SIGNALS)}).astype("int8")
    return (
        frame.drop(columns="catalog_object")
        .sort_values(["object_id", "signal_code", "ts", "channel_id"], kind="stable")
        .reset_index(drop=True)
    )


def history_bounds(events, queries, hours=HISTORY_HOURS):
    """Half-open source ranges per family in [query-hours,query), no labels.

    Store only two positions per family/query, not all sequence tensors. Bounds
    retain full counts; minibatches select the latest PER_SIGNAL per family.
    """
    queries = queries[KEYS].reset_index(drop=True)
    if hours <= 0 or queries.isna().any().any() or queries.duplicated(KEYS).any():
        raise ValueError("Invalid event-sequence queries")
    if len(events) >= np.iinfo(np.int32).max:
        raise ValueError("Event source exceeds int32 position capacity")
    result = np.zeros((len(queries), len(SIGNALS), 2), dtype=np.int32)
    groups = events.groupby(["object_id", "signal_code"], sort=False).indices
    source_times = events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for obj, rows in queries.groupby("object_id", sort=False):
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        for code in range(1, len(SIGNALS) + 1):
            ids = groups.get((obj, code))
            if ids is None:
                continue
            if ids[-1] - ids[0] + 1 != len(ids):
                raise ValueError("Event source groups must be contiguous")
            times = source_times[ids]
            if np.any(times[1:] < times[:-1]):
                raise ValueError("Event source is not chronological")
            result[rows.index, code - 1, 0] = ids[0] + np.searchsorted(
                times, at - int(hours * HOUR_NS), side="left"
            )
            result[rows.index, code - 1, 1] = ids[0] + np.searchsorted(times, at, side="left")
    return result


def validate_bounds(bounds, source_rows):
    b = np.asarray(bounds)
    if b.ndim != 3 or b.shape[1:] != (len(SIGNALS), 2) or not np.issubdtype(b.dtype, np.integer):
        raise ValueError("Invalid event-bound shape or type")
    if np.any(b < 0) or np.any(b > source_rows) or np.any(b[..., 0] > b[..., 1]):
        raise ValueError("Invalid event-bound positions")


def selected_source_mask(bounds, source_rows, per_signal=PER_SIGNAL):
    """Exactly the source rows referenced by these queries; no later vocabulary."""
    validate_bounds(bounds, source_rows)
    if per_signal <= 0:
        raise ValueError("Invalid per-family sequence length")
    stop = bounds[..., 1].ravel()
    start = np.maximum(bounds[..., 0], bounds[..., 1] - per_signal).ravel()
    valid = start < stop
    changes = np.zeros(source_rows + 1, dtype=np.int64)
    np.add.at(changes, start[valid], 1)
    np.add.at(changes, stop[valid], -1)
    return np.cumsum(changes[:-1]) > 0


def fit_event_codec(events, training_bounds, per_signal=PER_SIGNAL):
    selected = selected_source_mask(training_bounds, len(events), per_signal)
    train = events.loc[selected]
    return {
        "vocabulary": {
            name: {value: i + 1 for i, value in enumerate(sorted(train[name].astype(str).unique()))}
            for name in ("channel_id", "sensor_type")
        },
        "signal_codes": {name: i + 1 for i, name in enumerate(SIGNALS)},
        "training_onset_rows": int(selected.sum()),
        "training_queries": len(training_bounds),
        "per_signal": per_signal,
        "history_hours": HISTORY_HOURS,
        "scope": "Vocabulary only from actual retained training-query histories; unknown/padding0. Static catalog sensor types are assumed available, without historical catalog snapshots.",
    }


def encode_events(events, codec):
    if codec["signal_codes"] != {name: i + 1 for i, name in enumerate(SIGNALS)}:
        raise ValueError("Changed signal vocabulary")
    return np.column_stack(
        [
            events[name].astype(str).map(codec["vocabulary"][name]).fillna(0).to_numpy(dtype=np.int64)
            for name in ("channel_id", "sensor_type")
        ]
        + [events.signal_code.to_numpy(dtype=np.int64)]
    )


def event_batch(encoded, source_times, bounds, query_times, per_signal=PER_SIGNAL):
    """Oldest-first retained events, right padding, explicit ages/selected gaps.

    Family caps stop a busy signal from erasing every rarer family. Same-time
    ties are deterministic family/channel order, not claimed physical order.
    """
    validate_bounds(bounds, len(source_times))
    at = np.asarray(query_times, dtype=np.int64)
    if per_signal <= 0 or at.shape != (len(bounds),) or encoded.shape != (len(source_times), 3):
        raise ValueError("Invalid event minibatch")
    start = np.maximum(bounds[..., 0], bounds[..., 1] - per_signal)
    indices = start[..., None] + np.arange(per_signal)
    valid = indices < bounds[..., 1, None]
    width = per_signal * len(SIGNALS)
    indices, valid = indices.reshape(len(bounds), width), valid.reshape(len(bounds), width)
    safe = np.minimum(indices, max(len(source_times) - 1, 0))
    if len(source_times):
        times = np.where(valid, source_times[safe], np.iinfo(np.int64).max)
    else:
        times = np.full(indices.shape, np.iinfo(np.int64).max, dtype=np.int64)
    order = np.argsort(times, axis=1, kind="stable")
    safe, valid, times = (np.take_along_axis(x, order, axis=1) for x in (safe, valid, times))
    lengths = valid.sum(axis=1).astype(np.int64)
    categories = encoded[safe].copy() if len(encoded) else np.zeros((*indices.shape, 3), dtype=np.int64)
    categories[~valid] = 0
    # Substitute the query time before subtraction to avoid overflow at padding.
    causal_times = np.where(valid, times, at[:, None])
    ages = (at[:, None] - causal_times) / HOUR_NS
    if np.any(ages[valid] <= 0) or np.any(ages[valid] > HISTORY_HOURS):
        raise ValueError("Noncausal or out-of-window event token")
    previous = valid.copy()
    previous[:, 0] = False
    gaps = np.zeros(ages.shape, dtype=np.float32)
    gaps[:, 1:] = np.where(previous[:, 1:], (causal_times[:, 1:] - causal_times[:, :-1]) / HOUR_NS, 0)
    if np.any(gaps < 0):
        raise ValueError("Event minibatch order reversed")
    numeric = np.stack(
        [np.log1p(ages) / np.log1p(HISTORY_HOURS), np.log1p(gaps) / np.log1p(HISTORY_HOURS), previous],
        axis=-1,
    ).astype(np.float32)
    numeric[~valid] = 0
    return categories, numeric, lengths


def build(folder=FOLDER):
    paths = [PROCESSED / "channel-novelty-v18" / f"onsets-{year}.parquet" for year in range(2022, 2027)]
    catalog_path = PROCESSED / "channels.parquet"
    inputs = {catalog_path, *paths}
    for year in range(2022, 2027):
        metadata = PROCESSED / "channel-novelty-v18" / f"onsets-{year}.json"
        inputs.add(metadata)
        for category in ("inputs", "outputs"):
            for path, digest in read(metadata)[category].items():
                if sha256(Path(path)) != digest:
                    raise ValueError(f"Changed raw onset provenance: {path}")
                inputs.add(Path(path))
    hashes = {
        "inputs": {str(p): sha256(p) for p in sorted(inputs)},
        "code_hashes": {str(Path(__file__)): sha256(Path(__file__))},
    }
    manifest = folder / "build.json"
    if manifest.exists():
        old = read(manifest)
        if any(old[k] != v for k, v in hashes.items()) or any(
            sha256(Path(p)) != v for p, v in old["outputs"].items()
        ):
            raise ValueError("Event sequence source build changed")
        return old
    events = prepare_events(
        pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True), pd.read_parquet(catalog_path)
    )
    if not events.ts.lt(pd.Timestamp("2026-06-01")).all():
        raise ValueError("Post-May event source")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "events.parquet"
    events.to_parquet(target, index=False, compression="zstd")
    meta = {
        **hashes,
        "rows": len(events),
        "objects": int(events.object_id.nunique()),
        "channels": int(events.channel_id.nunique()),
        "before": "2026-06-01",
        "outputs": {str(target): sha256(target)},
    }
    write_json(manifest, meta)
    return meta


if __name__ == "__main__":
    print(build(), flush=True)
