"""Audit raw-sequence capacity on the oldest screening training periods only."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.experiments.channel_novelty_features import SIGNALS
from moscollector.experiments.event_sequence_data import (
    FOLDER,
    HISTORY_HOURS,
    PER_SIGNAL,
    encode_events,
    event_batch,
    fit_event_codec,
    history_bounds,
)
from moscollector.experiments.fresh_counts_research import source_files
from moscollector.experiments.goal90_research import read
from moscollector.experiments.onset_channel_features import FOLDER as ONSET_FOLDER
from moscollector.experiments.onset_training_data import augmented_slots
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import mask

build = read(FOLDER / "build.json")
for category in ("inputs", "outputs", "code_hashes"):
    for path, digest in build[category].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Changed raw-sequence build: {path}")
event_path, frame_path, trigger_path = (
    FOLDER / "events.parquet",
    PROCESSED / "features-channel-novelty.parquet",
    ONSET_FOLDER / "triggers.parquet",
)
events = pd.read_parquet(event_path, read_dictionary=["signal", "sensor_type"])
base = pd.read_parquet(frame_path, columns=["object_id", "as_of", "eligible"])
triggers = pd.read_parquet(trigger_path)
times = events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
by_object = {obj: group for obj, group in events.groupby("object_id", sort=False)}
sources = {FOLDER / "build.json", event_path, frame_path, trigger_path}
profiles, cache = {}, {}
for kind in ("access", "fire", "fault"):
    metadata = source_files(kind, "screen_1")[1]
    sources.add(metadata)
    dates = tuple(map(pd.Timestamp, read(metadata)["periods"]["train"]))
    if dates in cache:
        profiles[kind] = cache[dates]
        continue
    anchors = base.loc[mask(base, *dates)]
    queries = augmented_slots(anchors, triggers)
    bounds = history_bounds(events, queries)
    counts = bounds[..., 1].astype(np.int64) - bounds[..., 0]
    retained = np.minimum(counts, PER_SIGNAL)
    begin = np.maximum(bounds[..., 0], bounds[..., 1] - PER_SIGNAL)
    chosen_times = times[np.minimum(begin, len(events) - 1)]
    earliest = np.where(counts > 0, chosen_times, np.iinfo(np.int64).max).min(axis=1)
    at = queries.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    valid = retained.sum(axis=1) > 0
    ages = (at[valid] - earliest[valid]) / HOUR_NS
    cut = counts > PER_SIGNAL
    tied_cut = np.zeros(cut.shape, dtype=bool)
    tied_cut[cut] = times[begin[cut]] == times[begin[cut] - 1]
    codec = fit_event_codec(events, bounds)
    encoded = encode_events(events, codec)
    checked = 0
    flat_ages, flat_ties = [], 0
    for obj, rows in queries.groupby("object_id", sort=False):
        history = by_object[obj]
        t = np.sort(history.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        query_times = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        end = np.searchsorted(t, query_times, side="left")
        start = np.searchsorted(t, query_times - HISTORY_HOURS * HOUR_NS, side="left")
        count = end - start
        first = np.maximum(start, end - 64)
        available = count > 0
        flat_ages.extend(((query_times[available] - t[first[available]]) / HOUR_NS).tolist())
        overflow = count > 64
        flat_ties += int(np.sum(t[first[overflow]] == t[first[overflow] - 1]))
        ids = rows.index.to_numpy()[np.unique(np.linspace(0, len(rows) - 1, min(4, len(rows)), dtype=int))]
        for i in ids:
            now = queries.as_of.iloc[i]
            past = history.loc[history.ts.lt(now) & history.ts.ge(now - pd.Timedelta(hours=HISTORY_HOURS))]
            expected = (
                past.groupby("signal_code", observed=True, sort=True)
                .tail(PER_SIGNAL)
                .sort_values(["ts", "signal_code", "channel_id"], kind="stable")
            )
            cats, values, lengths = event_batch(encoded, times, bounds[i : i + 1], at[i : i + 1])
            assert lengths[0] == len(expected)
            np.testing.assert_array_equal(cats[0, : lengths[0]], encoded[expected.index])
            assert not cats[0, lengths[0] :].any() and not values[0, lengths[0] :].any()
            checked += 1
    profile = {
        "train_dates": [str(d) for d in dates],
        "original_anchors": len(anchors),
        "queries": len(queries),
        "no_history_fraction": float(np.mean(~valid)),
        "flat64_truncated_fraction": float(np.mean(counts.sum(axis=1) > 64)),
        "flat64_oldest_age_hours_quantiles_10_50_90": np.quantile(flat_ages, [0.1, 0.5, 0.9]).tolist(),
        "flat64_tied_cut_queries": flat_ties,
        "family16_oldest_age_hours_quantiles_10_50_90": np.quantile(ages, [0.1, 0.5, 0.9]).tolist(),
        "family16_retained_length_quantiles_50_90_99_max": np.quantile(
            retained.sum(axis=1), [0.5, 0.9, 0.99, 1]
        ).tolist(),
        "family16_any_tied_cut_queries": int(tied_cut.any(axis=1).sum()),
        "signal_presence_fraction": {
            name: float(np.mean(counts[:, i] > 0)) for i, name in enumerate(SIGNALS)
        },
        "training_vocabulary_sizes": {name: len(v) for name, v in codec["vocabulary"].items()},
        "referenced_training_onsets": codec["training_onset_rows"],
        "direct_sample_queries_checked": checked,
        "bounds_memory_bytes": bounds.nbytes,
    }
    cache[dates] = profile
    profiles[kind] = profile
    print(kind, profile, flush=True)
audit = {
    "scope": "Unlabelled input capacity audit on original oldest-screen training periods only. No event outcomes or test-month scores select sequence length. Fixed48h window,16 most recent onsets per each of6families; complete48h family counts supplied to both networks. Tied cutoffs and truncation remain explicitly limited; stable tie order is not physical causal order. Raw records are not independent incidents.",
    "profiles": profiles,
    "source_rows": len(events),
    "source_objects": int(events.object_id.nunique()),
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(history_bounds.__code__.co_filename),
            Path(augmented_slots.__code__.co_filename),
        )
    },
}
write_json(Path("artifacts/research-v37/input-audit.json"), audit)
