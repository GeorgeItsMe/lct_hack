"""Causal within-hour training snapshots; original labels/data stay frozen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.features import COUNT_COLUMNS
from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import connection, sha256, write_json
from moscollector.quarter_count_features import (
    BURST_SIGNALS,
    UPDATED_COLUMNS,
    aggregate_counts,
    refresh_counts,
)

BEFORE = "2026-06-01"


def extract(year, folder):
    if year == 2026:
        existing = PROCESSED / "quarter-counts-v21" / "counts-2026.json"
        meta = read(existing)
        for category in ("inputs", "outputs"):
            for path, digest in meta[category].items():
                if sha256(Path(path)) != digest:
                    raise ValueError("Original2026 quarter cache changed")
        return PROCESSED / "quarter-counts-v21" / "counts-2026.parquet"
    target, metadata = folder / f"counts-{year}.parquet", folder / f"counts-{year}.json"
    raw, catalog, hourly = (
        PROCESSED / name for name in (f"events-{year}.parquet", "channels.parquet", f"hourly-{year}.parquet")
    )
    inputs = {
        str(p): sha256(p)
        for p in (raw, catalog, hourly, Path(__file__), Path(aggregate_counts.__code__.co_filename))
    }
    if metadata.exists():
        meta = read(metadata)
        if meta["inputs"] != inputs or meta["outputs"] != {str(target): sha256(target)}:
            raise ValueError("Training quarter-count cache changed")
        return target
    folder.mkdir(parents=True, exist_ok=True)
    con = connection()
    con.read_parquet(str(raw)).filter(f"ts < TIMESTAMP '{BEFORE}'").create_view("source")
    con.read_parquet(str(catalog)).create_view("channels")
    aggregate_counts(con).write_parquet(str(target), compression="zstd")
    con.close()
    frame = pd.read_parquet(target)
    original = pd.read_parquet(hourly, columns=["object_id", "hour", *COUNT_COLUMNS])
    actual = (
        frame.assign(hour=frame.bucket.dt.floor("h"))
        .groupby(["object_id", "hour"], as_index=False)[COUNT_COLUMNS]
        .sum(min_count=1)
    )
    keys = ["object_id", "hour"]
    actual, original = (f.sort_values(keys).reset_index(drop=True) for f in (actual, original))
    pd.testing.assert_frame_equal(actual[keys], original[keys])
    np.testing.assert_array_equal(actual[COUNT_COLUMNS].to_numpy(), original[COUNT_COLUMNS].to_numpy())
    write_json(
        metadata,
        {
            "inputs": inputs,
            "outputs": {str(target): sha256(target)},
            "year": year,
            "rows": len(frame),
            "canonical_reports": int(frame.events.sum()),
            "whole_hour_rows_exact": len(original),
        },
    )
    print("Training count cache", year, len(frame), "hourly parity", len(original), flush=True)
    return target


def shifted_snapshots(frame):
    """One deterministic15/30/45min companion per observed original3h row.

    Requiring the following eligible3h point is an OFFLINE target-coverage
    condition, never a model input or runtime scheduling decision. No phase is
    selected using outcomes. Existing target columns are deliberately removed.
    """
    if frame.duplicated(["object_id", "as_of"]).any():
        raise ValueError("Duplicate source opportunities")
    original = frame[frame.eligible].sort_values(["object_id", "as_of"]).copy()
    following = original.groupby("object_id").as_of.shift(-1)
    selected = original[(following - original.as_of).eq(pd.Timedelta(hours=3))].copy()
    if not selected.as_of.eq(selected.as_of.dt.floor("h")).all():
        raise ValueError("Expected whole-hour original timestamps")
    selected["source_time"] = selected.as_of
    hours = selected.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64) // HOUR_NS
    phase = ((hours // 3 + selected.object_id.to_numpy(dtype=np.int64)) % 3 + 1) * 15
    selected["as_of"] += pd.to_timedelta(phase, unit="m")
    selected = selected.drop(columns=[name for name in selected if name.startswith("target_")])
    return selected.sort_values(["as_of", "object_id"]).reset_index(drop=True)


def paired_weights(base, shifted):
    """Preserve total training weight one per original source snapshot."""
    keys = ["object_id", "as_of"]
    if base.duplicated(keys).any() or shifted.duplicated(["object_id", "source_time"]).any():
        raise ValueError("Need one original and at most one companion")
    companions = shifted[["object_id", "source_time"]].rename(columns={"source_time": "as_of"})
    mapped = base[keys].merge(
        companions.assign(has_companion=True), on=keys, how="left", validate="one_to_one"
    )
    if mapped.has_companion.notna().sum() != len(shifted):
        raise ValueError("Companion crosses the selected temporal split")
    weights = np.where(mapped.has_companion.notna(), 0.5, 1.0)
    extra_weights = np.full(len(shifted), 0.5)
    assert weights.sum() + extra_weights.sum() == len(base)
    return weights, extra_weights


def build(folder):
    paths = [extract(year, folder) for year in range(2022, 2027)]
    counts = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    source = PROCESSED / "features-channel-novelty.parquet"
    frame = pd.read_parquet(source)
    assert frame.as_of.lt(pd.Timestamp(BEFORE)).all()
    checked_columns = [*UPDATED_COLUMNS, *(f"{s}_burst" for s in BURST_SIGNALS)]
    base = frame[frame.eligible][["object_id", "as_of", *checked_columns]].reset_index(drop=True)
    pd.testing.assert_frame_equal(base, refresh_counts(base, counts))
    shifted = shifted_snapshots(frame)
    shifted = refresh_counts(shifted, counts)
    assert shifted.as_of.dt.floor("h").eq(shifted.source_time).all()
    assert not any(name.startswith("target_") for name in shifted)
    target = folder / "shifted-features.parquet"
    shifted.to_parquet(target, index=False, compression="zstd")
    manifests = [folder / f"counts-{year}.json" for year in range(2022, 2026)] + [
        PROCESSED / "quarter-counts-v21" / "counts-2026.json"
    ]
    inputs = {
        str(p): sha256(p)
        for p in [source, *paths, *manifests, Path(__file__), Path(refresh_counts.__code__.co_filename)]
    }
    offsets = (
        ((shifted.as_of - shifted.source_time).dt.total_seconds() / 60)
        .astype(int)
        .value_counts()
        .sort_index()
    )
    write_json(
        folder / "build.json",
        {
            "inputs": inputs,
            "outputs": {str(target): sha256(target)},
            "source_valid_rows": len(base),
            "original_count_feature_parity": True,
            "checked_columns": checked_columns,
            "shifted_rows": len(shifted),
            "phase_minutes": offsets.to_dict(),
            "labels": "Original target columns removed; recompute24h counts at each shifted time inside each fit.",
            "weighting": "Half weight to original and companion, weight1 to unpaired original; split-specific purge precedes pairing.",
            "coverage_mask_is_offline_only": True,
            "before": BEFORE,
        },
    )
    print("Saved shifted training features", len(shifted), offsets.to_dict(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=PROCESSED / "phase-augmentation-v23")
    build(parser.parse_args().folder)
