"""Causal multi-resolution telemetry and recurrent-event features.

Inspired by ElasticPdM's unequal temporal bins, without copying its network or
using pretrained weights. Every bin is closed on the left and open at as_of.
Episode history is delayed by 70 minutes exactly as in the original feature set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.features import COUNT_COLUMNS, KINDS
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

BINS = ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 168), (168, 336), (336, 720))
SIGNALS = (
    "fault_reports",
    "power_reports",
    "unknown_reports",
    "smoke_reports",
    "access_reports",
    "flood_reports",
    "technical_codes",
)
HOUR_NS = 3_600_000_000_000


def recurrence_features(times: pd.DatetimeIndex, episodes: pd.DataFrame) -> pd.DataFrame:
    at = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    result = {}
    for kind in KINDS.values():
        ep = episodes[episodes.kind.eq(kind)].sort_values("start_ts")
        starts = ep.start_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        available = starts + 70 * 60_000_000_000
        right = np.searchsorted(available, at, side="left")
        for hours in (6, 48, 72, 336, 1440, 4320):
            left = np.searchsorted(available, at - hours * HOUR_NS, side="left")
            result[f"seq_{kind}_episodes_{hours}h"] = right - left
        if not len(available):
            for col in (
                "gap_last_h",
                "gap_mean5_h",
                "gap_mean20_h",
                "gap_cv20",
                "age_over_gap5",
                "age_over_gap20",
                "active_confirmed",
                "last_size",
            ):
                result[f"seq_{kind}_{col}"] = np.full(len(at), np.nan)
            continue
        gaps = np.r_[np.nan, np.diff(available) / HOUR_NS]
        gap_series = pd.Series(gaps)
        last_index = np.maximum(right - 1, 0)
        age = (at - available[last_index]) / HOUR_NS
        for n in (5, 20):
            mean = gap_series.rolling(n, min_periods=1).mean().to_numpy()[last_index]
            mean[right < 2] = np.nan
            result[f"seq_{kind}_gap_mean{n}_h"] = mean
            result[f"seq_{kind}_age_over_gap{n}"] = age / np.maximum(mean, 1)
        last_gap = gaps[last_index].copy()
        last_gap[right < 2] = np.nan
        result[f"seq_{kind}_gap_last_h"] = last_gap
        cv = (
            gap_series.rolling(20, min_periods=3).std()
            / gap_series.rolling(20, min_periods=3).mean().clip(lower=1)
        ).to_numpy()[last_index]
        cv[right < 4] = np.nan
        result[f"seq_{kind}_gap_cv20"] = cv
        size = ep.channel_count.to_numpy(dtype=float)[last_index]
        size[right == 0] = np.nan
        result[f"seq_{kind}_last_size"] = size
        # An end is available only when both the episode and its end are observed.
        # Comparing end timestamps to t does not expose their future duration.
        # group_episodes retains max(non-null end) for display even when another
        # member remains right-censored. Such a group has no observed completion.
        # Censoring is used only to normalize the end representation, never as a
        # predictive feature. This matches an online group with unknown ends.
        censored = ep.get("right_censored", pd.Series(False, index=ep.index)).to_numpy()
        ended = ep.end_ts.notna().to_numpy() & ~censored
        ends = ep.end_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ends = np.maximum(ends[ended], available[ended])
        ended_counts = np.searchsorted(np.sort(ends), at, side="left")
        result[f"seq_{kind}_active_confirmed"] = right - ended_counts
    return pd.DataFrame(result, index=times).astype(np.float32)


def object_sequence_features(times, hourly, episodes):
    """Pure transform for one object; future observations cannot affect a row."""
    times = pd.DatetimeIndex(times)
    index = pd.date_range(times.min() - pd.Timedelta(hours=720), times.max(), freq="h")
    history = hourly.set_index("hour").reindex(index)
    count = history[COUNT_COLUMNS].fillna(0)
    lagged = count.shift(1)
    select = index.get_indexer(times)
    if np.any(select < 0):
        raise ValueError("Forecasts must be on the hourly grid")
    extra = {}
    for low, high in BINS:
        value = lagged.shift(low).rolling(high - low, min_periods=high - low).sum()
        for col in COUNT_COLUMNS:
            extra[f"seq_{col}_bin_{low}_{high}h"] = value[col].to_numpy()[select] / (high - low)
    for window in (24, 168):
        for col in SIGNALS:
            extra[f"seq_{col}_max_{window}h"] = lagged[col].rolling(window).max().to_numpy()[select]
            extra[f"seq_{col}_active_hours_{window}h"] = (
                lagged[col].gt(0).rolling(window).sum().to_numpy()[select]
            )
    # Smooth sensor summaries causally; keep missing values for unseen sensors.
    for col in ("temperature", "temperature_max", "gas"):
        values = history[col].shift(1)
        for window in (24, 168):
            roll = values.rolling(window, min_periods=2)
            for stat in ("mean", "std", "min", "max"):
                extra[f"seq_{col}_{stat}_{window}h"] = getattr(roll, stat)().to_numpy()[select]
            extra[f"seq_{col}_observed_{window}h"] = values.rolling(window).count().to_numpy()[select]
    # Recurring daily schedules are learned from *past* same-hour observations.
    for col in ("events", "alarms", "access_reports", "smoke_reports", "fault_reports"):
        daily = lagged[col].rolling(24).sum()
        extra[f"seq_{col}_yesterday24h"] = daily.shift(24).to_numpy()[select]
        extra[f"seq_{col}_lastweek24h"] = daily.shift(168).to_numpy()[select]
        extra[f"seq_{col}_weekbefore24h"] = daily.shift(336).to_numpy()[select]
    telemetry = pd.DataFrame(extra, index=times).astype(np.float32)
    recurrence = recurrence_features(times, episodes)
    return pd.concat([telemetry, recurrence], axis=1)


def build(output: Path, before="2026-06-01"):
    features = pd.read_parquet(PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp(before))])
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp(before))]
    )
    years = sorted(features.as_of.dt.year.unique())
    hourly = pd.concat(
        [
            pd.read_parquet(PROCESSED / f"hourly-{y}.parquet", filters=[("hour", "<", pd.Timestamp(before))])
            for y in years
        ]
    )
    extras = []
    for obj, rows in features.groupby("object_id", sort=True):
        part = object_sequence_features(
            rows.as_of, hourly[hourly.object_id.eq(obj)], episodes[episodes.object_id.eq(obj)]
        )
        part.index = rows.index
        extras.append(part)
        print(f"sequence object={obj}, rows={len(rows)}, new_features={part.shape[1]}", flush=True)
    enriched = pd.concat([features, pd.concat(extras).reindex(features.index)], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False, compression="zstd")
    write_json(
        output.with_suffix(".json"),
        {
            "source_sha256": sha256(PROCESSED / "features.parquet"),
            "sha256": sha256(output),
            "before": before,
            "rows": len(enriched),
            "columns": list(enriched.columns),
            "causality": "hourly bins end strictly before as_of; episodes available strictly after start+70m",
            "sources": [str(PROCESSED / f"hourly-{y}.parquet") for y in years],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/features-sequence.parquet"))
    parser.add_argument("--before", default="2026-06-01")
    args = parser.parse_args()
    build(args.output, args.before)
