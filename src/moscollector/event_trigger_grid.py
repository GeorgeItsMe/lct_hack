"""Strictly delayed onset-triggered opportunities inside fixed observed hours."""

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS

KEYS = ["object_id", "as_of"]


def trigger_grid(hourly, onsets, seconds=60, signals=("fire",)):
    """Retain every original hour; add next-boundary onsets between adjacent hours.

    Hourly input is an offline evaluation-coverage mask, never a runtime input.
    No labels, future signal rows, inferred episode membership, or future
    channel inventory determine an onset's release time or trigger strength.
    A report exactly on a boundary is released at the *next* boundary too.
    """
    if not isinstance(seconds, int) or seconds <= 0 or 3600 % seconds:
        raise ValueError("Resolution must be a positive integer divisor of one hour")
    if not signals or len(set(signals)) != len(signals):
        raise ValueError("Expected distinct trigger signals")
    hours = hourly[KEYS].copy()
    if hours.isna().any().any() or hours.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate source hours")
    if not hours.as_of.eq(hours.as_of.dt.floor("h")).all():
        raise ValueError("Source forecasts must be whole hours")
    selected = onsets.loc[onsets.signal.isin(signals), ["object_id", "ts"]].copy()
    if selected.isna().any().any():
        raise ValueError("Missing trigger identity or time")
    step = seconds * 1_000_000_000
    raw = selected.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    selected["as_of"] = pd.to_datetime((raw // step + 1) * step)
    assert selected.as_of.gt(selected.ts).all()
    counts = selected.groupby(KEYS, as_index=False).agg(
        trigger_count=("ts", "size"), latest_onset=("ts", "max")
    )
    counts["source_time"] = counts.as_of.dt.floor("h")
    # A final isolated original hour remains a forecast but cannot authorize
    # additional slots after it, which could enlarge the eligible event cohort.
    coverage = hours.rename(columns={"as_of": "source_time"})
    next_hours = hours.assign(as_of=hours.as_of - pd.Timedelta(hours=1)).rename(
        columns={"as_of": "source_time"}
    )
    intervals = coverage.merge(next_hours, on=["object_id", "source_time"], validate="one_to_one")
    between = counts.loc[counts.as_of.ne(counts.source_time)].merge(
        intervals, on=["object_id", "source_time"], how="inner", validate="many_to_one"
    )
    exact = counts.loc[counts.as_of.eq(counts.source_time)].merge(
        coverage, on=["object_id", "source_time"], how="inner", validate="one_to_one"
    )
    observed = pd.concat([between, exact], ignore_index=True)
    slots = pd.concat([hours, observed[KEYS]], ignore_index=True).drop_duplicates(KEYS)
    slots = slots.merge(
        observed[KEYS + ["trigger_count", "latest_onset"]], on=KEYS, how="left", validate="one_to_one"
    )
    slots["trigger_count"] = slots.trigger_count.fillna(0).astype(np.int32)
    slots["source_time"] = slots.as_of.dt.floor("h")
    slots = slots.sort_values(KEYS).reset_index(drop=True)
    assert (
        slots.loc[slots.trigger_count.gt(0)].latest_onset.lt(slots.loc[slots.trigger_count.gt(0)].as_of).all()
    )
    return slots


def trigger_alerts(slots, cooldown_hours):
    """Pure past-warning cooldown; no event labels or event confirmations."""
    if cooldown_hours < 0 or not np.isfinite(cooldown_hours):
        raise ValueError("Invalid cooldown")
    if (
        slots.duplicated(KEYS).any()
        or slots[KEYS].isna().any().any()
        or not np.isfinite(slots.trigger_count).all()
        or slots.trigger_count.lt(0).any()
    ):
        raise ValueError("Invalid trigger slots")
    frame = slots.reset_index(drop=True)
    result = np.zeros(len(frame), dtype=bool)
    wait = int(cooldown_hours * HOUR_NS)
    for _, rows in frame.groupby("object_id", sort=False):
        previous = None
        for row in rows.loc[rows.trigger_count.gt(0)].sort_values("as_of").itertuples():
            at = row.as_of.value
            if previous is None or at - previous >= wait:
                result[row.Index] = True
                previous = at
    return result
