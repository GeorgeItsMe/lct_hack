"""Causal minute-trigger augmentation with preserved original snapshot mass."""

import numpy as np
import pandas as pd

from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.onset_channel_features import CATS, COLUMNS, KEYS
from moscollector.train import CATEGORICAL


def augmented_slots(base, triggers, spacing_hours=3, retain_quarters=False):
    """Offline coverage already filtered by split; never bridge a missing anchor."""
    if spacing_hours not in (1, 3) or retain_quarters and spacing_hours != 1:
        raise ValueError("Unsupported source spacing")
    hours = base[KEYS].copy()
    if hours.duplicated(KEYS).any() or hours.isna().any().any():
        raise ValueError("Duplicate or missing anchors")
    if not hours.as_of.eq(hours.as_of.dt.floor(f"{spacing_hours}h")).all():
        raise ValueError("Misaligned anchors")
    t = triggers[KEYS].copy()
    if t.duplicated(KEYS).any() or not t.as_of.eq(t.as_of.dt.floor("min")).all():
        raise ValueError("Invalid minute triggers")
    t["source_time"] = t.as_of.dt.floor(f"{spacing_hours}h")
    starts = hours.rename(columns={"as_of": "source_time"})
    successors = hours.assign(as_of=hours.as_of - pd.Timedelta(hours=spacing_hours)).rename(
        columns={"as_of": "source_time"}
    )
    intervals = starts.merge(successors, on=["object_id", "source_time"], validate="one_to_one")
    inside = t.loc[t.as_of.ne(t.source_time)].merge(
        intervals, on=["object_id", "source_time"], how="inner", validate="many_to_one"
    )
    anchors = subdivide_evaluation_slots(hours) if retain_quarters else hours
    out = (
        pd.concat([anchors, inside[KEYS]], ignore_index=True)
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    out["source_time"] = out.as_of.dt.floor(f"{spacing_hours}h")
    if out.as_of.lt(out.source_time).any():
        raise ValueError("Noncausal source time")
    return out


def snapshot_weights(slots):
    """One unit per original anchor: half anchor, half shared companions if any."""
    if slots.duplicated(KEYS).any():
        raise ValueError("Duplicate augmented row")
    anchor = slots.as_of.eq(slots.source_time)
    groups = slots.groupby(["object_id", "source_time"], sort=False)
    size = groups.as_of.transform("size").to_numpy()
    anchor_counts = slots.assign(is_anchor=anchor).groupby(["object_id", "source_time"]).is_anchor.sum()
    if not anchor_counts.eq(1).all():
        raise ValueError("Each augmented group must retain exactly one anchor")
    weights = np.where(anchor, np.where(size > 1, 0.5, 1), 0.5 / np.maximum(size - 1, 1))
    totals = slots.assign(weight=weights).groupby(["object_id", "source_time"]).weight.sum().to_numpy()
    np.testing.assert_allclose(totals, 1, rtol=0, atol=1e-12)
    return weights


def attach(base, slots, context, columns):
    """Original inputs remain held at source_time; fresh channel context is added."""
    fields = list(dict.fromkeys(["object_id", "as_of", *columns]))
    held = slots.merge(
        base[fields].rename(columns={"as_of": "source_time"}),
        on=["object_id", "source_time"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not held._merge.eq("both").all():
        raise ValueError("Missing source snapshot")
    held = held.drop(columns="_merge").merge(
        context[[*KEYS, *COLUMNS]], on=KEYS, how="left", validate="one_to_one", indicator=True
    )
    if not held._merge.eq("both").all():
        raise ValueError("Missing causal onset context")
    held = held.drop(columns="_merge")
    held["evt_source_age_h"] = (held.as_of - held.source_time).dt.total_seconds().astype(np.float32) / 3600
    if held.evt_source_age_h.lt(0).any() or held.evt_source_age_h.ge(3).any():
        raise ValueError("Stale or future original inputs")
    return held


def model_input(frame, columns):
    if any(
        name.startswith("target_")
        or name in ("as_of", "eligible", "source_time", "training_count", "training_weight")
        for name in columns
    ):
        raise ValueError("Forbidden predictor input")
    x = frame[columns].copy()
    for name in (*CATEGORICAL, *CATS):
        if name in x:
            x[name] = x[name].astype(object).fillna("__NONE__").astype(str)
    return x
