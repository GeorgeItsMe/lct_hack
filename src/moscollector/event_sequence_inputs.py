"""Shared causal current inputs and weighted targets for both event networks."""

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.event_sequence_data import HISTORY_HOURS, KEYS, history_bounds
from moscollector.fine_cadence_research import cohort
from moscollector.neural_sequence_data import encode
from moscollector.onset_training_data import augmented_slots, snapshot_weights
from moscollector.research import mask


def held_inputs(base, slots, columns):
    if any(c in ("as_of", "source_time", "eligible") or c.startswith("target_") for c in columns):
        raise ValueError("Forbidden current-network feature")
    names = list(dict.fromkeys([*KEYS, *columns]))
    held = slots.merge(
        base[names].rename(columns={"as_of": "source_time"}),
        on=["object_id", "source_time"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not held._merge.eq("both").all():
        raise ValueError("Missing original source snapshot")
    pd.testing.assert_frame_equal(held[KEYS], slots[KEYS])
    age = (held.as_of - held.source_time).dt.total_seconds().to_numpy() / 3600
    if np.any(age < 0) or np.any(age >= 3):
        raise ValueError("Stale or future event-network source")
    return held.drop(columns="_merge"), age


def input_arrays(base, slots, events, codec):
    rows, age = held_inputs(base, slots, codec["columns"])
    numeric, categorical = encode(rows, codec)
    bounds = history_bounds(events, slots)
    full_counts = bounds[..., 1].astype(np.int64) - bounds[..., 0]
    # Fixed units, fitted on no outcomes. Both networks receive these complete
    # counts even if only the sequence model receives the retained event tokens.
    extra = np.column_stack([np.log1p(age) / np.log(4), np.log1p(full_counts) / np.log(1001)])
    return {
        "numeric": np.concatenate([numeric, extra], axis=1).astype(np.float32),
        "categorical": categorical,
        "bounds": bounds,
        "query_times": slots.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    }


def period_rows(frame, dense, triggers, periods, part):
    dates = tuple(map(pd.Timestamp, periods[part]))
    reference = frame.loc[mask(frame, *dates)]
    fitting = part in ("train", "validation")
    base = reference if fitting else align_opportunities(dense.loc[mask(dense, *dates)], reference[KEYS])
    slots = augmented_slots(base, triggers, spacing_hours=3 if fitting else 1, retain_quarters=not fitting)
    if not (slots.as_of + pd.Timedelta(hours=25)).lt(dates[1]).all():
        raise ValueError("Event-network query crosses split purge")
    return base, slots


def training_part(base, slots, events, codec, episodes, kind):
    original = episode_counts(base, episodes)
    if not np.array_equal(original > 0, base[f"target_{kind}"].to_numpy().astype(bool)):
        raise ValueError("Original neural episode targets changed")
    data = input_arrays(base, slots, events, codec)
    counts, weight = episode_counts(slots, episodes), snapshot_weights(slots)
    np.testing.assert_allclose(weight.sum(), len(base), rtol=0, atol=1e-7)
    if cohort(slots, episodes, 1 / 60) != cohort(base, episodes, 3):
        raise ValueError("Event-network augmentation changed episode identities")
    data.update(target=counts.astype(np.float32), weight=weight)
    sizes = {
        "rows": len(slots),
        "original_rows": len(base),
        "weight_sum": float(weight.sum()),
        "positive_rows": int(np.sum(counts > 0)),
        "weighted_positive_rows": float(weight[counts > 0].sum()),
        "eligible_episodes": EventEvaluator(base, episodes, 3).events,
        "numeric_dim": data["numeric"].shape[1],
        "history_hours": HISTORY_HOURS,
        "queries_without_onsets": int(
            np.sum(np.sum(data["bounds"][..., 1] - data["bounds"][..., 0], axis=1) == 0)
        ),
    }
    return data, sizes
