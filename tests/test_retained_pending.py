from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from moscollector.fine_cadence_research import FinePendingSimulator, evaluator_for
from moscollector.retained_pending import RetainedPendingSimulator, select_retained


def inputs(hours, events, capacity=1, probability=1):
    start = pd.Timestamp("2025-01-01")
    rows = pd.DataFrame(
        {
            "object_id": 1,
            "as_of": start + pd.to_timedelta(hours, unit="h"),
            "expected_count": capacity,
            "probability": probability,
        }
    )
    episodes = pd.DataFrame({"object_id": 1, "start_ts": start + pd.to_timedelta(events, unit="h")})
    return rows, episodes


def direct(rows, episodes, capacity, probability, margin, floor):
    """Independent full-history reference: no pruning or skipped ticks."""
    flags = np.zeros(len(rows))
    for obj, group in rows.reset_index(drop=True).groupby("object_id"):
        events = sorted(t.to_pydatetime() for t in episodes.loc[episodes.object_id.eq(obj), "start_ts"])
        history, consumed, seen = [], set(), 0
        for i, row in group.sort_values("as_of").iterrows():
            now = row.as_of.to_pydatetime()
            while (
                seen < len(events)
                and (events[seen] + timedelta(minutes=70)).replace(minute=0, second=0, microsecond=0)
                + timedelta(hours=1)
                <= now
            ):
                event = events[seen]
                for j, issued in enumerate(history):
                    if j not in consumed and issued <= event < issued + timedelta(hours=24):
                        consumed.add(j)
                        break
                seen += 1
            active = sum(
                j not in consumed and now - issued < timedelta(hours=24) for j, issued in enumerate(history)
            )
            if probability[i] >= floor and capacity[i] >= active + margin:
                flags[i] = 1
                history.append(now)
    return flags


def test_late_confirmation_keeps_younger_warning_reserved():
    rows, eps = inputs([0, 2, 23, 24, 25], [23.5, 25.5], [1, 2, 2, 1, 1])
    args = (rows.expected_count, rows.probability, 1, 0)
    np.testing.assert_array_equal(FinePendingSimulator(rows, eps).alerts(*args), [1, 1, 0, 0, 1])
    np.testing.assert_array_equal(RetainedPendingSimulator(rows, eps).alerts(*args), [1, 1, 0, 0, 0])


def test_retired_record_does_not_consume_active_capacity():
    rows, eps = inputs([0, 24], [23.5])
    np.testing.assert_array_equal(
        RetainedPendingSimulator(rows, eps).alerts(rows.expected_count, rows.probability, 1, 0), [1, 1]
    )


def test_long_gap_resolves_all_confirmations_before_discarding_old_records():
    rows, eps = inputs([0, 2, 24, 100, 101], [23.5, 25.5, 99, 100], [1, 2, 2, 1, 1])
    args = (rows.expected_count.to_numpy(), rows.probability.to_numpy(), 1, 0)
    np.testing.assert_array_equal(RetainedPendingSimulator(rows, eps).alerts(*args), direct(rows, eps, *args))


def test_horizon_is_strict_at_24h_and_duplicate_events_resolve_separately():
    rows, eps = inputs([0, 0.25, 2, 3, 24, 25, 26], [1, 1, 24], [1, 2, 2, 2, 2, 2, 2])
    args = (rows.expected_count.to_numpy(), rows.probability.to_numpy(), 1, 0)
    np.testing.assert_array_equal(RetainedPendingSimulator(rows, eps).alerts(*args), direct(rows, eps, *args))


def test_unreleased_future_events_and_future_scores_do_not_change_prefix():
    rows, eps = inputs(np.arange(0, 80, 0.25), [23.5, 25.5, 40, 70], 1.5)
    changed = eps.copy()
    changed["start_ts"] = [
        t + timedelta(hours=8) if t > pd.Timestamp("2025-01-02 01:00") else t for t in changed.start_ts
    ]
    original = RetainedPendingSimulator(rows, eps).alerts(rows.expected_count, rows.probability, 0.5, 0)
    cap = rows.expected_count.to_numpy().copy()
    cap[rows.as_of > pd.Timestamp("2025-01-02 01:00")] = 100
    other = RetainedPendingSimulator(rows, changed).alerts(cap, rows.probability, 0.5, 0)
    np.testing.assert_array_equal(original[:101], other[:101])


def test_optimized_matches_unbounded_history_with_random_gaps_and_objects():
    rng = np.random.default_rng(714)
    for _ in range(5):
        parts, event_parts = [], []
        for obj in (1, 2):
            hours = np.sort(rng.choice(np.arange(0, 160, 0.25), size=130, replace=False))
            rows, eps = inputs(
                hours,
                np.sort(rng.choice(np.arange(160), size=55, replace=True)),
                rng.uniform(0, 8, len(hours)),
                rng.random(len(hours)),
            )
            parts.append(rows.assign(object_id=obj))
            event_parts.append(eps.assign(object_id=obj))
        rows, eps = pd.concat(parts, ignore_index=True), pd.concat(event_parts, ignore_index=True)
        rows = rows.sample(frac=1, random_state=7).reset_index(drop=True)
        for margin, floor in ((0.25, 0), (1, 0.5), (3, 0.99)):
            args = (rows.expected_count.to_numpy(), rows.probability.to_numpy(), margin, floor)
            np.testing.assert_array_equal(
                RetainedPendingSimulator(rows, eps).alerts(*args), direct(rows, eps, *args)
            )


@pytest.mark.parametrize("cadence,count", [(0.25, 456), (1, 120)])
def test_every_policy_grid_row_matches_direct_history(cadence, count):
    rng = np.random.default_rng(25)
    rows, eps = inputs(
        np.arange(0, 34, cadence),
        [1, 1, 12, 23.5, 25.5, 34],
        rng.uniform(0, 6, int(34 / cadence)),
        rng.random(int(34 / cadence)),
    )
    chosen, options = select_retained(rows, eps, cadence, 3)
    assert len(options) == count
    evaluator = evaluator_for(rows, eps, cadence, 3)
    for policy in options:
        flags = direct(
            rows,
            eps,
            rows.expected_count.to_numpy() * policy["capacity"],
            rows.probability.to_numpy(),
            policy["margin"],
            policy["floor"],
        )
        expected = evaluator.evaluate(flags, 0.5, cadence)
        assert all(policy[k] == v for k, v in expected.items())
    assert chosen["false_alerts_per_object_day"] <= 0.25


def test_no_positive_opportunities_and_empty_input_are_valid():
    for hours in ([0, 1], []):
        rows, eps = inputs(hours, [1], 0)
        chosen, options = select_retained(rows, eps, 1, 1)
        assert chosen["alerts"] == 0 and len(options) == 120


@pytest.mark.parametrize(
    "field,value",
    [("capacity", -1), ("capacity", np.nan), ("probability", 1.1), ("margin", np.nan), ("floor", np.inf)],
)
def test_invalid_values_are_rejected(field, value):
    rows, eps = inputs([0, 1], [1])
    args = dict(capacity=np.ones(2), probability=np.ones(2), margin=1, floor=0)
    args[field] = np.full(2, value) if field in ("capacity", "probability") else value
    with pytest.raises(ValueError, match="Invalid"):
        RetainedPendingSimulator(rows, eps).alerts(**args)
