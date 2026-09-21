import numpy as np
import pandas as pd
import pytest

from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import (
    FinePendingSimulator,
    cohort,
    evaluator_for,
    hold_hourly_scores,
)
from moscollector.goal90_research import PendingSimulator


def test_lagged_score_does_not_interpolate_future_or_cross_objects_and_gaps():
    origin = pd.Timestamp("2025-01-01")
    hourly = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 2],
            "as_of": [origin, origin + pd.Timedelta(hours=1), origin + pd.Timedelta(hours=4), origin],
            "raw": [2.0, 200.0, -5.0, -10.0],
        }
    )
    slots = subdivide_evaluation_slots(hourly)
    held = hold_hourly_scores(hourly.sample(frac=1, random_state=4), slots)
    assert held[held.object_id.eq(1)].raw.tolist() == [2, 2, 2, 2, 200, -5]
    assert held[held.object_id.eq(2)].raw.tolist() == [-10]
    assert held.source_age_minutes.max() == 45
    cutoff = origin + pd.Timedelta(minutes=45)
    past_slots = slots[slots.as_of.le(cutoff)]
    old = hold_hourly_scores(hourly, past_slots)
    changed = hourly.copy()
    changed.loc[changed.as_of.gt(cutoff), "raw"] = -1000
    pd.testing.assert_frame_equal(old, hold_hourly_scores(changed, past_slots))
    gap = pd.DataFrame({"object_id": [1], "as_of": [origin + pd.Timedelta(hours=2)]})
    with pytest.raises(ValueError, match="Stale"):
        hold_hourly_scores(hourly, gap)


def test_quarter_target_is_24_hours_from_new_time_and_exposure_is_unchanged():
    origin = pd.Timestamp("2025-01-01")
    hourly = pd.DataFrame(
        {"object_id": [1, 1], "as_of": [origin, origin + pd.Timedelta(hours=1)], "raw": [1.0, 3.0]}
    )
    quarter = hold_hourly_scores(hourly, subdivide_evaluation_slots(hourly))
    episodes = pd.DataFrame(
        {
            "object_id": [1, 1],
            "start_ts": [origin + pd.Timedelta(minutes=5), origin + pd.Timedelta(hours=24, minutes=10)],
        }
    )
    # The first event leaves and the next-day event enters the target. The
    # current raw input stays frozen, while the label uses the actual new time.
    assert episode_counts(quarter, episodes).tolist() == [1, 1, 1, 1, 1]
    assert episode_counts(quarter, episodes.iloc[:1]).tolist() == [1, 0, 0, 0, 0]
    assert episode_counts(quarter, episodes.iloc[1:]).tolist() == [0, 1, 1, 1, 1]
    assert cohort(quarter, episodes, 0.25) == cohort(hourly, episodes, 1)
    a = evaluator_for(hourly, episodes, 1, len(hourly) / 24)
    b = evaluator_for(quarter, episodes, 0.25, len(hourly) / 24)
    assert a.opportunity_days == b.opportunity_days


def test_confirmation_stays_on_original_hourly_information_boundary():
    origin = pd.Timestamp("2025-01-01")
    times = pd.date_range(origin, periods=13, freq="15min")
    frame = pd.DataFrame({"object_id": 1, "as_of": times})
    episodes = pd.DataFrame({"object_id": [1], "start_ts": [origin]})
    values = np.ones(len(frame))
    alerts = FinePendingSimulator(frame, episodes).alerts(values, values, 1, 0)
    assert frame.as_of[alerts.astype(bool)].tolist() == [origin, origin + pd.Timedelta(hours=2)]
    # Naively reusing strict70min at quarter points leaks the earlier
    # availability of confirmed labels; specifically prohibit that behavior.
    old = PendingSimulator(frame, episodes).alerts(values, values, 1, 0)
    assert old[5] == 1 and alerts[5] == 0
    at_boundary = episodes.copy()
    at_boundary["start_ts"] = origin + pd.Timedelta(minutes=50)
    alerts = FinePendingSimulator(frame, at_boundary).alerts(values, values, 1, 0)
    assert frame.as_of[alerts.astype(bool)].tolist() == [origin, origin + pd.Timedelta(hours=3)]
    future = pd.concat(
        [episodes, pd.DataFrame({"object_id": [1, 2], "start_ts": [origin + pd.Timedelta(hours=4), origin]})]
    )
    np.testing.assert_array_equal(
        FinePendingSimulator(frame, future).alerts(values, values, 1, 0),
        FinePendingSimulator(frame, episodes).alerts(values, values, 1, 0),
    )


def test_fine_confirmation_matches_original_hourly_simulator_with_gaps_and_expiry():
    rng = np.random.default_rng(912)
    origin = pd.Timestamp("2025-01-01")
    frame = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "object_id": obj,
                        "as_of": pd.date_range(origin, periods=80, freq="h").delete([5, 6, 20, 30]),
                    }
                )
                for obj in (1, 2)
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=7)
        .reset_index(drop=True)
    )
    episodes = pd.concat(
        [
            pd.DataFrame(
                {
                    "object_id": obj,
                    "start_ts": origin + pd.to_timedelta(rng.integers(-200, 5500, 80), unit="m"),
                }
            )
            for obj in (1, 2)
        ],
        ignore_index=True,
    )
    capacity, probability = rng.uniform(0, 10, len(frame)), rng.uniform(0, 1, len(frame))
    a, b = FinePendingSimulator(frame, episodes), PendingSimulator(frame, episodes)
    for margin in (0.25, 1, 3):
        for floor in (0, 0.75):
            np.testing.assert_array_equal(
                a.alerts(capacity, probability, margin, floor), b.alerts(capacity, probability, margin, floor)
            )
    quiet = pd.DataFrame({"object_id": [1] * 99, "as_of": pd.date_range(origin, periods=99, freq="15min")})
    empty = episodes.iloc[:0]
    alerts = FinePendingSimulator(quiet, empty).alerts(np.ones(len(quiet)), np.ones(len(quiet)), 1, 0)
    assert np.flatnonzero(alerts).tolist() == [0, 96]
    with pytest.raises(ValueError):
        a.alerts(np.full(len(frame), np.nan), probability, 1, 0)
