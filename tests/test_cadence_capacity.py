import numpy as np
import pandas as pd
import pytest

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.cadence_capacity import matched_slots, subdivide_evaluation_slots


def test_exact_capacity_preserves_horizon_boundaries_and_slot_limit():
    assert matched_slots([0], [0, 24 * HOUR_NS]).sum() == 1
    assert matched_slots([0], [24 * HOUR_NS]).sum() == 0
    assert matched_slots([0, HOUR_NS], [HOUR_NS // 2, 3 * HOUR_NS // 2, 2 * HOUR_NS]).sum() == 2
    assert matched_slots([], [0]).size == 0
    assert not matched_slots([0], []).any()
    with pytest.raises(ValueError):
        matched_slots([0, 0], [0])


def test_matching_capacity_agrees_with_exhaustive_small_assignment_search():
    rng = np.random.default_rng(19)
    for _ in range(40):
        times = np.sort(rng.choice(35, size=4, replace=False)) * HOUR_NS
        events = np.sort(rng.choice(45, size=5, replace=False)) * HOUR_NS

        def brute(i, used, times=times, events=events):
            if i == len(events):
                return 0
            choices = [brute(i + 1, used)]
            for j, at in enumerate(times):
                if j not in used and at <= events[i] < at + 24 * HOUR_NS:
                    choices.append(1 + brute(i + 1, used | {j}))
            return max(choices)

        assert int(matched_slots(times, events).sum()) == brute(0, set())


def test_subdivision_preserves_coverage_gaps_and_does_not_extend_final_horizon():
    frame = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 2],
            "as_of": pd.to_datetime(
                ["2025-01-01 01:00", "2025-01-01 00:00", "2025-01-01 04:00", "2025-01-01 00:00"]
            ),
        }
    )
    result = subdivide_evaluation_slots(frame)
    assert len(result) == 7
    assert result[result.object_id.eq(1)].as_of.dt.strftime("%H:%M").tolist() == [
        "00:00",
        "00:15",
        "00:30",
        "00:45",
        "01:00",
        "04:00",
    ]
    assert result[result.object_id.eq(2)].as_of.tolist() == [pd.Timestamp("2025-01-01")]
    with pytest.raises(ValueError):
        subdivide_evaluation_slots(frame, 7)
