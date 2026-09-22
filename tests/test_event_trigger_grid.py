import numpy as np
import pandas as pd
import pytest

from moscollector.event_trigger_grid import trigger_alerts, trigger_grid
from moscollector.fine_cadence_research import cohort


def hours(values=(0, 1, 3), obj=1):
    return pd.DataFrame(
        {"object_id": obj, "as_of": pd.Timestamp("2025-01-01") + pd.to_timedelta(values, unit="h")}
    )


def onsets(values, signals=None):
    return pd.DataFrame(
        {"object_id": 1, "ts": pd.to_datetime(values), "signal": signals or ["fire"] * len(values)}
    )


def test_next_boundary_is_strict_and_does_not_extend_last_hour_or_gaps():
    source = onsets(
        [
            "2025-01-01 00:00:00",
            "2025-01-01 00:00:59",
            "2025-01-01 00:59:59",
            "2025-01-01 01:00:00",
            "2025-01-01 02:59:59",
            "2025-01-01 03:01:00",
        ]
    )
    result = trigger_grid(hours(), source)
    assert result.as_of.dt.strftime("%H:%M").tolist() == ["00:00", "00:01", "01:00", "03:00"]
    assert result.trigger_count.tolist() == [0, 2, 1, 1]
    hourly = trigger_grid(hours(), source, seconds=3600)
    assert hourly.trigger_count.tolist() == [0, 3, 1]
    assert hourly.as_of.tolist() == hours().as_of.tolist()


def test_future_rows_labels_inventory_and_permutation_do_not_change_past_triggers():
    source = onsets(
        ["2025-01-01 00:00:00", "2025-01-01 00:02:01", "2025-01-01 00:40:01"], ["fire", "fault", "fire"]
    )
    before = trigger_grid(hours(), source)
    altered = pd.concat([source, onsets(["2025-01-01 00:50:00"])], ignore_index=True).sample(
        frac=1, random_state=3
    )
    altered["target_fault"], altered["channel_id"] = 999, 1234
    after = trigger_grid(hours().assign(eligible=False, target_fault=999), altered)
    cutoff = pd.Timestamp("2025-01-01 00:45")
    pd.testing.assert_frame_equal(before.loc[before.as_of.lt(cutoff)], after.loc[after.as_of.lt(cutoff)])
    assert before.trigger_count.sum() == 2


def test_causal_cooldown_is_per_object_and_uses_no_targets():
    source = onsets(["2025-01-01 00:00:00", "2025-01-01 00:30:00", "2025-01-02 00:00:00"])
    grid = trigger_grid(hours(tuple(range(26))), source)
    grid2 = (
        pd.concat([grid, grid.assign(object_id=2)], ignore_index=True)
        .sample(frac=1, random_state=8)
        .reset_index(drop=True)
    )
    alerts = trigger_alerts(grid2, 24)
    assert alerts.sum() == 4
    assert set(grid2.loc[alerts, "as_of"]) == {
        pd.Timestamp("2025-01-01 00:01"),
        pd.Timestamp("2025-01-02 00:01"),
    }
    np.testing.assert_array_equal(alerts, trigger_alerts(grid2.assign(target_fault=np.nan), 24))
    assert trigger_alerts(grid2, 0).sum() == 6


@pytest.mark.parametrize("seconds", [1, 60, 900, 3600])
def test_retains_exact_original_episode_cohort(seconds):
    h = hours()
    source = onsets(
        ["2025-01-01 00:14:59", "2025-01-01 00:58:40", "2025-01-01 01:02:00", "2025-01-01 03:30:00"]
    )
    eps = pd.DataFrame(
        {
            "object_id": 1,
            "start_ts": pd.to_datetime(
                ["2025-01-01 00:05:00", "2025-01-02 02:59:59", "2025-01-02 03:00:00", "2025-01-02 03:29:00"]
            ),
        }
    )
    assert cohort(h, eps, 1) == cohort(trigger_grid(h, source, seconds), eps, seconds / 3600)
    assert len(cohort(h, eps, 1)[1]) == 2


def test_invalid_resolution_and_duplicates_fail_closed():
    empty = onsets([])
    with pytest.raises(ValueError, match="Resolution"):
        trigger_grid(hours(), empty, 61)
    with pytest.raises(ValueError, match="duplicate"):
        trigger_grid(pd.concat([hours(), hours()]), empty)
    with pytest.raises(ValueError, match="cooldown"):
        trigger_alerts(trigger_grid(hours(), empty), float("nan"))
    with pytest.raises(ValueError, match="Invalid trigger"):
        trigger_alerts(trigger_grid(hours(), empty).assign(trigger_count=np.nan), 24)
