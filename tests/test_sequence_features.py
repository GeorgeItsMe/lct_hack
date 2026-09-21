import numpy as np
import pandas as pd

from moscollector.features import COUNT_COLUMNS
from moscollector.sequence_features import object_sequence_features, recurrence_features


def fixture():
    hours = pd.date_range("2025-01-01", periods=1000, freq="h")
    rng = np.random.default_rng(123)
    hourly = pd.DataFrame({c: rng.poisson(2, len(hours)) for c in COUNT_COLUMNS})
    hourly["hour"] = hours
    for col in ("temperature", "temperature_max", "gas"):
        hourly[col] = rng.normal(10, 1, len(hours))
    ep = pd.DataFrame(
        {
            "kind": ["fault"] * 5,
            "start_ts": hours[[100, 220, 320, 620, 800]],
            "end_ts": hours[[140, 260, 820, 840, 860]],
            "channel_count": [1, 2, 1, 1, 3],
        }
    )
    return hours, hourly, ep


def test_appending_future_and_changing_future_ends_cannot_change_past_features():
    hours, hourly, ep = fixture()
    cutoff = hours[750]
    times = hours[720:751:3]
    original = object_sequence_features(times, hourly, ep)
    changed_hourly = hourly.copy()
    changed_hourly.loc[
        changed_hourly.hour.ge(cutoff), COUNT_COLUMNS + ["temperature", "temperature_max", "gas"]
    ] = 9999
    changed_ep = ep.copy()
    changed_ep.loc[changed_ep.end_ts.ge(cutoff), "end_ts"] += pd.Timedelta(days=100)
    changed_ep.loc[changed_ep.start_ts.ge(cutoff), "channel_count"] = 999
    pd.testing.assert_frame_equal(original, object_sequence_features(times, changed_hourly, changed_ep))
    pd.testing.assert_frame_equal(
        original, object_sequence_features(times, hourly[hourly.hour.lt(cutoff)], ep[ep.start_ts.lt(cutoff)])
    )


def test_confirmation_and_observation_are_strict_boundaries():
    hours, hourly, ep = fixture()
    times = pd.DatetimeIndex(
        [ep.start_ts.iloc[0] + pd.Timedelta(minutes=70), ep.start_ts.iloc[0] + pd.Timedelta(minutes=71)]
    )
    out = recurrence_features(times, ep)
    assert out.seq_fault_episodes_6h.tolist() == [0, 1]
    assert out.seq_fault_active_confirmed.tolist() == [0, 1]
    at = pd.DatetimeIndex([hours[750]])
    out = object_sequence_features(at, hourly, ep)
    assert out.seq_events_bin_0_1h.iloc[0] == hourly.events.iloc[749]
    assert out.seq_events_bin_1_3h.iloc[0] == np.mean(hourly.events.iloc[747:749])


def test_partly_recovered_group_remains_unresolved_and_truncation_is_causal():
    ep = pd.DataFrame(
        {
            "kind": ["fault"],
            "start_ts": pd.to_datetime(["2025-01-01"]),
            "end_ts": pd.to_datetime(["2025-01-01 03:00"]),
            "channel_count": [2],
            "right_censored": [True],
        }
    )
    at = pd.DatetimeIndex(["2025-01-01 04:00"])
    assert recurrence_features(at, ep).seq_fault_active_confirmed.iloc[0] == 1
    # A future complete recovery cannot change what was observable at 04:00.
    complete = ep.copy()
    complete["end_ts"] = pd.Timestamp("2025-01-02")
    complete["right_censored"] = False
    pd.testing.assert_frame_equal(recurrence_features(at, ep), recurrence_features(at, complete))
