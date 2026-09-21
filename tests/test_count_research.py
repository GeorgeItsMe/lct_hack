import numpy as np
import pandas as pd

from moscollector.count_research import episode_counts, pending_alerts


def test_count_target_includes_start_excludes_horizon_and_is_per_object():
    at = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame({"object_id": [1, 2, 1], "as_of": [at, at, at + pd.Timedelta(hours=24)]})
    episodes = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 2],
            "start_ts": [at, at + pd.Timedelta(hours=23), at + pd.Timedelta(hours=24), at],
        }
    )
    np.testing.assert_array_equal(episode_counts(frame, episodes), [2, 1, 1])


def test_pending_alerts_resolve_only_after_confirmation_and_expire():
    at = pd.Timestamp("2026-01-01")
    pred = pd.DataFrame(
        {
            "object_id": [1] * 5,
            "as_of": at + pd.to_timedelta([0, 1, 2, 3, 27], unit="h"),
            "probability": [0.9] * 5,
            "expected_count": [0.9] * 5,
        }
    )
    episodes = pd.DataFrame({"object_id": [1], "start_ts": [at + pd.Timedelta(hours=1)]})
    # First warning remains pending at t=2h: the t=1h event is not confirmed yet.
    np.testing.assert_array_equal(pending_alerts(pred, episodes, 0.75, 0.5), [1, 0, 0, 1, 1])
    # Future outcomes cannot change any earlier decision.
    early = pending_alerts(pred.iloc[:3], episodes.iloc[:0], 0.75, 0.5)
    np.testing.assert_array_equal(early, pending_alerts(pred, episodes, 0.75, 0.5)[:3])


def test_one_observed_episode_cannot_resolve_two_pending_warnings():
    at = pd.Timestamp("2026-01-01")
    pred = pd.DataFrame(
        {
            "object_id": [1] * 5,
            "as_of": pd.date_range(at, periods=5, freq="h"),
            "probability": [1.0] * 5,
            "expected_count": [2.0] * 5,
        }
    )
    episodes = pd.DataFrame({"object_id": [1], "start_ts": [at + pd.Timedelta(hours=1.5)]})
    # At t=3h one warning resolves; exactly one remains before the new warning.
    np.testing.assert_array_equal(pending_alerts(pred, episodes, 0.75, 0.5), [1, 1, 0, 1, 0])
