import numpy as np
import pandas as pd
import pytest

from moscollector.count_research import pending_alerts
from moscollector.goal90_research import PendingSimulator, noncrossing, pooled, primary_score


def test_array_simulator_matches_existing_policy_on_shuffled_gaps_and_repeats():
    rng = np.random.default_rng(90)
    at = pd.Timestamp("2025-01-01")
    pred = (
        pd.DataFrame(
            {
                "object_id": np.repeat([1, 2, 3], 75),
                "as_of": np.tile(pd.date_range(at, periods=75, freq="h"), 3),
                "probability": rng.random(225),
                "expected_count": rng.random(225) * 6,
            }
        )
        .sample(frac=0.85, random_state=9)
        .reset_index(drop=True)
    )
    events = pd.DataFrame(
        {
            "object_id": rng.integers(1, 4, 150),
            "start_ts": at + pd.to_timedelta(rng.integers(-240, 6000, 150), unit="min"),
        }
    )
    for margin, floor in ((0.25, 0), (1, 0.5), (3, 0.9)):
        actual = PendingSimulator(pred, events).alerts(pred.expected_count, pred.probability, margin, floor)
        np.testing.assert_array_equal(actual, pending_alerts(pred, events, margin, floor))


def test_simulator_does_not_use_unconfirmed_future_events():
    at = pd.Timestamp("2025-01-01")
    pred = pd.DataFrame({"object_id": [1] * 4, "as_of": at + pd.to_timedelta([0, 1, 2, 3], unit="h")})
    episode = pd.DataFrame({"object_id": [1], "start_ts": [at + pd.Timedelta(hours=1)]})
    result = PendingSimulator(pred, episode).alerts(np.ones(4), np.ones(4), 1, 0)
    np.testing.assert_array_equal(result, [1, 0, 0, 1])
    prefix = PendingSimulator(pred.iloc[:3], episode.iloc[:0]).alerts(np.ones(3), np.ones(3), 1, 0)
    np.testing.assert_array_equal(prefix, result[:3])


def test_nonfinite_predictions_cannot_generate_warnings():
    pred = pd.DataFrame({"object_id": [1], "as_of": [pd.Timestamp("2025-01-01")]})
    episodes = pd.DataFrame({"object_id": [], "start_ts": pd.to_datetime([])})
    with pytest.raises(ValueError, match="Invalid capacity"):
        PendingSimulator(pred, episodes).alerts([np.nan], [1], 1, 0)


def test_goal_requires_both_metrics_and_pools_counts_not_month_averages():
    assert primary_score({"precision": 1, "recall": 0.45}) == 0.5
    assert primary_score({"precision": 0.45, "recall": 1}) == 0.5
    result = pooled(
        [
            {"true_alerts": 9, "alerts": 10, "eligible_episodes": 10},
            {"true_alerts": 0, "alerts": 1, "eligible_episodes": 90},
        ]
    )
    assert result["precision"] == 9 / 11
    assert result["recall"] == 0.09
    assert not result["both_90"]


def test_corrected_count_quantiles_are_finite_nonnegative_and_monotonic():
    actual = noncrossing([[2, -1, 1, 4, 3]], [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(actual, [[0, 1, 2, 3, 4]])
    with pytest.raises(ValueError, match="Invalid quantile"):
        noncrossing([[0, 1, 2, 3, np.inf]], [0] * 5)
