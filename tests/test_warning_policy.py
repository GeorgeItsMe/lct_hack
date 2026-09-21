import json

import numpy as np
import pandas as pd

from moscollector.count_research import pending_alerts
from moscollector.warning_policy import PendingWarningState, replay_pending


def test_incremental_policy_matches_research_after_serialization():
    rng = np.random.default_rng(812)
    start = pd.Timestamp("2026-01-01")
    pred = pd.DataFrame(
        {
            "object_id": np.repeat([1, 2], 240),
            "as_of": list(pd.date_range(start, periods=240, freq="h")) * 2,
            "probability": rng.random(480),
            "expected_count": rng.uniform(0, 3, 480),
        }
    )
    eps = pd.DataFrame(
        {
            "object_id": [1] * 70 + [2] * 60,
            "start_ts": start + pd.to_timedelta(rng.uniform(0, 241, 130), unit="h"),
        }
    )
    expected = pending_alerts(pred, eps, 0.75, 0.25)
    result, _ = replay_pending(pred, eps, 0.75, 0.25)
    np.testing.assert_array_equal(result, expected)
    first = pred[pred.as_of.lt(start + pd.Timedelta(hours=120))]
    second = pred[pred.as_of.ge(start + pd.Timedelta(hours=120))]
    a, state = replay_pending(first, eps, 0.75, 0.25)
    state = {k: PendingWarningState.from_dict(json.loads(json.dumps(v.to_dict()))) for k, v in state.items()}
    b, _ = replay_pending(second, eps, 0.75, 0.25, state)
    combined = pd.Series(np.r_[a, b], index=list(first.index) + list(second.index)).sort_index().to_numpy()
    np.testing.assert_array_equal(combined, expected)


def test_duplicate_snapshot_and_episode_cannot_issue_or_resolve_twice():
    state = PendingWarningState()
    at = pd.Timestamp("2026-01-01")
    assert state.step(at, 1, 2, [], 0.75, 0.5)
    assert not state.step(at, 1, 2, [], 0.75, 0.5)
    assert state.step(at + pd.Timedelta(hours=1), 1, 2, [], 0.75, 0.5)
    event = (at + pd.Timedelta(hours=1.5)).value
    assert state.step(at + pd.Timedelta(hours=3), 1, 2, [event], 0.75, 0.5)
    assert not state.step(at + pd.Timedelta(hours=4), 1, 2, [event], 0.75, 0.5)
    previous = state.to_dict()
    assert not state.step(at + pd.Timedelta(hours=2), 1, 50, [event], 0.75, 0.5)
    assert state.to_dict() == previous


def test_confirmation_after_expiry_is_consumed_by_original_warning():
    state = PendingWarningState()
    at = pd.Timestamp("2026-01-01")
    assert state.step(at, 1, 2, [], 0.75, 0.5)
    assert state.step(at + pd.Timedelta(hours=2), 1, 2, [], 0.75, 0.5)
    event = (at + pd.Timedelta(hours=23)).value
    state.step(at + pd.Timedelta(hours=24.5), 1, 0, [event], 0.75, 0.5)
    assert state.pending == [(at + pd.Timedelta(hours=2)).value]
    # Re-delivery after a later snapshot still cannot close the second warning.
    state.step(at + pd.Timedelta(hours=25), 1, 0, [event], 0.75, 0.5)
    assert state.pending == [(at + pd.Timedelta(hours=2)).value]
