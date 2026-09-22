import numpy as np
import pandas as pd
import pytest

from moscollector.fine_cadence_research import policy_alerts
from moscollector.goal90_research import pooled
from moscollector.onset_pending_research import compose, confirmation, screening


def inputs():
    start = pd.Timestamp("2025-01-01")
    frame = pd.DataFrame(
        {
            "object_id": 1,
            "as_of": start + pd.to_timedelta([0, 30, 70, 110, 120, 150, 180], unit="min"),
            "raw": -2.0,
            "probability": 0.02,
            "expected_count": 1.0,
        }
    )
    return frame.assign(raw=2.0, probability=0.9), frame


def test_composition_preserves_count_capacity_and_does_not_mutate_components():
    binary, count = inputs()
    before = count.copy(deep=True)
    out = compose(binary, count)
    np.testing.assert_array_equal(out.probability, binary.probability)
    np.testing.assert_array_equal(out.expected_count, count.expected_count)
    np.testing.assert_array_equal(out.raw_binary, binary.raw)
    np.testing.assert_array_equal(out.raw_count, count.raw)
    out.loc[0, "expected_count"] = 999
    pd.testing.assert_frame_equal(count, before)


def test_composition_rejects_misalignment_duplicates_and_invalid_values():
    binary, count = inputs()
    with pytest.raises(AssertionError):
        compose(binary.assign(as_of=binary.as_of + pd.Timedelta(minutes=1)), count)
    with pytest.raises(ValueError, match="duplicate"):
        compose(pd.concat([binary, binary]), pd.concat([count, count]))
    with pytest.raises(ValueError, match="Empty"):
        compose(binary.iloc[:0], count.iloc[:0])
    for value in (np.nan, -0.1, 1.1):
        with pytest.raises(ValueError, match="Invalid"):
            compose(binary.assign(probability=value), count)
    with pytest.raises(ValueError, match="Invalid"):
        compose(binary, count.assign(expected_count=-1))


def test_pending_composition_rearms_only_after_delayed_confirmation():
    binary, count = inputs()
    pred = compose(binary, count)
    start = pred.as_of.iloc[0]
    eps = pd.DataFrame({"object_id": [1, 1], "start_ts": start + pd.to_timedelta([10, 135], unit="min")})
    policy = {"capacity": 1, "margin": 1, "floor": 0.5}
    actual = policy_alerts(pred, eps, policy)
    np.testing.assert_array_equal(actual, [1, 0, 0, 0, 1, 0, 0])
    assert not policy_alerts(count, eps, policy).any()
    np.testing.assert_array_equal(policy_alerts(pred.iloc[:4], eps, policy), actual[:4])
    extra = pd.concat(
        [eps, pd.DataFrame({"object_id": [1], "start_ts": [start + pd.Timedelta(minutes=125)]})],
        ignore_index=True,
    )
    np.testing.assert_array_equal(policy_alerts(pred, extra, policy), actual)
    disabled = {**policy, "floor": 0}
    np.testing.assert_array_equal(policy_alerts(pred, eps, disabled), policy_alerts(count, eps, disabled))


def score(tp, alerts, events=100):
    return pooled([{"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}])


def row(candidate, control, direct=None, reference=None):
    return {
        "arms": {"pending_candidate": {"scores": candidate}, "count_control": {"scores": control}},
        "direct_control": direct or control,
        "reference": reference or control,
    }


def test_pending_gates_require_each_reference_and_preserve_f1():
    c, old = score(70, 80), score(50, 80)
    good = row(c, old)
    assert screening([good, good])["passed_screen"]
    assert confirmation([good] * 3)["research_eligible"]
    for stronger in (row(c, c), row(c, old, c), row(c, old, reference=c)):
        assert not screening([stronger] * 2)["passed_screen"]
        assert not confirmation([stronger, good, good])["passed_may"]
    assert not screening([row(score(103, 191, 190), score(95, 100, 190))] * 2)["passed_screen"]


def test_pending_confirmation_keeps_monthly_guard_when_pool_improves():
    may = row(score(70, 90), score(60, 80))
    december = row(score(90, 100), score(10, 100))
    march = row(score(40, 100), score(50, 100))
    assert confirmation([may, december, march]) == {
        "passed_may": True,
        "passed_stress": False,
        "research_eligible": False,
    }
    assert confirmation([may, december, row(score(45, 100), score(50, 100))])["research_eligible"]
