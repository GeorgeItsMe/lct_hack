import pytest

from moscollector.experiments.temporal_count_research import PLAN, screen


def score(tp, alerts, events=100):
    return {"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}


def row(candidate, controls):
    arms = {name: {"scores": controls} for name in PLAN["arms"]}
    arms["profile"] = {"scores": candidate}
    return {"arms": arms, "references": {"historical": controls, "recent": controls}}


def test_profile_must_beat_identical_total_with_uniform_time_profile():
    r = row(score(70, 100), score(50, 90))
    r["arms"]["uniform_new"]["scores"] = score(75, 100)
    assert not screen([r, r])["passed_screen"]


def test_profile_must_beat_old_forecasts_under_same_deadline_budget():
    r = row(score(70, 100), score(50, 90))
    r["arms"]["uniform_old"]["scores"] = score(75, 100)
    assert not screen([r])["passed_screen"]


def test_primary_improvement_does_not_allow_f1_loss():
    r = row(score(54, 100), score(90, 180))
    assert not screen([r])["passed_screen"]


def test_clear_gain_over_every_reference_can_pass_but_is_not_90_90():
    r = row(score(70, 90), score(50, 100))
    selected = screen([r, r])
    assert selected["passed_screen"]
    assert not selected["scores"]["profile"]["both_90"]


def test_missing_arm_or_reference_cannot_make_gate_easier():
    r = row(score(70, 90), score(50, 100))
    r["arms"].pop("uniform_old")
    with pytest.raises(ValueError, match="required arm"):
        screen([r])
    a, b = row(score(70, 90), score(50, 100)), row(score(70, 90), score(50, 100))
    b["references"].pop("recent")
    with pytest.raises(ValueError, match="reference"):
        screen([a, b])
    with pytest.raises(ValueError, match="No temporal"):
        screen([])
