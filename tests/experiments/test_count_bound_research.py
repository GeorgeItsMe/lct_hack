import pytest

from moscollector.experiments.count_bound_research import CANDIDATES, CONTROLS, selection


def score(tp, alerts=100, events=100):
    return {"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}


def row():
    return {
        "kind": "access",
        "arms": {name: {"scores": score(70 if name in CANDIDATES else 50)} for name in CANDIDATES + CONTROLS},
        "references": {"historical": score(50), "recent": score(50)},
    }


def test_fixed_tie_break_and_all_comparator_gates():
    r = row()
    assert selection([r])["selected"] == "projected_poisson"
    for name in CONTROLS:
        r = row()
        r["arms"][name]["scores"] = score(75)
        assert selection([r])["selected"] is None
    r = row()
    r["references"]["recent"] = score(75)
    assert selection([r])["selected"] is None


def test_only_passing_candidate_can_be_selected_without_f1_loss():
    r = row()
    r["arms"]["projected_poisson"]["scores"] = score(40)
    assert selection([r])["selected"] == "projected_tweedie"
    r = row()
    for a in CANDIDATES:
        r["arms"][a]["scores"] = score(54, 100)
    for a in CONTROLS:
        r["arms"][a]["scores"] = score(90, 180)
    for a in r["references"]:
        r["references"][a] = score(90, 180)
    assert selection([r])["selected"] is None


def test_missing_references_and_changed_cohorts_cannot_pass():
    r = row()
    del r["references"]["recent"]
    with pytest.raises(ValueError, match="reference"):
        selection([r])
    r = row()
    del r["arms"]["poisson_control"]
    with pytest.raises(ValueError, match="arm"):
        selection([r])
    r = row()
    r["references"]["recent"] = score(50, 100, 101)
    with pytest.raises(ValueError, match="cohort"):
        selection([r])
    with pytest.raises(ValueError, match="Missing"):
        selection([])
