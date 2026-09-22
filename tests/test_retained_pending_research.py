import pytest

from moscollector.retained_pending_research import ARMS, compare


def result(tp=90, alerts=100, episodes=100):
    return {
        "true_alerts": tp,
        "alerts": alerts,
        "eligible_episodes": episodes,
        "precision": tp / alerts,
        "recall": tp / episodes,
        "f1": 2 * tp / (alerts + episodes),
    }


def row():
    return {
        "arms": {a: {"scores": result(90 if a == "retained_selected" else 50)} for a in ARMS},
        "references": {"recent": result(50)},
    }


def test_corrected_policy_must_beat_every_reference_without_substitution():
    item = row()
    assert compare([item])["passed_screen"]
    item["references"]["recent"] = result(92)
    assert not compare([item])["passed_screen"]
    item = row()
    item["arms"]["retained_fixed_control"]["scores"] = result(92)
    assert not compare([item])["passed_screen"]


def test_comparison_rejects_missing_arms_references_or_changed_cohorts():
    item = row()
    del item["arms"]["legacy_control"]
    with pytest.raises(ValueError, match="arms"):
        compare([item])
    item = row()
    item["references"] = {}
    with pytest.raises(ValueError, match="reference"):
        compare([row(), item])
    item = row()
    item["references"]["recent"] = result(50, episodes=101)
    with pytest.raises(ValueError, match="cohort"):
        compare([item])


def test_pooled_counts_preserve_all_false_alerts_and_missed_events():
    first, second = row(), row()
    second["arms"]["retained_selected"]["scores"] = result(0, 100, 100)
    score = compare([first, second])["scores"]["retained_selected"]
    assert score["true_alerts"] == 90 and score["alerts"] == 200 and score["eligible_episodes"] == 200
    assert score["precision"] == score["recall"] == 0.45
