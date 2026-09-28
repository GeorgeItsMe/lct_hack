from moscollector.experiments.prefix_checkpoint_research import screening


def score(tp, alerts, events=100):
    return {"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}


def row(candidate, control, old):
    return {
        "arms": {v: {"scores": candidate} for v in ("current_mlp", "event_gru")},
        "old_neural": {v: old for v in ("current_mlp", "event_gru")},
        "controls": {v: control for v in ("fresh_count", "old_count", "reference")},
    }


def test_checkpoint_must_beat_strong_count_control_not_just_old_neural():
    result = screening([row(score(60, 80), score(70, 85), score(50, 80))] * 2)
    assert not result["passed_screen"]


def test_checkpoint_must_also_beat_its_old_architecture():
    result = screening([row(score(70, 85), score(50, 80), score(70, 85))] * 2)
    assert not result["passed_screen"]


def test_no_f1_loss_and_stable_architecture_tie_break():
    # Balanced gain in the primary minimum alone cannot offset an F1 loss.
    result = screening([row(score(66, 100), score(60, 61), score(40, 80))] * 2)
    assert not result["passed_screen"]
    result = screening([row(score(80, 90), score(50, 80), score(40, 80))] * 2)
    assert result["passed_screen"] and result["selected_variant"] == "current_mlp"
