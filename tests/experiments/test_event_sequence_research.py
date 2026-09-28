import pytest

pytest.importorskip("torch")

from moscollector.experiments.event_sequence_research import confirmation, screening  # noqa: E402
from moscollector.experiments.goal90_research import pooled  # noqa: E402


def score(tp, alerts, events=100):
    return pooled([{"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}])


def row(mlp, gru, fresh, old=None, reference=None):
    return {
        "arms": {
            "current_mlp": {"scores": mlp},
            "event_gru": {"scores": gru},
            "fresh_count": {"scores": fresh},
            "old_count": {"scores": old or fresh},
        },
        "reference": reference or fresh,
    }


def test_event_network_selection_preserves_mlp_win_and_every_count_reference():
    c, old = score(70, 80), score(50, 80)
    good = row(c, c, old)
    assert screening([good] * 2)["selected_variant"] == "current_mlp"
    assert screening([row(old, c, old)] * 2)["selected_variant"] == "event_gru"
    for stronger in (row(c, c, c), row(c, c, old, c), row(c, c, old, reference=c)):
        assert not screening([stronger] * 2)["passed_screen"]
    assert not screening([row(score(103, 191, 190), score(103, 191, 190), score(95, 100, 190))] * 2)[
        "passed_screen"
    ]


def test_event_network_confirmation_cannot_switch_variant_after_bad_month():
    old, better = score(50, 80), score(70, 80)
    may = row(better, better, old)
    december = row(old, better, old)
    march = row(old, better, old)
    assert not confirmation([may, december, march], "current_mlp")["research_eligible"]
    assert confirmation([may, december, march], "event_gru")["research_eligible"]
    with pytest.raises(ValueError, match="selected"):
        confirmation([may, december, march], None)


def test_event_network_confirmation_retains_per_month_f1_guard():
    may = row(score(70, 90), score(70, 90), score(60, 80))
    december = row(score(90, 100), score(90, 100), score(10, 100))
    march = row(score(40, 100), score(40, 100), score(50, 100))
    assert confirmation([may, december, march], "event_gru") == {
        "passed_may": True,
        "passed_stress": False,
        "research_eligible": False,
    }
