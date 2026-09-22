from moscollector.goal90_research import pooled
from moscollector.minute_cadence_research import confirmation, screening


def score(tp, alerts, episodes=100):
    return pooled([{"true_alerts": tp, "alerts": alerts, "eligible_episodes": episodes}])


def row(candidate, control, reference=None):
    return {
        "arms": {"minute_candidate": {"scores": candidate}, "quarter_control": {"scores": control}},
        "reference": reference or control,
    }


def test_screen_requires_improvement_against_both_references():
    c, old = score(70, 80), score(60, 80)
    assert screening([row(c, old), row(c, old)])["passed_screen"]
    assert not screening([row(c, old, c), row(c, old, c)])["passed_screen"]
    assert not screening([row(score(80, 200), old)] * 2)["passed_screen"]


def test_balanced_primary_gain_does_not_allow_f1_loss():
    old = score(95, 100, 190)
    c = score(103, 191, 190)
    assert min(c["precision"], c["recall"]) > 1.05 * min(old["precision"], old["recall"])
    assert c["f1"] < old["f1"]
    assert not screening([row(c, old)] * 2)["passed_screen"]


def test_confirmation_retains_per_month_guard_despite_pooled_gain():
    may = row(score(70, 90), score(60, 80))
    december = row(score(90, 100), score(10, 100))
    march = row(score(40, 100), score(50, 100))
    gates = confirmation([may, december, march])
    assert gates == {"passed_may": True, "passed_stress": False, "research_eligible": False}
    march = row(score(45, 100), score(50, 100))
    assert confirmation([may, december, march])["research_eligible"]
    bad_may = row(score(60, 100), score(60, 80))
    assert not confirmation([bad_may, december, march])["research_eligible"]
