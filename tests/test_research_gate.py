from copy import deepcopy

from moscollector.stability_research import passes_stability


def results(tp=10, false_alerts=70):
    return [
        {
            "test": {
                "true_alerts": tp,
                "alerts": tp + false_alerts,
                "eligible_episodes": 20,
                "false_alerts_per_object_day": 0.1,
                "f1": 2 * tp / (tp + false_alerts + 20),
            }
        }
        for _ in range(3)
    ]


def test_stability_requires_distributed_gain_and_rejects_unsafe_or_sparse_period():
    reference = results()
    candidate = results(false_alerts=60)
    assert passes_stability(reference, candidate)
    one_month = deepcopy(reference)
    one_month[0] = results(false_alerts=0)[0]
    assert not passes_stability(reference, one_month)
    candidate[2]["test"]["f1"] = 0.1
    assert not passes_stability(reference, candidate)
    for field, value in (("eligible_episodes", 9), ("false_alerts_per_object_day", 0.26)):
        candidate = results(false_alerts=60)
        candidate[1]["test"][field] = value
        assert not passes_stability(reference, candidate)
