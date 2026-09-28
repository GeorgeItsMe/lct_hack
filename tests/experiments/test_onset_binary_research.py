import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool

from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.goal90_research import pooled
from moscollector.experiments.onset_binary_policy import COOLDOWNS, select_threshold
from moscollector.experiments.onset_binary_research import BINARY_FIT, confirmation, screening
from moscollector.experiments.waiting_time_research import threshold_alerts


def test_explicit_weighted_prauc_controls_classifier_earlystop(tmp_path):
    x = np.arange(32).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 4)
    w = np.ones(32)
    w[y == 1] = np.linspace(0.1, 2, int(y.sum()))
    pool = Pool(x, y, weight=w)
    params = {
        **BINARY_FIT,
        "iterations": 4,
        "depth": 2,
        "thread_count": 2,
        "train_dir": str(tmp_path / "model"),
    }
    model = CatBoostClassifier(**params, custom_metric=["PRAUC:use_weights=false"], verbose=False)
    model.fit(pool, eval_set=pool)
    metrics = model.get_evals_result()["validation"]
    assert not np.allclose(metrics["PRAUC:use_weights=true"], metrics["PRAUC:use_weights=false"])
    assert model.best_iteration_ == int(np.argmax(metrics["PRAUC:use_weights=true"]))
    replay = model.eval_metrics(pool, ["PRAUC:use_weights=true"], tmp_dir=str(tmp_path))[
        "PRAUC:use_weights=true"
    ]
    np.testing.assert_allclose(replay, metrics["PRAUC:use_weights=true"][: model.tree_count_])


def test_direct_policy_is_not_limited_by_predicted_count_and_replays_all_cooldowns():
    start = pd.Timestamp("2025-01-01")
    pred = pd.DataFrame(
        {
            "object_id": 1,
            "as_of": pd.date_range(start, periods=60, freq="min"),
            "probability": 0.01,
            "expected_count": 0.0,
        }
    )
    pred.loc[3, "probability"] = 0.9
    eps = pd.DataFrame({"object_id": [1], "start_ts": [start + pd.Timedelta(minutes=4)]})
    selected, options = select_threshold(pred, eps, exposure=10)
    assert selected["true_alerts"] == selected["alerts"] == 1
    assert {p["cooldown_hours"] for p in options} == set(COOLDOWNS)
    evaluator = evaluator_for(pred, eps, 1 / 60, 10)
    for policy in options:
        alerts = threshold_alerts(pred, policy)
        replay = evaluator.evaluate(alerts, 0.5, policy["cooldown_hours"])
        assert all(policy[k] == v for k, v in replay.items())
        cutoff = 15
        np.testing.assert_array_equal(threshold_alerts(pred.iloc[:cutoff], policy), alerts[:cutoff])


def test_direct_policy_has_safe_empty_fallback_and_validates_probabilities():
    pred = pd.DataFrame({"object_id": [1], "as_of": pd.to_datetime(["2025-01-01"]), "probability": [0.7]})
    eps = pd.DataFrame({"object_id": pd.Series(dtype=int), "start_ts": pd.Series(dtype="datetime64[ns]")})
    policy, _ = select_threshold(pred, eps, exposure=1)
    assert policy["alerts"] == 0 and policy["threshold"] > 0.7
    for value in (np.nan, -0.1, 1.1):
        with pytest.raises(ValueError, match="Invalid"):
            select_threshold(pred.assign(probability=value), eps, 1)


def test_binary_screen_and_confirmation_require_every_reference():
    def score(tp, alerts):
        return pooled([{"true_alerts": tp, "alerts": alerts, "eligible_episodes": 100}])

    c, old = score(70, 80), score(50, 80)
    row = {
        "arms": {"binary_candidate": {"scores": c}, "count_direct": {"scores": old}},
        "old_pending": old,
        "reference": old,
    }
    assert screening([row, row])["passed_screen"]
    assert confirmation([row, row, row])["research_eligible"]
    stronger = {**row, "old_pending": c}
    assert not screening([stronger, stronger])["passed_screen"]
    assert not confirmation([stronger, row, row])["research_eligible"]
