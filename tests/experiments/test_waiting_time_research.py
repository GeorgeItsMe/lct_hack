import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.count_research import episode_counts
from moscollector.experiments.waiting_time_research import threshold_alerts, waiting_time_labels


def test_waiting_target_is_per_object_and_preserves_24h_boundaries():
    at = pd.Timestamp("2025-01-01")
    frame = pd.DataFrame({"object_id": [2, 1, 4, 3], "as_of": [at] * 4})
    episodes = pd.DataFrame(
        {
            "object_id": [1, 1, 2, 4],
            "start_ts": at + pd.to_timedelta([0, 1, 24, 23.5], unit="h"),
        }
    )
    labels, observed = waiting_time_labels(frame, episodes)
    np.testing.assert_array_equal(observed, episode_counts(frame, episodes) > 0)
    np.testing.assert_allclose(labels, [[24, -1], [1 / 60, 1 / 60], [23.5, 23.5], [24, -1]])
    # A later event beyond the censored horizon cannot change the target.
    shifted = episodes.copy()
    shifted.loc[shifted.object_id.eq(2), "start_ts"] = at + pd.Timedelta(days=100)
    np.testing.assert_array_equal(waiting_time_labels(frame, shifted)[0], labels)


def test_threshold_planner_reproduces_legacy_cooldown_matching_after_shuffle():
    at = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(15)
    pred = (
        pd.DataFrame(
            {
                "object_id": np.repeat([1, 2], 100),
                "as_of": np.tile(pd.date_range(at, periods=100, freq="h"), 2),
                "probability": rng.random(200),
            }
        )
        .sample(frac=0.9, random_state=15)
        .reset_index(drop=True)
    )
    episodes = pd.DataFrame(
        {
            "object_id": rng.integers(1, 3, 80),
            "start_ts": at + pd.to_timedelta(rng.integers(0, 7500, 80), unit="min"),
        }
    )
    evaluator = EventEvaluator(pred, episodes, 1)
    for cooldown in (1, 3, 12, 24):
        policy = {"threshold": 0.6, "cooldown_hours": cooldown}
        alert = threshold_alerts(pred, policy)
        old = evaluator.evaluate(pred.probability, 0.6, cooldown)
        new = evaluator.evaluate(alert, 0.5, 1)
        for field in ("true_alerts", "alerts", "eligible_episodes", "median_lead_hours"):
            assert old[field] == new[field]


def test_installed_aft_loss_accepts_right_censoring_and_orders_waiting_times():
    x = np.repeat([[0], [1]], 20, axis=0)
    y = np.repeat([[1.0, 1.0], [24.0, -1.0]], 20, axis=0)
    model = CatBoostRegressor(
        iterations=30,
        depth=2,
        learning_rate=0.15,
        loss_function="SurvivalAft:dist=Normal;scale=1",
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
        random_seed=15,
    )
    model.fit(x, y)
    raw = model.predict([[0], [1]], prediction_type="RawFormulaVal")
    assert np.isfinite(raw).all()
    assert raw[0] < raw[1]
