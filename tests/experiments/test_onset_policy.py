import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor, Pool

from moscollector.experiments.binary_gate_research import select_gate_policy
from moscollector.experiments.fine_cadence_research import FinePendingSimulator, evaluator_for, policy_alerts
from moscollector.experiments.onset_channel_features import CATS, COLUMNS
from moscollector.experiments.onset_policy import select_policy
from moscollector.experiments.onset_training_data import model_input


def sample():
    start = pd.Timestamp("2025-01-01")
    pred = pd.DataFrame({"object_id": 1, "as_of": pd.date_range(start, periods=140, freq="min")})
    rng = np.random.default_rng(933)
    pred["probability"] = rng.choice([0, 0.5, 0.8, 1], len(pred))
    pred["expected_count"] = rng.choice([0, 1, 3], len(pred))
    episodes = pd.DataFrame({"object_id": [1, 1], "start_ts": start + pd.to_timedelta([5, 6], unit="min")})
    return pred, episodes


def test_quarter_policy_exactly_replays_all_old_choices():
    pred, eps = sample()
    pred = pred.iloc[::15].reset_index(drop=True)
    assert select_policy(pred, eps, 10, 0.25) == select_gate_policy(pred, eps, 10)


def test_minute_policy_keeps_cadence_and_original_exposure_in_all_choices():
    pred, eps = sample()
    _, options = select_policy(pred, eps, exposure=10)
    assert len(options) == 456
    evaluator = evaluator_for(pred, eps, 1 / 60, 10)
    for policy in options:
        scores = evaluator.evaluate(policy_alerts(pred, eps, policy), 0.5, 1 / 60)
        assert all(policy[k] == value for k, value in scores.items())
        assert scores["false_alerts_per_object_day"] == scores["false_alerts"] / 10
    # Two minute-apart warnings may match distinct episodes, never one episode twice.
    values = np.zeros(len(pred))
    values[:2] = 1
    assert evaluator.evaluate(values, 0.5, 1 / 60)["true_alerts"] == 2
    assert evaluator.evaluate(values, 0.5, 0.25)["true_alerts"] == 1


def test_minute_predictions_do_not_accelerate_episode_confirmation():
    pred, eps = sample()
    values = FinePendingSimulator(pred, eps.iloc[:1]).alerts(np.ones(len(pred)), np.ones(len(pred)), 1, 0)
    # Episode at00:05 is visible at02:00, not at the next minute or01:15.
    np.testing.assert_array_equal(np.flatnonzero(values), [0, 120])


def test_dictionary_categories_and_weighted_poisson_are_usable(tmp_path):
    frame = pd.DataFrame(
        {c: ["__NONE__", "1", "2", "1"] if c in CATS else [np.nan, 0, 1, 2] for c in COLUMNS}
    )
    path = tmp_path / "context.parquet"
    frame.to_parquet(path, index=False)
    loaded = pd.read_parquet(path, read_dictionary=CATS)
    assert all(isinstance(loaded[c].dtype, pd.CategoricalDtype) for c in CATS)
    x = model_input(loaded, COLUMNS)
    model = CatBoostRegressor(
        iterations=2,
        depth=2,
        loss_function="Poisson",
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(Pool(x, [0, 1, 0, 2], weight=[1, 0.5, 0.25, 0.25], cat_features=CATS))
    assert np.isfinite(model.predict(x)).all()


def test_minute_policy_rejects_invalid_inputs():
    pred, eps = sample()
    for value in (np.nan, -1, 1.1):
        with pytest.raises(ValueError, match="Invalid"):
            select_policy(pred.assign(probability=value), eps, 10)
    with pytest.raises(ValueError, match="Unsupported"):
        select_policy(pred, eps, 10, 1)
