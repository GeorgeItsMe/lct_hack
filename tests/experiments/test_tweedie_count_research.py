import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from catboost.utils import eval_metric

from moscollector.experiments.peer_context_research import FIT
from moscollector.experiments.tweedie_count_research import (
    FIT_TWEEDIE,
    LOSS,
    check_native_parameters,
    comparison,
    count_support,
    training_pool,
)


def test_native_tweedie_uses_log_mean_not_direct_count():
    raw = np.array([-10.0, -1.0, 0.0, 2.0, 5.0])
    y = np.array([0.0, 0.0, 1.0, 3.0, 126.0])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 3.0])
    expected = np.average(2 * np.exp(raw / 2) + 2 * y * np.exp(-raw / 2), weights=weights)
    assert eval_metric(y, raw, LOSS, weight=weights)[0] == pytest.approx(expected, rel=1e-12)


def test_training_pool_retains_every_negative_and_original_target():
    rows = pd.DataFrame(
        {
            "object_id": [1] * 5,
            "parent_id": [2] * 5,
            "object_kind": ["test"] * 5,
            "feature": np.arange(5.0),
            "count_target": [0, 0, 1, 0, 4],
        }
    )
    pool = training_pool(rows, ["object_id", "parent_id", "object_kind", "feature"])
    assert pool.num_row() == 5
    np.testing.assert_array_equal(pool.get_label(), rows.count_target)
    assert count_support(rows.count_target)["count_mass"] == 5
    with pytest.raises(ValueError, match="future target"):
        training_pool(rows, ["object_id", "parent_id", "object_kind", "count_target"])
    with pytest.raises(ValueError, match="Duplicate"):
        training_pool(rows, ["object_id", "parent_id", "object_kind", "feature", "feature"])


@pytest.mark.parametrize("counts", [[], [-1, 0], [0, np.nan], [0, np.inf], [0, 0.5], [0, 1001]])
def test_invalid_counts_fail_without_silent_scaling_or_dropping(counts):
    with pytest.raises(ValueError, match="count target"):
        count_support(counts)


def test_all_zero_support_is_explicit():
    support = count_support(np.zeros(5))
    assert support["positive_rows"] == support["count_mass"] == 0
    assert support["variance_to_mean"] is None


def test_native_settings_match_poisson_and_weights_reload_exactly(tmp_path):
    x = np.arange(128).reshape(-1, 1)
    y = (x[:, 0] // 16) % 4
    old = CatBoostRegressor(
        **{**FIT, "iterations": 4, "leaf_estimation_method": "Newton", "leaf_estimation_iterations": 10},
        verbose=False,
    ).fit(x, y, eval_set=(x, y))
    new = CatBoostRegressor(**{**FIT_TWEEDIE, "iterations": 4}, verbose=False).fit(x, y, eval_set=(x, y))
    assert set(check_native_parameters(new, old)) == {"loss_function", "eval_metric"}
    path = tmp_path / "model.cbm"
    new.save_model(str(path))
    loaded = CatBoostRegressor()
    loaded.load_model(str(path))
    np.testing.assert_array_equal(
        loaded.predict(x, prediction_type="RawFormulaVal"), new.predict(x, prediction_type="RawFormulaVal")
    )
    assert check_native_parameters(loaded, old) == check_native_parameters(new, old)
    other = CatBoostRegressor(**{**FIT_TWEEDIE, "iterations": 4, "depth": 3}, verbose=False).fit(
        x, y, eval_set=(x, y)
    )
    with pytest.raises(ValueError, match="Unmatched"):
        check_native_parameters(other, old)


def score(tp, alerts, events=100):
    return {"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}


def row(candidate=None, control=None):
    candidate, control = candidate or score(70, 90), control or score(50, 100)
    return {
        "kind": "access",
        "arms": {
            "tweedie": {"scores": candidate},
            "poisson_corrected": {"scores": control},
            "poisson_legacy": {"scores": control},
        },
        "references": {"historical": control, "recent": control},
    }


def test_candidate_must_beat_both_queues_and_stronger_reference():
    assert comparison([row()])["passed_screen"]
    assert not comparison([row()])["scores"]["tweedie"]["both_90"]
    for arm in ("poisson_corrected", "poisson_legacy"):
        r = row()
        r["arms"][arm]["scores"] = score(75, 90)
        assert not comparison([r])["passed_screen"]
    r = row()
    r["references"]["recent"] = score(75, 90)
    assert not comparison([r])["passed_screen"]


def test_primary_gain_cannot_hide_f1_loss():
    assert not comparison([row(score(54, 100), score(90, 180))])["passed_screen"]


def test_incomplete_comparisons_or_changed_event_cohorts_fail():
    r = row()
    del r["arms"]["poisson_legacy"]
    with pytest.raises(ValueError, match="comparator"):
        comparison([r])
    r = row()
    del r["references"]["recent"]
    with pytest.raises(ValueError, match="reference"):
        comparison([r])
    r = row()
    r["references"]["recent"] = score(50, 100, 101)
    with pytest.raises(ValueError, match="cohort"):
        comparison([r])
    with pytest.raises(ValueError, match="Missing"):
        comparison([])
