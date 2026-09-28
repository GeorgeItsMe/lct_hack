import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from moscollector.experiments.count_validation import profile_count_loss


def test_profile_equals_independent_scalar_optimization():
    raw = np.array([-3.0, -1.0, 2.0, 0.5])
    target = np.array([0.0, 1.0, 3.0, 2.0])
    weight = np.array([0.5, 0.25, 0.25, 2.0])
    result = profile_count_loss(raw, target, weight)

    def objective(offset):
        return np.average(np.exp(raw + offset) - target * (raw + offset), weights=weight)

    optimum = minimize_scalar(objective, bracket=(-2, 2))
    assert optimum.success
    assert result["profiled_loss"] == pytest.approx(optimum.fun, abs=1e-12)
    assert result["optimal_log_scale"] == pytest.approx(optimum.x, abs=1e-7)
    assert result["raw_loss"] == pytest.approx(objective(0))
    assert result["raw_loss"] == pytest.approx(result["profiled_loss"] + result["scale_penalty"])


def test_shape_metric_is_invariant_to_global_count_scale():
    raw, y, weight = np.log([0.2, 4, 2]), [0, 6, 1], [1, 0.5, 2]
    original = profile_count_loss(raw, y, weight)
    for offset in (-100, -2, 3, 100):
        changed = profile_count_loss(raw + offset, y, weight)
        assert changed["profiled_loss"] == pytest.approx(original["profiled_loss"], abs=1e-12)
        assert changed["shape_gain_over_constant"] == pytest.approx(
            original["shape_gain_over_constant"], abs=1e-12
        )
        assert changed["optimal_log_scale"] == pytest.approx(original["optimal_log_scale"] - offset)


def test_constant_rate_has_zero_shape_skill_with_any_mean():
    for rate in (0.0001, 0.5, 20):
        result = profile_count_loss(np.full(4, np.log(rate)), [0, 0, 1, 3], [0.5, 1, 0.25, 0.25])
        assert result["shape_gain_over_constant"] == pytest.approx(0, abs=1e-14)
        assert result["profiled_loss"] == pytest.approx(result["constant_optimum_loss"])


def test_shifted_useful_count_can_lose_raw_comparison_to_constant():
    y = np.array([0, 0, 0, 10.0])
    useful = profile_count_loss(np.log([0.1, 0.1, 0.1, 10]) + 3, y, np.ones(4))
    constant = profile_count_loss(np.full(4, np.log(y.mean())), y, np.ones(4))
    assert useful["raw_loss"] > constant["raw_loss"]
    assert useful["profiled_loss"] < constant["profiled_loss"]
    assert useful["shape_gain_over_constant"] > 0


def test_weights_are_effective_and_repeated_rows_preserve_result():
    raw, y, weights = np.array([-3.0, 0, 2]), np.array([0.0, 1, 3]), np.array([2, 1, 4])
    weighted = profile_count_loss(raw, y, weights)
    duplicated = profile_count_loss(np.repeat(raw, weights), np.repeat(y, weights), np.ones(weights.sum()))
    for key in weighted:
        if isinstance(weighted[key], float):
            assert weighted[key] == pytest.approx(duplicated[key])
    assert weighted["profiled_loss"] != pytest.approx(profile_count_loss(raw, y, np.ones(3))["profiled_loss"])


def test_zero_events_are_explicitly_unsupported():
    result = profile_count_loss([-3, 0], [0, 0], [0.5, 0.25])
    assert result["support"] == "no_positive_count"
    assert result["optimal_log_scale"] is None
    assert result["profiled_loss"] == result["shape_gain_over_constant"] == 0


@pytest.mark.parametrize(
    "raw,target,weight",
    [
        ([], [], []),
        ([0], [1, 2], [1]),
        ([np.nan], [0], [1]),
        ([0], [-1], [1]),
        ([0], [0], [0]),
        ([0], [0], [np.inf]),
        ([[0]], [[0]], [[1]]),
        ([1000], [1], [1]),
    ],
)
def test_invalid_arrays_fail(raw, target, weight):
    with pytest.raises(ValueError):
        profile_count_loss(raw, target, weight)
