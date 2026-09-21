import numpy as np
import pytest

from moscollector.two_part_count_research import extra_calibration, factor_mean, positive_extra_targets


def test_positive_training_condition_does_not_remove_inference_rows():
    mask, target = positive_extra_targets([0, 1, 3, 0, 2])
    np.testing.assert_array_equal(mask, [False, True, True, False, True])
    np.testing.assert_array_equal(target, [0, 2, 1])
    # Calibration excludes zero-count rows only when estimating conditional recurrence.
    cal = extra_calibration([0, 1, 3, 0, 2], [100, 1, 2, 100, 3])
    assert cal["scale"] == 0.5
    assert cal["positive_rows"] == 3
    result = factor_mean([0, 0.2, 1, 0.7, 0.5], [100, 1, 2, 100, 3], cal["scale"])
    np.testing.assert_allclose(result, [0, 0.3, 2, 35.7, 1.25])
    assert len(result) == 5


def test_zero_and_missing_recurrence_have_explicit_calibration_behavior():
    assert extra_calibration([0, 0], [1, 5])["status"] == "unsupported_no_positive_calibration_rows"
    cal = extra_calibration([0, 1, 1], [5, 0, 0])
    assert cal["scale"] == 0
    np.testing.assert_array_equal(factor_mean([0.1, 1], [0, 100], cal["scale"]), [0.1, 1])
    assert extra_calibration([2], [0])["status"] == "unsupported_zero_prediction_denominator"


def test_invalid_counts_and_factor_inputs_are_rejected():
    for values in ([1, -1], [0.5], [np.nan], [[1, 2]]):
        with pytest.raises(ValueError):
            positive_extra_targets(values)
    for values in ([np.nan], [-1], [1, 2]):
        with pytest.raises(ValueError):
            extra_calibration([1], values)
    for p, extra, scale in (
        ([1.1], [1], 1),
        ([np.nan], [1], 1),
        ([1], [-1], 1),
        ([1], [1], -1),
        ([1], [1], np.inf),
        ([1, 0], [1], 1),
    ):
        with pytest.raises(ValueError):
            factor_mean(p, extra, scale)
