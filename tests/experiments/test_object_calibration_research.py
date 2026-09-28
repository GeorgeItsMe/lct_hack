import numpy as np
import pytest

from moscollector.experiments.object_calibration_research import correction_factors


def test_shrinkage_is_per_object_bounded_and_tends_to_identity():
    observed, predicted = [16, 16, 0, 0], [8, 8, 8, 8]
    weak = correction_factors([1, 1, 2, 2], observed, predicted, 5)
    strong = correction_factors([1, 1, 2, 2], observed, predicted, 80)
    assert 1 < strong[1] < weak[1]
    assert weak[2] < strong[2] < 1
    assert correction_factors([1, 1], [10000, 10000], [0, 0], 5)[1] == 4
    assert correction_factors([1, 1], [0, 0], [10000, 10000], 5)[1] == 0.25


def test_correct_counts_have_identity_and_invalid_inputs_fail():
    assert correction_factors([1, 2], [8, 0], [8, 0], 5) == {1: 1, 2: 1}
    with pytest.raises(ValueError, match="finite"):
        correction_factors([1], [np.nan], [1], 5)
    with pytest.raises(ValueError, match="Invalid"):
        correction_factors([1], [1, 2], [1], 5)
