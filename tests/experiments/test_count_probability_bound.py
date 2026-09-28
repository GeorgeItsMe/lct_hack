import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.count_probability_bound import bounded_count, project


def test_projection_only_raises_inconsistent_means_and_is_idempotent():
    mean, probability = np.array([0.0, 0.08, 0.5, 2.0]), np.array([0.0, 0.8, 0.2, 0.99])
    old_mean, old_probability = mean.copy(), probability.copy()
    result = bounded_count(mean, probability)
    np.testing.assert_array_equal(result, [0, 0.8, 0.5, 2])
    np.testing.assert_array_equal(bounded_count(result, probability), result)
    np.testing.assert_array_equal(mean, old_mean)
    np.testing.assert_array_equal(probability, old_probability)
    assert not np.shares_memory(mean, result)


def test_frame_projection_preserves_all_rows_and_other_values():
    table = pd.DataFrame(
        {
            "object_id": [1, 2, 1],
            "as_of": pd.date_range("2025-01-01", periods=3, freq="h"),
            "expected_count": [0.1, 0, 3],
            "probability": [0.9, 0, 0.2],
            "raw": [-2.0, -30.0, 1.0],
        },
        index=[7, 2, 3],
    )
    before = table.copy(deep=True)
    result, stats = project(table)
    pd.testing.assert_frame_equal(table, before, check_exact=True)
    pd.testing.assert_frame_equal(
        result.drop(columns=["expected_count", "source_expected_count"]),
        table.drop(columns="expected_count"),
        check_exact=True,
    )
    np.testing.assert_array_equal(result.source_expected_count, table.expected_count)
    assert stats["rows"] == 3 and stats["raised_rows"] == 1
    assert stats["maximum_raise"] == pytest.approx(0.8)
    assert stats["projected_count_sum"] == pytest.approx(3.9)
    with pytest.raises(ValueError, match="provenance"):
        project(result)


@pytest.mark.parametrize(
    "mean,probability",
    [
        ([-1], [0]),
        ([1], [1.1]),
        ([1], [-0.1]),
        ([np.nan], [0]),
        ([0], [np.inf]),
        ([0, 1], [0]),
        ([[0]], [[0]]),
    ],
)
def test_invalid_forecasts_are_rejected(mean, probability):
    with pytest.raises(ValueError, match="Invalid"):
        bounded_count(mean, probability)


def test_empty_arrays_are_valid_and_future_rows_do_not_change_prefix():
    assert len(bounded_count([], [])) == 0
    a = bounded_count([0, 0.1, 3], [0.9, 0.5, 0.2])
    b = bounded_count([0, 0.1, 300], [0.9, 0.5, 1])
    np.testing.assert_array_equal(a[:2], b[:2])
