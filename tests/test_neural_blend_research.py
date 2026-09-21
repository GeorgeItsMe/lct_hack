import numpy as np
import pandas as pd
import pytest

from moscollector.neural_blend_research import blend_predictions


def forecasts():
    return pd.DataFrame(
        {
            "object_id": [1, 2],
            "as_of": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "probability": [0.2, 0.8],
            "expected_count": [0.5, 3.0],
            "alert": [1, 0],
        }
    )


def test_blend_rejects_cross_object_and_time_misalignment():
    first = forecasts()
    for changed in (
        first.iloc[::-1],
        first.assign(object_id=[2, 1]),
        first.assign(as_of=first.as_of + pd.Timedelta(hours=1)),
    ):
        with pytest.raises(ValueError, match="opportunities differ"):
            blend_predictions({"a": first, "b": changed}, {"a": 0.5, "b": 0.5})
    duplicated = pd.concat([first.iloc[:1]] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        blend_predictions({"a": duplicated}, {"a": 1})


def test_blend_ignores_saved_decisions_and_keeps_probability_count_domains():
    a = forecasts()
    b = a.assign(probability=[0.8, 0.2], expected_count=[2.5, 1.0], alert=[0, 1])
    expected = blend_predictions({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    np.testing.assert_allclose(expected.probability, [0.5, 0.5])
    np.testing.assert_allclose(expected.expected_count, [1.5, 2.0])
    actual = blend_predictions(
        {"a": a.assign(alert=np.nan, future_target=100), "b": b.drop(columns="alert")}, {"a": 0.5, "b": 0.5}
    )
    pd.testing.assert_frame_equal(actual, expected)
    for weights in ({"a": -1, "b": 2}, {"a": 0.2, "b": 0.2}, {"a": np.nan}, {}):
        with pytest.raises(ValueError, match="convex"):
            blend_predictions({"a": a, "b": b}, weights)
    for changed in (a.assign(probability=1.1), a.assign(expected_count=-1), a.assign(expected_count=np.nan)):
        with pytest.raises(ValueError, match="Invalid component"):
            blend_predictions({"a": changed}, {"a": 1})
