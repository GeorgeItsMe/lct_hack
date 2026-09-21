import numpy as np
import pytest

from moscollector.object_kind_research import apply_calibration, calibration_pair, route_raw


def test_group_routing_preserves_shuffled_rows_and_unknown_fallback():
    original = [10, 20, 30, 40, 50]
    groups = ["guardObject", "unknown", "controlHouse", "guardObject", "controlHouse"]
    np.testing.assert_array_equal(route_raw(original, groups, {"guardObject": [1, 4]}), [1, 20, 30, 4, 50])
    np.testing.assert_array_equal(
        route_raw(original, groups, {"guardObject": [1, 4], "controlHouse": [3, 5]}), [1, 20, 3, 4, 5]
    )
    for overrides in ({"guardObject": [1]}, {"guardObject": [1, np.nan]}):
        with pytest.raises(ValueError):
            route_raw(original, groups, overrides)


def test_group_calibration_does_not_drop_unknown_or_zero_count_rows():
    global_cal = {"binary": {"slope": 1, "intercept": 0}, "scale": 2}
    group_cal = {"binary": {"slope": 0, "intercept": 0}, "scale": 0}
    p, n = apply_calibration(
        np.log([2, 3, 4]), ["known", "unknown", "known"], global_cal, {"known": group_cal}
    )
    np.testing.assert_allclose(p, [0.5, 0.75, 0.5])
    np.testing.assert_allclose(n, [0, 6, 0])
    assert calibration_pair([0, 0, 0], [0, 1, 2])["scale"] == 1
    assert calibration_pair([0, 0], [0, 0])["scale"] == 0


def test_calibration_rejects_invalid_data():
    for raw, counts in (([], []), ([1], [-1]), ([np.nan], [1]), ([1], [np.inf]), ([1, 2], [1])):
        with pytest.raises(ValueError):
            calibration_pair(raw, counts)
    with pytest.raises(ValueError):
        apply_calibration([np.nan], ["x"], {}, {})
