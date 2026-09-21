import numpy as np

from moscollector.uncertainty import bootstrap, ratios


def test_paired_identical_predictions_have_zero_difference_in_every_resample():
    result = bootstrap([[3, 5, 8, 3, 5], [0, 4, 0, 0, 4], [5, 8, 9, 5, 8]], 1000, 42)
    difference = result["percentile_95"]["f1_difference"]
    assert difference == {"low": 0, "high": 0, "valid_replicates": 1000}


def test_zero_denominators_are_not_perfect_or_failed_predictions():
    actual = ratios(np.array([[0, 0, 0, 0, 0], [0, 0, 2, 0, 1]]))
    assert all(np.isnan(values[0]) for values in actual.values())
    assert np.isnan(actual["precision"][1])
    assert actual["recall"][1] == actual["f1"][1] == 0
    result = bootstrap([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], 1000, 42)
    assert 0 < result["percentile_95"]["f1"]["valid_replicates"] < 1000
    assert result["percentile_95"]["f1"]["low"] == 1


def test_duplicate_cluster_draws_preserve_whole_history_and_known_difference():
    # All clusters have proportional counts. Repeating any whole cluster keeps
    # F1 = 2/3 and baseline F1 = 1/3, regardless of cluster size or draw.
    counts = [[2, 3, 3, 1, 3], [4, 6, 6, 2, 6], [6, 9, 9, 3, 9]]
    result = bootstrap(counts, 1000, 42)
    assert np.isclose(result["percentile_95"]["f1"]["low"], 2 / 3)
    assert np.isclose(result["percentile_95"]["f1"]["high"], 2 / 3)
    assert np.isclose(result["percentile_95"]["f1_difference"]["low"], 1 / 3)
    assert np.isclose(result["percentile_95"]["f1_difference"]["high"], 1 / 3)
