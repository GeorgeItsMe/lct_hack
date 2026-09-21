import numpy as np
import pandas as pd
import pytest

from moscollector.precision_research import choose_row_policy, goal_score, periods_for, row_operating
from moscollector.research import mask


def test_precision_objective_penalizes_each_unmet_goal():
    assert goal_score({"precision": 0.75, "recall": 0.5}) == 1
    assert goal_score({"precision": 1, "recall": 0.1}) == 0.2
    assert goal_score({"precision": 0.15, "recall": 1}) == pytest.approx(0.2)


def test_precision_policy_handles_ties_and_is_applied_without_test_labels():
    y = np.r_[np.ones(12), np.zeros(20)]
    p = np.r_[np.repeat(0.9, 10), np.repeat(0.6, 6), np.repeat(0.1, 16)]
    policy = choose_row_policy(y, p)
    assert policy["threshold"] == 0.6
    assert policy["precision"] == 0.75
    assert policy["recall"] == 1
    test = row_operating(np.zeros(32), p, policy["threshold"])
    assert test["precision"] == 0
    assert test["threshold"] == 0.6


def test_splits_purge_future_targets_before_each_boundary():
    periods = periods_for("2026-05-01", "fault")
    for begin, end in periods.values():
        times = pd.date_range(end - pd.Timedelta(hours=27), periods=12, freq="3h")
        frame = pd.DataFrame({"as_of": times, "eligible": True})
        kept = frame.loc[mask(frame, begin, end), "as_of"]
        assert ((kept + pd.Timedelta(hours=25)) < end).all()
