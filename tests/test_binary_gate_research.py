import numpy as np
import pandas as pd
import pytest

from moscollector.binary_gate_research import FLOORS, select_gate_policy
from moscollector.fine_cadence_research import evaluator_for, policy_alerts


def test_cached_threshold_masks_exactly_match_direct_probability_gating():
    start = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(103)
    pred = pd.concat(
        [
            pd.DataFrame({"object_id": obj, "as_of": pd.date_range(start, periods=120, freq="15min")})
            for obj in (1, 2)
        ],
        ignore_index=True,
    )
    pred["probability"] = rng.choice([0.0, 0.2, 0.5, 0.8, 1.0], len(pred))
    pred["expected_count"] = rng.choice([0.0, 0.5, 1.0, 3.0, 5.0], len(pred))
    pred = pred.sample(frac=1, random_state=5).reset_index(drop=True)
    episodes = pd.DataFrame(
        {"object_id": [1, 1, 1, 2, 2], "start_ts": start + pd.to_timedelta([0.5, 1.75, 27, 24, 28], unit="h")}
    )
    chosen, frontier = select_gate_policy(pred, episodes, exposure=2.5)
    assert len(frontier) == 4 * 6 * len(FLOORS)
    evaluator = evaluator_for(pred, episodes, 0.25, 2.5)
    for policy in frontier:
        direct = evaluator.evaluate(policy_alerts(pred, episodes, policy), 0.5, 0.25)
        for key, value in direct.items():
            assert policy[key] == value
    assert chosen["gate_disabled"] == (chosen["floor"] == 0)
    assert chosen["gated_policy_rows"] == int((pred.probability < chosen["floor"]).sum())


def test_cached_gates_do_not_hide_invalid_probabilities_or_counts():
    pred = pd.DataFrame(
        {
            "object_id": [1],
            "as_of": pd.to_datetime(["2025-01-01"]),
            "probability": [0.5],
            "expected_count": [1.0],
        }
    )
    episodes = pd.DataFrame(
        {"object_id": pd.Series(dtype=int), "start_ts": pd.Series(dtype="datetime64[ns]")}
    )
    for invalid in (np.nan, -0.1, 1.1):
        with pytest.raises(ValueError, match="Invalid"):
            select_gate_policy(pred.assign(probability=invalid), episodes, 1)
    with pytest.raises(ValueError, match="Invalid"):
        select_gate_policy(pred.assign(expected_count=-1), episodes, 1)
