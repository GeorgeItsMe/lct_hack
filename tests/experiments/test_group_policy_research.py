from itertools import product

import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.fine_cadence_research import policy_alerts
from moscollector.experiments.goal90_research import pooled, primary_score
from moscollector.experiments.group_policy_research import apply_group_policy, exposure_days, joint_choice


def counts(tp, alerts, events):
    return {"true_alerts": tp, "alerts": alerts, "eligible_episodes": events}


def test_joint_search_matches_brute_force_with_fixed_unknown_events_and_budget():
    rng = np.random.default_rng(27)
    for _ in range(40):
        frontiers = []
        for events in (30, 12):
            rows = [counts(0, 0, events)]
            for _ in range(5):
                tp = int(rng.integers(0, events + 1))
                rows.append(counts(tp, tp + int(rng.integers(0, 35)), events))
            frontiers.append(rows)
        fixed, exposure = counts(2, 6, 8), 80
        options = []
        for i, j in product(range(6), repeat=2):
            m = pooled([frontiers[0][i], frontiers[1][j], fixed])
            if (m["alerts"] - m["true_alerts"]) / exposure <= 0.25:
                options.append((i, j, m))
        supported = [r for r in options if r[2]["alerts"] >= 10]
        best = max(
            supported or options,
            key=lambda r: (primary_score(r[2]), r[2]["f1"], r[2]["recall"], r[2]["precision"]),
        )
        chosen = joint_choice(frontiers, exposure, fixed)
        assert chosen["indices"] == list(best[:2])
        assert all(chosen["scores"][k] == v for k, v in best[2].items())
        assert chosen["scores"]["eligible_episodes"] == 50
    tied = joint_choice([[counts(1, 1, 1)] * 2] * 2, 1)
    assert tied["indices"] == [0, 0]


def test_group_policy_matches_unsplit_policy_and_preserves_unknown_objects():
    at = pd.date_range("2025-01-01", periods=40, freq="15min")
    pred = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "object_id": obj,
                        "object_kind": group,
                        "as_of": at,
                        "source_time": at.floor("h"),
                        "expected_count": 2.0,
                        "probability": 0.8,
                    }
                )
                for obj, group in ((1, "controlHouse"), (2, "guardObject"), (3, None))
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=2)
        .reset_index(drop=True)
    )
    eps = pd.DataFrame(
        {
            "object_id": [1, 1, 2, 3],
            "start_ts": pd.to_datetime(
                ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00", "2025-01-01 01:00"]
            ),
        }
    )
    policy = {"capacity": 1, "margin": 1, "floor": 0.5}
    original = policy_alerts(pred, eps, policy)
    np.testing.assert_array_equal(
        apply_group_policy(pred, eps, {"controlHouse": policy, "guardObject": policy}, policy), original
    )
    actual = apply_group_policy(pred, eps, {"controlHouse": {**policy, "floor": 1}}, policy)
    assert not actual[pred.object_kind.eq("controlHouse")].any()
    np.testing.assert_array_equal(actual[pred.object_kind.isna()], original[pred.object_kind.isna()])
    assert exposure_days(pred) == 30 / 24
    pred.loc[pred.object_id.eq(1).idxmax(), "object_kind"] = "guardObject"
    with pytest.raises(ValueError):
        apply_group_policy(pred, eps, {}, policy)


def test_joint_search_rejects_invalid_or_changing_cohort():
    for rows in ([counts(2, 1, 2)], [counts(0, 0, 1), counts(0, 0, 2)], [counts(0.5, 1, 2)], []):
        with pytest.raises(ValueError):
            joint_choice([rows, [counts(0, 0, 1)]], 1)
    with pytest.raises(ValueError):
        joint_choice([[counts(0, 2, 1)]] * 2, 1)
    with pytest.raises(ValueError):
        joint_choice([[counts(0, 0, 1)]] * 2, 0)
