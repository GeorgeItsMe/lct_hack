import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.flood_research import (
    baseline_raw,
    evaluate_tables,
    fingerprint,
    fit_baseline,
    parts_for,
    periods_for,
)
from moscollector.experiments.flood_verification import verified_evidence
from moscollector.prepare import sha256, write_json
from moscollector.research import mask


def test_separate_three_month_periods_and_purge():
    periods = periods_for("2025-11-01")
    assert periods["train"] == (pd.Timestamp("2022-01-01"), pd.Timestamp("2025-02-01"))
    assert periods["validation"] == (pd.Timestamp("2025-02-01"), pd.Timestamp("2025-05-01"))
    assert periods["calibration"] == (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-08-01"))
    assert periods["policy"] == (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-11-01"))
    rows = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2025-10-30 22:59:59", "2025-10-30 23:00:00", "2025-11-01 00:00:00"]),
            "eligible": True,
        }
    )
    assert mask(rows, *periods["policy"]).tolist() == [True, False, False]
    with pytest.raises(ValueError, match="June"):
        periods_for("2026-06-01")


def test_smoothed_baseline_and_unseen_fallbacks():
    rows = pd.DataFrame(
        {"object_id": [1, 1, 2], "as_of": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-02"])}
    )
    model = fit_baseline(rows, [1, 3, 2])
    assert model["global_mean"] == 2
    assert model["object_rates"]["1"] == (4 + 240 * 2) / 242
    query = pd.DataFrame(
        {"object_id": [1, 1, 999], "as_of": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-06"])}
    )
    expected = [(4 + 32 * model["object_rates"]["1"]) / 34, model["object_rates"]["1"], 2]
    np.testing.assert_allclose(np.exp(baseline_raw(query, model)), expected)


def test_baseline_cannot_read_labels_or_other_future_columns():
    rows = pd.DataFrame({"object_id": [1, 2], "as_of": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    model = fit_baseline(rows, [0, 1])
    expected = baseline_raw(rows, model)
    poisoned = rows.assign(target_flood=999, eligible=False, count_target=-100, flood_reports_1h=np.inf)
    np.testing.assert_array_equal(expected, baseline_raw(poisoned, model))
    assert fit_baseline(poisoned, [0, 1]) == model


def test_baseline_empty_invalid_and_zero_counts():
    rows = pd.DataFrame({"object_id": [1], "as_of": pd.to_datetime(["2024-01-01"])})
    for counts in ([], [-1], [np.nan]):
        with pytest.raises(ValueError, match="Invalid"):
            fit_baseline(rows, counts)
    with pytest.raises(ValueError, match="Invalid"):
        fit_baseline(rows.iloc[:0], [])
    assert baseline_raw(rows, fit_baseline(rows, [0]))[0] == -20


def test_fingerprint_tracks_order_labels_features_and_dtypes():
    rows = pd.DataFrame({"a": [1, 2], "b": [3.0, np.nan]})
    digest = fingerprint(rows)
    assert fingerprint(rows.reset_index(drop=True)) == digest
    assert fingerprint(rows.iloc[::-1]) != digest
    assert fingerprint(rows.assign(a=[1, 3])) != digest
    assert fingerprint(rows.astype({"a": float})) != digest
    assert fingerprint(rows[["b", "a"]]) != digest


def tiny_parts():
    times = pd.to_datetime(["2024-01-01", "2025-02-02", "2025-05-02", "2025-08-02", "2025-11-02"])
    frame = pd.DataFrame({"object_id": 1, "as_of": times, "eligible": True, "target_flood": 0, "x": 1.0})
    episodes = pd.DataFrame(
        {"object_id": pd.Series(dtype="int64"), "start_ts": pd.Series(dtype="datetime64[ns]")}
    )
    return frame, episodes


def test_dense_parts_preserve_all_shared_negative_anchors():
    frame, episodes = tiny_parts()
    dense = frame.iloc[2:].copy()
    result = parts_for(frame, dense, episodes, "2025-11-01")
    assert all(len(r) == 1 for r in result.values())
    # The unchanged empty positive cohort must not hide a lost negative row.
    with pytest.raises(AssertionError):
        parts_for(frame, dense.iloc[1:], episodes, "2025-11-01")
    altered = dense.copy()
    altered.loc[altered.index[0], "x"] = 99
    with pytest.raises(AssertionError):
        parts_for(frame, altered, episodes, "2025-11-01")


def test_count_target_preserves_half_open_horizon():
    frame, _ = tiny_parts()
    t = frame.iloc[0].as_of
    episodes = pd.DataFrame({"object_id": [1, 1], "start_ts": [t, t + pd.Timedelta(hours=24)]})
    frame.loc[0, "target_flood"] = 1
    parts = parts_for(frame, frame.iloc[2:].copy(), episodes, "2025-11-01")
    assert parts["train"].count_target.tolist() == [1]
    frame.loc[0, "target_flood"] = 0
    with pytest.raises(ValueError, match="target"):
        parts_for(frame, frame.iloc[2:].copy(), episodes, "2025-11-01")


def test_future_test_predictions_do_not_select_policy():
    times = pd.date_range("2025-01-01", periods=8, freq="h")
    table = pd.DataFrame({"object_id": 1, "as_of": times, "probability": 0.8, "expected_count": 0.8})
    episodes = pd.DataFrame({"object_id": [1], "start_ts": [times[-1]]})
    first, curve = evaluate_tables({"policy": table.copy(), "test": table.copy()}, episodes)
    second, other = evaluate_tables(
        {"policy": table.copy(), "test": table.assign(probability=0, expected_count=0)}, episodes
    )
    assert len(curve) == 120 and curve == other
    assert first["policy"] == second["policy"]
    assert first["scores"]["test"] != second["scores"]["test"]


def test_proof_rejects_incomplete_or_changed_evidence(tmp_path):
    write_json(tmp_path / "plan.json", {"source_hashes": {}, "code_hashes": {}})
    proof = {
        "status": "exact_replay_passed",
        "folds": [],
        "models": 0,
        "forecast_and_frontier_files": 0,
        "plan_sha256": sha256(tmp_path / "plan.json"),
    }
    write_json(tmp_path / "weight-replay.json", proof)
    with pytest.raises(ValueError, match="Incomplete"):
        verified_evidence(tmp_path)
