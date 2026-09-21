import json

import numpy as np
import pandas as pd
import pytest

from moscollector import features
from moscollector.alert_diagnostics import EventEvaluator


def test_alternate_cadence_preserves_source_outputs_and_shared_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "PROCESSED", tmp_path)
    monkeypatch.setattr(features, "ARTIFACTS", tmp_path)
    index = pd.date_range("2027-01-01", periods=288, freq="h")
    rng = np.random.default_rng(12)
    hourly = pd.DataFrame({c: rng.integers(0, 3, len(index)) for c in features.COUNT_COLUMNS})
    hourly["events"] = 5
    hourly["object_id"] = 1
    hourly["hour"] = index
    for c in ("temperature", "temperature_max", "gas"):
        hourly[c] = rng.normal(10, 2, len(index))
    hourly["reporting_channels"] = 2
    hourly.to_parquet(tmp_path / "hourly-2027.parquet")
    pd.DataFrame({"channel_id": [1], "object_id": [1], "sensor_type": ["Датчик дыма"]}).to_parquet(
        tmp_path / "channels.parquet"
    )
    pd.DataFrame({"object_id": [1], "parent_id": [0], "object_kind": ["guardObject"]}).to_parquet(
        tmp_path / "objects.parquet"
    )
    episode = pd.DataFrame(
        {
            "channel_id": [1],
            "object_id": [1],
            "kind": ["fire"],
            "start_ts": [index[210]],
            "end_ts": [index[212]],
            "duration_seconds": [7200],
            "right_censored": [False],
            "event_id": [1],
            "sensor_name": ["sensor"],
        }
    )
    episode.to_parquet(tmp_path / "episodes-2027.parquet")
    (tmp_path / "audit-2027.json").write_text(
        json.dumps({"summary": {"start": str(index.min()), "end": str(index.max())}})
    )
    features.build_dataset([2027])
    frozen = {
        p: (tmp_path / p).read_bytes() for p in ("features.parquet", "episodes.parquet", "feature_audit.json")
    }
    features.build_dataset(
        [2027],
        step_hours=1,
        output_path=tmp_path / "hourly.parquet",
        episode_output_path=tmp_path / "hourly-episodes.parquet",
        audit_output_path=tmp_path / "hourly-audit.json",
        before="2027-01-13",
    )
    assert all((tmp_path / p).read_bytes() == content for p, content in frozen.items())
    base = pd.read_parquet(tmp_path / "features.parquet")
    dense = pd.read_parquet(tmp_path / "hourly.parquet")
    shared = dense[dense.as_of.isin(base.as_of)].reset_index(drop=True)
    pd.testing.assert_frame_equal(base, shared)


def test_cutoff_or_partial_output_override_cannot_overwrite_frozen_files():
    with pytest.raises(ValueError, match="all three"):
        features.build_dataset([2025], before="2026-06-01")


def test_alert_exposure_uses_actual_forecast_cadence():
    pred = pd.DataFrame({"object_id": [1] * 24, "as_of": pd.date_range("2026-01-01", periods=24, freq="h")})
    empty = pd.DataFrame({"object_id": [], "start_ts": pd.to_datetime([])})
    result = EventEvaluator(pred, empty, cadence_hours=1).evaluate(np.ones(24), 0.5, 24)
    assert result["false_alerts_per_object_day"] == 1
