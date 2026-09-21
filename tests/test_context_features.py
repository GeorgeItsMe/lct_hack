import numpy as np
import pandas as pd
import pytest

from moscollector.context_features import enrich_context


def fixture():
    row = {
        "channel_count": 10,
        "reporting_channels_24h": 2,
        "events_24h": 100,
        "hour": 12,
        "day_of_week": 2,
        "month": 5,
    }
    for signal in (
        "fault_reports",
        "power_reports",
        "unknown_reports",
        "smoke_reports",
        "flood_reports",
        "access_reports",
        "ambiguous",
        "technical_codes",
    ):
        for window in (6, 24, 168):
            row[f"{signal}_{window}h"] = 1
    for kind in ("fault", "fire", "flood", "access"):
        row[f"past_{kind}_recency_h"] = 24
        for window in (168, 720):
            row[f"past_{kind}_episodes_{window}h"] = 1
        row[f"target_{kind}"] = 0
    return pd.DataFrame(
        [
            {**row, "as_of": pd.Timestamp("2026-01-02"), "object_id": obj, "parent_id": parent}
            for obj, parent in [(1, 1), (2, 1), (3, 2)]
        ]
    )


def test_neighbor_context_excludes_self_other_parent_future_and_targets():
    data = fixture()
    data.loc[0, "fault_reports_24h"] = 10
    data.loc[1, "fault_reports_24h"] = 4
    data.loc[2, "fault_reports_24h"] = 1000
    before = enrich_context(data)
    assert before.loc[0, "ctx_neighbor_fault_reports_24h"] == pytest.approx(0.4)
    assert before.loc[1, "ctx_neighbor_fault_reports_24h"] == 1
    assert before.loc[2, "ctx_neighbor_fault_reports_24h"] == 0
    future = data.copy()
    future["as_of"] += pd.Timedelta(days=1)
    future["fault_reports_24h"] = 99999
    changed = pd.concat([data, future], ignore_index=True)
    changed["target_fault"] = 1
    actual = enrich_context(changed).iloc[:3].filter(like="ctx_")
    pd.testing.assert_frame_equal(before.filter(like="ctx_"), actual)


def test_context_is_snapshot_equivalent_and_rejects_duplicate_objects():
    data = fixture()
    with pytest.raises(ValueError):
        enrich_context(pd.concat([data, data]))
    result = enrich_context(data)
    assert np.isfinite(result.filter(like="ctx_").to_numpy()).all()
    assert result.loc[0, "ctx_fault_decay_24h"] == pytest.approx(np.exp(-1))
