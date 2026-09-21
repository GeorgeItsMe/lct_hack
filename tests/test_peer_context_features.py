import numpy as np
import pandas as pd
import pytest

from moscollector.peer_context_features import COLUMNS, SIGNALS, WINDOWS, peer_context


def source():
    frame = pd.DataFrame(
        {"object_id": [1, 2, 3, 4], "parent_id": ["a", "a", "b", "c"], "as_of": pd.Timestamp("2026-01-10")}
    )
    for hours in WINDOWS:
        for signal in SIGNALS:
            frame[f"{signal}_{hours}h"] = np.array([1, 2, 4, 0], dtype=float) * hours
    return frame


def test_peer_scopes_exclude_self_and_own_group():
    frame = source()
    expected = peer_context(frame)
    assert len(COLUMNS) == 64
    np.testing.assert_array_equal(expected.peer_parent_events_1h, [2, 1, 0, 0])
    np.testing.assert_array_equal(expected.peer_other_groups_events_1h, [4, 4, 3, 7])
    np.testing.assert_array_equal(expected.peer_other_groups_events_active_objects_6h, [1, 1, 2, 3])
    columns = [f"{s}_{h}h" for s in SIGNALS for h in WINDOWS]
    frame.loc[0, columns] *= 1000
    pd.testing.assert_series_equal(peer_context(frame).iloc[0], expected.iloc[0])
    shuffled = source().sample(frac=1, random_state=12)
    pd.testing.assert_frame_equal(peer_context(shuffled).sort_index(), expected)


def test_peer_context_ignores_future_rows_labels_and_eligibility():
    frame = source()
    expected = peer_context(frame)
    future = source().assign(as_of=pd.Timestamp("2026-01-11"), eligible=False, target_fault=1)
    for hours in WINDOWS:
        future[f"fault_reports_{hours}h"] = 999999
    extended = pd.concat([frame.assign(eligible=False, target_fault=1), future], ignore_index=True)
    pd.testing.assert_frame_equal(peer_context(extended).iloc[:4], expected)
    pd.testing.assert_frame_equal(peer_context(frame.assign(eligible=True, target_fault=0)), expected)


def test_missing_peer_windows_propagate_only_to_the_relevant_scope():
    frame = source()
    frame.loc[0, "fault_reports_6h"] = np.nan
    result = peer_context(frame)
    assert result.loc[0, "peer_parent_fault_reports_6h"] == 12
    assert np.isnan(result.loc[1, "peer_parent_fault_reports_6h"])
    assert result.loc[1, "peer_other_groups_fault_reports_6h"] == 24
    assert np.isnan(result.loc[2, "peer_other_groups_fault_reports_6h"])
    assert result.loc[2, "peer_parent_fault_reports_6h"] == 0
    assert np.isnan(result.loc[1, "peer_parent_fault_reports_active_objects_6h"])
    for changed in (source().assign(parent_id=None), pd.concat([source(), source().iloc[:1]])):
        with pytest.raises(ValueError, match="Missing or duplicate"):
            peer_context(changed)
    for value in (-1, np.inf):
        frame.loc[0, "events_1h"] = value
        with pytest.raises(ValueError, match="Invalid past"):
            peer_context(frame)
