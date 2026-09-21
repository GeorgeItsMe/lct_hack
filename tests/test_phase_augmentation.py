import numpy as np
import pandas as pd
import pytest

from moscollector.count_research import episode_counts
from moscollector.phase_augmentation import paired_weights, shifted_snapshots
from moscollector.phase_training_research import assemble
from moscollector.research import mask


def test_shifted_rows_require_adjacent_coverage_drop_labels_and_do_not_change_hour():
    at = pd.Timestamp("2025-01-01")
    frame = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 2, 2, 2],
            "as_of": at + pd.to_timedelta([0, 3, 9, 0, 3, 6], unit="h"),
            "eligible": [True, True, True, True, False, True],
            "target_access": [1] * 6,
            "temperature": [10, 20, 30, -10, -20, -30],
        }
    )
    out = shifted_snapshots(frame.sample(frac=1, random_state=1))
    assert len(out) == 1
    assert "target_access" not in out
    assert out.object_id.tolist() == [1]
    assert out.source_time.tolist() == [at]
    assert out.as_of.dt.floor("h").eq(out.source_time).all()
    assert (out.as_of - out.source_time).dt.total_seconds().iloc[0] in [900, 1800, 2700]
    assert out.temperature.tolist() == [10]
    changed = frame.copy()
    changed["target_access"] = 0
    pd.testing.assert_frame_equal(out, shifted_snapshots(changed))
    with pytest.raises(ValueError, match="Duplicate"):
        shifted_snapshots(pd.concat([frame, frame]))


def test_pair_weights_preserve_each_source_weight_and_reject_cross_split_companions():
    base = pd.DataFrame({"object_id": [1, 2, 3], "as_of": pd.to_datetime(["2025-01-01"] * 3)})
    shifted = base.iloc[[2, 0]].rename(columns={"as_of": "source_time"})
    a, b = paired_weights(base, shifted)
    np.testing.assert_array_equal(a, [0.5, 1.0, 0.5])
    np.testing.assert_array_equal(b, [0.5, 0.5])
    assert a.sum() + b.sum() == len(base)
    with pytest.raises(ValueError, match="split"):
        paired_weights(base.iloc[:2], shifted)


def test_shifted_targets_use_new_window_and_temporal_purge_is_reapplied():
    begin, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-04")
    source = end - pd.Timedelta(hours=25, minutes=15)
    base = pd.DataFrame({"object_id": [1], "as_of": [source], "eligible": [True]})
    shifted = base.assign(as_of=source + pd.Timedelta(minutes=30), source_time=source)
    assert mask(base, begin, end).tolist() == [True]
    assert mask(shifted, begin, end).tolist() == [False]
    episodes = pd.DataFrame({"object_id": [1], "start_ts": [source + pd.Timedelta(minutes=5)]})
    assert episode_counts(base, episodes).tolist() == [1]
    assert episode_counts(shifted, episodes).tolist() == [0]


def test_training_assembly_recomputes_count_labels_and_preserves_source_weight():
    at = pd.Timestamp("2025-01-01")
    base = pd.DataFrame(
        {
            "object_id": [1, 1],
            "as_of": [at, at + pd.Timedelta(hours=3)],
            "eligible": [True, True],
            "events_1h": [10, 20],
            "target_access": [1, 1],
        }
    )
    extra = pd.DataFrame(
        {
            "object_id": [1],
            "as_of": [at + pd.Timedelta(minutes=15)],
            "source_time": [at],
            "eligible": [True],
            "events_1h": [12],
        }
    )
    episodes = pd.DataFrame(
        {
            "object_id": [1, 1],
            "start_ts": [at + pd.Timedelta(minutes=5), at + pd.Timedelta(hours=3, minutes=10)],
        }
    )
    dates = (at, at + pd.Timedelta(days=3))
    combined, size = assemble(base, extra, episodes, ["object_id", "events_1h"], dates, "access")
    assert combined.training_count.tolist() == [2, 1, 1]
    assert combined.training_weight.tolist() == [0.5, 0.5, 1]
    assert size == {"original_rows": 2, "companion_rows": 1, "total_rows": 3, "weight_sum": 2.0}
    wrong = base.copy()
    wrong["target_access"] = 0
    with pytest.raises(ValueError, match="target changed"):
        assemble(wrong, extra, episodes, ["object_id", "events_1h"], dates, "access")
