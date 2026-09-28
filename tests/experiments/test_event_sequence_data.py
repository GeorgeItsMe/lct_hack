import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.channel_novelty_features import SIGNALS
from moscollector.experiments.event_sequence_data import (
    encode_events,
    event_batch,
    fit_event_codec,
    history_bounds,
    prepare_events,
    selected_source_mask,
)


def source():
    start = pd.Timestamp("2025-01-01")
    raw = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 1, 1, 1, 2],
            "channel_id": [1, 1, 2, 1, 3, 3, 4],
            "signal": ["access", "access", "fire", "access", "fault", "fault", "fire"],
            "ts": start + pd.to_timedelta([0, 1, 1, 2, 3, 4, 1], unit="h"),
        }
    )
    catalog = pd.DataFrame(
        {
            "channel_id": [1, 2, 3, 4],
            "object_id": [1, 1, 1, 2],
            "sensor_type": ["door", "smoke", "heat", "smoke"],
        }
    )
    return start, raw, catalog


def test_bounds_exclude_present_future_other_objects_and_ignore_target_columns():
    start, raw, cat = source()
    events = prepare_events(raw, cat)
    queries = pd.DataFrame(
        {
            "object_id": [1, 2, 1, 99],
            "as_of": start + pd.to_timedelta([2, 2, 3, 2], unit="h"),
            "target_fault": 1,
            "eligible": True,
        }
    )
    bounds = history_bounds(events, queries, hours=2)
    np.testing.assert_array_equal(
        bounds, history_bounds(events, queries.assign(target_fault=0, eligible=False), hours=2)
    )
    selected = selected_source_mask(bounds[:1], len(events), per_signal=16)
    assert set(events.loc[selected].channel_id) == {1, 2}
    assert events.loc[selected].ts.max() < queries.as_of.iloc[0]
    assert len(events.loc[selected]) == 3
    assert int((bounds[1, :, 1] - bounds[1, :, 0]).sum()) == 1
    assert int((bounds[3, :, 1] - bounds[3, :, 0]).sum()) == 0
    # [t-2h,t) includes the left endpoint but excludes the fault at exactly t.
    selected = selected_source_mask(bounds[2:3], len(events), per_signal=16)
    assert set(events.loc[selected].ts) == {start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=2)}


def test_family_cap_keeps_rare_signal_and_vocabulary_is_training_history_only():
    start, raw, cat = source()
    events = prepare_events(raw, cat)
    query = pd.DataFrame({"object_id": [1], "as_of": [start + pd.Timedelta(hours=3)]})
    bounds = history_bounds(events, query)
    codec = fit_event_codec(events, bounds, per_signal=1)
    assert codec["vocabulary"]["channel_id"] == {"1": 1, "2": 2}
    assert "heat" not in codec["vocabulary"]["sensor_type"]
    assert codec["training_onset_rows"] == 2
    encoded = encode_events(events, codec)
    assert np.all(encoded[events.channel_id.eq(3), 0] == 0)
    assert np.all(encoded[events.channel_id.eq(3), 1] == 0)
    categories, numeric, lengths = event_batch(
        encoded,
        events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        bounds,
        query.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        per_signal=1,
    )
    assert lengths.tolist() == [2]
    assert categories[0, :2, 2].tolist() == [SIGNALS.index("fire") + 1, SIGNALS.index("access") + 1]
    assert np.all(categories[0, 2:] == 0) and np.all(numeric[0, 2:] == 0)
    assert numeric[0, 0, 0] > numeric[0, 1, 0] > 0
    assert numeric[0, 0, 2] == 0 and numeric[0, 1, 2] == 1
    assert np.isfinite(numeric).all()


def test_future_extension_does_not_change_earlier_tokens_or_training_vocabulary():
    start, raw, cat = source()
    old = prepare_events(raw.loc[raw.ts.lt(start + pd.Timedelta(hours=3))], cat)
    extended = prepare_events(raw.sample(frac=1, random_state=9), cat)
    q = pd.DataFrame({"object_id": [1, 2], "as_of": [start + pd.Timedelta(hours=3)] * 2})
    outputs, codecs = [], []
    for events in (old, extended):
        bounds = history_bounds(events, q)
        codec = fit_event_codec(events, bounds)
        codecs.append(codec)
        outputs.append(
            event_batch(
                encode_events(events, codec),
                events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
                bounds,
                q.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            )
        )
    assert codecs[0] == codecs[1]
    for before, after in zip(*outputs, strict=True):
        np.testing.assert_array_equal(before, after)


def test_event_source_and_bounds_reject_malformed_or_noncausal_inputs():
    start, raw, cat = source()
    with pytest.raises(ValueError, match="ownership"):
        prepare_events(raw.assign(object_id=99), cat)
    with pytest.raises(ValueError, match="canonical"):
        prepare_events(pd.concat([raw, raw]), cat)
    events = prepare_events(raw, cat)
    q = pd.DataFrame({"object_id": [1], "as_of": [start + pd.Timedelta(hours=3)]})
    with pytest.raises(ValueError, match="queries"):
        history_bounds(events, pd.concat([q, q]))
    bounds = history_bounds(events, q)
    codec = fit_event_codec(events, bounds)
    times = events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    with pytest.raises(ValueError, match="Noncausal"):
        event_batch(encode_events(events, codec), times, bounds, np.array([start.value]))
    with pytest.raises(ValueError, match="positions"):
        selected_source_mask(bounds + len(events), len(events))


def test_empty_history_has_zero_tokens_counts_and_lengths():
    start, raw, cat = source()
    events = prepare_events(raw.iloc[:0], cat)
    q = pd.DataFrame({"object_id": [1], "as_of": [start]})
    bounds = history_bounds(events, q)
    codec = fit_event_codec(events, bounds)
    categories, numeric, lengths = event_batch(
        encode_events(events, codec), np.array([], dtype=np.int64), bounds, np.array([start.value])
    )
    assert not categories.any() and not numeric.any() and not lengths.any() and not bounds.any()
    empty = event_batch(
        encode_events(events, codec), np.array([], dtype=np.int64), bounds[:0], np.array([], dtype=np.int64)
    )
    assert empty[0].shape == (0, 96, 3) and empty[2].shape == (0,)
