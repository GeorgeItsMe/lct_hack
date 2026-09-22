import numpy as np
import pandas as pd
import pytest

from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort
from moscollector.onset_channel_features import channel_context, minute_onsets
from moscollector.onset_training_data import attach, augmented_slots, model_input, snapshot_weights
from moscollector.research import mask


def catalog():
    return pd.DataFrame(
        {"channel_id": [1, 2, 3], "object_id": [10, 10, 20], "sensor_type": ["smoke", "door", "power"]}
    )


def events():
    return pd.DataFrame(
        {
            "object_id": [10, 10, 10, 10],
            "channel_id": [1, 2, 1, 2],
            "signal": ["fire", "unknown", "fire", "fire"],
            "ts": pd.to_datetime(
                ["2024-12-31 23:00:00", "2025-01-01 00:01:00", "2025-01-01 01:00:00", "2025-01-01 01:00:00"]
            ),
        }
    )


def queries(times, obj=10):
    return pd.DataFrame({"object_id": obj, "as_of": pd.to_datetime(times)})


def test_context_preserves_channel_per_signal_and_strict_cutoff():
    q = queries(["2024-12-31 23:00:00", "2025-01-01 00:02:00", "2025-01-01 01:00:00", "2025-01-01 01:00:30"])
    extra = channel_context(q, events(), catalog())
    assert extra.evt_fire_channel.tolist() == ["__NONE__", "1", "1", "2"]
    assert extra.evt_unknown_channel.tolist() == ["__NONE__", "2", "2", "2"]
    assert extra.evt_fire_object_n_1m.tolist() == [0, 0, 0, 2]
    assert extra.evt_fire_channel_n_1h.iloc[-1] == 1
    assert extra.evt_fire_sensor_type.iloc[-1] == "door"
    assert np.isnan(extra.evt_fire_gap_h.iloc[-1])  # first fire onset on channel2
    one_channel = events().loc[events().channel_id.eq(1)]
    separated = channel_context(q, one_channel, catalog())
    assert separated.evt_fire_gap_h.iloc[-1] == 2
    assert separated.evt_fire_channel_n_24h.iloc[-1] == 2


def test_future_values_targets_and_permutation_cannot_change_context():
    q = queries(["2025-01-01 00:02:00", "2025-01-01 01:00:30"])
    base = channel_context(q, events(), catalog())
    later = events().iloc[:1].assign(ts=pd.Timestamp("2025-01-05"), signal="fault")
    changed = (
        pd.concat([events(), later])
        .sample(frac=1, random_state=12)
        .assign(target_fault=999, duration_seconds=99999)
    )
    actual = channel_context(q.assign(eligible=False, target_fault=0), changed, catalog())
    pd.testing.assert_frame_equal(actual, base)
    short = events().loc[events().ts.lt(q.as_of.iloc[0])]
    pd.testing.assert_frame_equal(channel_context(q.iloc[:1], short, catalog()), base.iloc[:1])


def test_unknown_and_stale_channels_stay_explicit_and_objects_do_not_mix():
    q = pd.concat([queries(["2025-03-01"]), queries(["2025-01-01"], 20)], ignore_index=True)
    extra = channel_context(q, events(), catalog())
    assert extra.evt_fire_channel.eq("__NONE__").all()
    assert extra.evt_fire_age_h.eq(721).all()
    assert extra.evt_fire_gap_h.isna().all()
    assert extra.evt_fire_object_n_24h.eq(0).all()
    with pytest.raises(ValueError, match="catalog"):
        channel_context(q, events().assign(object_id=20), catalog())
    with pytest.raises(ValueError, match="canonical"):
        channel_context(q, pd.concat([events(), events()]), catalog())


def test_trigger_grid_retains_original_cohort_and_snapshot_mass():
    base = queries(["2025-01-01 00:00:00", "2025-01-01 03:00:00", "2025-01-01 09:00:00"])
    source = events().copy()
    source.loc[0, "ts"] = pd.Timestamp("2025-01-01 00:00")
    triggers = minute_onsets(source)
    grid = augmented_slots(base, triggers)
    assert grid.as_of.dt.strftime("%H:%M").tolist() == ["00:00", "00:01", "00:02", "01:01", "03:00", "09:00"]
    weights = snapshot_weights(grid)
    np.testing.assert_allclose(weights, [0.5, 1 / 6, 1 / 6, 1 / 6, 1, 1])
    assert weights.sum() == 3
    eps = queries(["2025-01-01 00:30:00", "2025-01-02 08:59:59", "2025-01-02 09:00:00"]).rename(
        columns={"as_of": "start_ts"}
    )
    assert cohort(grid, eps, 1 / 60) == cohort(base, eps, 3)
    with pytest.raises(ValueError, match="anchor"):
        snapshot_weights(grid.loc[grid.as_of.ne(grid.source_time)])


def test_eval_grid_keeps_quarters_and_drops_gaps_and_last_hour_extensions():
    base = queries(["2025-01-01 00:00:00", "2025-01-01 01:00:00", "2025-01-01 03:00:00"])
    triggers = queries(
        ["2025-01-01 00:02:00", "2025-01-01 00:15:00", "2025-01-01 01:02:00", "2025-01-01 03:02:00"]
    )
    grid = augmented_slots(base, triggers, 1, True)
    assert grid.as_of.dt.strftime("%H:%M").tolist() == [
        "00:00",
        "00:02",
        "00:15",
        "00:30",
        "00:45",
        "01:00",
        "03:00",
    ]


def test_augmentation_reapplies_purge_and_does_not_predict_an_already_started_episode():
    base = pd.DataFrame(
        {"object_id": 10, "as_of": pd.date_range("2025-01-01", "2025-01-03", freq="3h"), "eligible": True}
    )
    dates = (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"))
    selected = base.loc[mask(base, *dates)]
    grid = augmented_slots(selected, minute_onsets(events()))
    assert (grid.as_of + pd.Timedelta(hours=25)).lt(dates[1]).all()
    eps = queries(["2025-01-01 01:00:00"]).rename(columns={"as_of": "start_ts"})
    counts = episode_counts(grid, eps)
    assert counts[grid.as_of.eq(pd.Timestamp("2025-01-01 00:00"))].item() == 1
    assert counts[grid.as_of.eq(pd.Timestamp("2025-01-01 01:01"))].item() == 0


def test_attach_keeps_frozen_base_values_and_no_old_labels():
    base = queries(["2025-01-01 00:00:00", "2025-01-01 03:00:00"]).assign(
        events_1h=[10, 20], target_fault=999
    )
    slots = augmented_slots(base, minute_onsets(events()))
    context = pd.concat([slots[["object_id", "as_of"]], channel_context(slots, events(), catalog())], axis=1)
    frame = attach(base, slots, context, ["object_id", "events_1h"])
    assert "target_fault" not in frame
    assert frame.loc[frame.source_time.eq(pd.Timestamp("2025-01-01")), "events_1h"].eq(10).all()
    assert frame.evt_source_age_h.between(0, 3, inclusive="left").all()
    with pytest.raises(ValueError, match="context"):
        attach(base, slots, context.iloc[:0], ["object_id", "events_1h"])
    with pytest.raises(ValueError, match="Forbidden"):
        model_input(base, ["target_fault"])
