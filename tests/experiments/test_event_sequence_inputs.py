import numpy as np
import pandas as pd
import pytest

from moscollector.experiments.event_sequence_data import prepare_events
from moscollector.experiments.event_sequence_inputs import held_inputs, input_arrays, training_part
from moscollector.experiments.neural_sequence_data import fit_codec
from moscollector.experiments.onset_training_data import augmented_slots
from moscollector.train import CATEGORICAL


def example():
    start = pd.Timestamp("2025-01-01")
    base = pd.DataFrame(
        {
            "object_id": 1,
            "as_of": pd.date_range(start, periods=3, freq="3h"),
            "parent_id": 9,
            "object_kind": "controlHouse",
            "reports": [10, 20, 30],
            "target_access": [1, 1, 0],
        }
    )
    triggers = pd.DataFrame({"object_id": 1, "as_of": start + pd.to_timedelta([1, 2, 4, 5], unit="h")})
    slots = augmented_slots(base, triggers)
    raw = pd.DataFrame(
        {
            "object_id": 1,
            "channel_id": 1,
            "signal": "access",
            "ts": start + pd.to_timedelta([0.5, 1, 3.5, 5], unit="h"),
        }
    )
    events = prepare_events(raw, pd.DataFrame({"object_id": [1], "channel_id": [1], "sensor_type": ["door"]}))
    codec = fit_codec(base, [*CATEGORICAL, "reports"])
    episodes = pd.DataFrame({"object_id": [1], "start_ts": [start + pd.Timedelta(hours=4)]})
    return base, slots, events, codec, episodes


def test_event_input_augmentation_retains_mass_and_recalculates_shifted_targets():
    base, slots, events, codec, episodes = example()
    data, sizes = training_part(base, slots, events, codec, episodes, "access")
    assert sizes["weight_sum"] == 3 and sizes["rows"] == 7 and sizes["eligible_episodes"] == 1
    np.testing.assert_array_equal(data["target"], [1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(data["weight"], [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 1])
    held, ages = held_inputs(base, slots, codec["columns"])
    assert held.reports.tolist() == [10, 10, 10, 20, 20, 20, 30]
    np.testing.assert_array_equal(ages, [0, 1, 2, 0, 1, 2, 0])
    assert data["numeric"].shape == (7, 9)
    np.testing.assert_allclose(data["numeric"][:, 2], np.log1p(ages) / np.log(4))


def test_future_labels_and_new_future_onsets_do_not_affect_current_inputs():
    base, slots, events, codec, _ = example()
    before = input_arrays(base, slots.iloc[:2].copy(), events, codec)
    changed = events.copy()
    changed.loc[changed.ts.gt(slots.as_of.iloc[1]), "channel_id"] = 99
    after = input_arrays(base.assign(target_access=123), slots.iloc[:2].copy(), changed, codec)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_held_inputs_reject_forbidden_features_missing_sources_and_wrong_targets():
    base, slots, events, codec, episodes = example()
    with pytest.raises(ValueError, match="Forbidden"):
        held_inputs(base, slots, [*codec["columns"], "target_access"])
    with pytest.raises(ValueError, match="Missing"):
        held_inputs(base.iloc[:1], slots, codec["columns"])
    with pytest.raises(ValueError, match="targets"):
        training_part(base.assign(target_access=0), slots, events, codec, episodes, "access")
