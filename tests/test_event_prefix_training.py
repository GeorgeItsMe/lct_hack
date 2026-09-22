import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from moscollector import event_prefix_training as prefix  # noqa: E402
from moscollector import event_sequence_training as old  # noqa: E402
from moscollector.event_sequence_data import (  # noqa: E402
    encode_events,
    fit_event_codec,
    history_bounds,
    prepare_events,
)


def example(history):
    events = prepare_events(
        pd.DataFrame(
            {
                "object_id": 1,
                "channel_id": [1, 1],
                "signal": ["fire", "access"],
                "ts": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            }
        ),
        pd.DataFrame({"object_id": [1], "channel_id": [1], "sensor_type": ["smoke"]}),
    )
    queries = pd.DataFrame({"object_id": 1, "as_of": pd.date_range("2025-01-01", periods=13, freq="6h")})
    bounds = history_bounds(events, queries)
    codec = fit_event_codec(events, bounds)
    source = {
        "encoded": encode_events(events, codec),
        "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    }
    rng = np.random.default_rng(13)
    row = {
        "numeric": rng.normal(size=(13, 5)).astype(np.float32),
        "categorical": np.zeros((13, 3), np.int64),
        "bounds": bounds,
        "query_times": queries.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "target": np.array([0, 1, 2, 3, 4, 0, 1, 0, 0, 2, 3, 0, 1], np.float32),
        "weight": np.array([1, 0.5, 0.25] * 4 + [1], np.float64),
    }
    data = {p: {k: v.copy() for k, v in row.items()} for p in ("train", "validation")}
    config = {
        "numeric_dim": 5,
        "category_sizes": [1, 1, 1],
        "event_category_sizes": [2, 2, 7],
        "use_history": history,
        "initial_log_mean": float(np.log(np.average(row["target"], weights=row["weight"]))),
    }
    settings = {**old.TRAINING, "epochs": 1, "batch_size": 3, "cpu_threads": 2}
    return data, source, config, settings


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("history", [False, True])
def test_extra_validation_keeps_original_trajectory_and_resumes_inside_epoch(
    tmp_path, monkeypatch, device, history
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    data, source, config, settings = example(history)
    _, original = old.fit_network(tmp_path / "old", data, source, config, {"plan": "toy"}, device, settings)
    model, full = prefix.fit_prefix(
        tmp_path / "full", data, source, config, {"plan": "toy"}, device, 2, settings
    )
    assert [r["step"] for r in full["history"]] == [0, 2, 4, 5]
    assert full["completed_rows"] == 13
    assert full["first_epoch_validation_loss"] == pytest.approx(
        original["history"][1]["validation_loss"], rel=1e-7, abs=1e-8
    )
    assert full["first_epoch_training_loss"] == pytest.approx(
        original["history"][1]["training_loss"], rel=1e-7, abs=1e-8
    )
    old_last = torch.load(tmp_path / "old/last.pt", map_location="cpu", weights_only=True)
    fine_last = torch.load(tmp_path / "full/last.pt", map_location="cpu", weights_only=True)
    for name, value in old_last["model"].items():
        torch.testing.assert_close(fine_last["model"][name], value, rtol=1e-7, atol=1e-8)
    save = prefix.save_torch

    def interrupt(path, state):
        save(path, state)
        if path.name == "last.pt" and state["step"] == 2:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(prefix, "save_torch", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        prefix.fit_prefix(tmp_path / "resumed", data, source, config, {"plan": "toy"}, device, 2, settings)
    monkeypatch.setattr(prefix, "save_torch", save)
    restored, actual = prefix.fit_prefix(
        tmp_path / "resumed", data, source, config, {"plan": "toy"}, device, 2, settings
    )
    assert actual["history"] == full["history"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    changed = {p: {k: v.copy() for k, v in row.items()} for p, row in data.items()}
    changed["train"]["weight"][0] += 0.01
    with pytest.raises(ValueError, match="Changed"):
        prefix.fit_prefix(tmp_path / "resumed", changed, source, config, {"plan": "toy"}, device, 2, settings)


def test_interval_is_quarter_of_original_not_augmented_epoch():
    assert prefix.check_interval(254976, 512) == 125
    assert prefix.check_interval(774530, 512) == 379
    assert prefix.check_interval(1, 512) == 1
    with pytest.raises(ValueError):
        prefix.check_interval(0, 512)


def test_invalid_interval_and_weights_fail_before_fitting(tmp_path):
    data, source, config, settings = example(False)
    with pytest.raises(ValueError, match="interval"):
        prefix.fit_prefix(tmp_path, data, source, config, {}, "cpu", 0, settings)
    data["validation"]["weight"][0] = 0
    with pytest.raises(ValueError, match="weights"):
        prefix.fit_prefix(tmp_path, data, source, config, {}, "cpu", 1, settings)
