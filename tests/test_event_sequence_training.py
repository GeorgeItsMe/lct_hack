import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from moscollector import event_sequence_training as training  # noqa: E402
from moscollector.event_sequence_data import (  # noqa: E402
    encode_events,
    fit_event_codec,
    history_bounds,
    prepare_events,
)
from moscollector.event_sequence_model import EventCountNetwork  # noqa: E402


def example(use_history):
    start = pd.Timestamp("2025-01-01")
    raw = pd.DataFrame(
        {
            "object_id": 1,
            "channel_id": [1, 2] * 3,
            "signal": ["access", "fire"] * 3,
            "ts": pd.date_range(start, periods=6, freq="h"),
        }
    )
    catalog = pd.DataFrame({"channel_id": [1, 2], "object_id": 1, "sensor_type": ["door", "smoke"]})
    events = prepare_events(raw, catalog)
    q = pd.DataFrame(
        {"object_id": 1, "as_of": pd.date_range(start + pd.Timedelta(hours=1), periods=12, freq="h")}
    )
    bounds = history_bounds(events, q)
    codec = fit_event_codec(events, bounds)
    source = {
        "encoded": encode_events(events, codec),
        "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    }
    rng = np.random.default_rng(13)
    rows = {
        "numeric": rng.normal(size=(12, 9)).astype(np.float32),
        "categorical": np.ones((12, 3), dtype=np.int64),
        "bounds": bounds,
        "query_times": q.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "target": np.array([0, 1, 3, 2, 0, 4, 1, 0, 6, 2, 0, 1], dtype=np.float32),
        "weight": np.array([0.5, 0.25, 0.25, 1] * 3, dtype=np.float64),
    }
    data = {part: {k: v.copy() for k, v in rows.items()} for part in ("train", "validation")}
    config = {
        "numeric_dim": 9,
        "category_sizes": [4, 3, 2],
        "event_category_sizes": [3, 3, 7],
        "use_history": use_history,
        "initial_log_mean": float(np.log(np.average(rows["target"], weights=rows["weight"]))),
    }
    settings = {**training.TRAINING, "epochs": 3, "patience": 5, "batch_size": 4, "cpu_threads": 2}
    return data, source, config, settings


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("use_history", [False, True])
def test_weighted_event_network_restores_optimizer_and_rejects_changed_inputs(
    tmp_path, monkeypatch, device, use_history
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    data, source, config, settings = example(use_history)
    model, reference = training.fit_network(
        tmp_path / "full", data, source, config, {"plan": "toy"}, device, settings
    )
    save = training.save_torch

    def interrupted(path, state):
        save(path, state)
        if path.name == "last.pt" and state["epoch"] == 1:
            raise RuntimeError("Simulated interruption")

    monkeypatch.setattr(training, "save_torch", interrupted)
    with pytest.raises(RuntimeError, match="Simulated"):
        training.fit_network(tmp_path / "resumed", data, source, config, {"plan": "toy"}, device, settings)
    assert not (tmp_path / "resumed/fit.json").exists()
    monkeypatch.setattr(training, "save_torch", save)
    resumed, actual = training.fit_network(
        tmp_path / "resumed", data, source, config, {"plan": "toy"}, device, settings
    )
    assert actual["completed_epochs"] == reference["completed_epochs"] == 3
    assert actual["best_epoch"] == reference["best_epoch"]
    for r, a in zip(reference["history"], actual["history"], strict=True):
        for key in ("validation_loss", "training_loss"):
            if key in r:
                assert a[key] == pytest.approx(r[key], rel=1e-6, abs=1e-7)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=1e-6, atol=1e-7)
    full = torch.load(tmp_path / "full/last.pt", map_location="cpu", weights_only=True)
    restored = torch.load(tmp_path / "resumed/last.pt", map_location="cpu", weights_only=True)
    for name, value in full["model"].items():
        torch.testing.assert_close(restored["model"][name], value, rtol=1e-6, atol=1e-7)
    data["train"]["weight"][0] += 0.1
    with pytest.raises(ValueError, match="inputs"):
        training.fit_network(tmp_path / "resumed", data, source, config, {"plan": "toy"}, device, settings)


def test_validation_loss_uses_original_weights():
    data, source, config, _ = example(False)
    config["initial_log_mean"] = float(np.log(2))
    model = EventCountNetwork(**config)
    raw = training.predict(model, data["validation"], source, "cpu")
    expected = np.average(
        np.exp(raw) - data["validation"]["target"] * raw, weights=data["validation"]["weight"]
    )
    assert training.validation_loss(model, data["validation"], source, "cpu") == expected
    assert expected != pytest.approx(np.mean(np.exp(raw) - data["validation"]["target"] * raw))
