import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Optional self-supervised research environment")

from moscollector.neural_count_model import CountNetwork  # noqa: E402
from moscollector.self_supervised_model import (  # noqa: E402
    HistoryPredictor,
    masked_huber,
    pretext_batch,
    pretext_targets,
    telemetry_indices,
    transfer_history,
)


def seed_model():
    torch.manual_seed(42)
    return CountNetwork(4, [2, 2, 2], True, -1)


def toy_data():
    rng = np.random.default_rng(7)
    history = np.concatenate([rng.normal(size=(16, 2)), np.zeros((16, 2))], axis=1).astype(np.float32)
    positions = np.tile(np.arange(16, dtype=np.int32), (24, 1))
    ages = np.tile(np.arange(16, 0, -1, dtype=np.float32) * 3, (24, 1))
    positions[::3, 4:] = -1
    ages[::3, 4:] = 0
    numeric = np.concatenate([rng.normal(size=(24, 2)) + 1, np.zeros((24, 2))], axis=1).astype(np.float32)
    data = {
        "numeric": numeric,
        "categorical": np.zeros((24, 3), dtype=np.int64),
        "positions": positions,
        "ages": ages,
        "target": (np.arange(24) % 3).astype(np.float32),
    }
    return data, history


def test_pretext_targets_exclude_calendar_static_and_incident_columns():
    codec = {
        "numeric": [
            "hour",
            "channel_count",
            "past_fault_recency_h",
            "events_6h",
            "temperature",
            "nov_fault_channels_6h",
        ]
    }
    assert telemetry_indices(codec) == [3, 4, 5]
    with pytest.raises(ValueError, match="Forbidden"):
        telemetry_indices({"numeric": ["target_fault"]})
    data, _ = toy_data()
    data["numeric"][0, 2] = 1
    data["positions"][1] = -1
    target, valid = pretext_targets(data, np.arange(24), [0, 1])
    assert target.shape == valid.shape == (24, 2)
    assert not valid[0, 0] and valid[0, 1] and not valid[1].any()
    changed = {**data, "target": np.full(24, 99999)}
    _, after = pretext_targets(changed, np.arange(24), [0, 1])
    np.testing.assert_array_equal(after, valid)


def test_masked_huber_ignores_unobserved_values_and_gradients():
    pred = torch.tensor([[1.0, 999.0], [2.0, -999.0]], requires_grad=True)
    target = torch.tensor([[0.0, float("nan")], [3.0, float("nan")]])
    mask = torch.tensor([[True, False], [True, False]])
    loss, count = masked_huber(pred, target, mask)
    assert loss.item() == 0.5 and count == 2
    loss.backward()
    assert torch.equal(pred.grad[:, 1], torch.zeros(2))
    with pytest.raises(ValueError, match="No observed"):
        masked_huber(pred, target, torch.zeros_like(mask))


def test_pretext_never_reads_current_inputs_and_transfers_only_history():
    base = seed_model()
    predictor = HistoryPredictor(base, 2)
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.add_(0.1)
    tokens, lengths = torch.randn(3, 16, 5), torch.tensor([0, 4, 16])
    expected = predictor(tokens, lengths)
    tokens[0] = 999
    tokens[1, 4:] = -999
    torch.testing.assert_close(predictor(tokens, lengths), expected, rtol=0, atol=0)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    transfer_history(predictor, base)
    for name, value in base.state_dict().items():
        if name.startswith(("history_projection.", "gru.")):
            torch.testing.assert_close(value, predictor.state_dict()[name], rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not any(name.startswith("decoder.") for name in base.state_dict())


def test_pretext_learning_uses_no_count_targets():
    torch.set_num_threads(2)
    predictor = HistoryPredictor(seed_model(), 2)
    data, history = toy_data()
    ids = np.arange(24)
    before, _ = pretext_batch(predictor, data, ids, history, "cpu", [0, 1])
    alternate, _ = pretext_batch(
        predictor, {**data, "target": np.full(24, np.nan)}, ids, history, "cpu", [0, 1]
    )
    torch.testing.assert_close(before, alternate, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.01)
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = pretext_batch(predictor, data, ids, history, "cpu", [0, 1])
        loss.backward()
        optimizer.step()
    after, _ = pretext_batch(predictor, data, ids, history, "cpu", [0, 1])
    assert after.item() < before.item()


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("stage", ["pretext", "count"])
def test_both_stages_resume_same_optimizer_and_best_weights(tmp_path, monkeypatch, device, stage):
    from moscollector import self_supervised_training as study

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.set_num_threads(2)
    config = {**study.TRAINING, "epochs": 2, "patience": 10, "batch_size": 8}
    monkeypatch.setattr(study, "TRAINING", config)
    monkeypatch.setattr(study, "PRETRAIN", {**config, "seed": 43})
    data, history = toy_data()
    sets = {"train": data, "validation": data}
    indices = [0, 1] if stage == "pretext" else None

    def model():
        base = seed_model()
        pretext = HistoryPredictor(base, 2)
        with torch.no_grad():
            pretext.gru.weight_ih_l0.add_(0.1)
        if stage == "pretext":
            return pretext
        transfer_history(pretext, base)
        return base

    reference, expected = study.fit_stage(
        tmp_path / "full", model(), sets, history, device, stage, {"plan": "toy"}, indices
    )
    save = study.save_torch

    def interrupted(path, state):
        save(path, state)
        if path.name == "last.pt":
            raise RuntimeError("Simulated interruption")

    monkeypatch.setattr(study, "save_torch", interrupted)
    with pytest.raises(RuntimeError, match="Simulated"):
        study.fit_stage(tmp_path / "resumed", model(), sets, history, device, stage, {"plan": "toy"}, indices)
    monkeypatch.setattr(study, "save_torch", save)
    resumed, actual = study.fit_stage(
        tmp_path / "resumed", model(), sets, history, device, stage, {"plan": "toy"}, indices
    )
    assert actual["completed_epochs"] == expected["completed_epochs"] == 2
    assert actual["best_epoch"] == expected["best_epoch"]
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
