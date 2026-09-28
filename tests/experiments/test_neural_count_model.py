import pytest

torch = pytest.importorskip("torch", reason="Optional neural research environment")

from moscollector.experiments.neural_count_model import CountNetwork  # noqa: E402


def test_padding_and_empty_history_cannot_change_forecast():
    torch.manual_seed(42)
    model = CountNetwork(4, [3, 2, 2], True, -1).eval()
    torch.nn.init.normal_(model.head[-1].weight, std=0.1)
    current, cat = torch.randn(3, 4), torch.zeros(3, 3, dtype=torch.long)
    history, lengths = torch.randn(3, 16, 5), torch.tensor([0, 4, 16])
    expected = model(current, cat, history, lengths)
    history[0] = 1000
    history[1, 4:] = -1000
    torch.testing.assert_close(model(current, cat, history, lengths), expected, atol=0, rtol=0)


def test_both_networks_train_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    for use_history in (False, True):
        torch.manual_seed(42)
        model = CountNetwork(4, [3, 2, 2], use_history, -1)
        x, cats = torch.randn(8, 4), torch.zeros(8, 3, dtype=torch.long)
        history, lengths = torch.randn(8, 16, 5), torch.tensor([0, 1, 2, 4, 8, 12, 16, 16])
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        target = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0])
        first = torch.nn.functional.poisson_nll_loss(model(x, cats, history, lengths), target).item()
        for _ in range(20):
            opt.zero_grad()
            loss = torch.nn.functional.poisson_nll_loss(model(x, cats, history, lengths), target)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        assert torch.nn.functional.poisson_nll_loss(model(x, cats, history, lengths), target).item() < first
        path = tmp_path / f"model-{use_history}.pt"
        torch.save(model.state_dict(), path)
        restored = CountNetwork(4, [3, 2, 2], use_history, -1)
        restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(
            restored(x, cats, history, lengths), model(x, cats, history, lengths), atol=0, rtol=0
        )


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("variant", ["mlp", "gru"])
def test_interrupted_epoch_checkpoint_resumes_same_training(tmp_path, monkeypatch, device, variant):
    import numpy as np

    from moscollector.experiments import neural_count_research as study
    from moscollector.prepare import write_json

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable in this process")
    torch.set_num_threads(2)
    monkeypatch.setattr(study, "TRAINING", {**study.TRAINING, "epochs": 2, "patience": 10, "batch_size": 8})
    rng = np.random.default_rng(3)
    data = {
        "numeric": rng.normal(size=(24, 4)).astype(np.float32),
        "categorical": np.zeros((24, 3), dtype=np.int64),
        "positions": np.full((24, 16), -1, dtype=np.int32),
        "ages": np.zeros((24, 16), dtype=np.float32),
        "target": (np.arange(24) % 4).astype(np.float32),
    }
    for row in range(24):
        length = (0, 4, 16)[row % 3]
        data["positions"][row, :length] = np.arange(length)
        data["ages"][row, :length] = np.arange(length, 0, -1) * 3
    sets = {"train": data, "validation": data}
    codec = {"vocabulary": {c: {"known": 1} for c in ("object_id", "parent_id", "object_kind")}}
    history = rng.normal(size=(16, 4)).astype(np.float32)
    for run in ("full", "resumed"):
        write_json(tmp_path / run / "codec.json", codec)
    reference, ref_meta = study.fit_network(
        tmp_path / f"full/models/{variant}", variant, sets, history, codec, {}, device, "toy"
    )
    original_save = study.save_torch

    def interrupt_after_atomic_checkpoint(path, value):
        original_save(path, value)
        if path.name == "last.pt":
            raise RuntimeError("Simulated process interruption")

    monkeypatch.setattr(study, "save_torch", interrupt_after_atomic_checkpoint)
    with pytest.raises(RuntimeError, match="Simulated"):
        study.fit_network(tmp_path / f"resumed/models/{variant}", variant, sets, history, codec, {}, device, "toy")
    monkeypatch.setattr(study, "save_torch", original_save)
    resumed, meta = study.fit_network(
        tmp_path / f"resumed/models/{variant}", variant, sets, history, codec, {}, device, "toy"
    )
    assert meta["completed_epochs"] == ref_meta["completed_epochs"] == 2
    assert meta["best_epoch"] == ref_meta["best_epoch"]
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, atol=0, rtol=0)
