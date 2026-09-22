import numpy as np
import pytest

torch = pytest.importorskip("torch")

from moscollector.event_sequence_model import EventCountNetwork  # noqa: E402


def model(use_history=True):
    torch.manual_seed(13)
    m = EventCountNetwork(9, [6, 4, 3], [10, 5, 7], use_history, -1)
    torch.nn.init.normal_(m.backbone.head[-1].weight, std=0.1)
    return m


def inputs():
    numeric = torch.randn(3, 9)
    categories = torch.tensor([[1, 2, 1]] * 3)
    events = torch.ones(3, 6, 3, dtype=torch.long)
    continuous = torch.rand(3, 6, 3)
    lengths = torch.tensor([0, 2, 6])
    return numeric, categories, events, continuous, lengths


def test_event_gru_padding_and_empty_history_do_not_change_predictions():
    m = model()
    args = list(inputs())
    expected = m(*args).detach()
    args[2][0] = 4
    args[3][0] = 123
    args[2][1, 2:] = 4
    args[3][1, 2:] = 123
    torch.testing.assert_close(m(*args), expected)
    assert torch.isfinite(expected).all()
    with pytest.raises(ValueError, match="Missing"):
        m(args[0], args[1])
    control = model(False)
    torch.testing.assert_close(control(*args), control(args[0], args[1]))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_event_gru_gradients_unknown_embeddings_and_cpu_parity(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    cpu = model()
    args = inputs()
    expected = cpu(*args).detach().numpy()
    m = model().to(device)
    m.load_state_dict(cpu.state_dict())
    output = m(*(a.to(device) for a in args))
    np.testing.assert_allclose(output.detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-6)
    loss = (torch.exp(output) - 2 * output).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.event_embeddings.parameters())
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)
    optimizer.step()
    for layer in m.event_embeddings:
        assert not layer.weight[0].detach().cpu().numpy().any()
