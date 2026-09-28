"""Synthetic end-to-end forward/backward timing before any v37 task outcomes."""

import time
from pathlib import Path

import numpy as np
import torch

from moscollector.experiments.event_sequence_data import STEPS
from moscollector.experiments.event_sequence_model import EventCountNetwork
from moscollector.experiments.neural_count_model import CountNetwork
from moscollector.prepare import sha256, write_json

if not torch.backends.mps.is_available():
    raise RuntimeError("Explicit local MPS preflight requires an available GPU")
torch.set_num_threads(4)
records = []
for device in ("cpu", "mps"):
    for use_history in (False, True):
        torch.manual_seed(42)
        model = EventCountNetwork(289, [100, 8, 4], [12000, 64, 7], use_history, 0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        numeric = torch.randn(512, 289, device=device)
        categorical = torch.ones(512, 3, dtype=torch.long, device=device)
        events = torch.ones(512, STEPS, 3, dtype=torch.long, device=device)
        continuous = torch.rand(512, STEPS, 3, device=device)
        lengths = torch.arange(512, device=device) % (STEPS + 1)
        target = torch.poisson(torch.ones(512, device=device))
        weight = torch.linspace(0.01, 1, 512, device=device)
        elapsed = []
        for step in range(11):
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(numeric, categorical, events, continuous, lengths)
            loss = (weight * (torch.exp(output) - target * output)).mean() / weight.mean()
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite event preflight loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            if device == "mps":
                torch.mps.synchronize()
            if step >= 3:
                elapsed.append(time.perf_counter() - start)
        record = {
            "device": device,
            "variant": "event_gru" if use_history else "current_mlp",
            "batch_size": 512,
            "sequence_steps": STEPS,
            "timed_steps": len(elapsed),
            "median_step_seconds": float(np.median(elapsed)),
            "mean_step_seconds": float(np.mean(elapsed)),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        records.append(record)
        print(record, flush=True)
data = {
    "scope": "Synthetic timing and finite-gradient preflight only, not model quality or complete training-time measurement. Includes optimizer updates and explicit MPS synchronization, excludes CPU history materialization and transfer. Maximum numeric width and generous catalog sizes; actual fold codecs are training-only.",
    "torch": str(torch.__version__),
    "chosen_device": "mps",
    "records": records,
    "cpu_mps_forward_backward_tests": "8 data/model tests passed, including CPU/MPS nonconstant forward parity, padding invariance, causal source extension, family caps, actual-training-only vocabularies and gradients; no predictive model fitted yet.",
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(EventCountNetwork.__init__.__code__.co_filename),
            Path(CountNetwork.__init__.__code__.co_filename),
        )
    },
}
write_json(Path("artifacts/research-v37/compute-preflight.json"), data)
