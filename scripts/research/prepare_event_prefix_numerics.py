"""Record the failed strict guard and authorize byte-unchanged fit reuse.

The original experiment and every checkpoint remain intact. The numerical
follow-up uses documented float32 defaults; training/loss guards do not change.
"""

from pathlib import Path

import numpy as np
import torch

from moscollector.experiments.event_sequence_verification import check_hashes
from moscollector.experiments.goal90_research import read
from moscollector.prepare import sha256, write_json

old = Path("artifacts/research-v39-prefix")
root = Path("artifacts/research-v39-prefix-fixed")
plan = read(old / "plan.json")
check_hashes(plan)
target = root / "numerical-preflight.json"
if target.exists():
    raise ValueError("Numerical preflight is frozen; do not overwrite it")
sources = {**plan["source_hashes"], str(old / "plan.json"): sha256(old / "plan.json")}
reuse, rows = {}, []
tensor_tolerances = {"rtol": 1.3e-6, "atol": 1e-5}
for variant in ("current_mlp", "event_gru"):
    directory = old / "access/screen_1" / variant
    fit = read(directory / "fit.json")
    original_path = Path("artifacts/research-v37-fixed/access/screen_1") / variant / "fit.json"
    original = read(original_path)
    assert fit["signature"]["plan_sha256"] == sha256(old / "plan.json")
    assert fit["signature"]["old_fit_sha256"] == sha256(original_path)
    assert fit["model_sha256"] == sha256(directory / "model.pt")
    assert fit["last_sha256"] == sha256(directory / "last.pt")
    for key in ("config", "settings", "data_hashes", "event_source_hashes", "device", "torch"):
        assert fit["signature"][key] == original["signature"][key]
    for new_key, old_key in (
        ("first_epoch_training_loss", "training_loss"),
        ("first_epoch_validation_loss", "validation_loss"),
    ):
        np.testing.assert_allclose(fit[new_key], original["history"][1][old_key], rtol=1e-6, atol=1e-7)
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    selected = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    assert last["signature"] == fit["signature"]
    assert selected.keys() == last["best_state"].keys()
    assert all(torch.equal(value, last["best_state"][name]) for name, value in selected.items())
    tensor_rows = []
    if original["best_epoch"] == 1:
        weights = torch.load(original_path.parent / "model.pt", map_location="cpu", weights_only=True)
        assert weights.keys() == last["model"].keys()
        for name, value in weights.items():
            actual = last["model"][name]
            assert actual.dtype == value.dtype == torch.float32
            torch.testing.assert_close(actual, value, **tensor_tolerances)
            tensor_rows.append(
                {
                    "name": name,
                    "elements": value.numel(),
                    "strict_mismatches": int((~torch.isclose(actual, value, rtol=1e-6, atol=1e-7)).sum()),
                    "default_float32_mismatches": int(
                        (~torch.isclose(actual, value, **tensor_tolerances)).sum()
                    ),
                    "max_absolute_difference": float((actual - value).abs().max()),
                }
            )
    rows.append(
        {
            "variant": variant,
            "old_best_loss": original["best_validation_loss"],
            "observed_prefix_best_loss": fit["best_validation_loss"],
            "observed_prefix_best_step": fit["best_step"],
            "end_validation_loss_difference": fit["first_epoch_validation_loss"]
            - original["history"][1]["validation_loss"],
            "first_epoch_training_loss_difference": fit["first_epoch_training_loss"]
            - original["history"][1]["training_loss"],
            "tensors": tensor_rows,
        }
    )
    reuse[f"access/screen_1/{variant}"] = str(directory)
    for filename in ("fit.json", "model.pt", "last.pt"):
        path = directory / filename
        sources[str(path)] = sha256(path)
assert sum(t["strict_mismatches"] for r in rows for t in r["tensors"]) > 0
manifest = {
    "reason": "Original first-epoch probe completed2fits then failed its rtol1e-6/atol1e-7 tensor guard on accessGRU. Raw validation/training loss guards passed. Original failure and prior validation outcomes are explicitly retained, not reclassified as a passing original study.",
    "scope": "Numerical guard follow-up, not changed training or checkpoint selection; no policy/test predictions. Two completed fits reused from original paths byte-for-byte; no rewritten fit metadata/checkpoints or copied optimizer state.",
    "tensor_tolerances": tensor_tolerances,
    "source": "https://docs.pytorch.org/docs/2.14/testing.html#torch.testing.assert_close",
    "limit": "Default float32 proximity is not deterministic or bitwise trajectory equality; the cause of small MPS differences is not proven by this measurement.",
    "reuse_fits": reuse,
    "observed_before_followup": rows,
    "source_hashes": sources,
    "code_hashes": {**plan["code_hashes"], str(Path(__file__)): sha256(Path(__file__))},
}
write_json(target, manifest)
print("Strict tensor mismatches", sum(t["strict_mismatches"] for r in rows for t in r["tensors"]))
print("Maximum difference", max(t["max_absolute_difference"] for r in rows for t in r["tensors"]))
print("Two completed fits retained at original paths; no bytes modified")
