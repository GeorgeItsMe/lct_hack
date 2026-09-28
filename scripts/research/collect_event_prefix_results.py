"""Audit all completed prefix fits, retaining failed numerical guard outcomes.

This does not rewrite either probe plan, its weights, or its unsuccessful run.
Numerical trajectory equivalence is not a prerequisite for separately testing
a genuinely fitted candidate, but cannot be claimed after a failed guard.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from moscollector.experiments.event_sequence_data import FOLDER as EVENT_FOLDER
from moscollector.experiments.event_sequence_data import encode_events
from moscollector.experiments.event_sequence_inputs import period_rows, training_part
from moscollector.experiments.event_sequence_model import EventCountNetwork
from moscollector.experiments.event_sequence_training import array_digest, validation_loss
from moscollector.experiments.event_sequence_verification import check_hashes
from moscollector.experiments.goal90_research import read
from moscollector.experiments.onset_channel_features import FOLDER as ONSET_FOLDER
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v39-prefix-fixed")
parent = Path("artifacts/research-v37-fixed")
plan = read(root / "plan.json")
check_hashes(plan)
if plan["torch"] != str(torch.__version__) or not torch.backends.mps.is_available():
    raise ValueError("Need original saved MPS backend")
torch.set_num_threads(4)
frame, dense = (
    pd.read_parquet(PROCESSED / n)
    for n in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
)
events = pd.read_parquet(EVENT_FOLDER / "events.parquet", read_dictionary=["signal", "sensor_type"])
triggers = pd.read_parquet(ONSET_FOLDER / "triggers.parquet")
episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
sources = dict(plan["source_hashes"])
records = []
for kind in ("access", "fire", "fault"):
    eps = episodes.loc[episodes.kind.eq(kind)]
    for fold in ("screen_1", "screen_2"):
        directory = parent / kind / fold
        prepared = read(directory / "data.json")
        codec, event_codec = (read(directory / n) for n in ("current-codec.json", "event-codec.json"))
        base, slots = period_rows(frame, dense, triggers, prepared["periods"], "validation")
        data, sizes = training_part(base, slots, events, codec, eps, kind)
        source = {
            "encoded": encode_events(events, event_codec),
            "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        }
        hashes = {k: array_digest(v) for k, v in data.items()}
        source_hashes = {k: array_digest(v) for k, v in source.items()}
        for variant in ("current_mlp", "event_gru"):
            relative = Path(kind) / fold / variant
            fit_dir = Path(plan["reuse_fits"].get(str(relative), str(root / relative)))
            meta = read(fit_dir / "fit.json")
            old = read(parent / relative / "fit.json")
            if (
                meta["signature"]["data_hashes"]["validation"] != hashes
                or meta["signature"]["event_source_hashes"] != source_hashes
                or old["signature"]["sizes"]["validation"] != sizes
            ):
                raise ValueError("Changed validation data or vocabulary")
            for key in ("data_hashes", "event_source_hashes", "config", "settings", "device", "torch"):
                if meta["signature"][key] != old["signature"][key]:
                    raise ValueError("Changed original training input signatures")
            if (
                sha256(fit_dir / "model.pt") != meta["model_sha256"]
                or sha256(fit_dir / "last.pt") != meta["last_sha256"]
            ):
                raise ValueError("Changed completed model or optimizer checkpoint")
            last = torch.load(fit_dir / "last.pt", map_location="cpu", weights_only=True)
            weights = torch.load(fit_dir / "model.pt", map_location="cpu", weights_only=True)
            if (
                last["signature"] != meta["signature"]
                or last["next_start"] != meta["completed_rows"]
                or last["step"] != meta["completed_steps"]
                or weights.keys() != last["best_state"].keys()
                or any(not torch.equal(v, last["best_state"][k]) for k, v in weights.items())
            ):
                raise ValueError("Invalid completed first epoch or selected state")
            model = EventCountNetwork(**old["config"]).to("mps")
            model.load_state_dict(weights)
            val = validation_loss(model, data, source, "mps")
            np.testing.assert_allclose(val, meta["best_validation_loss"], rtol=1e-7, atol=1e-8)
            loss_match = bool(
                np.isclose(
                    meta["first_epoch_validation_loss"],
                    old["history"][1]["validation_loss"],
                    rtol=1e-6,
                    atol=1e-7,
                )
                and np.isclose(
                    meta["first_epoch_training_loss"],
                    old["history"][1]["training_loss"],
                    rtol=1e-6,
                    atol=1e-7,
                )
            )
            tensor_match, max_tensor_difference = None, None
            if old["best_epoch"] == 1:
                old_weights = torch.load(
                    parent / relative / "model.pt", map_location="cpu", weights_only=True
                )
                if old_weights.keys() != last["model"].keys():
                    raise ValueError("Architecture mismatch")
                tensor_match = True
                max_tensor_difference = 0.0
                for name, value in last["model"].items():
                    if value.dtype != torch.float32 or old_weights[name].dtype != torch.float32:
                        raise ValueError("Changed float32dtype")
                    tensor_match = tensor_match and bool(
                        torch.isclose(value, old_weights[name], **plan["tensor_tolerances"]).all()
                    )
                    max_tensor_difference = max(
                        max_tensor_difference, float((value - old_weights[name]).abs().max())
                    )
            record = {
                "kind": kind,
                "fold": fold,
                "variant": variant,
                "original_training_rows": old["signature"]["sizes"]["train"]["original_rows"],
                "augmented_training_rows": old["signature"]["sizes"]["train"]["rows"],
                "check_every_steps": meta["signature"]["every_steps"],
                "checks": len(meta["history"]),
                "best_step": meta["best_step"],
                "completed_steps": meta["completed_steps"],
                "best_prefix_validation_loss": val,
                "old_best_validation_loss": old["best_validation_loss"],
                "old_best_epoch": old["best_epoch"],
                "first_epoch_trajectory_losses_match": loss_match,
                "first_epoch_tensors_match_old_selected": tensor_match,
                "end_validation_loss_difference": meta["first_epoch_validation_loss"]
                - old["history"][1]["validation_loss"],
                "end_training_loss_difference": meta["first_epoch_training_loss"]
                - old["history"][1]["training_loss"],
                "max_final_tensor_difference": max_tensor_difference,
                "improves_old_selected": val
                < old["best_validation_loss"] - meta["signature"]["settings"]["min_delta"],
                "fit_directory": str(fit_dir),
                "imported_unchanged": fit_dir != root / relative,
                "tensor_tolerances": plan["tensor_tolerances"],
                "fit_sha256": sha256(fit_dir / "fit.json"),
                "weights_sha256": sha256(fit_dir / "model.pt"),
                "validation_arrays_and_selected_weights_replayed": True,
            }
            records.append(record)
            for filename in ("fit.json", "model.pt", "last.pt"):
                path = fit_dir / filename
                sources[str(path)] = sha256(path)
            print(
                "COLLECTED PREFIX",
                kind,
                fold,
                variant,
                "improved",
                record["improves_old_selected"],
                "original_loss_guard_passed",
                loss_match,
                flush=True,
            )
            del model, last, weights
            gc.collect()
        del data, source, base, slots
        gc.collect()
sources.update({str(p): sha256(p) for p in root.rglob("*") if p.suffix in (".json", ".pt")})
result = {
    "scope": "Post-run audit of12completed one-epoch training runs. Original strict probe failed a tensor guard; float32follow-up failed one original loss guard. Those outcomes remain failures of trajectory-equivalence checks. Candidate inputs and selected validation weights are verified separately; no event inference, causal monitoring attribution or90/90claim.",
    "plan_sha256": sha256(root / "plan.json"),
    "models": len(records),
    "rows": records,
    "improved_models": sum(r["improves_old_selected"] for r in records),
    "all_original_loss_guards_passed": all(r["first_epoch_trajectory_losses_match"] for r in records),
    "trajectory_equivalence_claimed": False,
    "event_precision_recall_evaluated": False,
    "serving_changed": False,
    "goal_achieved": False,
    "source_hashes": sources,
    "code_hashes": {**plan["code_hashes"], str(Path(__file__)): sha256(Path(__file__))},
}
if len(records) != 12:
    raise ValueError("Incomplete prefix training coverage")
check_hashes(result)
write_json(Path("artifacts/event_prefix_audit.json"), result)
print(
    "AUDIT COMPLETE",
    result["improved_models"],
    "improved; all_original_loss_guards_passed",
    result["all_original_loss_guards_passed"],
    flush=True,
)
