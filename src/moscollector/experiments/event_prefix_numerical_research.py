"""Numerical follow-up of the first-epoch probe; old failed run remains frozen."""

import argparse
import gc
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from moscollector.experiments.event_prefix_training import check_interval, fit_prefix
from moscollector.experiments.event_sequence_data import FOLDER as EVENT_FOLDER
from moscollector.experiments.event_sequence_data import encode_events, fit_event_codec
from moscollector.experiments.event_sequence_inputs import period_rows, training_part
from moscollector.experiments.event_sequence_training import array_digest, validation_loss
from moscollector.experiments.event_sequence_verification import check_hashes
from moscollector.experiments.goal90_research import read
from moscollector.experiments.neural_count_research import TRAINING
from moscollector.experiments.neural_sequence_data import fit_codec
from moscollector.experiments.onset_channel_features import FOLDER as ONSET_FOLDER
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import mask

PARENT = Path("artifacts/research-v37-fixed")
ROOT = Path("artifacts/research-v39-prefix-fixed")
KINDS, FOLDS, VARIANTS = ("access", "fire", "fault"), ("screen_1", "screen_2"), ("current_mlp", "event_gru")
PLAN = {
    "scope": "First-epoch optimization diagnostic, NOT event forecast evaluation or a new blind test.",
    "motivation": "All12v37networks use2.869-2.999times original optimizer steps between validations. A separate audit of initial/best/final saved states found0/12final states improved selected states after profiling out global count scale, and no hidden shape skill in3epoch0-selected final states. It did not observe intermediate states.",
    "population": "All12networks:3kinds x2originalscreenfolds x2architectures; no outcome-dependent omission. Flood remains part of the full unmet goal and its separatev38result is unchanged.",
    "training": TRAINING,
    "updates": "Replay exactly ONE full first augmented epoch per network from original seed42 and shuffle43. Identical input arrays, codecs, targets, weights, initialization, architecture, float32weightedPoisson, AdamW and gradients. Validation never stops/changes updates. Existing studies/weights remain untouched.",
    "monitoring": "Check weighted raw Poisson at step0, every ceil(ceil(original_anchor_rows/512)/4) optimizer steps, and final partial interval. Select lower loss by original min_delta1e-5. Interval depends only on training row count, never validation/test outcomes.",
    "trajectory_check": "End-of-epoch validation AND accumulated training loss retain rtol1e-6,atol1e-7. If original best_epoch==1, compare float32final tensors with old selected state at documented PyTorch float32defaults rtol1.3e-6,atol1e-5; retain dtype/shape checks. This means numerical proximity, NOT exact deterministic trajectory. Original strict-guard failure and observed validation outcomes retained in numerical-preflight.json.",
    "tensor_tolerances": {"rtol": 1.3e-6, "atol": 1e-5},
    "numerical_followup": "Original probe stopped after2complete first epochs on a float32tensor guard. Reuse those exact model/fit/checkpoint files by explicit original path, without mutation. Training, monitoring cadence and best-loss criterion unchanged. No event test outcomes inspected to revise the guard.",
    "comparison": "Report whether best monitored first-epoch state improves the OLD selected best from ALL original epochs by>1e-5. Recompute selected prefix validation loss from saved weights. No policy/calibration/test forecast or90/90claim, even when loss improves.",
    "resume": "Atomic checkpoints at each validation contain model, optimizer, next shuffled-row offset, accumulated loss and best state. Tests compare unchanged original trajectory and interrupted resume on CPU/MPS.",
    "device": "mps",
}


def lock_plan(root):
    target = root / "plan.json"
    if target.exists():
        plan = read(target)
        if any(plan[k] != v for k, v in PLAN.items()) or plan["torch"] != str(torch.__version__):
            raise ValueError("Changed first-epoch diagnostic plan")
        check_hashes(plan)
        return plan
    scale_path = Path("artifacts/neural_count_scale_audit.json")
    scale = read(scale_path)
    check_hashes(scale)
    if scale["models"] != 12 or scale["device"] != "mps" or scale["torch"] != str(torch.__version__):
        raise ValueError("Incomplete prior diagnostic or changed backend")
    numeric_path = root / "numerical-preflight.json"
    numeric = read(numeric_path)
    check_hashes(numeric)
    if numeric["tensor_tolerances"] != PLAN["tensor_tolerances"]:
        raise ValueError("Changed numerical preflight tolerances")
    files = (
        Path(__file__),
        Path("src/moscollector/experiments/event_prefix_training.py"),
        Path("tests/test_event_prefix_training.py"),
    )
    plan = {
        **PLAN,
        "created_at": datetime.now(UTC).isoformat(),
        "torch": str(torch.__version__),
        "source_hashes": {
            **scale["source_hashes"],
            **numeric["source_hashes"],
            str(scale_path): sha256(scale_path),
            str(numeric_path): sha256(numeric_path),
        },
        "reuse_fits": numeric["reuse_fits"],
        "code_hashes": {
            **scale["code_hashes"],
            **numeric["code_hashes"],
            **{str(p): sha256(p) for p in files},
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(target, plan)
    return plan


def run(root):
    if not torch.backends.mps.is_available():
        raise ValueError("Saved MPS backend unavailable")
    plan = lock_plan(root)
    frame, dense = (
        pd.read_parquet(PROCESSED / n)
        for n in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    if not frame.as_of.lt("2026-06-01").all() or not dense.as_of.lt("2026-06-01").all():
        raise ValueError("Unexpected June inputs")
    events = pd.read_parquet(EVENT_FOLDER / "events.parquet", read_dictionary=["signal", "sensor_type"])
    triggers = pd.read_parquet(ONSET_FOLDER / "triggers.parquet")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    records = []
    for kind in KINDS:
        eps = episodes.loc[episodes.kind.eq(kind)]
        for fold in FOLDS:
            original_dir = PARENT / kind / fold
            prepared = read(original_dir / "data.json")
            periods = prepared["periods"]
            anchors = frame.loc[mask(frame, *map(pd.Timestamp, periods["train"]))]
            codec = fit_codec(anchors, prepared["features"])
            if codec != read(original_dir / "current-codec.json"):
                raise ValueError("Changed training-only current codec")
            data, sizes = {}, {}
            for part in ("train", "validation"):
                base, slots = period_rows(frame, dense, triggers, periods, part)
                data[part], sizes[part] = training_part(base, slots, events, codec, eps, kind)
            event_codec = fit_event_codec(events, data["train"]["bounds"])
            if event_codec != read(original_dir / "event-codec.json"):
                raise ValueError("Changed retained training-only event vocabulary")
            source = {
                "encoded": encode_events(events, event_codec),
                "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            }
            hashes = {p: {k: array_digest(v) for k, v in row.items()} for p, row in data.items()}
            source_hashes = {k: array_digest(v) for k, v in source.items()}
            interval = check_interval(sizes["train"]["original_rows"], TRAINING["batch_size"])
            for variant in VARIANTS:
                prior = original_dir / variant / "fit.json"
                old = read(prior)
                if (
                    old["signature"]["sizes"] != sizes
                    or old["signature"]["data_hashes"] != hashes
                    or old["signature"]["event_source_hashes"] != source_hashes
                    or old["signature"]["settings"] != TRAINING
                ):
                    raise ValueError("Training inputs/settings differ from original trajectory")
                directory = root / kind / fold / variant
                directory.mkdir(parents=True, exist_ok=True)
                fit_directory = Path(plan["reuse_fits"].get(f"{kind}/{fold}/{variant}", str(directory)))
                training_plan = (
                    read(fit_directory / "fit.json")["signature"]["plan_sha256"]
                    if fit_directory != directory
                    else sha256(root / "plan.json")
                )
                model, meta = fit_prefix(
                    fit_directory,
                    data,
                    source,
                    old["config"],
                    {
                        "plan_sha256": training_plan,
                        "old_fit_sha256": sha256(prior),
                        "kind": kind,
                        "fold": fold,
                        "variant": variant,
                    },
                    "mps",
                    interval,
                )
                np.testing.assert_allclose(
                    meta["first_epoch_validation_loss"],
                    old["history"][1]["validation_loss"],
                    rtol=1e-6,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    meta["first_epoch_training_loss"],
                    old["history"][1]["training_loss"],
                    rtol=1e-6,
                    atol=1e-7,
                )
                total_steps = math.ceil(len(data["train"]["target"]) / TRAINING["batch_size"])
                expected_steps = sorted({0, *range(interval, total_steps + 1, interval), total_steps})
                if [row["step"] for row in meta["history"]] != expected_steps or meta[
                    "completed_steps"
                ] != total_steps:
                    raise ValueError("Incomplete within-epoch monitoring")
                # fit_prefix returns only the selected prefix weights. Reload
                # them to ensure the validation claim is reproducible on disk.
                model.load_state_dict(
                    torch.load(fit_directory / "model.pt", map_location="mps", weights_only=True)
                )
                val = validation_loss(model, data["validation"], source, "mps")
                np.testing.assert_allclose(val, meta["best_validation_loss"], rtol=1e-7, atol=1e-8)
                final_matches_old_weights = None
                if old["best_epoch"] == 1:
                    end = torch.load(fit_directory / "last.pt", map_location="cpu", weights_only=True)[
                        "model"
                    ]
                    saved = torch.load(
                        original_dir / variant / "model.pt", map_location="cpu", weights_only=True
                    )
                    if end.keys() != saved.keys():
                        raise ValueError("Final state architecture differs")
                    for name, value in end.items():
                        if value.dtype != torch.float32 or saved[name].dtype != torch.float32:
                            raise ValueError("Expected original float32weights")
                        torch.testing.assert_close(value, saved[name], **PLAN["tensor_tolerances"])
                    final_matches_old_weights = True
                row = {
                    "kind": kind,
                    "fold": fold,
                    "variant": variant,
                    "original_training_rows": sizes["train"]["original_rows"],
                    "augmented_training_rows": sizes["train"]["rows"],
                    "check_every_steps": interval,
                    "checks": len(meta["history"]),
                    "best_step": meta["best_step"],
                    "completed_steps": total_steps,
                    "best_prefix_validation_loss": val,
                    "old_best_validation_loss": old["best_validation_loss"],
                    "old_best_epoch": old["best_epoch"],
                    "first_epoch_trajectory_losses_match": True,
                    "first_epoch_tensors_match_old_selected": final_matches_old_weights,
                    "improves_old_selected": val < old["best_validation_loss"] - TRAINING["min_delta"],
                    "fit_directory": str(fit_directory),
                    "imported_unchanged": fit_directory != directory,
                    "tensor_tolerances": PLAN["tensor_tolerances"],
                    "fit_sha256": sha256(fit_directory / "fit.json"),
                    "weights_sha256": sha256(fit_directory / "model.pt"),
                }
                records.append(row)
                write_json(directory / "comparison.json", row)
                print("PREFIX COMPARED", row, flush=True)
                del model
                gc.collect()
            del data, source, anchors, base, slots
            gc.collect()
    check_hashes(plan)
    if len(records) != 12:
        raise ValueError("Incomplete first-epoch diagnostic")
    sources = dict(plan["source_hashes"])
    sources.update({str(p): sha256(p) for p in root.rglob("*") if p.suffix in (".json", ".pt")})
    report = {
        "scope": PLAN["scope"],
        "plan_sha256": sha256(root / "plan.json"),
        "models": 12,
        "rows": records,
        "improved_models": sum(r["improves_old_selected"] for r in records),
        "event_precision_recall_evaluated": False,
        "serving_changed": False,
        "goal_achieved": False,
        "source_hashes": sources,
        "code_hashes": plan["code_hashes"],
    }
    write_json(Path("artifacts/event_prefix_audit.json"), report)
    print(
        "PREFIX COMPLETE",
        report["improved_models"],
        "of12improvevalidation; no event quality claim",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    run(parser.parse_args().output)
