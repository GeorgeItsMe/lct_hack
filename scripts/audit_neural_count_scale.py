"""Audit existing v37 initial/best/final states on validation only; no fitting."""

import argparse
import gc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from moscollector.count_validation import profile_count_loss
from moscollector.event_sequence_data import FOLDER as EVENT_FOLDER
from moscollector.event_sequence_data import encode_events
from moscollector.event_sequence_inputs import period_rows, training_part
from moscollector.event_sequence_verification import check_hashes, verified_evidence
from moscollector.goal90_research import read
from moscollector.onset_channel_features import FOLDER as ONSET_FOLDER
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

PARENT = Path("artifacts/research-v37-fixed")
ROOT = Path("artifacts/research-v39-diagnostic")
KINDS, FOLDS, VARIANTS = ("access", "fire", "fault"), ("screen_1", "screen_2"), ("current_mlp", "event_gru")
SPEC = {
    "scope": "Post-result diagnostic of existing neural count validation. No training, test/policy inference, calibration deployment, checkpoint replacement or90/90 claim.",
    "states": ["initial_constant", "selected_best", "final_epoch"],
    "population": "All12v37models, original weighted validation queries only; no new checkpoints or epoch search. Initial, selected best and final saved state assessed even if identical. Exact input hashes and saved raw validation losses must replay.",
    "decomposition": "For weighted mean E and raw log rate z, L=E[exp(z)]-E[y*z]. Optimal validation-only log scale=log(E[y])-log(E[exp(z)]). Profiled loss is constant optimum minus shape gain E[y*z]-E[y]*log(E[exp(z)]). Raw minus profiled is the nonnegative global scale penalty. This is an analytic diagnostic, not independent calibration.",
    "ranking": "Weighted row average precision also reported descriptively; overlapping rows are not independent events and AP is not the90/90 event objective.",
    "comparison": "Record final state's improvement over selected state in profiled loss by>1e-5 (original stopping min_delta), and whether an epoch0-selected model has positive final shape gain by>1e-5. No automatic promotion, new event forecasts or causal attribution.",
    "source": "https://docs.pytorch.org/docs/2.14/generated/torch.nn.PoissonNLLLoss.html",
}


def frozen_plan(root, device, torch_version):
    target = root / "plan.json"
    if target.exists():
        plan = read(target)
        if (
            any(plan[k] != v for k, v in SPEC.items())
            or plan["device"] != device
            or plan["torch"] != torch_version
        ):
            raise ValueError("Changed diagnostic specification/backend")
        check_hashes(plan)
        return plan
    parent = verified_evidence(PARENT)
    if parent["device"] != device or parent["torch"] != torch_version:
        raise ValueError("Diagnostic must replay the saved training backend")
    sources, origins = dict(parent["source_hashes"]), {}
    for kind in KINDS:
        for fold in FOLDS:
            for variant in VARIANTS:
                directory = PARENT / kind / fold / variant
                fit_path = directory / "fit.json"
                fit = read(fit_path)
                origin = Path(fit["reused_from"]["fit_file"]) if "reused_from" in fit else fit_path
                checkpoint = origin.parent / "last.pt"
                if not checkpoint.exists():
                    raise ValueError(f"No authoritative final checkpoint: {checkpoint}")
                name = f"{kind}/{fold}/{variant}"
                origins[name] = {"fit": str(fit_path), "original_fit": str(origin), "last": str(checkpoint)}
                for path in (fit_path, origin, checkpoint):
                    sources[str(path)] = sha256(path)
    for path in (PARENT / "weight-replay.json", PARENT / "plan.json"):
        sources[str(path)] = sha256(path)
    new_code = (
        Path(__file__),
        Path("src/moscollector/count_validation.py"),
        Path("tests/test_count_validation.py"),
    )
    plan = {
        **SPEC,
        "created_at": datetime.now(UTC).isoformat(),
        "device": device,
        "torch": torch_version,
        "origins": origins,
        "source_hashes": sources,
        "code_hashes": {**parent["code_hashes"], **{str(p): sha256(p) for p in new_code}},
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(target, plan)
    return plan


def describe(raw, data):
    return {
        **profile_count_loss(raw, data["target"], data["weight"]),
        "weighted_row_average_precision": float(
            average_precision_score(data["target"] > 0, raw, sample_weight=data["weight"])
        ),
    }


def run(root, device):
    import torch

    from moscollector.event_sequence_model import EventCountNetwork
    from moscollector.event_sequence_training import array_digest, predict
    from moscollector.neural_count_research import TRAINING

    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Saved MPS backend unavailable")
    torch.set_num_threads(TRAINING["cpu_threads"])
    plan = frozen_plan(root, device, str(torch.__version__))
    frame, dense = (
        pd.read_parquet(PROCESSED / n)
        for n in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    if not frame.as_of.lt("2026-06-01").all() or not dense.as_of.lt("2026-06-01").all():
        raise ValueError("Unexpected June input")
    events = pd.read_parquet(EVENT_FOLDER / "events.parquet", read_dictionary=["signal", "sensor_type"])
    triggers = pd.read_parquet(ONSET_FOLDER / "triggers.parquet")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    records, outputs = [], {}
    for kind in KINDS:
        eps = episodes.loc[episodes.kind.eq(kind)]
        for fold in FOLDS:
            directory = PARENT / kind / fold
            prepared = read(directory / "data.json")
            codec, event_codec = (read(directory / n) for n in ("current-codec.json", "event-codec.json"))
            base, slots = period_rows(frame, dense, triggers, prepared["periods"], "validation")
            data, sizes = training_part(base, slots, events, codec, eps, kind)
            source = {
                "encoded": encode_events(events, event_codec),
                "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            }
            data_hashes = {k: array_digest(v) for k, v in data.items()}
            event_hashes = {k: array_digest(v) for k, v in source.items()}
            for variant in VARIANTS:
                name = f"{kind}/{fold}/{variant}"
                paths = plan["origins"][name]
                fit, original_fit = read(Path(paths["fit"])), read(Path(paths["original_fit"]))
                if (
                    sizes != fit["signature"]["sizes"]["validation"]
                    or data_hashes != fit["signature"]["data_hashes"]["validation"]
                    or event_hashes != fit["signature"]["event_source_hashes"]
                ):
                    raise ValueError("Validation arrays/weights/targets or event vocabulary changed")
                checkpoint = torch.load(paths["last"], map_location="cpu", weights_only=True)
                if (
                    checkpoint["signature"] != original_fit["signature"]
                    or checkpoint["epoch"] != fit["completed_epochs"]
                    or checkpoint["best_epoch"] != fit["best_epoch"]
                    or checkpoint["best_loss"] != fit["best_validation_loss"]
                ):
                    raise ValueError("Final checkpoint provenance differs from completed fit")
                state = torch.load(directory / variant / "model.pt", map_location="cpu", weights_only=True)
                if state.keys() != checkpoint["best_state"].keys() or any(
                    not torch.equal(v, checkpoint["best_state"][k]) for k, v in state.items()
                ):
                    raise ValueError("Final checkpoint's best state differs from selected weights")
                model = EventCountNetwork(**fit["config"]).to(device)
                model.load_state_dict(state)
                raw_best = predict(model, data, source, device)
                model.load_state_dict(checkpoint["model"])
                raw_final = predict(model, data, source, device)
                raw_initial = np.full(
                    len(raw_best),
                    np.float32(np.clip(fit["config"]["initial_log_mean"], -12, 8)),
                    dtype=np.float64,
                )
                states = {
                    key: describe(raw, data)
                    for key, raw in zip(SPEC["states"], (raw_initial, raw_best, raw_final), strict=True)
                }
                for key, expected in zip(
                    SPEC["states"],
                    (
                        fit["history"][0]["validation_loss"],
                        fit["best_validation_loss"],
                        fit["history"][-1]["validation_loss"],
                    ),
                    strict=True,
                ):
                    np.testing.assert_allclose(states[key]["raw_loss"], expected, rtol=1e-7, atol=1e-8)
                destination = root / f"{kind}-{fold}-{variant}.parquet"
                table = (
                    slots[["object_id", "as_of"]]
                    .copy()
                    .assign(
                        target=data["target"],
                        weight=data["weight"],
                        raw_initial=raw_initial,
                        raw_best=raw_best,
                        raw_final=raw_final,
                    )
                )
                table.to_parquet(destination, index=False, compression="zstd")
                outputs[str(destination)] = sha256(destination)
                record = {
                    "kind": kind,
                    "fold": fold,
                    "variant": variant,
                    "validation_rows": len(table),
                    "eligible_validation_episodes": sizes["eligible_episodes"],
                    "best_epoch": fit["best_epoch"],
                    "final_epoch": fit["completed_epochs"],
                    "weighted_positive_fraction": float(
                        np.average(data["target"] > 0, weights=data["weight"])
                    ),
                    "states": states,
                    "profiled_final_improves_selected": states["final_epoch"]["profiled_loss"]
                    < states["selected_best"]["profiled_loss"] - TRAINING["min_delta"],
                    "epoch0_selected_but_final_has_shape_skill": fit["best_epoch"] == 0
                    and states["final_epoch"]["shape_gain_over_constant"] > TRAINING["min_delta"],
                    "validation_arrays_and_original_losses_replayed": True,
                }
                records.append(record)
                print(
                    "VALIDATION SCALE",
                    name,
                    {
                        k: (
                            round(v["raw_loss"], 6),
                            round(v["profiled_loss"], 6),
                            round(v["weighted_row_average_precision"], 6),
                        )
                        for k, v in states.items()
                    },
                    flush=True,
                )
                del model, checkpoint, state, table
                gc.collect()
            del data, source, base, slots
            gc.collect()
    check_hashes(plan)
    # Recompute every statistic from persisted arrays independently of GPU inference.
    for row in records:
        path = root / f"{row['kind']}-{row['fold']}-{row['variant']}.parquet"
        table = pd.read_parquet(path)
        data = {"target": table.target.to_numpy(), "weight": table.weight.to_numpy()}
        for key, column in zip(SPEC["states"], ("raw_initial", "raw_best", "raw_final"), strict=True):
            if describe(table[column].to_numpy(), data) != row["states"][key]:
                raise ValueError("Saved diagnostic statistics do not replay")
    result = {
        **SPEC,
        "plan_sha256": sha256(root / "plan.json"),
        "device": device,
        "torch": str(torch.__version__),
        "models": len(records),
        "states_assessed": len(records) * 3,
        "rows": records,
        "final_profile_improves_selected_models": sum(r["profiled_final_improves_selected"] for r in records),
        "epoch0_selected_but_final_has_shape_skill_models": sum(
            r["epoch0_selected_but_final_has_shape_skill"] for r in records
        ),
        "source_hashes": {
            **plan["source_hashes"],
            **outputs,
            str(root / "plan.json"): sha256(root / "plan.json"),
        },
        "code_hashes": plan["code_hashes"],
    }
    if len(records) != 12:
        raise ValueError("Incomplete validation audit")
    write_json(Path("artifacts/neural_count_scale_audit.json"), result)
    print(
        "COMPLETE",
        result["models"],
        result["final_profile_improves_selected_models"],
        result["epoch0_selected_but_final_has_shape_skill_models"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    run(args.output, args.device)
