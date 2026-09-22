"""Screen validation-selected early checkpoints against all existing controls."""

import argparse
import gc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.count_research import episode_counts
from moscollector.event_sequence_data import FOLDER as EVENT_FOLDER
from moscollector.event_sequence_data import encode_events
from moscollector.event_sequence_inputs import input_arrays, period_rows
from moscollector.event_sequence_verification import check_hashes
from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.goal90_research import pooled, primary_score, read
from moscollector.neural_optimization_verification import verify_prefix
from moscollector.onset_channel_features import FOLDER as ONSET_FOLDER
from moscollector.onset_policy import select_policy
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.train import calibrate, calibrated

PARENT = Path("artifacts/research-v37-fixed")
ROOT = Path("artifacts/research-v39")
KINDS, FOLDS, VARIANTS = ("access", "fire", "fault"), ("screen_1", "screen_2"), ("current_mlp", "event_gru")
REFERENCES = ("fresh_count", "old_count", "reference")
PLAN = {
    "scope": "Adaptive retrospective first-epoch checkpoint screening; not independent future validation.",
    "candidate": "For EACH architecture/fold, keep original v37best unless the completed first-epoch probe's best raw weighted validation loss improves it by>1e-5. This choice uses validation only and is frozen before any new calibration/policy/test predictions. No new fitting, mixing or hyperparameter search here.",
    "inputs": "Exact v37current codec, raw-event codec, original feature columns, strictly-past histories, augmented opportunities,25hpurge, original event cohort and hourly exposure. No event omission, relabeling or June evaluation.",
    "calibration_policy": "For changed checkpoints: independently refit preceding-period sigmoid and mean-count scale, then the same456pending policies on the next preceding period. Same delayed confirmation and24hone-to-one event evaluation. Unchanged weights reuse their fully verified v37tables and policies byte-for-byte; recompute metrics/calibration/policy to confirm exact parity.",
    "controls": "Original same-architecture v37model PLUS fresh_count,old_count,historical reference from the verified v37comparison. The stronger controls remain mandatory, not just the old neural baseline.",
    "screen": "Both architectures and all3kinds on originalNov/Feb. Eligible only if pooled min(P/.9,R/.9,1)>1.05*each of4references and no pooledF1loss. Select highest primary,F1,R,P; current_mlp first on exact tie. No replacement after outcomes.",
    "continuation": "Only a passing architecture may enter a separately linked confirmation: same fixed recipe onMay,Dec,Mar. May primary improves/F1>=95%every control; pooledDec/Mar primary>1.05*all,noF1loss,eachmonthF1>=90%controls. Requires original-trajectory plus monitored training where missing; no automatic promotion or completion claim from screen alone.",
    "limits": "Source first-epoch probe failed a strict tensor guard; its float32follow-up completed12trainings but failed one original end-loss guard. Separate post-run audit verifies all candidate inputs/selected validation weights while retaining both failures. Treat candidates as monitored retraining, NOT an isolated causal effect of check cadence or proof of trajectory equivalence. No event test outcomes informed numerical handling. Validation loss does not establish event quality or90/90. Floodv38and original serving models remain in scope unchanged.",
}


def screening(rows):
    candidates = {v: pooled([r["arms"][v]["scores"] for r in rows]) for v in VARIANTS}
    old = {v: pooled([r["old_neural"][v] for r in rows]) for v in VARIANTS}
    controls = {v: pooled([r["controls"][v] for r in rows]) for v in REFERENCES}
    eligible = [
        v
        for v in VARIANTS
        if all(
            primary_score(candidates[v]) > 1.05 * primary_score(ref) and candidates[v]["f1"] >= ref["f1"]
            for ref in (*controls.values(), old[v])
        )
    ]
    winner = (
        max(
            eligible,
            key=lambda v: (
                primary_score(candidates[v]),
                candidates[v]["f1"],
                candidates[v]["recall"],
                candidates[v]["precision"],
            ),
        )
        if eligible
        else None
    )
    return {
        "candidates": candidates,
        "old_neural": old,
        "controls": controls,
        "eligible_variants": eligible,
        "selected_variant": winner,
        "passed_screen": winner is not None,
    }


def prepare(root):
    target = root / "plan.json"
    if target.exists():
        plan = read(target)
        if any(plan[k] != v for k, v in PLAN.items()):
            raise ValueError("Changed early-checkpoint screening plan")
        check_hashes(plan)
        return plan
    probe = verify_prefix()
    selections = {}
    for row in probe["rows"]:
        kind, fold, variant = (row[k] for k in ("kind", "fold", "variant"))
        chosen = (
            Path(row["fit_directory"]) if row["improves_old_selected"] else PARENT / kind / fold / variant
        )
        selections[f"{kind}/{fold}/{variant}"] = {
            "uses_prefix": row["improves_old_selected"],
            "weights": str(chosen / "model.pt"),
            "weights_sha256": sha256(chosen / "model.pt"),
            "config": read(PARENT / kind / fold / variant / "fit.json")["config"],
        }
    files = (
        Path(__file__),
        Path("src/moscollector/prefix_checkpoint_verification.py"),
        Path("src/moscollector/neural_optimization_verification.py"),
        Path("tests/test_prefix_checkpoint_research.py"),
    )
    plan = {
        **PLAN,
        "created_at": datetime.now(UTC).isoformat(),
        "selections": selections,
        "device": "mps",
        "torch": read(PARENT / "plan.json")["torch"],
        "source_hashes": {
            **probe["source_hashes"],
            "artifacts/event_prefix_audit.json": sha256(Path("artifacts/event_prefix_audit.json")),
        },
        "code_hashes": {**probe["code_hashes"], **{str(p): sha256(p) for p in files}},
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(target, plan)
    return plan


def evaluate(root, plan, kind, fold, frame, dense, events, triggers, episodes, device):
    import torch

    from moscollector.event_sequence_model import EventCountNetwork
    from moscollector.event_sequence_training import predict

    directory = root / kind / fold
    directory.mkdir(parents=True, exist_ok=True)
    parent = PARENT / kind / fold
    archived = read(parent / "result.json")
    prepared = read(parent / "data.json")
    codec = read(parent / "current-codec.json")
    event_codec = read(parent / "event-codec.json")
    source = {
        "encoded": encode_events(events, event_codec),
        "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    }
    selected = {v: plan["selections"][f"{kind}/{fold}/{v}"] for v in VARIANTS}
    models = {}
    for variant, item in selected.items():
        if sha256(Path(item["weights"])) != item["weights_sha256"]:
            raise ValueError("Changed validation-selected weights")
        if item["uses_prefix"]:
            models[variant] = EventCountNetwork(**item["config"]).to(device)
            models[variant].load_state_dict(
                torch.load(item["weights"], map_location=device, weights_only=True)
            )
    tables = {v: {} for v in VARIANTS}
    for part in ("calibration", "policy", "test"):
        # All opportunities are shared between the original architectures.
        old = {v: pd.read_parquet(parent / f"{v}-{part}.parquet") for v in VARIANTS}
        pd.testing.assert_frame_equal(
            old[VARIANTS[0]][["object_id", "as_of"]], old[VARIANTS[1]][["object_id", "as_of"]]
        )
        data = None
        if models:
            base, slots = period_rows(frame, dense, triggers, prepared["periods"], part)
            pd.testing.assert_frame_equal(
                slots[["object_id", "as_of"]], old[VARIANTS[0]][["object_id", "as_of"]]
            )
            if (
                cohort(slots, episodes, 1 / 60) != cohort(base, episodes, 1)
                or len(base) / 24 != archived["exposure"][part]
            ):
                raise ValueError("Changed checkpoint event cohort/exposure")
            data = input_arrays(base, slots, events, codec)
        for variant in VARIANTS:
            table = old[variant].drop(columns="alert", errors="ignore").copy()
            if variant in models:
                table["raw"] = predict(models[variant], data, source, device)
            tables[variant][part] = table
        del data
    arms = {}
    for variant, predictions in tables.items():
        cal = predictions["calibration"]
        counts = episode_counts(cal, episodes)
        calibration = calibrate(cal.raw.to_numpy(), counts > 0)
        scale = float(counts.sum() / np.exp(np.clip(cal.raw, -20, 20)).sum())
        for table in predictions.values():
            table["probability"] = calibrated(table.raw, calibration)
            table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
        policy, frontier = select_policy(predictions["policy"], episodes, archived["exposure"]["policy"])
        for part in ("policy", "test"):
            predictions[part]["alert"] = policy_alerts(predictions[part], episodes, policy)
        score = evaluator_for(predictions["test"], episodes, 1 / 60, archived["exposure"]["test"]).evaluate(
            predictions["test"].alert, 0.5, 1 / 60
        )
        record = {"scores": score, "policy": policy, "calibration": calibration, "rate_scale": scale}
        if not selected[variant]["uses_prefix"]:
            if record != archived["arms"][variant] or frontier != read(parent / f"{variant}-frontier.json"):
                raise ValueError("Unchanged neural checkpoint policy differs")
            for part, table in predictions.items():
                pd.testing.assert_frame_equal(
                    table, pd.read_parquet(parent / f"{variant}-{part}.parquet"), check_exact=True
                )
        for part, table in predictions.items():
            table.to_parquet(directory / f"{variant}-{part}.parquet", index=False)
        write_json(directory / f"{variant}-frontier.json", frontier)
        arms[variant] = record
        print(
            "CHECKPOINT SCORE",
            kind,
            fold,
            variant,
            "changed",
            selected[variant]["uses_prefix"],
            score,
            flush=True,
        )
    result = {
        "kind": kind,
        "fold": fold,
        "plan_sha256": sha256(root / "plan.json"),
        "selected_checkpoints": selected,
        "arms": arms,
        "old_neural": {v: archived["arms"][v]["scores"] for v in VARIANTS},
        "controls": {
            r: archived["reference"] if r == "reference" else archived["arms"][r]["scores"]
            for r in REFERENCES
        },
        "exposure": archived["exposure"],
    }
    if any(
        a["scores"]["eligible_episodes"] != archived["reference"]["eligible_episodes"] for a in arms.values()
    ):
        raise ValueError("Changed original event denominator")
    write_json(directory / "result.json", result)
    del models, tables, source
    gc.collect()
    return result


def run(root):
    import torch

    plan = prepare(root)
    if str(torch.__version__) != plan["torch"] or not torch.backends.mps.is_available():
        raise ValueError("Saved checkpoint backend unavailable")
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
    report = {}
    for kind in KINDS:
        eps = episodes.loc[episodes.kind.eq(kind)]
        rows = [evaluate(root, plan, kind, f, frame, dense, events, triggers, eps, "mps") for f in FOLDS]
        selected = screening(rows)
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "status": "requires_linked_confirmation" if selected["passed_screen"] else "screen_rejected",
            "goal_achieved": False,
            "serving_changed": False,
        }
        write_json(root / "report.json", report)
        print("CHECKPOINT SCREEN", kind, selected, flush=True)
    check_hashes(plan)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    run(parser.parse_args().output)
