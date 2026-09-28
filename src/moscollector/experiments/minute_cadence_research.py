"""v34: follow up the frozen v33 minute-grid controls without fitting new trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.experiments.fresh_counts_research import anchor, source_files
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.onset_channel_features import FOLDER, KEYS
from moscollector.experiments.onset_count_research import verify_quarter_control
from moscollector.experiments.onset_policy import select_policy
from moscollector.experiments.onset_training_data import augmented_slots
from moscollector.experiments.peer_context_research import control_source
from moscollector.experiments.quarter_count_features import carry_features
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import calibrate, calibrated, model_input

KINDS = ("access", "fire", "fault")
ARMS = ("minute_candidate", "quarter_control")
REFERENCES = ("quarter_control", "reference")
PRIOR = Path("artifacts/research-v33")
PLAN = {
    "scope": "adaptive_retrospective_followup_of_known_control_results_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "motivation": "V33 learned channel-onset count models failed. Its frozen old-weight minute control kept11/46 fault true alerts with31 fewer false alerts than the quarter control. This known screening observation motivates a separately locked cadence follow-up. Screening is reused evidence, not newly independent support. All three kinds remain in scope and are reported; no goal is narrowed to fault alone.",
    "weights": "Exact existing v9 access/fire and v19 fault count weights and66/94/144 inputs. No tree fitting, new features, resampling or changed24h episode labels. Raw logmean scores are held from the latest whole hour in both arms.",
    "grid": "Minute candidate retains all original quarter-hour slots and adds next-strict-minute triggers from all six onset families only between adjacent eligible hourly snapshots. Quarter control uses the original quarter grid. Same event identities,25h purge and original hourly exposure. Offline coverage eligibility is not a runtime signal. Unknown transport latency remains a limitation.",
    "calibration_and_policy": "Same per-grid sigmoid and mean-count scaling on the original calibration period. Same456 gate/capacity/margin choices on the policy period; same joint90/90 objective, support and FP budget. Candidate minimum cooldown1min, control15min; confirmation remains floor(start+70min to hour)+1hour with24h pending expiry. This is a grid plus per-grid calibration/policy comparison, not a gain in trained model quality.",
    "reuse": "Regenerate all six screening forecast grids and raw scores from frozen weights, refit preceding-period calibration, then require exact parity with v33 minute_control/quarter_control probabilities, mean counts and policies. Replay saved policies and metrics instead of selecting from test outcomes again. Report the same known screening scores. Additional periods rerun the same456 policy search without substituting variants.",
    "screen": "Pooled Nov/Feb min(P/.9,R/.9,1) improves>5% over quarter control and historical anchor, with noF1 loss. Only passing kinds reach May and Dec/Mar.",
    "confirmation": "May improves primary and retains>=95%F1 against both. Pooled Dec/Mar primary improves>5% with no pooledF1 loss and each month'sF1>=90% of each reference. No new variant or gate chosen after confirmation outcomes.",
    "limits": "No June selection/evaluation or new labels. Historical periods already used by other studies, not an independent future test. Sensor episodes remain proxies, flood unsupported, full90/90 requires all directions. No automatic activation.",
}


def metric(row, name):
    return row["reference"] if name == "reference" else row["arms"][name]["scores"]


def screening(rows):
    scores = {name: pooled([metric(row, name) for row in rows]) for name in (*ARMS, "reference")}
    c = scores["minute_candidate"]
    return {
        **scores,
        "passed_screen": all(
            primary_score(c) > 1.05 * primary_score(scores[name]) and c["f1"] >= scores[name]["f1"]
            for name in REFERENCES
        ),
    }


def confirmation(rows):
    may = metric(rows[0], "minute_candidate")
    stress = pooled([metric(row, "minute_candidate") for row in rows[1:]])
    passed_may = all(
        primary_score(may) > primary_score(metric(rows[0], name))
        and may["f1"] >= 0.95 * metric(rows[0], name)["f1"]
        for name in REFERENCES
    )
    passed_stress = all(
        primary_score(stress) > 1.05 * primary_score(pooled([metric(row, name) for row in rows[1:]]))
        and stress["f1"] >= pooled([metric(row, name) for row in rows[1:]])["f1"]
        and all(metric(row, "minute_candidate")["f1"] >= 0.9 * metric(row, name)["f1"] for row in rows[1:])
        for name in REFERENCES
    )
    return {
        "passed_may": passed_may,
        "passed_stress": passed_stress,
        "research_eligible": bool(passed_may and passed_stress),
    }


def forecasts(kind, fold, frame, dense, triggers, episodes):
    weights, metadata, _ = source_files(kind, fold)
    meta = read(metadata)
    model = CatBoostRegressor()
    model.load_model(str(weights))
    tables, exposure, sizes = {name: {} for name in ARMS}, {}, {}
    for part in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, meta["periods"][part]))
        reference = frame.loc[mask(frame, *dates), KEYS]
        hours = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        hourly = hours[KEYS].copy()
        hourly["raw"] = model.predict(
            model_input(hours, meta["features"]), prediction_type="RawFormulaVal", thread_count=2
        )
        if not np.isfinite(hourly.raw).all():
            raise ValueError("Invalid frozen count score")
        minute = augmented_slots(hours, triggers, spacing_hours=1, retain_quarters=True)
        quarter = subdivide_evaluation_slots(hours)
        for name, grid in zip(ARMS, (minute, quarter), strict=True):
            pred = carry_features(hourly, grid[KEYS], ["raw"])
            cadence = 1 / 60 if name == "minute_candidate" else 0.25
            if cohort(pred, episodes, cadence) != cohort(hourly, episodes, 1):
                raise ValueError("Minute follow-up changes event identities")
            tables[name][part] = pred
        exposure[part] = len(hours) / 24
        sizes[part] = {"hourly": len(hours), "minute": len(minute), "quarter": len(quarter)}
    calibrations = {}
    for name in ARMS:
        cal = tables[name]["calibration"]
        counts = episode_counts(cal, episodes)
        calibration = calibrate(cal.raw.to_numpy(), counts > 0)
        scale = float(counts.sum() / np.exp(np.clip(cal.raw, -20, 20)).sum())
        calibrations[name] = {"calibration": calibration, "rate_scale": scale}
        for pred in tables[name].values():
            pred["probability"] = calibrated(pred.raw, calibration)
            pred["expected_count"] = np.exp(np.clip(pred.raw, -20, 20)) * scale
    return tables, calibrations, exposure, sizes


def evaluate(root, kind, fold, frame, dense, triggers, episodes):
    directory = root / kind / fold
    target = directory / "result.json"
    weights, metadata, _ = source_files(kind, fold)
    if target.exists():
        old = read(target)
        if old["model_sha256"] != sha256(weights) or old["plan_sha256"] != sha256(root / "plan.json"):
            raise ValueError("Changed cadence result provenance")
        return old
    tables, calibrations, exposure, sizes = forecasts(kind, fold, frame, dense, triggers, episodes)
    reused = read(PRIOR / kind / fold / "result.json") if fold.startswith("screen_") else None
    directory.mkdir(parents=True, exist_ok=True)
    arms = {}
    for name in ARMS:
        cadence = 1 / 60 if name == "minute_candidate" else 0.25
        pred = tables[name]
        if reused is not None:
            old_name = "minute_control" if name == "minute_candidate" else name
            source = PRIOR / kind / fold
            for part in pred:
                saved = pd.read_parquet(source / f"{old_name}-{part}.parquet")
                pd.testing.assert_frame_equal(pred[part][KEYS], saved[KEYS])
                for column in ("raw", "probability", "expected_count"):
                    np.testing.assert_allclose(pred[part][column], saved[column], rtol=1e-10, atol=1e-10)
            old = reused["arms"][old_name]
            if any(old[k] != v for k, v in calibrations[name].items()):
                raise ValueError("Reused calibration changed")
            policy, frontier = old["policy"], read(source / f"{old_name}-frontier.json")
        else:
            print("START minute cadence policy", kind, fold, name, flush=True)
            policy, frontier = select_policy(pred["policy"], episodes, exposure["policy"], cadence)
        for part in ("policy", "test"):
            pred[part]["alert"] = policy_alerts(pred[part], episodes, policy)
        policy_metrics = evaluator_for(pred["policy"], episodes, cadence, exposure["policy"]).evaluate(
            pred["policy"].alert, 0.5, cadence
        )
        if any(v != policy[k] for k, v in policy_metrics.items()):
            raise ValueError("Selected policy replay changed")
        scores = evaluator_for(pred["test"], episodes, cadence, exposure["test"]).evaluate(
            pred["test"].alert, 0.5, cadence
        )
        if reused is not None and scores != reused["arms"][old_name]["scores"]:
            raise ValueError("Reused test metrics changed")
        parity = (
            verify_quarter_control(control_source(kind, fold), policy, scores, pred)
            if name == "quarter_control"
            else None
        )
        for part, table in pred.items():
            table.to_parquet(directory / f"{name}-{part}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {
            "scores": scores,
            "policy": policy,
            **calibrations[name],
            "cadence_hours": cadence,
            "archived_quarter_parity": parity,
        }
        print("DONE minute cadence", kind, fold, name, scores, flush=True)
    historical = anchor(kind, fold)
    if any(arm["scores"]["eligible_episodes"] != historical["eligible_episodes"] for arm in arms.values()):
        raise ValueError("Historical event denominator changed")
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "reference": historical,
        "model_sha256": sha256(weights),
        "metadata_sha256": sha256(metadata),
        "plan_sha256": sha256(root / "plan.json"),
        "exposure": exposure,
        "opportunities": sizes,
        "reused_known_screening": reused is not None,
        "same_episode_cohort": True,
        "new_weights": False,
    }
    write_json(target, result)
    return result


def lock_plan(root):
    previous = read(PRIOR / "plan.json")
    for category in ("source_hashes", "code_hashes"):
        for path, digest in previous[category].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed original onset study: {path}")
    sources = {PRIOR / name for name in ("plan.json", "selection.json", "report.json", "error-audit.json")}
    for kind in KINDS:
        for fold in ("screen_1", "screen_2"):
            base = PRIOR / kind / fold
            sources.add(base / "result.json")
            for arm in ("minute_control", "quarter_control"):
                sources.add(base / f"{arm}-frontier.json")
                sources.update(base / f"{arm}-{part}.parquet" for part in ("calibration", "policy", "test"))
    code = {Path(__file__), Path("tests/test_minute_cadence_research.py")}
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "source_hashes": {
                    **previous["source_hashes"],
                    **{str(p): sha256(p) for p in sorted(sources)},
                },
                "code_hashes": {**previous["code_hashes"], **{str(p): sha256(p) for p in sorted(code)}},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    if target.exists() and read(target) != plan:
        raise ValueError("Changed minute cadence study; use a new directory")
    if not target.exists():
        write_json(target, plan)


def run(root, stage):
    lock_plan(root)
    frame, dense = (
        pd.read_parquet(PROCESSED / name)
        for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    triggers = pd.read_parquet(FOLDER / "triggers.parquet")
    if not all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense, triggers)):
        raise ValueError("Post-May cadence inputs")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selected = {}
        for kind in KINDS:
            rows = [
                evaluate(root, kind, fold, frame, dense, triggers, episodes.loc[episodes.kind.eq(kind)])
                for fold in ("screen_1", "screen_2")
            ]
            selected[kind] = screening(rows)
            write_json(root / "selection.json", selected)
            print("SELECT minute cadence", kind, selected[kind], flush=True)
        return
    selected = read(root / "selection.json")
    if set(selected) != set(KINDS):
        raise ValueError("Complete all screening first")
    report = {}
    for kind in KINDS:
        if not selected[kind]["passed_screen"]:
            report[kind] = {
                "selection": selected[kind],
                "research_eligible": False,
                "status": "screen_failed",
            }
            continue
        rows = [
            evaluate(root, kind, fold, frame, dense, triggers, episodes.loc[episodes.kind.eq(kind)])
            for fold in ("confirmation", *STRESS)
        ]
        gates = confirmation(rows)
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected[kind],
            "periods": rows,
            "five_period_pooled": {
                name: pooled([metric(row, name) for row in rows]) for name in (*ARMS, "reference")
            },
            **gates,
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT minute cadence", kind, gates, report[kind]["five_period_pooled"], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v34"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
