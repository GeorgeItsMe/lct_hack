"""v35: weighted occurrence classification on the frozen minute-onset features."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities
from moscollector.count_extension_research import validate_opportunities
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort, evaluator_for
from moscollector.fresh_counts_research import anchor, source_files
from moscollector.goal90_research import STRESS, pooled, primary_score, read
from moscollector.minute_cadence_research import evaluate as evaluate_pending
from moscollector.onset_binary_policy import COOLDOWNS, FIXED, QUANTILES, select_threshold
from moscollector.onset_channel_features import CATS, COLUMNS, FOLDER, KEYS
from moscollector.onset_count_research import fit as fit_count
from moscollector.onset_training_data import attach, augmented_slots, model_input, snapshot_weights
from moscollector.paths import PROCESSED
from moscollector.peer_context_research import FIT
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated
from moscollector.waiting_time_research import threshold_alerts

KINDS = ("access", "fire", "fault")
ARMS = ("binary_candidate", "count_direct")
REFERENCES = ("count_direct", "old_pending", "reference")
BINARY_FIT = {**FIT, "loss_function": "Logloss", "eval_metric": "PRAUC:use_weights=true"}
PLAN = {
    "scope": "adaptive_retrospective_onset_binary_study_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "hypothesis": "v33 Poisson onset models fail and v34 cadence alone leaves fault months nearly silent. Test occurrence classification to reduce dominance of repeated-event counts, with direct probability warnings unconstrained by a predicted-count capacity. This does not assert that low mean counts alone explain missed events.",
    "training": BINARY_FIT,
    "rows_and_features": "Exactlyv33 augmented train/validation grids, original snapshot mass, 127/155/205 inputs and25h purge. Binary target is unchanged count24>0 at each new timestamp. No class balancing, extra resampling or feature substitutions. Original binary targets match, actual augmented counts/weights and event cohorts are checked. Weighted Logloss training and explicitly weighted PRAUC earlystop replace Poisson loss/earlystop; this is a classifier-family comparison, not a loss-only claim.",
    "controls": "Count_direct uses the samev33 Poisson features/rows/weights and the same new direct-threshold policy. Reuse all6 v33screen count models and require raw-score/calibration parity. If a classifier passes screening, fit only missing additional-period Poisson controls with byte-frozenv33fit code under a separate parent-linked plan. Old_pending is thev34 frozen-old-weight minute policy, reused where complete or evaluated with unchangedv34code for missing additional periods. Historical count anchor is a third reference. No v33/v34 study is extended or overwritten.",
    "inference": "Same quarter-plus-minute event-trigger opportunities for every model, same original event identities and hourly exposure. Original features held from current hour, fresh strictly-past channel context at the query. Per-arm sigmoid calibration on the original preceding period and unchanged24h binary target. No count capacity or pending state in either new direct arm; fixed causal alert cooldown, at most one warning per opportunity.",
    "thresholds": FIXED,
    "quantiles": QUANTILES,
    "cooldowns_hours": COOLDOWNS,
    "policy": "Policy-period only; same fixed thresholds plus own probability quantiles and same8cooldowns for binary/count_direct. Joint min(P/.9,R/.9,1),thenF1,R,P; originalFP budget.25/objectday,>=10alerts/events support.1.01 provides explicit no-warning fallback. Event matching remains one-to-one24h; no test outcomes choose thresholds.",
    "screen": "Nov/Feb pooled binary primary improves>5% over count_direct,old_pending and historical reference with noF1 loss. Only passing kinds continue. No post-outcome substitutions.",
    "confirmation": "May primary improves and F1>=95% of every reference. Dec/Mar pooled primary improves>5%, no pooledF1 loss, each month'sF1>=90% of every reference. Research gates are not90/90 achievement or automatic activation.",
    "limitations": "No June or new labels. Historical months adaptively reused; proxy sensor episodes, static catalog and unknown delivery latency remain limitations. Flood unsupported and full solution goal unchanged. Direct repeated warnings may increase workload; report all false warnings.",
    "metric_documentation": "https://catboost.ai/docs/en/concepts/loss-functions-multiclassification#PRAUC",
}


def metric(row, name):
    return row[name] if name in ("old_pending", "reference") else row["arms"][name]["scores"]


def screening(rows):
    scores = {
        name: pooled([metric(row, name) for row in rows]) for name in (*ARMS, "old_pending", "reference")
    }
    c = scores["binary_candidate"]
    return {
        **scores,
        "passed_screen": all(
            primary_score(c) > 1.05 * primary_score(scores[name]) and c["f1"] >= scores[name]["f1"]
            for name in REFERENCES
        ),
    }


def confirmation(rows):
    may, stress = (
        metric(rows[0], "binary_candidate"),
        pooled([metric(row, "binary_candidate") for row in rows[1:]]),
    )
    passed_may = all(
        primary_score(may) > primary_score(metric(rows[0], name))
        and may["f1"] >= 0.95 * metric(rows[0], name)["f1"]
        for name in REFERENCES
    )
    passed_stress = all(
        primary_score(stress) > 1.05 * primary_score(pooled([metric(row, name) for row in rows[1:]]))
        and stress["f1"] >= pooled([metric(row, name) for row in rows[1:]])["f1"]
        and all(metric(row, "binary_candidate")["f1"] >= 0.9 * metric(row, name)["f1"] for row in rows[1:])
        for name in REFERENCES
    )
    return {
        "passed_may": passed_may,
        "passed_stress": passed_stress,
        "research_eligible": bool(passed_may and passed_stress),
    }


def fit_classifier(root, kind, fold, frame, dense, context, triggers, episodes):
    directory = root / kind / fold / "binary"
    weights, target = directory / "model.cbm", directory / "fit.json"
    old_weights, old_metadata, _ = source_files(kind, fold)
    old = read(old_metadata)
    columns = [*old["features"], *COLUMNS, "evt_source_age_h"]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate onset feature")
    signature = {
        "plan_sha256": sha256(root / "plan.json"),
        "kind": kind,
        "fold": fold,
        "features": columns,
        "periods": old["periods"],
        "fit": BINARY_FIT,
    }
    if target.exists():
        meta = read(target)
        if meta["signature"] != signature or sha256(weights) != meta["model_sha256"]:
            raise ValueError("Binary onset model inputs or weights changed")
        return meta
    periods = {name: tuple(map(pd.Timestamp, dates)) for name, dates in old["periods"].items()}
    validate_opportunities(frame, dense, periods)
    pools, sizes = {}, {}
    for name in ("train", "validation"):
        base = frame.loc[mask(frame, *periods[name])]
        original_counts = episode_counts(base, episodes)
        if not np.array_equal(original_counts > 0, base[f"target_{kind}"].to_numpy().astype(bool)):
            raise ValueError("Original onset target differs from archived binary label")
        slots = augmented_slots(base, triggers)
        if not (slots.as_of + pd.Timedelta(hours=25)).lt(periods[name][1]).all():
            raise ValueError("Augmented target crosses split purge")
        rows = attach(base, slots, context, old["features"])
        counts, weight = episode_counts(rows, episodes), snapshot_weights(slots)
        if cohort(rows, episodes, 1 / 60) != cohort(base, episodes, 3):
            raise ValueError("Training augmentation changes event cohort")
        np.testing.assert_allclose(weight.sum(), len(base), rtol=0, atol=1e-7)
        pools[name] = Pool(
            model_input(rows, columns),
            (counts > 0).astype(np.int8),
            cat_features=[*CATEGORICAL, *CATS],
            weight=weight,
        )
        sizes[name] = {
            "rows": len(rows),
            "original_rows": len(base),
            "weight_sum": float(weight.sum()),
            "positive_rows": int(np.sum(counts > 0)),
            "weighted_positive_rows": float(weight[counts > 0].sum()),
            "eligible_episodes": EventEvaluator(base, episodes, 3).events,
        }
        del rows, base, slots
        gc.collect()
    directory.mkdir(parents=True, exist_ok=True)
    model = CatBoostClassifier(**BINARY_FIT, verbose=200)
    print("START binary onset fit", kind, fold, len(columns), sizes, flush=True)
    model.fit(pools["train"], eval_set=pools["validation"])
    temporary = weights.with_suffix(".cbm.tmp")
    model.save_model(str(temporary))
    temporary.replace(weights)
    meta = {
        "signature": signature,
        "features": columns,
        "periods": old["periods"],
        "sizes": sizes,
        "best_iteration": model.best_iteration_,
        "model_sha256": sha256(weights),
        "original_model_sha256": sha256(old_weights),
        "original_target_parity": True,
        "same_training_episode_cohort": True,
        "added_columns": [*COLUMNS, "evt_source_age_h"],
    }
    write_json(target, meta)
    print("DONE binary onset fit", kind, fold, "best", model.best_iteration_, flush=True)
    return meta


def control_sources(root, kind, fold, frame, dense, context, triggers, episodes):
    count = Path("artifacts/research-v33") / kind / fold / "onset"
    if not (count / "fit.json").exists():
        if fold.startswith("screen_"):
            raise ValueError("Missing predeclared screening count control")
        fit_count(root / "count_controls", kind, fold, frame, dense, context, triggers, episodes)
        count = root / "count_controls" / kind / fold / "onset"
    count_meta = read(count / "fit.json")
    if sha256(count / "model.cbm") != count_meta["model_sha256"]:
        raise ValueError("Count control weights changed")
    pending = Path("artifacts/research-v34") / kind / fold
    if not (pending / "result.json").exists():
        if fold.startswith("screen_"):
            raise ValueError("Missing screening pending control")
        evaluate_pending(root / "pending_controls", kind, fold, frame, dense, triggers, episodes)
        pending = root / "pending_controls" / kind / fold
    return count, count_meta, pending


def forecast_tables(kind, fold, frame, dense, context, triggers, episodes, binary_path, count_path, features):
    meta = read(source_files(kind, fold)[1])
    binary, count = CatBoostClassifier(), CatBoostRegressor()
    binary.load_model(str(binary_path))
    count.load_model(str(count_path))
    tables, exposure = {name: {} for name in ARMS}, {}
    for part in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, meta["periods"][part]))
        reference = frame.loc[mask(frame, *dates), KEYS]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        slots = augmented_slots(rows, triggers, spacing_hours=1, retain_quarters=True)
        values = attach(rows, slots, context, meta["features"])
        x = model_input(values, features)
        for name, model in (("binary_candidate", binary), ("count_direct", count)):
            pred = slots[KEYS].copy()
            pred["raw"] = model.predict(x, prediction_type="RawFormulaVal", thread_count=2)
            if not np.isfinite(pred.raw).all() or cohort(pred, episodes, 1 / 60) != cohort(rows, episodes, 1):
                raise ValueError("Invalid direct onset forecasts or changed cohort")
            tables[name][part] = pred
        exposure[part] = len(rows) / 24
    calibrations = {}
    for name in ARMS:
        cal = tables[name]["calibration"]
        calibration = calibrate(cal.raw.to_numpy(), episode_counts(cal, episodes) > 0)
        calibrations[name] = calibration
        for pred in tables[name].values():
            pred["probability"] = calibrated(pred.raw, calibration)
    return tables, calibrations, exposure


def evaluate(root, kind, fold, frame, dense, context, triggers, episodes):
    directory = root / kind / fold
    meta = fit_classifier(root, kind, fold, frame, dense, context, triggers, episodes)
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
        if result["fit"] != meta:
            raise ValueError("Changed classifier result")
        return result
    count, count_meta, pending = control_sources(root, kind, fold, frame, dense, context, triggers, episodes)
    if meta["features"] != count_meta["features"] or meta["sizes"] != count_meta["sizes"]:
        raise ValueError("Classifier and Poisson use different rows, weights or inputs")
    tables, calibrations, exposure = forecast_tables(
        kind,
        fold,
        frame,
        dense,
        context,
        triggers,
        episodes,
        directory / "binary/model.cbm",
        count / "model.cbm",
        meta["features"],
    )
    if fold.startswith("screen_"):
        old = Path("artifacts/research-v33") / kind / fold
        for part, current in tables["count_direct"].items():
            saved = pd.read_parquet(old / f"onset_candidate-{part}.parquet")
            pd.testing.assert_frame_equal(current[KEYS], saved[KEYS])
            for column in ("raw", "probability"):
                np.testing.assert_allclose(current[column], saved[column], rtol=1e-10, atol=1e-10)
    arms = {}
    for name in ARMS:
        pred = tables[name]
        print("START binary onset policy", kind, fold, name, flush=True)
        policy, frontier = select_threshold(pred["policy"], episodes, exposure["policy"])
        for part in ("policy", "test"):
            pred[part]["alert"] = threshold_alerts(pred[part], policy)
        scores = evaluator_for(pred["test"], episodes, 1 / 60, exposure["test"]).evaluate(
            pred["test"].alert, 0.5, policy["cooldown_hours"]
        )
        for part, table in pred.items():
            table.to_parquet(directory / f"{name}-{part}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {"scores": scores, "policy": policy, "calibration": calibrations[name]}
        print("DONE binary onset policy", kind, fold, name, scores, flush=True)
    pending_result = read(pending / "result.json")
    historical = anchor(kind, fold)
    old_pending = pending_result["arms"]["minute_candidate"]["scores"]
    if (
        any(a["scores"]["eligible_episodes"] != historical["eligible_episodes"] for a in arms.values())
        or old_pending["eligible_episodes"] != historical["eligible_episodes"]
    ):
        raise ValueError("Binary reference event denominator changed")
    if pending_result["exposure"] != exposure:
        raise ValueError("Binary references have different exposure")
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "fit": meta,
        "count_control": str(count),
        "count_fit": count_meta,
        "pending_control": str(pending),
        "pending_result_sha256": sha256(pending / "result.json"),
        "old_pending": old_pending,
        "reference": historical,
        "exposure": exposure,
        "same_episode_cohort": True,
        "same_training_rows_features_weights": True,
        "original_count_forecast_parity": fold.startswith("screen_"),
    }
    write_json(directory / "result.json", result)
    return result


def lock_plan(root):
    prior_root = Path("artifacts/research-v34")
    prior = read(prior_root / "plan.json")
    for category in ("source_hashes", "code_hashes"):
        for path, digest in prior[category].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed prior cadence source: {path}")
    sources = {prior_root / name for name in ("plan.json", "selection.json", "report.json")}
    sources.update(prior_root.glob("*/*/result.json"))
    sources.update(prior_root.glob("*/*/minute_candidate-*.parquet"))
    for kind in KINDS:
        for fold in ("screen_1", "screen_2"):
            base = Path("artifacts/research-v33") / kind / fold
            sources.update(base / "onset" / name for name in ("fit.json", "model.cbm"))
            sources.update(
                base / f"onset_candidate-{part}.parquet" for part in ("calibration", "policy", "test")
            )
    code = {
        Path(__file__),
        Path(select_threshold.__code__.co_filename),
        Path(threshold_alerts.__code__.co_filename),
        Path("tests/test_onset_binary_research.py"),
    }
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "source_hashes": {**prior["source_hashes"], **{str(p): sha256(p) for p in sorted(sources)}},
                "code_hashes": {**prior["code_hashes"], **{str(p): sha256(p) for p in sorted(code)}},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Changed binary onset study; use new directory")
    if not path.exists():
        write_json(path, plan)
    for name, source in (
        ("count_controls", Path("artifacts/research-v33/plan.json")),
        ("pending_controls", prior_root / "plan.json"),
    ):
        target = root / name / "plan.json"
        control_plan = {
            "parent_plan_sha256": sha256(path),
            "frozen_source_plan_sha256": sha256(source),
            "purpose": "Only missing additional-period matched controls for screening-passing classifier kinds; original studies stay frozen.",
        }
        if target.exists() and read(target) != control_plan:
            raise ValueError("Changed binary control plan")
        if not target.exists():
            write_json(target, control_plan)


def run(root, stage):
    lock_plan(root)
    frame, dense = (
        pd.read_parquet(PROCESSED / name)
        for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    context = pd.read_parquet(FOLDER / "context.parquet", read_dictionary=CATS)
    triggers = pd.read_parquet(FOLDER / "triggers.parquet")
    if not all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense, context, triggers)):
        raise ValueError("Post-May binary onset inputs")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selected = {}
        for kind in KINDS:
            rows = [
                evaluate(
                    root, kind, fold, frame, dense, context, triggers, episodes.loc[episodes.kind.eq(kind)]
                )
                for fold in ("screen_1", "screen_2")
            ]
            selected[kind] = screening(rows)
            write_json(root / "selection.json", selected)
            print("SELECT binary onset", kind, selected[kind], flush=True)
        return
    selected = read(root / "selection.json")
    if set(selected) != set(KINDS):
        raise ValueError("Complete all binary onset screening first")
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
            evaluate(root, kind, fold, frame, dense, context, triggers, episodes.loc[episodes.kind.eq(kind)])
            for fold in ("confirmation", *STRESS)
        ]
        gates = confirmation(rows)
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected[kind],
            "periods": rows,
            "five_period_pooled": {
                name: pooled([metric(row, name) for row in rows])
                for name in (*ARMS, "old_pending", "reference")
            },
            **gates,
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT binary onset", kind, gates, report[kind]["five_period_pooled"], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v35"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
