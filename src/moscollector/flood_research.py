"""V38: fixed flood count model with adequately supported, separate time windows.

Research only. All original sensor episodes remain targets. Five previously
used months are reported, with no post-screen replacement or serving change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_capacity import matched_slots
from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import evaluator_for, policy_alerts, select
from moscollector.goal90_research import STRESS, pooled, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

ROOT = Path("artifacts/research-v38")
CUTOFF = pd.Timestamp("2026-06-01")
FIVE = {**FOLDS, **STRESS}
ARMS = ("poisson", "object_weekday")
SCHEMA = Path("artifacts/research-v9/screen_1/fire.json")
PLAN = {
    "scope": "adaptive_retrospective_flood_research_not_new_blind_test",
    "kind": "flood",
    "hypothesis": "Separate three-month validation, calibration and policy windows provide enough past flood episodes to fit and assess the previously disabled rare head.",
    "support_basis": "artifacts/flood_support_audit.json: window lengths chosen from support only, without fitting candidate models. Original short pulses retained.",
    "folds": FIVE,
    "periods": "Train from2022 until test-9months; validation, calibration and policy each3months; test1month. Every part purges25h before its end. No June telemetry reads.",
    "features": "The original94 extended features in archived v9 schema order. The fire metadata supplies names only, no fire model weights or outcomes. No novelty or target/eligibility features.",
    "training": "Original3h eligible anchors. CatBoost Poisson depth6,l2=8,lr=.04,max1000,seed42,CPU4,Poisson early stopping100; no weighting or hyperparameter search.",
    "baseline": "Train-only smoothed24h count rate. Object=(sum+240*globalmean)/(n+240); object-weekday=(sum+32*objectrate)/(n+32). Only object_id/as_of at prediction; unseen object uses global mean, unseen weekday uses object rate. Fixed pseudocounts represent30days and4weekday-weeks on3h anchors.",
    "calibration": "Each arm separately: sigmoid of log count plus mean-count scale on hourly calibration rows; neither policy nor test refits these.",
    "opportunities": "Original hourly research cache, only between consecutive eligible3h anchors with gap<=3h. Exact same event identities,94shared features and targets. Same hourly exposure for both arms.",
    "policy": "Fixed120 FinePendingSimulator rules: capacity .5/1/1.5/2, margin .25/.5/1/2/3/5, floor0/.5/.75/.9/.95. Select on preceding3months only by min(P/.9,R/.9,1),F1,R,P; FP<=.25/object-day, prefer>=10alerts, supported requires>=10events. Strict70min whole-hour release, one-to-one matching,24h expiry.",
    "evaluation": "Fit this single fixed family on ALL five folds, irrespective of Nov/Feb results (only5flood events there). Report all40eligible test episodes and each month. Compare against the fixed object-weekday baseline and disabled zero-alert head; no test-based model or rule selection.",
    "promotion": "No automatic activation. P/R90% refers to events and remains a whole-solution objective across all incident types. Flood proxy counts are not independent physical incident evidence.",
    "source": "https://catboost.ai/docs/en/concepts/loss-functions-regression#Poisson",
    "poisson_limit": "Nonnegative count loss, not an assertion that incidents follow a Poisson process.",
}


def periods_for(test_begin):
    test = pd.Timestamp(test_begin)
    if test + pd.DateOffset(months=1) > CUTOFF:
        raise ValueError("June is outside this study")
    starts = [test - pd.DateOffset(months=n) for n in (9, 6, 3)]
    points = [pd.Timestamp("2022-01-01"), *starts, test, test + pd.DateOffset(months=1)]
    if any(a >= b for a, b in zip(points, points[1:], strict=False)):
        raise ValueError("Invalid flood periods")
    return dict(
        zip(
            ("train", "validation", "calibration", "policy", "test"),
            zip(points[:-1], points[1:], strict=True),
            strict=True,
        )
    )


def fingerprint(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps([(c, str(frame[c].dtype)) for c in frame], ensure_ascii=False).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def fit_baseline(rows, counts):
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (len(rows),) or not len(rows) or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Invalid training counts")
    frame = rows[["object_id", "as_of"]].copy()
    frame["count"] = counts
    frame["weekday"] = frame.as_of.dt.dayofweek
    mean = float(counts.mean())
    objects, weekdays = {}, {}
    for obj, group in frame.groupby("object_id", sort=True):
        rate = float((group["count"].sum() + 240 * mean) / (len(group) + 240))
        objects[str(int(obj))] = rate
        for day, subset in group.groupby("weekday", sort=True):
            weekdays[f"{int(obj)}:{int(day)}"] = float(
                (subset["count"].sum() + 32 * rate) / (len(subset) + 32)
            )
    return {"global_mean": mean, "object_rates": objects, "weekday_rates": weekdays}


def baseline_raw(rows, fitted):
    # Reading exactly two columns prevents a hidden dependency on future labels.
    rates = [
        fitted["weekday_rates"].get(
            f"{int(obj)}:{int(day)}", fitted["object_rates"].get(str(int(obj)), fitted["global_mean"])
        )
        for obj, day in zip(rows.object_id, rows.as_of.dt.dayofweek, strict=True)
    ]
    return np.log(np.maximum(np.asarray(rates), np.exp(-20)))


def load_data():
    columns = read(SCHEMA)["features"]
    if len(columns) != 94 or len(set(columns)) != 94 or not set(CATEGORICAL).issubset(columns):
        raise ValueError("Unexpected original extended schema")
    if any(c.startswith("target_") or c in ("as_of", "eligible") for c in columns):
        raise ValueError("Future information in features")
    used = [*columns, "as_of", "eligible", "target_flood"]
    frame = pd.read_parquet(PROCESSED / "features.parquet", columns=used, filters=[("as_of", "<", CUTOFF)])
    dense = pd.read_parquet(PROCESSED / "features-hourly-research.parquet", columns=used)
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", CUTOFF), ("kind", "==", "flood")]
    )
    if not dense.as_of.lt(CUTOFF).all() or not episodes.episode_id.is_unique:
        raise ValueError("Invalid research data bounds/episode identities")
    return frame, dense, episodes, columns


def parts_for(frame, dense, episodes, test_begin):
    result = {}
    for part, bounds in periods_for(test_begin).items():
        reference = frame.loc[mask(frame, *bounds)].reset_index(drop=True)
        if part in ("train", "validation"):
            rows = reference
        else:
            rows = align_opportunities(dense.loc[mask(dense, *bounds)], reference)
            assert_same_episode_cohort(rows, reference, episodes)
            # A cohort check alone can miss absent negative rows or shared anchors.
            keys = ["object_id", "as_of"]
            shared = rows.merge(reference[keys], on=keys, validate="one_to_one")
            pd.testing.assert_frame_equal(
                shared[reference.columns].sort_values(keys).reset_index(drop=True),
                reference.sort_values(keys).reset_index(drop=True),
                check_exact=True,
            )
        counts = episode_counts(rows, episodes)
        if not len(rows) or not np.array_equal(counts > 0, rows.target_flood.astype(bool)):
            raise ValueError("Missing coverage or changed binary flood target")
        rows = rows.copy()
        rows["count_target"] = counts
        result[part] = rows
    return result


def part_signature(rows, episodes, cadence):
    evaluator = EventEvaluator(rows, episodes, cadence)
    return {
        "rows": len(rows),
        "positive_rows": int(rows.count_target.gt(0).sum()),
        "count_target_sum": int(rows.count_target.sum()),
        "eligible_episodes": evaluator.events,
        "fingerprint": fingerprint(rows),
        "cadence_hours": cadence,
    }


def validate_hashes(record):
    for category in ("source_hashes", "code_hashes"):
        for path, expected in record[category].items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"Changed {category}: {path}")


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    if target.exists():
        existing = read(target)
        if any(existing[k] != v for k, v in PLAN.items()):
            raise ValueError("Frozen study specification changed")
        validate_hashes(existing)
        return existing
    parity = read(Path("artifacts/hourly_feature_parity.json"))
    for path, expected in parity["source_files_unchanged"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("Changed original evidence")
    if sha256(PROCESSED / "features-hourly-research.parquet") != parity["output_sha256"]:
        raise ValueError("Changed original hourly cache")
    frame, dense, episodes, columns = load_data()
    audited = read(Path("artifacts/flood_support_audit.json"))
    preflight = {}
    for fold, begin in FIVE.items():
        parts = parts_for(frame, dense, episodes, begin)
        signatures = {
            p: part_signature(r, episodes, 3 if p in ("train", "validation") else 1) for p, r in parts.items()
        }
        support = next(
            r
            for r in audited["windows"]
            if r["fold"] == fold and r["validation_months"] == r["calibration_and_policy_months_each"] == 3
        )
        if any(signatures[p]["eligible_episodes"] != support["parts"][p]["eligible_episodes"] for p in parts):
            raise ValueError("Flood episode population differs from support audit")
        if any(
            signatures[p]["eligible_episodes"] < 10 for p in ("train", "validation", "calibration", "policy")
        ):
            raise ValueError("Insufficient non-test event support")
        evaluator = EventEvaluator(parts["test"], episodes, 1)
        capacity = sum(int(matched_slots(times, events).sum()) for _, _, times, events in evaluator.groups)
        preflight[fold] = {"parts": signatures, "test_hourly_matching_capacity": capacity}
        print("PREFLIGHT", fold, {p: v["eligible_episodes"] for p, v in signatures.items()}, flush=True)
    source_paths = [
        PROCESSED / "features.parquet",
        PROCESSED / "features-hourly-research.parquet",
        PROCESSED / "episodes.parquet",
        SCHEMA,
        Path("artifacts/flood_support_audit.json"),
        Path("artifacts/hourly_feature_parity.json"),
    ]
    code_names = (
        "flood_research",
        "flood_verification",
        "features",
        "alert_diagnostics",
        "cadence_capacity",
        "cadence_research",
        "count_research",
        "fine_cadence_research",
        "goal90_research",
        "research",
        "train",
        "paths",
        "prepare",
    )
    code_paths = [Path(f"src/moscollector/{name}.py") for name in code_names]
    code_paths += [Path("tests/test_flood_research.py")]
    plan = {
        **PLAN,
        "created_at": datetime.now(UTC).isoformat(),
        "feature_columns": columns,
        "environment": {p: version(p) for p in ("catboost", "numpy", "pandas", "scikit-learn")},
        "preflight": preflight,
        "source_hashes": {str(p): sha256(p) for p in source_paths},
        "code_hashes": {str(p): sha256(p) for p in code_paths},
        "capacity_limit": "Future-informed matching diagnostic only, not a deployable prediction score.",
    }
    write_json(target, plan)
    return plan


def score_tables(parts, columns, model=None, baseline=None):
    tables = {}
    for part in ("calibration", "policy", "test"):
        rows = parts[part]
        raw = (
            model.predict(model_input(rows, columns), prediction_type="RawFormulaVal", thread_count=4)
            if model is not None
            else baseline_raw(rows, baseline)
        )
        table = rows[["object_id", "as_of"]].copy()
        table["raw"] = raw
        table["count_target"] = rows.count_target.to_numpy()
        tables[part] = table
    cal = tables["calibration"]
    calibration = calibrate(cal.raw.to_numpy(), cal.count_target.to_numpy() > 0)
    scale = float(cal.count_target.sum() / np.exp(np.clip(cal.raw.to_numpy(), -20, 20)).sum())
    for table in tables.values():
        table["probability"] = calibrated(table.raw, calibration)
        table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
    return tables, calibration, scale


def evaluate_tables(tables, episodes):
    policy, frontier = select(tables["policy"], episodes, 1, len(tables["policy"]) / 24)
    scores = {}
    for part in ("policy", "test"):
        table = tables[part]
        table["alert"] = policy_alerts(table, episodes, policy)
        scores[part] = evaluator_for(table, episodes, 1, len(table) / 24).evaluate(table.alert, 0.5, 1)
    return {"policy": policy, "scores": scores}, frontier


def fit_fold(root, plan, frame, dense, episodes, columns, fold):
    directory = root / fold
    directory.mkdir(exist_ok=True)
    parts = parts_for(frame, dense, episodes, FIVE[fold])
    signatures = {
        p: part_signature(r, episodes, 3 if p in ("train", "validation") else 1) for p, r in parts.items()
    }
    if signatures != plan["preflight"][fold]["parts"]:
        raise ValueError("Input arrays changed after preflight")
    fit_path, weights = directory / "fit.json", directory / "model.cbm"
    model = CatBoostRegressor()
    if fit_path.exists():
        fit = read(fit_path)
        if (
            fit["plan_sha256"] != sha256(root / "plan.json")
            or fit["parts"] != signatures
            or sha256(weights) != fit["model_sha256"]
        ):
            raise ValueError("Stale cached flood fit")
        model.load_model(str(weights))
    else:
        model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            l2_leaf_reg=8,
            learning_rate=0.04,
            loss_function="Poisson",
            eval_metric="Poisson",
            random_seed=42,
            thread_count=4,
            cat_features=CATEGORICAL,
            allow_writing_files=False,
            early_stopping_rounds=100,
            verbose=200,
        )
        print("START FLOOD", fold, len(parts["train"]), flush=True)
        model.fit(
            model_input(parts["train"], columns),
            parts["train"].count_target,
            eval_set=(model_input(parts["validation"], columns), parts["validation"].count_target),
        )
        model.save_model(str(weights))
        fit = {
            "fold": fold,
            "plan_sha256": sha256(root / "plan.json"),
            "parts": signatures,
            "model_sha256": sha256(weights),
            "best_iteration": int(model.best_iteration_),
            "tree_count": int(model.tree_count_),
            "validation_history": model.evals_result_,
            "features": columns,
        }
        write_json(fit_path, fit)
    baseline = fit_baseline(parts["train"], parts["train"].count_target)
    write_json(directory / "baseline.json", baseline)
    arms = {}
    for arm in ARMS:
        tables, calibration, scale = score_tables(
            parts,
            columns,
            model=model if arm == "poisson" else None,
            baseline=baseline,
        )
        evaluated, frontier = evaluate_tables(tables, episodes)
        for part, table in tables.items():
            table.to_parquet(directory / f"{arm}-{part}.parquet", index=False)
        write_json(directory / f"{arm}-frontier.json", frontier)
        arms[arm] = {"calibration": calibration, "rate_scale": scale, **evaluated}
    zero = EventEvaluator(parts["test"], episodes, 1).evaluate(np.zeros(len(parts["test"])), 0.5, 1)
    result = {
        "fold": fold,
        "plan_sha256": sha256(root / "plan.json"),
        "periods": {k: list(map(str, v)) for k, v in periods_for(FIVE[fold]).items()},
        "arms": arms,
        "disabled_head": zero,
    }
    write_json(directory / "result.json", result)
    print("DONE FLOOD", fold, {a: v["scores"]["test"] for a, v in arms.items()}, flush=True)
    return result


def make_report(root, results):
    summary = {a: pooled([r["arms"][a]["scores"]["test"] for r in results]) for a in ARMS}
    return {
        "scope": PLAN["scope"],
        "plan_sha256": sha256(root / "plan.json"),
        "periods": results,
        "five_period_pooled": summary,
        "disabled_head": pooled([r["disabled_head"] for r in results]),
        "all_original_test_episodes_retained": all(v["eligible_episodes"] == 40 for v in summary.values()),
        "goal_achieved": False,
        "serving_changed": False,
        "limitation": "40proxy sensor episodes across five already used months;1/4/1/17/17. No independent physical incident labels. This study does not establish the whole-solution90/90 objective.",
    }


def run(root=ROOT):
    plan = prepare(root)
    frame, dense, episodes, columns = load_data()
    results = [fit_fold(root, plan, frame, dense, episodes, columns, fold) for fold in FIVE]
    report = make_report(root, results)
    if not report["all_original_test_episodes_retained"]:
        raise ValueError("Lost flood test episodes")
    validate_hashes(plan)
    write_json(root / "report.json", report)
    print(json.dumps(report["five_period_pooled"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("prepare", "run"), default="run")
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.output)
