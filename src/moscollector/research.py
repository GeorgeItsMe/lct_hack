"""Predeclared temporal comparison, isolated from the deployed frozen June test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from moscollector.context_features import enrich_context
from moscollector.paths import PROCESSED
from moscollector.prepare import write_json
from moscollector.train import (
    CATEGORICAL,
    alert_metrics,
    calibrate,
    calibrated,
    choose_policy,
    feature_columns,
    model_input,
    row_metrics,
)

KINDS = ("fault", "fire", "access")
CONFIGS = ("reference", "context", "context_weighted")
FOLDS = {"screen_1": "2025-11-01", "screen_2": "2026-02-01", "confirmation": "2026-05-01"}
PLAN = {
    "created_at": datetime.now(UTC).isoformat(),
    "scope": "retrospective_rolling_origin_backtest_not_new_blind_test",
    "june": "Excluded from reads, tuning and selection; original release preserved.",
    "configs": {
        "reference": "Original family: base/all history for access, extended/14 months for fault and fire.",
        "context": "Same data and hyperparameters, plus causal normalized, recurrence and neighbor context.",
        "context_weighted": "Context variant with fixed positive training weight 3. Independent calibration unchanged.",
    },
    "folds": FOLDS,
    "protocol": "Train before test-4 months; early stopping next 2 months; calibration next month; policy next month; evaluate next month. Purge 25 hours at each boundary.",
    "selection": "Per kind: best pooled event F1 on the two screen folds, then precision. Require >5% relative F1 gain over reference before confirmation.",
    "promotion": "Selected candidate must also beat reference event F1 on confirmation, with >=10 eligible episodes and false alerts <=0.25/object/day. June cannot select a candidate.",
    "limits": "All these dates existed in earlier research. Forecasts are out of time within each fold, but this is retrospective confirmation, not a newly unseen test.",
}


def mask(frame, start, end):
    return frame.eligible & frame.as_of.ge(start) & (frame.as_of + pd.Timedelta(hours=25)).lt(end)


def fit_one(frame, episodes, folder, kind, config, test_begin):
    target = folder / f"{kind}.json"
    if target.exists():
        return json.loads(target.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    test_begin = pd.Timestamp(test_begin)
    train_end = test_begin - pd.DateOffset(months=4)
    cal_begin = test_begin - pd.DateOffset(months=2)
    policy_begin = test_begin - pd.DateOffset(months=1)
    start = pd.Timestamp("2022-01-01") if kind == "access" else train_end - pd.DateOffset(months=14)
    periods = {
        "train": (start, train_end),
        "validation": (train_end, cal_begin),
        "calibration": (cal_begin, policy_begin),
        "policy": (policy_begin, test_begin),
        "test": (test_begin, test_begin + pd.DateOffset(months=1)),
    }
    masks = {name: mask(frame, *bounds) for name, bounds in periods.items()}
    original_features = config in ("reference", "regularized")
    base = feature_columns(frame, "base" if kind == "access" and original_features else "extended")
    columns = [c for c in base if not original_features or not c.startswith("ctx_")]
    x = model_input(frame, columns)
    y = frame[f"target_{kind}"].to_numpy()
    model = CatBoostClassifier(
        iterations=1000 if config == "regularized" else 700,
        depth=4 if config == "regularized" else 6,
        learning_rate=0.04 if config == "regularized" else 0.055,
        loss_function="Logloss",
        eval_metric="PRAUC",
        l2_leaf_reg=30 if config == "regularized" else 8,
        random_seed=42,
        thread_count=4,
        cat_features=CATEGORICAL,
        allow_writing_files=False,
        early_stopping_rounds=80,
        verbose=100,
    )
    started = time.monotonic()
    weights = np.where(y[masks["train"]] > 0, 3.0, 1.0) if config == "context_weighted" else None
    if config == "regularized":
        age_days = (train_end - frame.loc[masks["train"], "as_of"]).dt.total_seconds() / 86400
        weights = np.exp2(-age_days.to_numpy() / 180)
        weights /= weights.mean()
    print(f"START {folder.name} {kind}: train={masks['train'].sum()}, features={len(columns)}", flush=True)
    model.fit(
        x[masks["train"]],
        y[masks["train"]],
        sample_weight=weights,
        eval_set=(x[masks["validation"]], y[masks["validation"]]),
    )
    calibration = calibrate(
        model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal"), y[masks["calibration"]]
    )
    predictions = {}
    for period in ("policy", "test"):
        pred = frame.loc[masks[period], ["object_id", "as_of"]].copy()
        pred["probability"] = calibrated(
            model.predict(x[masks[period]], prediction_type="RawFormulaVal"), calibration
        )
        predictions[period] = pred
    policy, _ = choose_policy(predictions["policy"], episodes[episodes.kind.eq(kind)])
    result = {
        "kind": kind,
        "config": config,
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "features": columns,
        "best_iteration": model.best_iteration_,
        "calibration": calibration,
        "threshold": policy["threshold"],
        "policy": policy,
        "test": alert_metrics(predictions["test"], episodes[episodes.kind.eq(kind)], policy["threshold"]),
        "test_rows": row_metrics(y[masks["test"]], predictions["test"].probability.to_numpy()),
        "training_rows": int(masks["train"].sum()),
        "seconds": time.monotonic() - started,
    }
    model.save_model(str(folder / f"{kind}.cbm"))
    predictions["test"].to_parquet(folder / f"{kind}-predictions.parquet", index=False)
    write_json(target, result)
    print(
        f"DONE {folder.name} {kind}: F1={result['test']['f1']:.4f}, seconds={result['seconds']:.1f}",
        flush=True,
    )
    return result


def pooled(results):
    tp = sum(r["test"]["true_alerts"] for r in results)
    alerts = sum(r["test"]["alerts"] for r in results)
    events = sum(r["test"]["eligible_episodes"] for r in results)
    return {
        "f1": 2 * tp / (alerts + events) if alerts + events else 0,
        "precision": tp / alerts if alerts else 0,
        "tp": tp,
        "alerts": alerts,
        "events": events,
    }


def run(output: Path, stage: str):
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    source_hash = hashlib.sha256((PROCESSED / "features.parquet").read_bytes()).hexdigest()
    if not plan_path.exists():
        write_json(plan_path, {**PLAN, "source_sha256": source_hash})
    elif json.loads(plan_path.read_text())["source_sha256"] != source_hash:
        raise ValueError("Feature source changed; use a new research directory")
    # Physically exclude June data from this research process.
    frame = pd.read_parquet(
        PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
    )
    frame = enrich_context(frame)
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        for fold in ("screen_1", "screen_2"):
            for config in CONFIGS:
                for kind in KINDS:
                    fit_one(frame, episodes, output / fold / config, kind, config, FOLDS[fold])
        selection = {}
        for kind in KINDS:
            scores = {
                config: pooled(
                    [
                        json.loads((output / fold / config / f"{kind}.json").read_text())
                        for fold in ("screen_1", "screen_2")
                    ]
                )
                for config in CONFIGS
            }
            winner = max(CONFIGS, key=lambda c: (scores[c]["f1"], scores[c]["precision"]))
            gain = scores[winner]["f1"] > scores["reference"]["f1"] * 1.05
            selection[kind] = {
                "scores": scores,
                "candidate": winner if gain else "reference",
                "passes_screen": gain,
            }
        write_json(
            output / "selection.json", {"selected_at": datetime.now(UTC).isoformat(), "models": selection}
        )
        print(json.dumps(selection, indent=2), flush=True)
    else:
        selection = json.loads((output / "selection.json").read_text())["models"]
        report = {}
        for kind in KINDS:
            candidate = selection[kind]["candidate"]
            reference = fit_one(
                frame,
                episodes,
                output / "confirmation" / "reference",
                kind,
                "reference",
                FOLDS["confirmation"],
            )
            challenger = fit_one(
                frame, episodes, output / "confirmation" / candidate, kind, candidate, FOLDS["confirmation"]
            )
            improves = (
                candidate != "reference"
                and challenger["test"]["f1"] > reference["test"]["f1"]
                and challenger["test"]["eligible_episodes"] >= 10
                and challenger["test"]["false_alerts_per_object_day"] <= 0.25
            )
            report[kind] = {
                "candidate": candidate,
                "reference": reference["test"],
                "challenger": challenger["test"],
                "promote": improves,
            }
        write_json(output / "confirmation.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v4"))
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    args = parser.parse_args()
    run(args.output, args.stage)
