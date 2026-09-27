"""Research v6: sequence features, fresher training, explicit P/R objectives.

No promotion happens here. The original 24h episode target and June release are
immutable. Policy-only and model gains are reported separately on the same rows.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve

from moscollector.alert_diagnostics import EventEvaluator, row_frontier
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, feature_columns, model_input

CONFIGS = ("recent_reference", "sequence", "sequence_regularized", "recurrence")
KINDS = ("fault", "fire", "access", "flood")
PLAN = {
    "scope": "adaptive_retrospective_research_not_new_blind_test",
    "target": "Unchanged grouped sensor proxy episode starts within [t,t+24h).",
    "goals": {"precision": 0.75, "recall": 0.5, "brief_precision": 0.7},
    "folds": FOLDS,
    "configs": {
        "recent_reference": "Original features, fresher train ending 2 months before test; depth6/l2=8.",
        "sequence": "Same fit plus 239 causal multi-resolution, variation and recurrence features.",
        "sequence_regularized": "Sequence with depth4/l2=30 and 180-day training weight half-life.",
        "recurrence": "Small model with calendar, static metadata and confirmed past episode history; depth4/l2=30.",
    },
    "periods": "Train until test-2mo (14mo for fault/fire/flood; from 2022 for access). Next 1mo early stopping; next 14d calibration; rest (~14-17d) policy. 25h purge each boundary.",
    "training": "CatBoost Logloss, early stop PRAUC, max1000, patience100, lr.04, seed42, 4 CPU threads.",
    "policy": "Search 3/6/12/24h cooldown only in policy period. Budget <=.25 false alerts/object/day. Maximize min(P/.75,R/.5,1), then recall, precision, F1; require >=10 alerts. If none satisfy alert count, mark unsupported and use closest supported-in-budget candidate without claiming success.",
    "row_policy": "Separately choose maximum row Recall at empirical row Precision>=.75 and >=10 positive predictions on policy period. If unavailable, minimize proportional P/R deficit. This is a different unit from incident evaluation and never substitutes for it.",
    "selection": "Choose best pooled event goal_score across two screening months. Require >5% relative score gain over recent_reference and event F1 not lower; then one May confirmation of chosen model vs reference. No June reads.",
    "confirmation": "Candidate improves event goal_score on May and does not lower F1 >5%; report each P/R and support even if gate fails. No automatic deployment.",
    "attribution": "Report both fixed24h and adaptive cooldown for every model; compare previous reference with policy-only adjustment separately.",
    "external_training_data": "None; provided telemetry only; all weights trained from scratch.",
}


def goal_score(metrics):
    return min(metrics["precision"] / 0.75, metrics["recall"] / 0.5, 1.0)


def choose_event_policy(pred, episodes, cooldowns=(3, 6, 12, 24)):
    evaluator = EventEvaluator(pred, episodes)
    rows = [r for c in cooldowns for r in evaluator.frontier(pred.probability, c)]
    budget = [r for r in rows if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    chosen = dict(
        max(supported or budget, key=lambda r: (goal_score(r), r["recall"], r["precision"], r["f1"]))
    )
    chosen["status"] = "supported_policy" if supported and evaluator.events >= 10 else "low_support_policy"
    chosen["goal_score"] = goal_score(chosen)
    return chosen


def choose_row_policy(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    # The PR curve uses >= threshold. Count ties explicitly for support.
    counts = len(p) - np.searchsorted(np.sort(p), thresholds, side="left")
    feasible = np.where((precision[:-1] >= 0.75) & (counts >= 10))[0]
    if len(feasible):
        i = feasible[np.argmax(recall[feasible])]
        status = "precision_target_met_on_policy"
    else:
        supported = np.where(counts >= 10)[0]
        if not len(supported):
            return {"threshold": 1.01, "status": "insufficient_rows"}
        score = np.minimum(precision[supported] / 0.75, recall[supported] / 0.5)
        i = supported[np.argmax(score)]
        status = "precision_target_unmet_on_policy"
    return {
        "threshold": float(thresholds[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "status": status,
    }


def row_operating(y, p, threshold):
    predicted = np.asarray(p) >= threshold
    tp = int(np.sum(np.asarray(y)[predicted]))
    positives = int(np.sum(y))
    count = int(np.sum(predicted))
    return {
        "precision": tp / count if count else 0,
        "recall": tp / positives if positives else 0,
        "true_positives": tp,
        "predicted_positives": count,
        "positives": positives,
        "rows": len(y),
        "threshold": threshold,
    }


def columns_for(frame, config, kind):
    cols = feature_columns(frame, "extended")
    if config == "recent_reference":
        cols = [c for c in cols if not c.startswith("seq_")]
        if kind == "access":
            cols = [c for c in cols if not c.startswith("past_") and not c.endswith(("_recency_h", "_burst"))]
    if config == "recurrence":
        static = CATEGORICAL + [
            "channel_count",
            "temperature_channels",
            "smoke_channels",
            "pump_channels",
            "water_channels",
            "hour",
            "day_of_week",
            "month",
            "weekend",
        ]
        cols = [
            c
            for c in cols
            if c in static or c.startswith("past_") or any(c.startswith(f"seq_{k}_") for k in KINDS)
        ]
    return cols


def periods_for(test_begin, kind):
    test = pd.Timestamp(test_begin)
    train_end = test - pd.DateOffset(months=2)
    cal_begin = test - pd.DateOffset(months=1)
    policy_begin = cal_begin + pd.Timedelta(days=14)
    start = pd.Timestamp("2022-01-01") if kind == "access" else train_end - pd.DateOffset(months=14)
    return {
        "train": (start, train_end),
        "validation": (train_end, cal_begin),
        "calibration": (cal_begin, policy_begin),
        "policy": (policy_begin, test),
        "test": (test, test + pd.DateOffset(months=1)),
    }


def fit_one(frame, episodes, folder, kind, config, test_begin):
    target = folder / f"{kind}.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    periods = periods_for(test_begin, kind)
    masks = {k: mask(frame, *v) for k, v in periods.items()}
    columns = columns_for(frame, config, kind)
    # Only materialize data used by this fit, to bound memory.
    selected = np.logical_or.reduce(list(masks.values()))
    df = frame.loc[selected].reset_index(drop=True)
    masks = {k: mask(df, *v) for k, v in periods.items()}
    x = model_input(df, columns)
    y = df[f"target_{kind}"].to_numpy()
    regularized = config in ("sequence_regularized", "recurrence")
    model = CatBoostClassifier(
        iterations=1000,
        depth=4 if regularized else 6,
        l2_leaf_reg=30 if regularized else 8,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=42,
        thread_count=4,
        cat_features=CATEGORICAL,
        allow_writing_files=False,
        early_stopping_rounds=100,
        verbose=200,
    )
    weights = None
    if config == "sequence_regularized":
        age = (periods["train"][1] - df.loc[masks["train"], "as_of"]).dt.total_seconds() / 86400
        weights = np.exp2(-age.to_numpy() / 180)
        weights /= weights.mean()
    started = time.monotonic()
    print(f"START {folder} {kind}: n={int(masks['train'].sum())}, features={len(columns)}", flush=True)
    model.fit(
        x[masks["train"]],
        y[masks["train"]],
        sample_weight=weights,
        eval_set=(x[masks["validation"]], y[masks["validation"]]),
    )
    cal = calibrate(
        model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal"), y[masks["calibration"]]
    )
    predictions = {}
    for period in ("policy", "test"):
        pred = df.loc[masks[period], ["object_id", "as_of"]].copy()
        pred["probability"] = calibrated(
            model.predict(x[masks[period]], prediction_type="RawFormulaVal"), cal
        )
        predictions[period] = pred
    eps = episodes[episodes.kind.eq(kind)]
    policies = {
        "fixed24": choose_event_policy(predictions["policy"], eps, (24,)),
        "adaptive": choose_event_policy(predictions["policy"], eps),
    }
    evaluator = EventEvaluator(predictions["test"], eps)
    result = {
        "kind": kind,
        "config": config,
        "features": columns,
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "training_rows": int(masks["train"].sum()),
        "best_iteration": model.best_iteration_,
        "calibration": cal,
        "policies": policies,
        "event": {
            name: evaluator.evaluate(predictions["test"].probability, p["threshold"], p["cooldown_hours"])
            for name, p in policies.items()
        },
    }
    row_policy = choose_row_policy(y[masks["policy"]], predictions["policy"].probability.to_numpy())
    result["row_policy"] = row_policy
    result["row_test"] = row_operating(
        y[masks["test"]], predictions["test"].probability, row_policy["threshold"]
    )
    result["row_test_ranking"] = row_frontier(y[masks["test"]], predictions["test"].probability)
    result["seconds"] = time.monotonic() - started
    model.save_model(str(folder / f"{kind}.cbm"))
    for name, pred in predictions.items():
        pred.to_parquet(folder / f"{kind}-{name}.parquet", index=False)
    write_json(target, result)
    m = result["event"]["adaptive"]
    print(
        f"DONE {folder.name} {kind}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}; row={result['row_test']}; {result['seconds']:.1f}s",
        flush=True,
    )
    return result


def pool(results, policy="adaptive"):
    metrics = [r["event"][policy] for r in results]
    tp = sum(m["true_alerts"] for m in metrics)
    alerts = sum(m["alerts"] for m in metrics)
    events = sum(m["eligible_episodes"] for m in metrics)
    result = {
        "precision": tp / alerts if alerts else 0,
        "recall": tp / events if events else 0,
        "f1": 2 * tp / (alerts + events) if alerts + events else 0,
        "true_alerts": tp,
        "alerts": alerts,
        "eligible_episodes": events,
    }
    result["goal_score"] = goal_score(result)
    return result


def run(output: Path, stage: str, kinds=KINDS, configs=CONFIGS):
    output.mkdir(parents=True, exist_ok=True)
    feature_path = PROCESSED / "features-sequence.parquet"
    source_hash = sha256(feature_path)
    plan_path = output / "plan.json"
    if not plan_path.exists():
        write_json(
            plan_path,
            {
                **PLAN,
                "created_at": datetime.now(UTC).isoformat(),
                "source_sha256": source_hash,
                "code_sha256": sha256(Path(__file__)),
            },
        )
    elif json.loads(plan_path.read_text(encoding="utf-8"))["source_sha256"] != source_hash:
        raise ValueError("Source changed; start a new research directory")
    frame = pd.read_parquet(feature_path, filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        for fold in ("screen_1", "screen_2"):
            for kind in kinds:
                for config in configs:
                    fit_one(frame, episodes, output / fold / config, kind, config, FOLDS[fold])
        selection = {}
        for kind in kinds:
            scores = {
                c: pool(
                    [
                        json.loads((output / f / c / f"{kind}.json").read_text(encoding="utf-8"))
                        for f in ("screen_1", "screen_2")
                    ]
                )
                for c in configs
            }
            winner = max(configs, key=lambda c: (scores[c]["goal_score"], scores[c]["f1"]))
            base = scores["recent_reference"]
            passes = (
                scores[winner]["goal_score"] > base["goal_score"] * 1.05
                and scores[winner]["f1"] >= base["f1"]
            )
            selection[kind] = {
                "scores": scores,
                "candidate": winner if passes else "recent_reference",
                "passes": passes,
            }
        write_json(output / "selection.json", selection)
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    else:
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        report = {}
        for kind in kinds:
            candidate = selection[kind]["candidate"]
            results = {
                c: fit_one(frame, episodes, output / "confirmation" / c, kind, c, FOLDS["confirmation"])
                for c in dict.fromkeys(("recent_reference", candidate))
            }
            base = results["recent_reference"]["event"]["adaptive"]
            chosen = results[candidate]["event"]["adaptive"]
            report[kind] = {
                "candidate": candidate,
                "reference": base,
                "candidate_metrics": chosen,
                "passes": candidate != "recent_reference"
                and goal_score(chosen) > goal_score(base)
                and chosen["f1"] >= base["f1"] * 0.95,
                "target_met": chosen["precision"] >= 0.75 and chosen["recall"] >= 0.5,
            }
            write_json(output / "confirmation.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research-v6b"))
    p.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    args = p.parse_args()
    run(args.output, args.stage, args.kinds, args.configs)
