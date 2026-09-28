"""Research v7: semantic device modes and different model families.

All choices, feature provenance, time splits and the selection gate are recorded
before fitting. No new package or external/pretrained model is required.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from moscollector.alert_diagnostics import EventEvaluator, row_frontier
from moscollector.experiments.numeric_transforms import signed_log
from moscollector.paths import PROCESSED
from moscollector.precision_research import (
    KINDS,
    choose_event_policy,
    choose_row_policy,
    goal_score,
    periods_for,
    pool,
    row_operating,
)
from moscollector.precision_research import (
    fit_one as fit_catboost,
)
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, feature_columns, model_input

CONFIGS = ("device_catboost", "device_histogram", "device_loglinear")
PLAN = {
    "scope": "adaptive_retrospective_research_not_new_blind_test",
    "reason": "Raw log inspection found device modes (arming, battery, pump/fan, persistent bad-state counts) not represented by the original aggregate features.",
    "features": "333 v6 features + 84 device-mode features. Canonical conflict handling; all observations strictly before t; per-year state reset. No label redefinition.",
    "configs": {
        "device_catboost": "v6 sequence fit (depth6, l2=8, max1000, lr=.04) with extra device state features.",
        "device_histogram": "Sklearn histogram gradient boosting, 15 leaves, min_leaf=100, l2=10, max400, lr=.05, seed42. Early stop AP on explicit temporal validation, patience40. Four threads.",
        "device_loglinear": "Signed log1p numeric values; training-only median imputation/missing indicators/standardization; categorical one-hot. L2 logistic C=.01, class_weight=balanced, max_iter=300, seed42. Four threads.",
    },
    "periods": "Identical to v6 periods: test-2mo train cutoff; validation1mo; calibration14d; policy remaining days; purge25h.",
    "policy": "Identical to v6: joint P=.75,R=.5 goal score, FP budget, report both 24h and selected 3/6/12/24h cooldown; row policy separate.",
    "selection": "Best pooled event goal_score on November/February, then event F1. Require >5% gain vs v6 recent_reference with no F1 loss. Compare selected candidate vs v6 reference on May only after selection; goal_score must improve and F1 cannot lose >5%. No automatic deployment.",
    "june": "Excluded from all research reads. Original frozen files unchanged.",
    "external_data": "None",
}


def fit_alternative(frame, episodes, folder, kind, config, test_begin):
    target = folder / f"{kind}.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    if config == "device_catboost":
        result = fit_catboost(frame, episodes, folder, kind, "sequence", test_begin)
        result["config"] = config
        write_json(target, result)
        return result
    started = time.monotonic()
    periods = periods_for(test_begin, kind)
    selected = np.logical_or.reduce([mask(frame, *v) for v in periods.values()])
    df = frame.loc[selected].reset_index(drop=True)
    masks = {k: mask(df, *v) for k, v in periods.items()}
    columns = feature_columns(df)
    x = model_input(df, columns)
    y = df[f"target_{kind}"].to_numpy()
    print(f"START {folder} {kind}: n={int(masks['train'].sum())}, features={len(columns)}", flush=True)
    if config == "device_histogram":
        # Category dictionaries are fitted only on the training rows.
        category_levels = {}
        for col in CATEGORICAL:
            category_levels[col] = sorted(x.loc[masks["train"], col].unique().tolist())
            x[col] = pd.Categorical(x[col], categories=category_levels[col])
        model = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=10,
            early_stopping=True,
            n_iter_no_change=40,
            scoring="average_precision",
            random_state=42,
            categorical_features=CATEGORICAL,
        )
        with threadpool_limits(limits=4):
            model.fit(
                x[masks["train"]],
                y[masks["train"]],
                X_val=x[masks["validation"]],
                y_val=y[masks["validation"]],
            )
        extras = {"iterations": model.n_iter_, "category_levels": category_levels}
    else:
        cats = CATEGORICAL + ["hour", "day_of_week", "month"]
        numeric = [c for c in columns if c not in cats]
        transform = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(
                        FunctionTransformer(signed_log),
                        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                        StandardScaler(),
                    ),
                    numeric,
                ),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ]
        )
        model = make_pipeline(
            transform,
            LogisticRegression(
                C=0.01, class_weight="balanced", max_iter=300, solver="lbfgs", random_state=42
            ),
        )
        with threadpool_limits(limits=4):
            model.fit(x[masks["train"]], y[masks["train"]])
        extras = {"iterations": int(model[-1].n_iter_[0])}
    with threadpool_limits(limits=4):
        cal = calibrate(model.decision_function(x[masks["calibration"]]), y[masks["calibration"]])
        predictions = {}
        for name in ("policy", "test"):
            pred = df.loc[masks[name], ["object_id", "as_of"]].copy()
            pred["probability"] = calibrated(model.decision_function(x[masks[name]]), cal)
            predictions[name] = pred
    eps = episodes[episodes.kind.eq(kind)]
    policies = {
        "fixed24": choose_event_policy(predictions["policy"], eps, (24,)),
        "adaptive": choose_event_policy(predictions["policy"], eps),
    }
    evaluator = EventEvaluator(predictions["test"], eps)
    row_policy = choose_row_policy(y[masks["policy"]], predictions["policy"].probability.to_numpy())
    result = {
        "kind": kind,
        "config": config,
        "features": columns,
        **extras,
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "training_rows": int(masks["train"].sum()),
        "calibration": cal,
        "policies": policies,
        "event": {
            name: evaluator.evaluate(predictions["test"].probability, p["threshold"], p["cooldown_hours"])
            for name, p in policies.items()
        },
        "row_policy": row_policy,
        "row_test": row_operating(y[masks["test"]], predictions["test"].probability, row_policy["threshold"]),
        "row_test_ranking": row_frontier(y[masks["test"]], predictions["test"].probability),
        "seconds": time.monotonic() - started,
    }
    joblib.dump(model, folder / f"{kind}.joblib", compress=3)
    for name, pred in predictions.items():
        pred.to_parquet(folder / f"{kind}-{name}.parquet", index=False)
    write_json(target, result)
    m = result["event"]["adaptive"]
    print(
        f"DONE {config} {kind}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}; {result['seconds']:.1f}s",
        flush=True,
    )
    return result


def run(output: Path, stage: str, kinds=KINDS, configs=CONFIGS, reference=Path("artifacts/research-v6b")):
    output.mkdir(parents=True, exist_ok=True)
    feature_path = PROCESSED / "features-device.parquet"
    source_hash = sha256(feature_path)
    plan_path = output / "plan.json"
    if not plan_path.exists():
        write_json(
            plan_path,
            {
                **PLAN,
                "created_at": datetime.now(UTC).isoformat(),
                "source_sha256": source_hash,
                "reference_directory": str(reference),
                "code_sha256": sha256(Path(__file__)),
            },
        )
    elif json.loads(plan_path.read_text(encoding="utf-8"))["source_sha256"] != source_hash:
        raise ValueError("Feature source changed; start a new research directory")
    frame = pd.read_parquet(feature_path, filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        for fold in ("screen_1", "screen_2"):
            for kind in kinds:
                for config in configs:
                    fit_alternative(frame, episodes, output / fold / config, kind, config, FOLDS[fold])
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
            baseline = pool(
                [
                    json.loads(
                        (reference / f / "recent_reference" / f"{kind}.json").read_text(encoding="utf-8")
                    )
                    for f in ("screen_1", "screen_2")
                ]
            )
            winner = max(configs, key=lambda c: (scores[c]["goal_score"], scores[c]["f1"]))
            passes = (
                scores[winner]["goal_score"] > baseline["goal_score"] * 1.05
                and scores[winner]["f1"] >= baseline["f1"]
            )
            selection[kind] = {"scores": scores, "baseline": baseline, "candidate": winner, "passes": passes}
        write_json(output / "selection.json", selection)
        print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    else:
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        report = {}
        for kind in kinds:
            item = selection[kind]
            if not item["passes"]:
                report[kind] = {"passes": False, "reason": "screening_failed"}
                continue
            candidate = item["candidate"]
            result = fit_alternative(
                frame, episodes, output / "confirmation" / candidate, kind, candidate, FOLDS["confirmation"]
            )
            base = json.loads(
                (reference / "confirmation/recent_reference" / f"{kind}.json").read_text(encoding="utf-8")
            )["event"]["adaptive"]
            chosen = result["event"]["adaptive"]
            report[kind] = {
                "candidate": candidate,
                "reference": base,
                "candidate_metrics": chosen,
                "passes": goal_score(chosen) > goal_score(base) and chosen["f1"] >= base["f1"] * 0.95,
                "target_met": chosen["precision"] >= 0.75 and chosen["recall"] >= 0.5,
            }
            write_json(output / "confirmation.json", report)
        write_json(output / "confirmation.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research-v7b"))
    p.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    p.add_argument("--reference", type=Path, default=Path("artifacts/research-v6b"))
    args = p.parse_args()
    run(args.output, args.stage, args.kinds, args.configs, args.reference)
