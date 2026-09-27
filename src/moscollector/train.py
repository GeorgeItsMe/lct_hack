"""Training, independent calibration, alert policy tuning, and frozen temporal test."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from moscollector.alert_policy import alert_metrics  # noqa: F401 — re-exported for research code
from moscollector.features import KINDS
from moscollector.inference import CATEGORICAL, calibrated, model_input
from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.prepare import write_json

LOG = logging.getLogger(__name__)
SPLITS = {
    "train": ("2019-01-01", "2026-03-01"),
    "validation": ("2026-03-01", "2026-05-01"),
    "calibration": ("2026-05-01", "2026-05-16"),
    "policy": ("2026-05-16", "2026-06-01"),
    "test": ("2026-06-01", "2026-07-01"),
}


def feature_columns(df, feature_set="extended"):
    columns = [x for x in df.columns if not x.startswith("target_") and x not in ("as_of", "eligible")]
    if feature_set == "base":
        columns = [
            x for x in columns if not x.startswith("past_") and not x.endswith(("_recency_h", "_burst"))
        ]
    return columns


def split_mask(df, name):
    begin, end = map(pd.Timestamp, SPLITS[name])
    # Purge 25h so future labels and duration confirmation never enter the next split.
    return df.eligible & df.as_of.ge(begin) & (df.as_of + pd.Timedelta(hours=25)).lt(end)


def calibrate(raw, target):
    if len(np.unique(target)) < 2 or int(np.sum(target)) < 5:
        return {"slope": 1.0, "intercept": 0.0, "status": "insufficient_calibration_events"}
    regression = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    regression.fit(np.asarray(raw).reshape(-1, 1), target)
    return {
        "slope": float(regression.coef_[0, 0]),
        "intercept": float(regression.intercept_[0]),
        "status": "sigmoid_independent_period",
    }


def row_metrics(y, p):
    result = {
        "rows": len(y),
        "positives": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "pr_auc": float(average_precision_score(y, p)) if np.sum(y) else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }
    bins = np.linspace(0, 1, 11)
    result["calibration_curve"] = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        mask = (p >= low) & (p < (high if high < 1 else 1.00001))
        if mask.any():
            result["calibration_curve"].append(
                {
                    "predicted": float(p[mask].mean()),
                    "observed": float(np.asarray(y)[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    precision, recall, thresholds = precision_recall_curve(y, p)
    ids = np.unique(np.linspace(0, len(thresholds) - 1, min(100, len(thresholds))).astype(int))
    result["pr_curve"] = [
        {"precision": float(precision[i]), "recall": float(recall[i]), "threshold": float(thresholds[i])}
        for i in ids
    ]
    return result


def choose_policy(pred, eps):
    candidates = np.unique(
        np.r_[
            np.linspace(0.02, 0.95, 20), np.quantile(pred.probability, [0.75, 0.9, 0.95, 0.98, 0.995]), 1.01
        ]
    )
    rows = []
    for t in candidates:
        rows.append({"threshold": float(t), **alert_metrics(pred, eps, float(t))})
    if rows[0]["eligible_episodes"] < 10:
        disabled = next(r for r in rows if r["threshold"] == 1.01)
        disabled["status"] = "insufficient_policy_events"
        return disabled, rows
    # The budget is chosen before opening the final test: <= 1 false warning / 4 object-days.
    allowed = [r for r in rows if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    best = max(allowed or rows, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
    best["status"] = "validated_policy"
    return best, rows


def train(kinds: list[str], evaluate_test: bool = False, feature_set="extended"):
    begin = time.monotonic()
    df = pd.read_parquet(PROCESSED / "features.parquet")
    eps = pd.read_parquet(PROCESSED / "episodes.parquet")
    cols = feature_columns(df, feature_set)
    x = model_input(df, cols)
    masks = {name: split_mask(df, name) for name in SPLITS}
    model_dir = ARTIFACTS / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = ARTIFACTS / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "splits": SPLITS,
        "label": "proxy_sensor_episodes",
        "feature_cutoff": "strictly_before_as_of",
        "horizon_hours": 24,
        "purge_hours": 25,
        "step_hours": 3,
        "source_feature_sha256": hashlib.sha256((PROCESSED / "features.parquet").read_bytes()).hexdigest(),
        "models": {},
        "test_opened": False,
    }
    for kind in kinds:
        y = df[f"target_{kind}"].to_numpy()
        LOG.info(
            "Training %s on %s rows (%s positive)",
            kind,
            int(masks["train"].sum()),
            int(y[masks["train"]].sum()),
        )
        model = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.055,
            loss_function="Logloss",
            eval_metric="PRAUC",
            l2_leaf_reg=8,
            random_seed=42,
            thread_count=4,
            cat_features=CATEGORICAL,
            allow_writing_files=False,
            early_stopping_rounds=80,
            verbose=100,
        )
        model.fit(
            x[masks["train"]], y[masks["train"]], eval_set=(x[masks["validation"]], y[masks["validation"]])
        )
        model.save_model(str(model_dir / f"{kind}.cbm"))
        raw_cal = model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal")
        calibration = calibrate(raw_cal, y[masks["calibration"]])
        policy_mask = masks["policy"]
        policy_pred = df.loc[policy_mask, ["object_id", "as_of"]].copy()
        policy_pred["probability"] = calibrated(
            model.predict(x[policy_mask], prediction_type="RawFormulaVal"), calibration
        )
        policy, curve = choose_policy(policy_pred, eps[eps.kind.eq(kind)])
        prior = float(y[masks["train"]].mean())
        baseline = (
            df.loc[masks["train"], ["object_id", f"target_{kind}"]]
            .groupby("object_id")[f"target_{kind}"]
            .agg(["sum", "count"])
        )
        baseline["rate"] = (baseline["sum"] + 20 * prior) / (baseline["count"] + 20)
        baseline_rates = {str(k): float(v) for k, v in baseline.rate.items()}
        bp = policy_pred.copy()
        bp["probability"] = bp.object_id.astype(str).map(baseline_rates).fillna(prior)
        bpolicy, _ = choose_policy(bp, eps[eps.kind.eq(kind)])
        entry = {
            "kind": kind,
            "calibration": calibration,
            "threshold": policy["threshold"],
            "policy_period": policy,
            "policy_curve": curve,
            "features": cols,
            "best_iteration": model.best_iteration_,
            "training_rows": int(masks["train"].sum()),
            "training_positive_rows": int(y[masks["train"]].sum()),
            "calibration_rows": int(masks["calibration"].sum()),
            "baseline": {
                "global_rate": prior,
                "object_rates": baseline_rates,
                "threshold": bpolicy["threshold"],
            },
            "feature_importance": sorted(
                [
                    {"feature": n, "importance": float(v)}
                    for n, v in zip(cols, model.feature_importances_, strict=True)
                ],
                key=lambda v: -v["importance"],
            ),
        }
        write_json(model_dir / f"{kind}.json", entry)
        report["models"][kind] = {k: v for k, v in entry.items() if k not in ("features", "baseline")}
        LOG.info("Frozen %s policy: %s", kind, policy)
    write_json(ARTIFACTS / "validation_report.json", report)
    if evaluate_test:
        evaluate(kinds)
    LOG.info("Training completed in %.1fs", time.monotonic() - begin)


def evaluate(kinds):
    df = pd.read_parquet(PROCESSED / "features.parquet")
    eps = pd.read_parquet(PROCESSED / "episodes.parquet")
    test = split_mask(df, "test")
    report = json.loads((ARTIFACTS / "validation_report.json").read_text(encoding="utf-8"))
    report["test_opened"] = True
    report["test_opened_at"] = datetime.now(UTC).isoformat()
    predictions = []
    for kind in kinds:
        meta = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text(encoding="utf-8"))
        model = CatBoostClassifier()
        model.load_model(str(ARTIFACTS / "models" / f"{kind}.cbm"))
        start = time.monotonic()
        all_prob = calibrated(
            model.predict(model_input(df, meta["features"]), prediction_type="RawFormulaVal"),
            meta["calibration"],
        )
        elapsed = time.monotonic() - start
        pred = df.loc[test, ["object_id", "as_of"]].copy()
        pred["probability"] = all_prob[test]
        metrics, matches = alert_metrics(pred, eps[eps.kind.eq(kind)], meta["threshold"], with_matches=True)
        bp = pred.copy()
        bp["probability"] = (
            bp.object_id.astype(str)
            .map(meta["baseline"]["object_rates"])
            .fillna(meta["baseline"]["global_rate"])
        )
        baseline = alert_metrics(bp, eps[eps.kind.eq(kind)], meta["baseline"]["threshold"])
        report["models"][kind]["test"] = {
            "alerts": metrics,
            "rows": row_metrics(df.loc[test, f"target_{kind}"].to_numpy(), all_prob[test]),
            "baseline_alerts": baseline,
            "baseline_rows": row_metrics(
                df.loc[test, f"target_{kind}"].to_numpy(), bp.probability.to_numpy()
            ),
            "inference_ms_per_row": elapsed / len(df) * 1000,
        }
        out = (
            df[["object_id", "as_of", "eligible", f"target_{kind}"]]
            .copy()
            .rename(columns={f"target_{kind}": "target"})
        )
        out["kind"] = kind
        out["probability"] = all_prob
        out["threshold"] = meta["threshold"]
        out["split"] = "history"
        for name in SPLITS:
            out.loc[split_mask(df, name), "split"] = name
        predictions.append(out)
        write_json(ARTIFACTS / "predictions" / f"test-matches-{kind}.json", matches)
        LOG.info("TEST %s: %s", kind, metrics)
    pd.concat(predictions, ignore_index=True).to_parquet(
        ARTIFACTS / "predictions" / "all.parquet", index=False, compression="zstd"
    )
    write_json(ARTIFACTS / "evaluation_report.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinds", nargs="+", choices=list(KINDS.values()), default=list(KINDS.values()))
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--feature-set", choices=["base", "extended"], default="extended")
    p.add_argument("--train-since", default="2019-01-01", help="Lower time bound for the training sample")
    args = p.parse_args()
    SPLITS["train"] = (args.train_since, SPLITS["train"][1])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.evaluate_only:
        evaluate(args.kinds)
    else:
        train(args.kinds, args.evaluate_test, args.feature_set)


if __name__ == "__main__":
    main()
