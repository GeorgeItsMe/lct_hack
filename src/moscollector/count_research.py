"""Research v9: forecast episode multiplicity and budget outstanding warnings.

The evaluation target, episode grouping and 24h horizon remain unchanged. The
auxiliary regression target is the number of those same episodes in the window.
Observed outcomes can resolve a pending warning only after the existing 70min
confirmation delay. Nothing in this module activates a production model.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort, choose_policy
from moscollector.paths import PROCESSED
from moscollector.precision_research import columns_for, goal_score, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

KINDS = ("fault", "fire", "access")
PLAN = {
    "scope": "adaptive_retrospective_count_forecasting_not_new_blind_test",
    "hypothesis": "Binary probability cannot distinguish one impending episode from multiple episodes. Count regression and a causal outstanding-warning budget may reduce duplicate false alerts while covering recurrent episodes.",
    "reference": "v8 selected hourly classifier family for each kind; selection already completed before this study.",
    "training": "Original v6 recent-reference features and time splits. CatBoost Poisson, depth6, l2=8, lr=.04, max1000, early stop Poisson100, seed42, CPU4. Count>0 must exactly equal original binary labels on all used rows.",
    "calibration": "On separate calibration period: sigmoid of raw log rate for binary probability; mean-count multiplicative correction sum(observed)/sum(predicted).",
    "opportunities": "Same hourly cohort as v8, same 24h event matching, 70min delay before a past episode can resolve a pending warning.",
    "policies": "Two choices, selected on preceding policy period: ordinary probability threshold/cooldown1,2,3,6,12,24; pending budget emits if expected_count >= outstanding + margin and p>=floor. Margin .25,.5,.75,1,1.5,2,3; p floor0,.25,.5,.75. At most one warning/hour. Outstanding warnings expire after24h or resolve one-to-one to confirmed intervening episodes.",
    "selection": "Best pooled goal_score over Nov/Feb, then F1; require >5% goal_score gain over v8 selected classifier and no F1 loss. Only passing kinds proceed to May; require improvement in goal_score and no >5%F1 loss. Targets .75/.5 reported separately. No automatic activation.",
    "june": "Never read.",
    "external_training_data": "None",
    "sources": [
        "https://catboost.ai/docs/en/concepts/loss-functions-regression#Poisson",
        "https://arxiv.org/pdf/1505.07661",
    ],
    "not_claimed": "This is neither a Hawkes/RPP likelihood implementation nor proof that episodes follow a Poisson process. Poisson is used as a nonnegative count regression loss.",
}


def episode_counts(frame, episodes):
    result = np.zeros(len(frame), dtype=np.int32)
    for obj, rows in frame.reset_index(drop=True).groupby("object_id"):
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        events = (
            episodes.loc[episodes.object_id.eq(obj), "start_ts"]
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        result[rows.index] = np.searchsorted(events, at + 24 * HOUR_NS, side="left") - np.searchsorted(
            events, at, side="left"
        )
    return result


def pending_alerts(predictions, episodes, margin, probability_floor):
    """Causal planner: outcomes after t-70min cannot affect the decision at t."""
    frame = predictions.reset_index(drop=True)
    result = np.zeros(len(frame), dtype=np.float64)
    delay = 70 * 60 * 1_000_000_000
    for obj, rows in frame.groupby("object_id"):
        rows = rows.sort_values("as_of")
        events = (
            episodes.loc[episodes.object_id.eq(obj), "start_ts"]
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        next_event = 0
        pending = []
        for row in rows.itertuples():
            now = pd.Timestamp(row.as_of).value
            # Resolve each newly confirmed episode to at most one previous warning.
            while next_event < len(events) and events[next_event] + delay < now:
                observed = events[next_event]
                next_event += 1
                for i, issued in enumerate(pending):
                    if issued <= observed < issued + 24 * HOUR_NS:
                        pending.pop(i)
                        break
            pending = [issued for issued in pending if now - issued < 24 * HOUR_NS]
            if row.probability >= probability_floor and row.expected_count >= len(pending) + margin:
                result[row.Index] = 1
                pending.append(now)
    return result


def fit_one(frame, dense, episodes, directory, kind, test_begin):
    meta_path = directory / f"{kind}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    periods = periods_for(test_begin, kind)
    used = np.logical_or.reduce([mask(frame, *periods[k]) for k in ("train", "validation", "calibration")])
    training = frame.loc[used].reset_index(drop=True)
    masks = {k: mask(training, *periods[k]) for k in ("train", "validation", "calibration")}
    columns = columns_for(training, "recent_reference", kind)
    eps = episodes[episodes.kind.eq(kind)]
    counts = episode_counts(training, eps)
    y = training[f"target_{kind}"].to_numpy()
    if not np.array_equal(counts > 0, y.astype(bool)):
        raise ValueError("Count target does not match original episode definition")
    x = model_input(training, columns)
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
    print("START count", directory.name, kind, int(masks["train"].sum()), flush=True)
    model.fit(
        x[masks["train"]],
        counts[masks["train"]],
        eval_set=(x[masks["validation"]], counts[masks["validation"]]),
    )
    raw = model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal")
    cal = calibrate(raw, y[masks["calibration"]])
    rate_scale = float(counts[masks["calibration"]].sum() / np.exp(np.clip(raw, -20, 20)).sum())
    predictions = {}
    for period in ("policy", "test"):
        reference = frame.loc[mask(frame, *periods[period]), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *periods[period])], reference)
        pred = rows[["object_id", "as_of"]].copy()
        raw = model.predict(model_input(rows, columns), prediction_type="RawFormulaVal")
        pred["probability"] = calibrated(raw, cal)
        pred["expected_count"] = np.exp(np.clip(raw, -20, 20)) * rate_scale
        assert_same_episode_cohort(pred, reference, eps)
        predictions[period] = pred
    policy_pred = predictions["policy"]
    evaluator = EventEvaluator(policy_pred, eps, cadence_hours=1)
    regular = choose_policy(policy_pred, eps, 1)
    policies = {"regular": regular}
    pending_options = []
    for margin in (0.25, 0.5, 0.75, 1, 1.5, 2, 3):
        for floor in (0, 0.25, 0.5, 0.75):
            alerts = pending_alerts(policy_pred, eps, margin, floor)
            pending_options.append(
                {"margin": margin, "probability_floor": floor, **evaluator.evaluate(alerts, 0.5, 1)}
            )
    budget = [r for r in pending_options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    policies["pending"] = (
        max(supported or budget, key=lambda r: (goal_score(r), r["recall"], r["precision"], r["f1"]))
        if budget
        else {
            "margin": 1e12,
            "probability_floor": 1.01,
            **evaluator.evaluate(np.zeros(len(policy_pred)), 0.5, 1),
        }
    )
    policies["pending"]["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    test_pred = predictions["test"]
    evaluator = EventEvaluator(test_pred, eps, cadence_hours=1)
    scores = {
        "regular": evaluator.evaluate(test_pred.probability, regular["threshold"], regular["cooldown_hours"])
    }
    pending = policies["pending"]
    test_pred["pending_alert"] = pending_alerts(
        test_pred, eps, pending["margin"], pending["probability_floor"]
    )
    scores["pending"] = evaluator.evaluate(test_pred.pending_alert, 0.5, 1)
    result = {
        "kind": kind,
        "features": columns,
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "training_rows": int(masks["train"].sum()),
        "binary_target_parity": True,
        "best_iteration": model.best_iteration_,
        "calibration": cal,
        "rate_scale": rate_scale,
        "policies": policies,
        "scores": scores,
        "cadence_hours": 1,
    }
    model.save_model(str(directory / f"{kind}.cbm"))
    for period, pred in predictions.items():
        pred.to_parquet(directory / f"{kind}-{period}.parquet", index=False)
    write_json(meta_path, result)
    print("DONE count", directory.name, kind, scores, flush=True)
    return result


def pooled(results, policy):
    rows = [r["scores"][policy] for r in results]
    tp, alerts, events = (sum(r[k] for r in rows) for k in ("true_alerts", "alerts", "eligible_episodes"))
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


def run(output, stage):
    output.mkdir(parents=True, exist_ok=True)
    source = PROCESSED / "features.parquet"
    checksum = sha256(source)
    plan = output / "plan.json"
    if not plan.exists():
        write_json(
            plan,
            {
                **PLAN,
                "created_at": datetime.now(UTC).isoformat(),
                "source_sha256": checksum,
                "code_sha256": sha256(Path(__file__)),
            },
        )
    elif json.loads(plan.read_text())["source_sha256"] != checksum:
        raise ValueError("Source changed")
    cutoff = pd.Timestamp("2026-06-01")
    frame = pd.read_parquet(source, filters=[("as_of", "<", cutoff)])
    dense = pd.read_parquet(
        PROCESSED / "features-dense-rich.parquet",
        columns=frame.columns.tolist(),
        filters=[("as_of", "<", cutoff)],
    )
    episodes = pd.read_parquet(PROCESSED / "episodes.parquet", filters=[("start_ts", "<", cutoff)])
    if stage == "screen":
        baseline = json.loads(Path("artifacts/research-v8/selection.json").read_text())
        selection = {}
        for kind in KINDS:
            results = [
                fit_one(frame, dense, episodes, output / fold, kind, FOLDS[fold])
                for fold in ("screen_1", "screen_2")
            ]
            scores = {p: pooled(results, p) for p in ("regular", "pending")}
            candidate = max(scores, key=lambda p: (scores[p]["goal_score"], scores[p]["f1"]))
            ref = baseline[kind]["scores"][baseline[kind]["candidate"]]["1"]
            selected = scores[candidate]
            selection[kind] = {
                "candidate": candidate,
                "scores": scores,
                "reference": ref,
                "passes": selected["goal_score"] > ref["goal_score"] * 1.05 and selected["f1"] >= ref["f1"],
            }
            write_json(output / "selection.json", selection)
    else:
        selection = json.loads((output / "selection.json").read_text())
        baseline = json.loads(Path("artifacts/research-v8/confirmation.json").read_text())
        confirmation = {}
        for kind, selected in selection.items():
            if not selected["passes"]:
                confirmation[kind] = {"passes": False, "status": "screening_failed"}
                continue
            result = fit_one(frame, dense, episodes, output / "confirmation", kind, FOLDS["confirmation"])
            scores = result["scores"][selected["candidate"]]
            ref = baseline[kind]["hourly"]
            confirmation[kind] = {
                "candidate": selected["candidate"],
                "scores": scores,
                "reference": ref,
                "passes": goal_score(scores) > goal_score(ref) and scores["f1"] >= ref["f1"] * 0.95,
                "target_met": scores["precision"] >= 0.75 and scores["recall"] >= 0.5,
            }
        write_json(output / "confirmation.json", confirmation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v9"))
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    args = parser.parse_args()
    run(args.output, args.stage)
