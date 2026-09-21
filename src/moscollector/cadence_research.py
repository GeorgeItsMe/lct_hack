"""Research v8: isolate forecast/alert cadence with unchanged weights and labels."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.paths import PROCESSED
from moscollector.precision_research import fit_one, goal_score
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import calibrated, model_input

MODELS = {
    "legacy_reference": (Path("artifacts/research-v4"), "reference"),
    "recent_reference": (Path("artifacts/research-v6b"), "recent_reference"),
    "sequence": (Path("artifacts/research-v6b"), "sequence"),
    "device_catboost": (Path("artifacts/research-v7b"), "device_catboost"),
}
KINDS = ("fault", "fire", "access")
PLAN = {
    "scope": "adaptive_retrospective_cadence_study_not_new_blind_test",
    "hypothesis": "A 3h grid/24h cooldown misses repeated episodes even with good ranking; evaluate hourly recalculation with independent preceding-period policy selection.",
    "unchanged": "Same model weights, same calibration, same grouped episodes and 24h target. No resampling or synthetic training rows.",
    "models": list(MODELS),
    "kinds": KINDS,
    "folds": FOLDS,
    "cohort": "Hourly forecasts only between consecutive eligible 3h opportunities (gap <=3h). Assert identical eligible episode count for each model/period. All shared-time feature columns must match exactly; shared-time probabilities must match archived values.",
    "policy": "On each model's original policy interval only: compare 3h cadence (cooldowns3/6/12/24) with1h cadence(cooldowns1/2/3/6/12/24). Maximize min(P/.75,R/.5,1), then recall,precision,F1. False alerts <=.25/object-day; require10alerts; flag<10eligible episodes. Exposure uses actual cadence.",
    "selection": "Choose best pooled hourly event goal_score over Nov/Feb, then F1. Confirm only that family's 1h vs3h version in May. Require >5% goal_score improvement on May and no >5%F1 loss; target achievement reported separately.",
    "june": "Excluded; no changes to original June release or production.",
    "warning": "Cadence/calculated-policy gains are not gains from model weights. More frequent warnings can increase workload; report alerts and lead time.",
}


def align_opportunities(dense: pd.DataFrame, reference: pd.DataFrame):
    """Add only points lying between adjacent reference opportunities, no gaps."""
    keep = np.zeros(len(dense), dtype=bool)
    frame = dense.reset_index(drop=True)
    refs = {
        int(obj): rows.as_of.sort_values().to_numpy(dtype="datetime64[ns]").astype(np.int64)
        for obj, rows in reference.groupby("object_id")
    }
    for obj, rows in frame.groupby("object_id"):
        old = refs.get(int(obj))
        if old is None or not len(old):
            continue
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        left = np.searchsorted(old, at, side="right") - 1
        right = np.searchsorted(old, at, side="left")
        inside = (left >= 0) & (right < len(old))
        span = old[np.minimum(right, len(old) - 1)] - old[np.maximum(left, 0)]
        keep[rows.index] = inside & (span <= 3 * 3_600_000_000_000)
    return frame.loc[keep].reset_index(drop=True)


def choose_policy(pred, episodes, cadence):
    evaluator = EventEvaluator(pred, episodes, cadence_hours=cadence)
    cooldowns = (1, 2, 3, 6, 12, 24) if cadence == 1 else (3, 6, 12, 24)
    rows = [r for c in cooldowns for r in evaluator.frontier(pred.probability, c)]
    budget = [r for r in rows if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    chosen = dict(
        max(supported or budget, key=lambda r: (goal_score(r), r["recall"], r["precision"], r["f1"]))
    )
    chosen["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    return chosen


def assert_same_episode_cohort(dense, reference, episodes):
    """Compare event identities, not just denominators that could cancel out."""

    def cohort(predictions, cadence):
        return {
            obj: tuple(events)
            for obj, _, _, events in EventEvaluator(predictions, episodes, cadence).groups
            if len(events)
        }

    if cohort(dense, 1) != cohort(reference, 3):
        raise ValueError("Hourly and three-hour forecasts have different eligible episodes")


def evaluate_one(frame, episodes, output, family, fold, kind):
    target = output / fold / family / f"{kind}.json"
    if target.exists():
        return json.loads(target.read_text())
    root, config = MODELS[family]
    directory = root / fold / config
    meta_path = directory / f"{kind}.json"
    if not meta_path.exists():
        # Confirmation may need a family not selected by the earlier 3h study.
        if fold != "confirmation" or family not in ("sequence", "device_catboost"):
            raise FileNotFoundError(f"Complete screening model first: {meta_path}")
        source = "features-device.parquet" if family == "device_catboost" else "features-sequence.parquet"
        training_frame = pd.read_parquet(
            PROCESSED / source, filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
        )
        fitted = fit_one(training_frame, episodes, directory, kind, "sequence", FOLDS[fold])
        fitted["config"] = config
        write_json(meta_path, fitted)
        del training_frame
    meta = json.loads(meta_path.read_text())
    model = CatBoostClassifier()
    model.load_model(str(directory / f"{kind}.cbm"))
    old_table = pd.read_parquet(
        PROCESSED / "features.parquet",
        columns=["object_id", "as_of", "eligible"],
        filters=[("as_of", "<", pd.Timestamp("2026-06-01"))],
    )
    eps = episodes[episodes.kind.eq(kind)]
    predictions = {}
    for period in ("policy", "test"):
        start, end = map(pd.Timestamp, meta["periods"][period])
        reference = old_table.loc[mask(old_table, start, end), ["object_id", "as_of"]]
        dense = align_opportunities(frame.loc[mask(frame, start, end)], reference)
        pred = dense[["object_id", "as_of"]].copy()
        pred["probability"] = calibrated(
            model.predict(
                model_input(dense, meta["features"]), prediction_type="RawFormulaVal", thread_count=2
            ),
            meta["calibration"],
        )
        shared = reference.merge(pred, on=["object_id", "as_of"], validate="one_to_one")
        assert len(shared) == len(reference)
        assert_same_episode_cohort(pred, shared, eps)
        predictions[period] = {1: pred, 3: shared}
    old_predictions = directory / (
        f"{kind}-predictions.parquet" if family == "legacy_reference" else f"{kind}-test.parquet"
    )
    saved = pd.read_parquet(old_predictions)
    compared = saved.merge(
        predictions["test"][3],
        on=["object_id", "as_of"],
        suffixes=("_saved", "_hourly"),
        validate="one_to_one",
    )
    assert len(compared) == len(saved)
    difference = float(
        np.max(np.abs((compared.probability_saved - compared.probability_hourly).to_numpy()), initial=0)
    )
    if difference > 1e-10:
        raise ValueError(f"Prediction parity failed: {difference}")
    policies = {c: choose_policy(predictions["policy"][c], eps, c) for c in (3, 1)}
    scores = {
        str(c): EventEvaluator(predictions["test"][c], eps, c).evaluate(
            predictions["test"][c].probability, policies[c]["threshold"], policies[c]["cooldown_hours"]
        )
        for c in (3, 1)
    }
    result = {
        "family": family,
        "fold": fold,
        "kind": kind,
        "scores": scores,
        "policies": policies,
        "identical_episode_cohort": True,
        "shared_prediction_max_difference": difference,
        "source_model_sha256": sha256(directory / f"{kind}.cbm"),
        "model_metadata": str(meta_path),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    predictions["test"][1].to_parquet(target.with_name(f"{kind}-hourly.parquet"), index=False)
    write_json(target, result)
    print(
        fold,
        family,
        kind,
        "3h/1h",
        [(round(scores[str(c)]["precision"], 3), round(scores[str(c)]["recall"], 3)) for c in (3, 1)],
        flush=True,
    )
    return result


def pooled(results, cadence):
    m = [r["scores"][str(cadence)] for r in results]
    tp, alerts, events = (sum(x[k] for x in m) for k in ("true_alerts", "alerts", "eligible_episodes"))
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
    source = PROCESSED / "features-dense-rich.parquet"
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
        raise ValueError("Dense feature source changed")
    frame = pd.read_parquet(source)
    if not frame.as_of.lt(pd.Timestamp("2026-06-01")).all():
        raise ValueError("June is excluded from this research")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            scores = {}
            for family in MODELS:
                results = [
                    evaluate_one(frame, episodes, output, family, fold, kind)
                    for fold in ("screen_1", "screen_2")
                ]
                scores[family] = {str(c): pooled(results, c) for c in (3, 1)}
            candidate = max(MODELS, key=lambda c: (scores[c]["1"]["goal_score"], scores[c]["1"]["f1"]))
            selection[kind] = {"candidate": candidate, "scores": scores}
            write_json(output / "selection.json", selection)
    else:
        selection = json.loads((output / "selection.json").read_text())
        report = {}
        for kind in KINDS:
            family = selection[kind]["candidate"]
            result = evaluate_one(frame, episodes, output, family, "confirmation", kind)
            old, new = result["scores"]["3"], result["scores"]["1"]
            report[kind] = {
                "candidate": family,
                "three_hour": old,
                "hourly": new,
                "passes": goal_score(new) > goal_score(old) * 1.05 and new["f1"] >= old["f1"] * 0.95,
                "target_met": new["precision"] >= 0.75 and new["recall"] >= 0.5,
            }
            write_json(output / "confirmation.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/research-v8"))
    p.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    args = p.parse_args()
    run(args.output, args.stage)
