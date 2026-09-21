"""One bounded follow-up after context failed: simpler trees and recency weighting.

All periods have already been seen. This is explicitly an adaptive retrospective
comparison, never a new blind confirmation or a replacement for the frozen June test.
"""

import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
from catboost import CatBoostClassifier

from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.prepare import write_json
from moscollector.research import FOLDS, KINDS, fit_one, mask, pooled
from moscollector.train import alert_metrics, calibrate, calibrated, choose_policy, model_input


def fit_blend(frame, episodes, folder, kind, reference_folder, reference, candidate):
    """Fixed equal-logit blend; calibration and policy each use their own past month."""
    target = folder / f"{kind}-blend.json"
    if target.exists():
        return json.loads(target.read_text())
    if reference["features"] != candidate["features"] or reference["periods"] != candidate["periods"]:
        raise ValueError("Blend members must use identical features and temporal boundaries")
    models = []
    for location in (reference_folder, folder):
        model = CatBoostClassifier()
        model.load_model(str(location / f"{kind}.cbm"))
        models.append(model)
    raw, predictions = {}, {}
    for period in ("calibration", "policy", "test"):
        selected = mask(frame, *reference["periods"][period])
        x = model_input(frame.loc[selected], reference["features"])
        raw[period] = sum(model.predict(x, prediction_type="RawFormulaVal") for model in models) / 2
        predictions[period] = frame.loc[selected, ["object_id", "as_of", f"target_{kind}"]].copy()
    calibration = calibrate(raw["calibration"], predictions["calibration"][f"target_{kind}"].to_numpy())
    for period in ("policy", "test"):
        predictions[period]["probability"] = calibrated(raw[period], calibration)
    policy, _ = choose_policy(predictions["policy"], episodes[episodes.kind.eq(kind)])
    result = {
        "kind": kind,
        "config": "blend",
        "features": reference["features"],
        "periods": reference["periods"],
        "calibration": calibration,
        "threshold": policy["threshold"],
        "policy": policy,
        "test": alert_metrics(predictions["test"], episodes[episodes.kind.eq(kind)], policy["threshold"]),
        "members": ["reference", "regularized"],
        "raw_logit_weights": [0.5, 0.5],
    }
    write_json(target, result)
    predictions["test"].to_parquet(folder / f"{kind}-blend-predictions.parquet", index=False)
    print(f"DONE blend {kind}: F1={result['test']['f1']:.4f}", flush=True)
    return result


def passes_stability(reference, candidate):
    pairs = list(zip(reference, candidate, strict=True))
    return (
        len(pairs) == 3
        and sum(c["test"]["f1"] > r["test"]["f1"] * 1.05 for r, c in pairs) >= 2
        and all(c["test"]["f1"] >= r["test"]["f1"] * 0.95 for r, c in pairs)
        and all(
            c["test"]["eligible_episodes"] >= 10 and c["test"]["false_alerts_per_object_day"] <= 0.25
            for _, c in pairs
        )
        and pooled(candidate)["f1"] > pooled(reference)["f1"] * 1.05
    )


def run():
    output = ARTIFACTS / "research-v5"
    source = ARTIFACTS / "research-v4"
    output.mkdir(parents=True, exist_ok=True)
    if not (source / "confirmation.json").exists():
        raise ValueError("Finish and freeze the preceding research round first")
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "adaptive_retrospective_not_independent_validation",
        "reason": "Context candidate failed May confirmation; no previous candidate promoted.",
        "variant": "Original 66/94 features and history ranges; depth=4, L2=30, learning_rate=.04, max_iterations=1000, seed=42; normalized sample weights with 180-day half-life.",
        "second_variant": "Fixed 50/50 raw-logit ensemble of reference and regularized models; separately calibrated on the original calibration period.",
        "folds": FOLDS,
        "gate": "At least two periods gain >5% relative event F1; no period loses >5%; pooled F1 gain >5%; >=10 eligible episodes each; false alerts <=.25/object/day each.",
        "selection": "Only candidates passing every gate can replace a head; choose highest pooled F1 among them. No extra variants or blend-weight search.",
        "june": "Never loaded. Previous frozen models and June report remain unchanged.",
        "source_sha256": hashlib.sha256((PROCESSED / "features.parquet").read_bytes()).hexdigest(),
    }
    previous = json.loads((source / "plan.json").read_text())
    if plan["source_sha256"] != previous["source_sha256"]:
        raise ValueError("Reference and candidate must use identical source features")
    plan_path = output / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if any(old.get(k) != plan[k] for k in plan if k != "created_at"):
            raise ValueError("Plan changed; refusing to reuse existing results")
    else:
        write_json(plan_path, plan)
    frame = pd.read_parquet(
        PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
    )
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    report = {}
    for kind in KINDS:
        reference, candidate, blends = [], [], []
        for fold, date in FOLDS.items():
            reference.append(json.loads((source / fold / "reference" / f"{kind}.json").read_text()))
            candidate.append(fit_one(frame, episodes, output / fold, kind, "regularized", date))
            blends.append(
                fit_blend(
                    frame,
                    episodes,
                    output / fold,
                    kind,
                    source / fold / "reference",
                    reference[-1],
                    candidate[-1],
                )
            )
        variants = {}
        for name, members in (("regularized", candidate), ("blend", blends)):
            variants[name] = {
                "periods": {
                    fold: {"reference": r["test"], "candidate": c["test"]}
                    for fold, r, c in zip(FOLDS, reference, members, strict=True)
                },
                "pooled_reference": pooled(reference),
                "pooled_candidate": pooled(members),
                "passes_gate": passes_stability(reference, members),
            }
        passed = [name for name, result in variants.items() if result["passes_gate"]]
        selected = (
            max(passed, key=lambda name: variants[name]["pooled_candidate"]["f1"]) if passed else "reference"
        )
        report[kind] = {"variants": variants, "selected": selected}
        write_json(output / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
