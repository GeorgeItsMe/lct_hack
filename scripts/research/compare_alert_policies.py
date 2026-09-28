"""Attribute gains from alert cadence without fitting/changing a model."""

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.paths import PROCESSED
from moscollector.precision_research import choose_event_policy
from moscollector.prepare import write_json
from moscollector.research import FOLDS, mask
from moscollector.train import calibrated, model_input

frame = pd.read_parquet(PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
report = {
    "scope": "policy_only_on_previously_trained_models_no_new_weights",
    "selection_period": "Only original policy month; test threshold never optimized.",
    "results": {},
}
for fold in FOLDS:
    for kind in ("fault", "fire", "access"):
        folder = Path("artifacts/research-v4") / fold / "reference"
        meta = json.loads((folder / f"{kind}.json").read_text())
        model = CatBoostClassifier()
        model.load_model(str(folder / f"{kind}.cbm"))
        selected = frame.loc[mask(frame, *map(pd.Timestamp, meta["periods"]["policy"]))]
        pred = selected[["object_id", "as_of"]].copy()
        pred["probability"] = calibrated(
            model.predict(
                model_input(selected, meta["features"]), prediction_type="RawFormulaVal", thread_count=2
            ),
            meta["calibration"],
        )
        eps = episodes[episodes.kind.eq(kind)]
        policies = {
            "original": {"threshold": meta["threshold"], "cooldown_hours": 24},
            "precision_goal_fixed24": choose_event_policy(pred, eps, (24,)),
            "precision_goal_adaptive": choose_event_policy(pred, eps),
        }
        test = pd.read_parquet(folder / f"{kind}-predictions.parquet")
        evaluator = EventEvaluator(test, eps)
        scores = {
            name: evaluator.evaluate(test.probability, p["threshold"], p["cooldown_hours"])
            for name, p in policies.items()
        }
        report["results"][f"{fold}/{kind}"] = {"policies": policies, "scores": scores}
        write_json(Path("artifacts/policy_comparison.json"), report)
        print(
            fold,
            kind,
            [(name, round(m["precision"], 3), round(m["recall"], 3)) for name, m in scores.items()],
            flush=True,
        )
