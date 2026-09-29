"""Blind June check of the active new-batch heads at the service cadence (1 hour).

Builds 2026 hourly features with the production builder into a temporary folder,
checks exact parity with the 3-hour training features on shared rows, and scores
June once with the event evaluator used for model selection. June never entered
training, calibration or rule selection. Needs the full prepared dataset.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.count_research import pending_alerts
from moscollector.features import build_dataset
from moscollector.inference import calibrated, model_input
from moscollector.model_registry import active_version, load_bundle
from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.prepare import write_json
from moscollector.research import mask

BEGIN, END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01")
KEEP = (
    "precision",
    "recall",
    "f1",
    "alerts",
    "true_alerts",
    "eligible_episodes",
    "false_alerts_per_object_day",
    "median_lead_hours",
)


def main():
    version = active_version()
    heads = load_bundle(version)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        build_dataset(
            [2026],
            step_hours=1,
            output_path=folder / "f.parquet",
            episode_output_path=folder / "e.parquet",
            audit_output_path=folder / "a.json",
        )
        hourly = pd.read_parquet(folder / "f.parquet")
    three = pd.read_parquet(PROCESSED / "features.parquet", filters=[("as_of", ">=", BEGIN)])
    shared = three.merge(hourly, on=["object_id", "as_of"], suffixes=("", "_h"))
    for col in three.columns.drop(["object_id", "as_of"]):
        a = pd.to_numeric(shared[col], errors="coerce").astype(float).fillna(-9)
        b = pd.to_numeric(shared[col + "_h"], errors="coerce").astype(float).fillna(-9)
        assert np.allclose(a, b), col
    test = hourly[mask(hourly, BEGIN, END)].reset_index(drop=True)
    rows = test[["object_id", "as_of"]]
    episodes = pd.read_parquet(PROCESSED / "episodes.parquet")
    report = {
        "version": version,
        "cadence_hours": 1,
        "period": [str(BEGIN), str(END)],
        "parity_rows": len(shared),
        "results": {},
    }
    for kind in ("fault", "fire", "access"):
        eps = episodes[episodes.kind.eq(kind)]
        evaluator = EventEvaluator(rows, eps, cadence_hours=1)
        if kind in heads:
            head = heads[kind]
            pred = rows.copy()
            pred["probability"] = head.probability(test)
            pred["expected_count"] = head.expected_count(test)
            policy = head.meta["alert_policy"]
            scores = pending_alerts(pred, eps, policy["margin"], policy["probability_floor"])
            result = evaluator.evaluate(scores, 0.5, 1)
        else:
            meta = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text(encoding="utf-8"))
            model = CatBoostClassifier()
            model.load_model(str(ARTIFACTS / "models" / f"{kind}.cbm"))
            prob = calibrated(
                model.predict(model_input(test, meta["features"]), prediction_type="RawFormulaVal"),
                meta["calibration"],
            )
            result = evaluator.evaluate(prob, meta["threshold"], 24)
        report["results"][kind] = {k: result[k] for k in KEEP}
        print(kind, report["results"][kind], flush=True)
    write_json(ARTIFACTS / "june_hourly_evaluation.json", report)


if __name__ == "__main__":
    main()
