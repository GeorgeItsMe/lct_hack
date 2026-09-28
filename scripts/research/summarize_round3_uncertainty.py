"""Paired parent-cluster sensitivity of fixed count models versus v8 classifiers."""

import json
from pathlib import Path

import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import assert_same_episode_cohort
from moscollector.paths import PROCESSED
from moscollector.prepare import write_json
from moscollector.uncertainty import COUNT_COLUMNS, bootstrap

root = Path("artifacts")
selection = json.loads((root / "research-v9/selection.json").read_text())
confirmation = json.loads((root / "research-v9/confirmation.json").read_text())
reference = json.loads((root / "research-v8/selection.json").read_text())
episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
objects = pd.read_parquet(PROCESSED / "objects.parquet")
report = {
    "scope": "conditional_post_selection_sensitivity_not_independent_validation",
    "method": "Paired resampling of entire parent histories across Nov/Feb/May; 5000 replicates.",
    "limitations": [
        "Does not include training, calibration or model-selection uncertainty.",
        "Only 16 parent clusters; cross-parent dependence may remain.",
        "Repeated historical periods cannot establish future-month performance.",
    ],
    "models": {},
}
for kind, item in selection.items():
    if not confirmation[kind].get("passes"):
        continue
    policy_type = item["candidate"]
    rows = []
    for fold in ("screen_1", "screen_2", "confirmation"):
        folder = root / "research-v9" / fold
        metadata = json.loads((folder / f"{kind}.json").read_text())
        candidate = pd.read_parquet(folder / f"{kind}-test.parquet")
        base_folder = root / "research-v8" / fold / reference[kind]["candidate"]
        base_meta = json.loads((base_folder / f"{kind}.json").read_text())
        baseline = pd.read_parquet(base_folder / f"{kind}-hourly.parquet")
        keys = ["object_id", "as_of"]
        pd.testing.assert_frame_equal(
            candidate[keys].reset_index(drop=True), baseline[keys].reset_index(drop=True)
        )
        eps = episodes[episodes.kind.eq(kind)]
        assert_same_episode_cohort(candidate, baseline, eps)
        for obj, pred in candidate.groupby("object_id"):
            events = eps[eps.object_id.eq(obj)]
            policy = metadata["policies"][policy_type]
            if policy_type == "pending":
                scores, threshold, cooldown = pred.pending_alert, 0.5, 1
            else:
                scores, threshold, cooldown = pred.probability, policy["threshold"], policy["cooldown_hours"]
            c = EventEvaluator(pred, events, 1).evaluate(scores, threshold, cooldown)
            b = baseline[baseline.object_id.eq(obj)]
            p = base_meta["policies"]["1"]
            r = EventEvaluator(b, events, 1).evaluate(b.probability, p["threshold"], p["cooldown_hours"])
            assert c["eligible_episodes"] == r["eligible_episodes"]
            rows.append(
                {
                    "object_id": obj,
                    "model_tp": c["true_alerts"],
                    "model_alerts": c["alerts"],
                    "episodes": c["eligible_episodes"],
                    "baseline_tp": r["true_alerts"],
                    "baseline_alerts": r["alerts"],
                }
            )
    counts = pd.DataFrame(rows).groupby("object_id")[COUNT_COLUMNS].sum().reset_index()
    counts = counts.merge(objects[["object_id", "parent_id"]], validate="one_to_one")
    grouped = counts.groupby("parent_id")[COUNT_COLUMNS].sum()
    report["models"][kind] = {
        "selected_policy": policy_type,
        "counts": counts.to_dict("records"),
        "by_parent": bootstrap(grouped.to_numpy()),
        "by_object": bootstrap(counts[COUNT_COLUMNS].to_numpy()),
    }
write_json(root / "research_round3_uncertainty.json", report)
for kind, value in report["models"].items():
    print(kind, value["by_parent"]["percentile_95"])
