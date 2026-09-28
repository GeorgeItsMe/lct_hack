"""Conditional paired sensitivity of selected v10 count models across five months."""

import json
from pathlib import Path

import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.paths import PROCESSED
from moscollector.prepare import write_json
from moscollector.uncertainty import COUNT_COLUMNS, bootstrap

root = Path("artifacts/research-v10b")
selection = json.loads((root / "selection.json").read_text())
# Require completion of the predeclared additional periods before summarizing.
stress = json.loads((root / "stress.json").read_text())
episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
objects = pd.read_parquet(PROCESSED / "objects.parquet")
report = {
    "scope": "conditional_post_selection_sensitivity_not_independent_validation",
    "method": "5000 paired bootstrap samples of full parent histories across five retrospective periods.",
    "limitations": [
        "Conditional on already selected and fitted models; excludes training and selection uncertainty.",
        "Sixteen parent clusters; dependence between parents is not covered.",
        "The additional months appeared in earlier training/calibration, so they are not blind data.",
    ],
    "models": {},
}
for kind, selected in selection.items():
    if not selected["passes"]:
        continue
    assert "folds" in stress[kind]
    rows = []
    for fold in ("screen_1", "screen_2", "confirmation", "stress_1", "stress_2"):
        candidate_folder = root / fold / selected["candidate"]
        reference_folder = Path("artifacts/research-v9") / fold
        if not (reference_folder / f"{kind}.json").exists():
            reference_folder = root / fold / "reference"
        candidate = pd.read_parquet(candidate_folder / f"{kind}-test.parquet")
        reference = pd.read_parquet(reference_folder / f"{kind}-test.parquet")
        pd.testing.assert_frame_equal(candidate[["object_id", "as_of"]], reference[["object_id", "as_of"]])
        for obj, pred in candidate.groupby("object_id"):
            events = episodes[episodes.kind.eq(kind) & episodes.object_id.eq(obj)]
            base = reference[reference.object_id.eq(obj)]
            c = EventEvaluator(pred, events, 1).evaluate(pred.pending_alert, 0.5, 1)
            r = EventEvaluator(base, events, 1).evaluate(base.pending_alert, 0.5, 1)
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
        "candidate": selected["candidate"],
        "counts": counts.to_dict("records"),
        "by_parent": bootstrap(grouped.to_numpy()),
    }
    print(kind, report["models"][kind]["by_parent"]["percentile_95"], flush=True)
write_json(Path("artifacts/research_round4_uncertainty.json"), report)
