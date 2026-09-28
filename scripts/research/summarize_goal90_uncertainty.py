"""Conditional paired sensitivity; separates warning-policy and quantile gains."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.experiments.goal90_research import STRESS, apply_policy, base_directory, read
from moscollector.paths import PROCESSED
from moscollector.prepare import write_json
from moscollector.research import FOLDS
from moscollector.uncertainty import COUNT_COLUMNS, bootstrap, interval

root = Path("artifacts/research-v11")
completed = read(root / "report.json")
family = completed["selection"]["selected"]
episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet",
    filters=[("kind", "==", "access"), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
)
objects = pd.read_parquet(PROCESSED / "objects.parquet")
report = {
    "scope": "conditional_post_selection_sensitivity_not_independent_validation",
    "method": "5000 paired samples of complete parent-node histories across all five months, seed20260921; per-object counts recomputed from fixed predictions.",
    "limits": "Conditional on fitted and adaptively selected models. Does not include training/selection uncertainty, dependence across parents or future calendar shifts. Bootstrap frequencies are not probabilities of future success.",
    "comparisons": {},
}
for comparison in ("archived_operational_family", "mean_with_90_policy"):
    rows = []
    for fold in (*FOLDS, *STRESS):
        candidate = pd.read_parquet(root / fold / family / "evaluated.parquet")
        reference = pd.read_parquet(base_directory(fold) / "access-test.parquet")
        keys = ["object_id", "as_of"]
        candidate = candidate.sort_values(keys).reset_index(drop=True)
        reference = reference.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(candidate[["object_id", "as_of"]], reference[["object_id", "as_of"]])
        if comparison == "mean_with_90_policy":
            meta = read(Path("artifacts/research-v12") / fold / "uncorrected.json")
            reference["pending_alert"] = apply_policy(reference, episodes, "mean_retarget", meta["policy"])
            actual = EventEvaluator(reference, episodes, 1).evaluate(reference.pending_alert, 0.5, 1)
            for field in ("true_alerts", "alerts", "eligible_episodes"):
                assert actual[field] == meta["scores"][field]
        for obj, pred in candidate.groupby("object_id"):
            eps = episodes[episodes.object_id.eq(obj)]
            baseline = reference[reference.object_id.eq(obj)]
            c = EventEvaluator(pred, eps, 1).evaluate(pred.candidate_alert, 0.5, 1)
            b = EventEvaluator(baseline, eps, 1).evaluate(baseline.pending_alert, 0.5, 1)
            assert c["eligible_episodes"] == b["eligible_episodes"]
            rows.append(
                {
                    "object_id": obj,
                    "model_tp": c["true_alerts"],
                    "model_alerts": c["alerts"],
                    "episodes": c["eligible_episodes"],
                    "baseline_tp": b["true_alerts"],
                    "baseline_alerts": b["alerts"],
                }
            )
    counts = pd.DataFrame(rows).groupby("object_id")[COUNT_COLUMNS].sum().reset_index()
    counts = counts.merge(objects[["object_id", "parent_id"]], validate="one_to_one")
    grouped = counts.groupby("parent_id")[COUNT_COLUMNS].sum().to_numpy()
    tp, alerts, events = grouped.sum(axis=0)[:3]
    expected = completed["five_period_pooled"]
    assert (int(tp), int(alerts), int(events)) == (
        expected["true_alerts"],
        expected["alerts"],
        expected["eligible_episodes"],
    )
    stats = bootstrap(grouped)
    rng = np.random.default_rng(20260921)
    sampled = grouped[rng.integers(0, len(grouped), size=(5000, len(grouped)))].sum(axis=1).astype(float)
    a, b, n, c, d = sampled.T
    with np.errstate(invalid="ignore", divide="ignore"):
        p, r, old_p, old_r = a / b, a / n, c / d, c / n
        candidate_goal = np.minimum(np.minimum(p, r) / 0.9, 1)
        reference_goal = np.minimum(np.minimum(old_p, old_r) / 0.9, 1)
    stats["paired_differences_95"] = {
        "precision": interval(p - old_p),
        "recall": interval(r - old_r),
        "primary_score": interval(candidate_goal - reference_goal),
    }
    report["comparisons"][comparison] = {"counts": counts.to_dict("records"), "by_parent": stats}
    print(comparison, stats["percentile_95"]["f1_difference"], stats["paired_differences_95"], flush=True)
write_json(Path("artifacts/goal90_uncertainty.json"), report)
