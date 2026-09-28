"""Paired parent-history sensitivity for fixed quarter-hour access policies."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.experiments.fine_cadence_research import cohort
from moscollector.experiments.goal90_research import STRESS, apply_policy, read
from moscollector.experiments.waiting_time_research import base_directory
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS
from moscollector.uncertainty import COUNT_COLUMNS, bootstrap, interval

root = Path("artifacts/research-v20")
completed = read(root / "report.json")["access"]
if "five_period_pooled" not in completed:
    raise ValueError("Complete five-month access evaluation first")
sources = {root / "report.json", PROCESSED / "episodes.parquet", PROCESSED / "objects.parquet"}
episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet",
    filters=[("kind", "==", "access"), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
)
objects = pd.read_parquet(PROCESSED / "objects.parquet")
report = {
    "scope": "conditional_post_selection_sensitivity_not_independent_validation",
    "method": "5000 paired samples of complete parent-node histories across all five months, seed20260921; fixed-policy event counts recomputed at each arm's actual cadence; exact episode-cohort parity.",
    "limits": "Conditional on adaptively selected models/calibrations/policies. Does not include training/selection uncertainty, dependence across parents or future calendar shifts. Frequencies are not probabilities of future success. Archived operational comparison is a retrospective model-family comparison, not a new blind test of serving weights.",
    "comparisons": {},
}
for comparison in ("recalibrated_hourly_control", "mean_with_90_policy", "archived_operational_family"):
    rows = []
    for fold in (*FOLDS, *STRESS):
        directory = root / "access" / fold
        cp = directory / "quarter_candidate-test.parquet"
        sources.add(cp)
        candidate = pd.read_parquet(cp)
        if comparison == "recalibrated_hourly_control":
            bp = directory / "hourly_control-test.parquet"
            baseline = pd.read_parquet(bp)
        else:
            bp = base_directory("access", fold) / "access-test.parquet"
            baseline = pd.read_parquet(bp)
            if comparison == "mean_with_90_policy":
                mp = Path("artifacts/research-v12") / fold / "uncorrected.json"
                sources.add(mp)
                meta = read(mp)
                baseline["pending_alert"] = apply_policy(baseline, episodes, "mean_retarget", meta["policy"])
            baseline["alert"] = baseline.pending_alert
        sources.add(bp)
        assert cohort(candidate, episodes, 0.25) == cohort(baseline, episodes, 1)
        # Repeated matching uses chronological event identities, not row-wise
        # outcomes that would change with the number of forecast opportunities.
        for obj, pred in candidate.groupby("object_id"):
            eps = episodes[episodes.object_id.eq(obj)]
            reference = baseline[baseline.object_id.eq(obj)]
            c = EventEvaluator(pred, eps, 0.25).evaluate(pred.alert, 0.5, 0.25)
            b = EventEvaluator(reference, eps, 1).evaluate(reference.alert, 0.5, 1)
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
    counts = counts.merge(objects[["object_id", "parent_id"]], validate="one_to_one", how="left")
    assert counts.parent_id.notna().all()
    grouped = counts.groupby("parent_id")[COUNT_COLUMNS].sum().to_numpy()
    a, b, n, c, d = map(int, grouped.sum(axis=0))
    expected = completed["five_period_pooled"]
    assert (a, b, n) == tuple(expected[k] for k in ("true_alerts", "alerts", "eligible_episodes"))
    if comparison == "archived_operational_family":
        source = Path("artifacts/access_additional_validation.json")
        sources.add(source)
        expected = read(source)["five_period_pooled"]
    else:
        expected = completed[
            "five_period_control" if comparison == "recalibrated_hourly_control" else "five_period_reference"
        ]
    assert (c, d, n) == tuple(expected[k] for k in ("true_alerts", "alerts", "eligible_episodes"))
    stats = bootstrap(grouped)
    rng = np.random.default_rng(20260921)
    sampled = grouped[rng.integers(0, len(grouped), size=(5000, len(grouped)))].sum(axis=1).astype(float)
    a, b, n, c, d = sampled.T
    with np.errstate(invalid="ignore", divide="ignore"):
        p, r, old_p, old_r = a / b, a / n, c / d, c / n
        delta_goal = np.minimum(np.minimum(p, r) / 0.9, 1) - np.minimum(np.minimum(old_p, old_r) / 0.9, 1)
    stats["paired_differences_95"] = {
        "precision": interval(p - old_p),
        "recall": interval(r - old_r),
        "primary_score": interval(delta_goal),
    }
    report["comparisons"][comparison] = {"counts": counts.to_dict("records"), "by_parent": stats}
    print(comparison, stats["percentile_95"]["f1_difference"], stats["paired_differences_95"], flush=True)
report["source_hashes"] = {str(p): sha256(p) for p in sorted(sources)}
report["code_hashes"] = {
    str(p): sha256(p)
    for p in {
        Path(__file__),
        Path(cohort.__code__.co_filename),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(bootstrap.__code__.co_filename),
    }
}
write_json(Path("artifacts/fine_cadence_uncertainty.json"), report)
