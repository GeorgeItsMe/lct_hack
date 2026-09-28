"""Describe v35 warning errors and lead times without changing its frozen choices."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.goal90_research import read
from moscollector.experiments.onset_binary_research import metric
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v35")
report = read(root / "report.json")
source = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(source, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
sources = {source, root / "plan.json", root / "report.json"}
rows, paired = [], []
expected_periods = {
    (kind, fold)
    for kind, outcome in report.items()
    for fold in (
        ["screen_1", "screen_2"]
        + (["confirmation", "stress_1", "stress_2"] if outcome["selection"]["passed_screen"] else [])
    )
}
paths = sorted(root.glob("*/*/result.json"))
if {(read(p)["kind"], read(p)["fold"]) for p in paths} != expected_periods:
    raise ValueError("Binary error audit requires all declared periods")
for result_path in paths:
    result = read(result_path)
    sources.add(result_path)
    eps = episodes.loc[episodes.kind.eq(result["kind"])]
    matches = {}
    for name in ("binary_candidate", "count_direct", "old_pending"):
        path = (
            Path(result["pending_control"]) / "minute_candidate-test.parquet"
            if name == "old_pending"
            else result_path.parent / f"{name}-test.parquet"
        )
        sources.add(path)
        pred = pd.read_parquet(path)
        expected = metric(result, name)
        evaluator = evaluator_for(pred, eps, 1 / 60, result["exposure"]["test"])
        assert evaluator.evaluate(pred.alert, 0.5, expected["cooldown_hours"]) == expected
        empty, redundant, found = 0, 0, {}
        alerts = pred.alert.to_numpy()
        for obj, ids, times, events in evaluator.groups:
            next_event = 0
            for at in times[alerts[ids] >= 0.5]:
                next_event = max(next_event, int(np.searchsorted(events, at)))
                if next_event < len(events) and events[next_event] - at < 24 * HOUR_NS:
                    found[(str(obj), int(events[next_event]))] = float((events[next_event] - at) / HOUR_NS)
                    next_event += 1
                else:
                    future = np.searchsorted(events, at + 24 * HOUR_NS) - np.searchsorted(events, at)
                    empty += int(future == 0)
                    redundant += int(future > 0)
        assert len(found) == expected["true_alerts"]
        assert empty + redundant == expected["false_alerts"]
        leads = np.asarray(list(found.values()))
        row = {
            "kind": result["kind"],
            "fold": result["fold"],
            "arm": name,
            "true_alerts": len(found),
            "false_empty": empty,
            "false_redundant": redundant,
            "eligible_episodes": expected["eligible_episodes"],
            "missed_episodes": expected["missed_episodes"],
            "lead_under_1min": int(np.sum(leads < 1 / 60)),
            "lead_1_to_15min": int(np.sum((leads >= 1 / 60) & (leads < 0.25))),
            "lead_15_to_60min": int(np.sum((leads >= 0.25) & (leads < 1))),
            "lead_at_least_1h": int(np.sum(leads >= 1)),
        }
        assert sum(v for k, v in row.items() if k.startswith("lead_")) == len(found)
        rows.append(row)
        matches[name] = found
    candidate = matches["binary_candidate"]
    for name in ("count_direct", "old_pending"):
        reference = matches[name]
        common = sorted(candidate.keys() & reference.keys())
        paired.append(
            {
                "kind": result["kind"],
                "fold": result["fold"],
                "reference": name,
                "both_find": len(common),
                "candidate_only": len(candidate.keys() - reference.keys()),
                "reference_only": len(reference.keys() - candidate.keys()),
                "candidate_earlier": sum(candidate[k] > reference[k] for k in common),
                "reference_earlier": sum(candidate[k] < reference[k] for k in common),
                "same_warning_time": sum(candidate[k] == reference[k] for k in common),
                "median_paired_lead_difference_hours": (
                    float(np.median([candidate[k] - reference[k] for k in common])) if common else None
                ),
            }
        )
totals = (
    pd.DataFrame(rows)
    .drop(columns="fold")
    .groupby(["kind", "arm"], as_index=False)
    .sum(numeric_only=True)
)
audit = {
    "scope": "Descriptive completed historical audit, not independent validation or policy selection. All original episode denominators and false warnings remain. Empty means no eligible episode within24h; redundant means future episodes already matched. Lead bands do not redefine the target. Paired lead comparisons use only identical events found by both arms, and report every newly found and lost event separately.",
    "periods": rows,
    "totals": totals.to_dict("records"),
    "paired_lead_comparisons": paired,
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(EventEvaluator.__init__.__code__.co_filename),
            Path(evaluator_for.__code__.co_filename),
            Path(metric.__code__.co_filename),
        )
    },
}
write_json(root / "error-audit.json", audit)
print(totals.to_string(index=False), flush=True)
