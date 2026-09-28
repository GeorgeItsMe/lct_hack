"""Describe completed v33 warnings with unchanged full event denominators."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.goal90_research import read
from moscollector.experiments.onset_count_research import ARMS
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v33")
source = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(source, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
sources = {source, root / "plan.json"}
rows = []
for result_path in sorted(root.glob("*/*/result.json")):
    result = read(result_path)
    sources.add(result_path)
    eps = episodes.loc[episodes.kind.eq(result["kind"])]
    for name in ARMS:
        path = result_path.parent / f"{name}-test.parquet"
        sources.add(path)
        pred = pd.read_parquet(path)
        cadence = result["arms"][name]["cadence_hours"]
        expected = result["arms"][name]["scores"]
        evaluator = evaluator_for(pred, eps, cadence, result["original_hourly_exposure"]["test"])
        assert evaluator.evaluate(pred.alert, 0.5, cadence) == expected
        hits, empty, redundant, leads = 0, 0, 0, []
        for _, ids, times, events in evaluator.groups:
            next_event = 0
            for at in times[pred.alert.to_numpy()[ids] >= 0.5]:
                next_event = max(next_event, int(np.searchsorted(events, at)))
                if next_event < len(events) and events[next_event] - at < 24 * HOUR_NS:
                    hits += 1
                    leads.append((events[next_event] - at) / HOUR_NS)
                    next_event += 1
                else:
                    future = np.searchsorted(events, at + 24 * HOUR_NS) - np.searchsorted(events, at)
                    empty += int(future == 0)
                    redundant += int(future > 0)
        assert hits == expected["true_alerts"]
        assert empty + redundant == expected["false_alerts"]
        leads = np.asarray(leads)
        row = {
            "kind": result["kind"],
            "fold": result["fold"],
            "arm": name,
            "true_alerts": hits,
            "false_empty": empty,
            "false_redundant": redundant,
            "eligible_episodes": expected["eligible_episodes"],
            "missed_episodes": expected["missed_episodes"],
            "lead_under_1min": int(np.sum(leads < 1 / 60)),
            "lead_1_to_15min": int(np.sum((leads >= 1 / 60) & (leads < 0.25))),
            "lead_15_to_60min": int(np.sum((leads >= 0.25) & (leads < 1))),
            "lead_at_least_1h": int(np.sum(leads >= 1)),
            "probability_gate_disabled": result["arms"][name]["policy"]["floor"] == 0,
        }
        assert (
            sum(
                row[k] for k in ("lead_under_1min", "lead_1_to_15min", "lead_15_to_60min", "lead_at_least_1h")
            )
            == hits
        )
        rows.append(row)
if not rows:
    raise ValueError("No completed onset periods")
complete = (root / "report.json").exists()
if complete:
    report = read(root / "report.json")
    expected_folds = {
        (kind, fold)
        for kind, outcome in report.items()
        for fold in (
            ["screen_1", "screen_2"]
            + (["confirmation", "stress_1", "stress_2"] if outcome["selection"]["passed_screen"] else [])
        )
    }
    assert {(row["kind"], row["fold"]) for row in rows} == expected_folds
    sources.add(root / "report.json")
table = pd.DataFrame(rows)
totals = (
    table.drop(columns=["fold", "probability_gate_disabled"])
    .groupby(["kind", "arm"], as_index=False)
    .sum(numeric_only=True)
)
audit = {
    "scope": "Descriptive audit of completed historical predictions; no policy selection, event exclusion or claim of independent validation. Empty false warning has no future eligible episode in its24h window; redundant warning has future episodes already matched. Both remain false warnings. Lead bands are descriptive and do not redefine the24h target.",
    "completed_study": complete,
    "periods": rows,
    "totals": totals.to_dict("records"),
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(EventEvaluator.__init__.__code__.co_filename),
            Path(evaluator_for.__code__.co_filename),
        )
    },
}
write_json(root / "error-audit.json", audit)
print(totals.to_string(index=False), flush=True)
print("completed_study", complete, flush=True)
