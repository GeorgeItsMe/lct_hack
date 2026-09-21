"""Describe remaining errors of frozen v20; no test-label policy selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.goal90_research import STRESS, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

episode_path = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(
    episode_path, filters=[("kind", "==", "access"), ("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
sources = {episode_path}
rows, objects = [], []
for fold in (*FOLDS, *STRESS):
    directory = Path("artifacts/research-v20/access") / fold
    path = directory / "quarter_candidate-test.parquet"
    sources.update((path, directory / "result.json"))
    pred = pd.read_parquet(path)
    expected = read(directory / "result.json")["scores"]
    evaluator = EventEvaluator(pred, episodes, 0.25)
    leads = []
    fold_rows = []
    for obj, ids, times, events in evaluator.groups:
        next_event = true = empty = redundant = 0
        matched = np.zeros(len(events), dtype=bool)
        for at in times[pred.alert.to_numpy()[ids] >= 0.5]:
            next_event = max(next_event, int(np.searchsorted(events, at, side="left")))
            if next_event < len(events) and events[next_event] - at < 24 * HOUR_NS:
                leads.append((events[next_event] - at) / HOUR_NS)
                matched[next_event] = True
                true += 1
                next_event += 1
            else:
                future = np.searchsorted(events, at + 24 * HOUR_NS, side="left") - np.searchsorted(
                    events, at, side="left"
                )
                empty += int(future == 0)
                redundant += int(future > 0)
        item = {
            "fold": fold,
            "object_id": obj,
            "true_alerts": true,
            "empty_false_alerts": empty,
            "redundant_false_alerts": redundant,
            "missed_episodes": int((~matched).sum()),
            "eligible_episodes": len(events),
        }
        objects.append(item)
        fold_rows.append(item)
    sums = {
        k: sum(item[k] for item in fold_rows)
        for k in (
            "true_alerts",
            "empty_false_alerts",
            "redundant_false_alerts",
            "missed_episodes",
            "eligible_episodes",
        )
    }
    assert sums["true_alerts"] == expected["true_alerts"]
    assert sums["empty_false_alerts"] + sums["redundant_false_alerts"] == expected["false_alerts"]
    assert sums["missed_episodes"] == expected["missed_episodes"]
    rows.append(
        {
            "fold": fold,
            **sums,
            "true_alerts_lead_under_1h": int(np.sum(np.asarray(leads) < 1)),
            "median_lead_hours": float(np.median(leads)),
        }
    )
totals = {
    key: sum(row[key] for row in rows)
    for key in (
        "true_alerts",
        "empty_false_alerts",
        "redundant_false_alerts",
        "missed_episodes",
        "eligible_episodes",
        "true_alerts_lead_under_1h",
    )
}
by_object = pd.DataFrame(objects).groupby("object_id").sum(numeric_only=True).reset_index()
report = {
    "scope": "Descriptive analysis of already-used v20 access folds, not new validation or an implementable policy.",
    "interpretation": "Empty: no future eligible episode in the unchanged24h window. Redundant: future episodes exist but have already been matched to earlier warnings. All false warnings and missed episodes remain in the reported score.",
    "perfect_removal_of_only_redundant_false_alerts_not_model": {
        "precision": totals["true_alerts"] / (totals["true_alerts"] + totals["empty_false_alerts"]),
        "recall_unchanged": totals["true_alerts"] / totals["eligible_episodes"],
        "uses_future_outcomes": True,
    },
    "totals": totals,
    "periods": rows,
    "top_missed_objects": by_object.nlargest(10, "missed_episodes").to_dict("records"),
    "all_objects": by_object.to_dict("records"),
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p) for p in (Path(__file__), Path(EventEvaluator.__init__.__code__.co_filename))
    },
}
write_json(Path("artifacts/fine_cadence_error_audit.json"), report)
print(totals)
print(report["perfect_removal_of_only_redundant_false_alerts_not_model"])
