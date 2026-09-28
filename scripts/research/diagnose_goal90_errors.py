"""Describe existing access errors; never select thresholds from test labels."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.experiments.goal90_research import STRESS, base_directory, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

episode_path = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(
    episode_path,
    filters=[("kind", "==", "access"), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
)
rows, objects = [], []
for fold in (*FOLDS, *STRESS):
    source = base_directory(fold) / "access-test.parquet"
    pred = pd.read_parquet(source)
    evaluator = EventEvaluator(pred, episodes, 1)
    false_empty = false_redundant = tp = 0
    missed = []
    for obj, ids, times, events in evaluator.groups:
        next_event = true = empty = redundant = 0
        matched = np.zeros(len(events), dtype=bool)
        for t in times[pred.pending_alert.to_numpy()[ids] >= 0.5]:
            next_event = max(next_event, int(np.searchsorted(events, t)))
            if next_event < len(events) and events[next_event] - t < 24 * HOUR_NS:
                matched[next_event] = True
                next_event += 1
                true += 1
            else:
                count = np.searchsorted(events, t + 24 * HOUR_NS) - np.searchsorted(events, t)
                empty += count == 0
                redundant += count > 0
        tp += true
        false_empty += empty
        false_redundant += redundant
        objects.append(
            {
                "fold": fold,
                "object_id": obj,
                "true_alerts": true,
                "empty_false_alerts": int(empty),
                "redundant_false_alerts": int(redundant),
                "missed_episodes": int((~matched).sum()),
                "episodes": len(events),
            }
        )
        for j in np.flatnonzero(~matched):
            missed.append({"object_id": obj, "at": int(events[j])})
    expected = read(base_directory(fold) / "access.json")["scores"]["pending"]
    assert tp == expected["true_alerts"]
    assert false_empty + false_redundant == expected["false_alerts"]
    assert len(missed) == expected["missed_episodes"]
    rows.append(
        {
            "fold": fold,
            "true_alerts": tp,
            "false_alerts_no_future_episode": int(false_empty),
            "false_alerts_future_episodes_already_matched": int(false_redundant),
            "missed_episodes": len(missed),
            "source_sha256": sha256(source),
        }
    )
by_object = pd.DataFrame(objects).groupby("object_id").sum(numeric_only=True).reset_index()
by_object["false_alerts"] = by_object.empty_false_alerts + by_object.redundant_false_alerts
result = {
    "scope": "Descriptive error analysis of already-used five retrospective months; not model selection or a new test.",
    "kind": "access",
    "episode_source_sha256": sha256(episode_path),
    "periods": rows,
    "totals": {
        k: sum(r[k] for r in rows)
        for k in (
            "true_alerts",
            "false_alerts_no_future_episode",
            "false_alerts_future_episodes_already_matched",
            "missed_episodes",
        )
    },
    "top_missed_objects": by_object.nlargest(10, "missed_episodes").to_dict("records"),
    "top_false_alert_objects": by_object.nlargest(10, "false_alerts").to_dict("records"),
    "all_objects": by_object.to_dict("records"),
    "interpretation": "A redundant warning may have real future episodes, but all were already assigned to earlier warnings by the unchanged one-to-one evaluator. No error category is excluded from reported metrics.",
}
target = Path("artifacts/goal90_error_diagnostics.json")
write_json(target, result)
print(result["totals"])
print("top missed", result["top_missed_objects"][:3])
