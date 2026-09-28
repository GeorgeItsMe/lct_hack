"""Diagnostic upper capacity of the current hourly opportunity grid."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.cadence_research import assert_same_episode_cohort
from moscollector.experiments.cadence_capacity import matched_slots, subdivide_evaluation_slots
from moscollector.experiments.goal90_research import STRESS
from moscollector.experiments.waiting_time_research import KINDS, base_directory
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

source = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(source, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
inputs = {str(source): sha256(source)}
rows = []
for kind in KINDS:
    eps = episodes[episodes.kind.eq(kind)]
    for fold in (*FOLDS, *STRESS):
        path = base_directory(kind, fold) / f"{kind}-test.parquet"
        inputs[str(path)] = sha256(path)
        pred = pd.read_parquet(path)
        evaluator = EventEvaluator(pred, eps, 1)
        alerts = np.zeros(len(pred), dtype=np.int8)
        capacity = 0
        for _, ids, times, events in evaluator.groups:
            assert np.all(np.diff(times) >= HOUR_NS)
            selected = matched_slots(times, events)
            capacity += int(selected.sum())
            alerts[ids] = selected
        replay = evaluator.evaluate(alerts, 0.5, 1)
        assert replay["true_alerts"] == replay["alerts"] == capacity
        rows.append(
            {
                "kind": kind,
                "fold": fold,
                "matching_capacity": capacity,
                "eligible_episodes": evaluator.events,
                "capacity_fraction": capacity / evaluator.events if evaluator.events else None,
            }
        )
table = pd.DataFrame(rows)
pooled = table.groupby("kind")[["matching_capacity", "eligible_episodes"]].sum().reset_index()
pooled["capacity_fraction"] = pooled.matching_capacity / pooled.eligible_episodes
report = {
    "scope": "Exact combinatorial capacity of fixed hourly slots using full future outcomes. NOT model performance, a deployed schedule, or a bound on models with a different cadence. No forecast precision claim.",
    "method": "Maximum bipartite matching: one side original eligible events, other side original hourly opportunities; edge if0<=event-t<24h. Slots separated by>=1h. Replay selected slots through original evaluator and assert cardinality agreement.",
    "inputs": inputs,
    "periods": rows,
    "pooled": pooled.to_dict("records"),
    "implementation_test": "Matches exhaustive assignment search on40 small random graphs; explicit boundary tests.",
}
write_json(Path("artifacts/hourly_capacity_audit.json"), report)
print(pooled.to_string(index=False))
quarter_rows = []
for kind in KINDS:
    eps = episodes[episodes.kind.eq(kind)]
    for fold in (*FOLDS, *STRESS):
        original = pd.read_parquet(base_directory(kind, fold) / f"{kind}-test.parquet")
        pred = subdivide_evaluation_slots(original, 15)
        assert_same_episode_cohort(pred, original, eps)
        evaluator = EventEvaluator(pred, eps, 0.25)
        alerts = np.zeros(len(pred), dtype=np.int8)
        capacity = 0
        for _, ids, times, events in evaluator.groups:
            assert np.all(np.diff(times) >= HOUR_NS // 4)
            selected = matched_slots(times, events)
            alerts[ids] = selected
            capacity += int(selected.sum())
        replay = evaluator.evaluate(alerts, 0.5, 0.25)
        assert replay["true_alerts"] == replay["alerts"] == capacity
        quarter_rows.append(
            {
                "kind": kind,
                "fold": fold,
                "matching_capacity": capacity,
                "eligible_episodes": evaluator.events,
                "capacity_fraction": capacity / evaluator.events if evaluator.events else None,
            }
        )
quarter = pd.DataFrame(quarter_rows)
quarter_pooled = quarter.groupby("kind")[["matching_capacity", "eligible_episodes"]].sum().reset_index()
quarter_pooled["capacity_fraction"] = quarter_pooled.matching_capacity / quarter_pooled.eligible_episodes
report["quarter_hour"] = {
    "scope": "Same cohort, offline densification between adjacent eligible hourly slots only. No new model or policy quality is implied.",
    "periods": quarter_rows,
    "pooled": quarter_pooled.to_dict("records"),
}
report["code_hashes"] = {
    str(p): sha256(p)
    for p in (
        Path(__file__),
        Path(matched_slots.__code__.co_filename),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(assert_same_episode_cohort.__code__.co_filename),
    )
}
write_json(Path("artifacts/hourly_capacity_audit.json"), report)
print("15min capacity", quarter_pooled.to_string(index=False))
