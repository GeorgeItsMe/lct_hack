"""Describe overlap of episodes found by frozen v28 models; never tune a policy."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

episode_path = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(episode_path, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
sources = {episode_path, Path("artifacts/research-v28/report.json")}
periods = []
for kind in ("access", "fire", "fault"):
    for fold in ("screen_1", "screen_2"):
        directory = Path("artifacts/research-v28") / kind / fold
        result = read(directory / "result.json")
        sources.add(directory / "result.json")
        matched, cohorts = {}, {}
        for variant in ("global_control", "mlp", "gru"):
            path = directory / f"{variant}-test.parquet"
            sources.add(path)
            pred = pd.read_parquet(path)
            evaluator = EventEvaluator(pred, episodes.loc[episodes.kind.eq(kind)], 0.25)
            found, cohort = set(), set()
            for obj, ids, times, events in evaluator.groups:
                cohort.update((obj, int(at)) for at in events)
                next_event = 0
                for at in times[pred.alert.to_numpy()[ids] >= 0.5]:
                    next_event = max(next_event, int(np.searchsorted(events, at, side="left")))
                    if next_event < len(events) and events[next_event] - at < 24 * HOUR_NS:
                        found.add((obj, int(events[next_event])))
                        next_event += 1
            expected = result["arms"][variant]["scores"]
            assert len(cohort) == evaluator.events == expected["eligible_episodes"]
            assert len(found) == expected["true_alerts"]
            assert int(pred.alert.sum()) == expected["alerts"]
            matched[variant], cohorts[variant] = found, cohort
        assert cohorts["global_control"] == cohorts["mlp"] == cohorts["gru"]
        control, cohort = matched["global_control"], cohorts["global_control"]
        row = {"kind": kind, "fold": fold, "eligible_episodes": len(cohort), "pairs": {}}
        for variant in ("mlp", "gru"):
            neural = matched[variant]
            row["pairs"][variant] = {
                "both": len(control & neural),
                "control_only": len(control - neural),
                "neural_only": len(neural - control),
                "neither": len(cohort - control - neural),
                "union": len(control | neural),
            }
            assert sum(
                row["pairs"][variant][k] for k in ("both", "control_only", "neural_only", "neither")
            ) == len(cohort)
        row["all_three_union"] = len(set.union(*matched.values()))
        periods.append(row)
totals = {}
for kind in ("access", "fire", "fault"):
    rows = [r for r in periods if r["kind"] == kind]
    events = sum(r["eligible_episodes"] for r in rows)
    totals[kind] = {
        "eligible_episodes": events,
        "pairs": {
            variant: {key: sum(r["pairs"][variant][key] for r in rows) for key in rows[0]["pairs"][variant]}
            for variant in ("mlp", "gru")
        },
        "all_three_union": sum(r["all_three_union"] for r in rows),
        "all_three_union_fraction": sum(r["all_three_union"] for r in rows) / events,
    }
report = {
    "scope": "Descriptive overlap of already-used Nov/Feb event outcomes. No weights or thresholds selected here.",
    "interpretation": "Union counts distinct eligible episode identities independently matched by the frozen branches. This uses future outcomes, is not a causal ensemble's performance, does not remove its false warnings and is not an upper bound for new forecasts or policies.",
    "periods": periods,
    "totals": totals,
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p) for p in (Path(__file__), Path(EventEvaluator.__init__.__code__.co_filename))
    },
}
write_json(Path("artifacts/neural_complementarity_audit.json"), report)
print(totals)
