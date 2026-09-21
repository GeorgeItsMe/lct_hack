"""Describe completed v31 fault misses; no policy selection or relabeling."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.fine_cadence_research import evaluator_for
from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v31")
outcome = read(root / "report.json")["fault"]
if not outcome["selection"]["passed_screen"] or len(outcome["periods"]) != 5:
    raise ValueError("Complete the declared fault confirmation before this descriptive audit")
episode_path = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(episode_path, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
episodes = episodes.loc[episodes.kind.eq("fault")]
sources = {episode_path, root / "plan.json", root / "report.json"}
rows = []
variants = ("pretrained", "global_control", "scratch_gru")
for fold in ("screen_1", "screen_2", "confirmation", "stress_1", "stress_2"):
    directory = root / "fault" / fold
    result = read(directory / "result.json")
    sources.add(directory / "result.json")
    source = Path(result["source_control_directory"])
    found, all_events, candidate_groups, shared = {}, None, None, None
    for variant in variants:
        path = (
            directory / "pretrained-test.parquet"
            if variant == "pretrained"
            else source / ("gru-test.parquet" if variant == "scratch_gru" else "global_control-test.parquet")
        )
        sources.add(path)
        pred = pd.read_parquet(path)
        keys = pred[["object_id", "as_of"]]
        if shared is None:
            shared = keys
        else:
            pd.testing.assert_frame_equal(keys, shared)
        evaluator = evaluator_for(pred, episodes, 0.25, result["exposure_days"])
        expected = result["scores"] if variant == "pretrained" else result["controls"][variant]
        assert evaluator.evaluate(pred.alert, 0.5, 0.25) == expected
        hits, cohort = set(), set()
        for obj, ids, times, events in evaluator.groups:
            assert len(times) < 2 or np.all(np.diff(times) >= HOUR_NS // 4)
            cohort.update((obj, int(at)) for at in events)
            next_event = 0
            for at in times[pred.alert.to_numpy()[ids] >= 0.5]:
                next_event = max(next_event, int(np.searchsorted(events, at)))
                if next_event < len(events) and events[next_event] - at < 24 * HOUR_NS:
                    hits.add((obj, int(events[next_event])))
                    next_event += 1
        assert len(hits) == expected["true_alerts"]
        assert len(cohort) == evaluator.events == expected["eligible_episodes"]
        if all_events is None:
            all_events, candidate_groups = cohort, evaluator.groups
        else:
            assert all_events == cohort
        found[variant] = hits
    for obj, _, times, events in candidate_groups:
        history = (
            episodes.loc[episodes.object_id.eq(obj), "start_ts"]
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        available = ((history + 70 * 60 * 1_000_000_000) // HOUR_NS + 1) * HOUR_NS
        for event in events:
            at = times[np.searchsorted(times, event, side="right") - 1]
            previous = np.searchsorted(available, at, side="right") - 1
            if previous < 0:
                regime = "no_observed_prior_episode"
            else:
                age = (at - history[previous]) / HOUR_NS
                regime = (
                    "within_1d"
                    if age <= 24
                    else "1_to_7d"
                    if age <= 168
                    else ("7_to_30d" if age <= 720 else "over_30d")
                )
            rows.append(
                {
                    "fold": fold,
                    "object_id": obj,
                    "start_ts": pd.Timestamp(event).isoformat(),
                    "regime": regime,
                    "episodes": 1,
                    **{variant: int((obj, int(event)) in found[variant]) for variant in variants},
                }
            )
table = pd.DataFrame(rows)
columns = ["episodes", *variants]
summary = table.groupby("regime")[columns].sum().reset_index()
totals = {name: int(value) for name, value in table[columns].sum().items()}
for variant in variants:
    assert totals[variant] == outcome["five_period_pooled"][variant]["true_alerts"]
assert totals["episodes"] == outcome["five_period_pooled"]["pretrained"]["eligible_episodes"]
report = {
    "scope": "Descriptive audit of already-used historical outcomes, not a new test, trained ensemble, or predictability bound. No thresholds selected, events removed, or labels changed.",
    "regime": "At the latest eligible forecast at/before each episode, use only earlier episodes whose confirmation is available at floor(start+70min to hour)+1h. Regime is descriptive at this snapshot, not necessarily at the matched earlier warning. Quiet history does not rule out raw-sensor precursors.",
    "summary": summary.to_dict("records"),
    "totals": totals,
    "per_period": table.groupby(["fold", "regime"])[columns].sum().reset_index().to_dict("records"),
    "cases": rows,
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
write_json(Path("artifacts/self_supervised_fault_error_audit.json"), report)
print(summary.to_string(index=False))
