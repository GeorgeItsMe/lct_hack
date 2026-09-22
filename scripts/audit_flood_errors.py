"""Describe every V38 test episode and warning; no new fitting or selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.flood_research import ARMS, FIVE, ROOT
from moscollector.flood_verification import verified_evidence
from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

verified_evidence(ROOT)
episode_path = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(
    episode_path, filters=[("kind", "==", "flood"), ("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
sources = {episode_path, ROOT / "report.json", ROOT / "weight-replay.json"}
cases, warnings = [], []
for fold in FIVE:
    result_path = ROOT / fold / "result.json"
    result = read(result_path)
    sources.add(result_path)
    for arm in ARMS:
        pred_path = ROOT / fold / f"{arm}-test.parquet"
        sources.add(pred_path)
        pred = pd.read_parquet(pred_path)
        evaluator = EventEvaluator(pred, episodes, 1)
        expected = result["arms"][arm]["scores"]["test"]
        assert evaluator.evaluate(pred.alert, 0.5, 1) == expected
        period_cases, period_warnings = [], []
        for obj, ids, times, events in evaluator.groups:
            next_event, hits = 0, {}
            for at in times[pred.alert.to_numpy()[ids] >= 0.5]:
                next_event = max(next_event, int(np.searchsorted(events, at)))
                hit = next_event < len(events) and events[next_event] - at < 24 * HOUR_NS
                row = {"fold": fold, "arm": arm, "object_id": obj, "as_of": str(pd.Timestamp(at))}
                if hit:
                    lead = float((events[next_event] - at) / HOUR_NS)
                    hits[int(events[next_event])] = lead
                    row.update({"reason": "matched", "lead_hours": lead})
                    next_event += 1
                else:
                    future = np.searchsorted(events, at + 24 * HOUR_NS) - np.searchsorted(events, at)
                    row.update(
                        {
                            "reason": "already_matched_episode" if future else "no_future_episode",
                            "lead_hours": None,
                        }
                    )
                period_warnings.append(row)
            history = episodes.loc[episodes.object_id.eq(obj)].sort_values("start_ts")
            all_times = history.start_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            for event in events:
                index = int(np.searchsorted(all_times, event))
                assert index < len(history) and all_times[index] == event
                prior_gap = float((event - all_times[index - 1]) / HOUR_NS) if index else None
                regime = (
                    "within_24h"
                    if prior_gap is not None and prior_gap <= 24
                    else "within_7d"
                    if prior_gap is not None and prior_gap <= 168
                    else "after_7d_or_first"
                )
                period_cases.append(
                    {
                        "fold": fold,
                        "arm": arm,
                        "object_id": obj,
                        "episode_id": str(history.iloc[index].episode_id),
                        "start_ts": str(pd.Timestamp(event)),
                        "matched": int(event) in hits,
                        "lead_hours": hits.get(int(event)),
                        "previous_onset_gap_hours": prior_gap,
                        "recurrence": regime,
                        "duration_seconds": float(history.iloc[index].duration_seconds),
                    }
                )
        assert len(period_cases) == expected["eligible_episodes"]
        assert sum(c["matched"] for c in period_cases) == expected["true_alerts"]
        assert len(period_warnings) == expected["alerts"]
        cases.extend(period_cases)
        warnings.extend(period_warnings)
table = pd.DataFrame(cases)
alerts = pd.DataFrame(warnings)
assert not table.duplicated(["arm", "fold", "episode_id"]).any()
assert table.groupby("arm").size().to_dict() == dict.fromkeys(ARMS, 40)
summary = {}
for arm in ARMS:
    sub = table.loc[table.arm.eq(arm)]
    issued = alerts.loc[alerts.arm.eq(arm)]
    summary[arm] = {
        "warnings_by_reason": issued.reason.value_counts().to_dict(),
        "matched_lead_at_least_1h": int((issued.reason.eq("matched") & issued.lead_hours.ge(1)).sum()),
        "by_recurrence": sub.groupby("recurrence")
        .agg(episodes=("matched", "size"), matched=("matched", "sum"))
        .reset_index()
        .to_dict("records"),
        "by_object": sub.groupby("object_id")
        .agg(episodes=("matched", "size"), matched=("matched", "sum"))
        .reset_index()
        .to_dict("records"),
    }
audit = {
    "scope": "Descriptive post-result audit of ALL40original test episodes and both fixed arms. No model/rule selection, event removal, June reads, physical flood validation or prediction upper-bound claim.",
    "recurrence_note": "Gap between original episode start times. It is a descriptive grouping, not a causal feature or claim that the preceding episode was already confirmed at warning time.",
    "cases": cases,
    "warnings": warnings,
    "summary": summary,
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p) for p in (Path(__file__), Path(EventEvaluator.__init__.__code__.co_filename))
    },
}
write_json(Path("artifacts/flood_count_error_audit.json"), audit)
print(summary)
