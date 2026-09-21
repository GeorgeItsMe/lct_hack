"""Describe preceding fire states of fault channels, without prediction claims.

The channel is selected with hindsight from the observed fault. This is a case
description, not a usable warning rule, a controlled association or a recall
ceiling. No event is removed from the evaluation cohort.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.goal90_research import STRESS
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS
from moscollector.waiting_time_research import base_directory

paths = [PROCESSED / f"episodes-{year}.parquet" for year in (2025, 2026)]
channel = pd.concat(
    [pd.read_parquet(p, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]) for p in paths],
    ignore_index=True,
)
faults = channel[channel.kind.eq("fault")]
fires = channel[channel.kind.eq("fire")]
preceding = fires.set_index(["channel_id", "end_ts"]).start_ts.to_dict()
episode_path = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(
    episode_path, filters=[("kind", "==", "fault"), ("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
rows = []
group_members = episodes.set_index(["object_id", "start_ts"]).channel_ids
for fold in (*FOLDS, *STRESS):
    pred = pd.read_parquet(base_directory("fault", fold) / "fault-test.parquet")
    evaluator = EventEvaluator(pred, episodes, 1)
    for obj, _, times, events in evaluator.groups:
        history = (
            episodes.loc[episodes.object_id.eq(obj), "start_ts"]
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        local = faults[faults.object_id.eq(obj)]
        for event in events:
            at = times[np.searchsorted(times, event, side="right") - 1]
            previous = np.searchsorted(history, at - 70 * 60 * 1_000_000_000, side="left") - 1
            quiet = previous < 0 or at - history[previous] > 168 * HOUR_NS
            start = pd.Timestamp(event)
            members = local[local.start_ts.ge(start) & local.start_ts.le(start + pd.Timedelta(minutes=10))]
            if members.empty:
                raise ValueError("Eligible grouped fault has no source channel episode")
            if set(members.channel_id) != set(group_members.loc[(obj, start)]):
                raise ValueError("Source channel membership differs from grouped fault label")
            leads, visible, smoke_previous, missing = [], False, False, 0
            for member in members.itertuples():
                if member.previous_code != 2:
                    continue
                smoke_previous = True
                fire_start = preceding.get((member.channel_id, member.start_ts))
                if fire_start is None:
                    missing += 1
                    continue
                leads.append((start - fire_start).total_seconds() / 3600)
                visible |= fire_start.value < at < member.start_ts.value
            rows.append(
                {
                    "fold": fold,
                    "object_id": obj,
                    "start_ts": start,
                    "quiet_over_week_or_no_history": bool(quiet),
                    "members": len(members),
                    "member_preceded_by_fire_state": smoke_previous,
                    "preceding_fire_visible_at_latest_forecast": bool(visible),
                    "missing_preceding_fire_channel_episodes": missing,
                    "max_precursor_lead_h": max(leads) if leads else None,
                }
            )
    assert sum(row["fold"] == fold for row in rows) == evaluator.events
frame = pd.DataFrame(rows)
assert len(frame) == 217
summary = []
for quiet, group in frame.groupby("quiet_over_week_or_no_history"):
    summary.append(
        {
            "quiet_over_week_or_no_history": bool(quiet),
            "episodes": len(group),
            "preceded_by_fire_state": int(group.member_preceded_by_fire_state.sum()),
            "fire_visible_at_latest_forecast": int(group.preceding_fire_visible_at_latest_forecast.sum()),
            "missing_source_members": int(group.missing_preceding_fire_channel_episodes.sum()),
            "median_lead_h_when_found": float(group.max_precursor_lead_h.median()),
        }
    )
report = {
    "scope": "Descriptive case-only audit on already-used five months; faulty channels selected with hindsight. No controls, no warning policy or quality estimate; not a bound on predictability. No relabeling.",
    "precursor": "A directly preceding positive fire-state channel episode ending at the fault onset. Visible means it had started strictly before the latest eligible forecast. Missing source fire episode is not proof of no precursor.",
    "inputs": {str(p): sha256(p) for p in (*paths, episode_path)},
    "before": "2026-06-01",
    "all_217_source_memberships_verified": True,
    "summary": summary,
    "cases": rows,
}
write_json(Path("artifacts/fault_precursor_audit.json"), report)
print(summary)
