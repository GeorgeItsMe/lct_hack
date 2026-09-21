"""Describe missed episodes by causally observable recurrence history.

This does NOT prove an information-theoretic predictability limit, remove hard
events, or select thresholds. It diagnoses already-used historical folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.goal90_research import STRESS, apply_policy, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS
from moscollector.waiting_time_research import KINDS, base_directory, reference_result

episode_path = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(episode_path, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
rows = []
for kind in KINDS:
    eps = episodes[episodes.kind.eq(kind)]
    for fold in (*FOLDS, *STRESS):
        pred = pd.read_parquet(base_directory(kind, fold) / f"{kind}-test.parquet")
        policy_path = (
            Path("artifacts/research-v12") / fold / "uncorrected.json"
            if kind == "access"
            else Path("artifacts/research-v13-policy") / f"{kind}-{fold}.json"
        )
        policy = read(policy_path)["policy"]
        alerts = apply_policy(pred, eps, "mean_retarget", policy)
        evaluator = EventEvaluator(pred, eps, 1)
        metrics = evaluator.evaluate(alerts, 0.5, 1)
        expected = reference_result(kind, fold)
        assert all(metrics[k] == expected[k] for k in ("true_alerts", "alerts", "eligible_episodes"))
        matched_total = episode_total = 0
        for obj, ids, times, events in evaluator.groups:
            matched = np.zeros(len(events), dtype=bool)
            next_event = 0
            for at in times[alerts[ids] >= 0.5]:
                next_event = max(next_event, int(np.searchsorted(events, at)))
                if next_event < len(events) and events[next_event] - at < 24 * HOUR_NS:
                    matched[next_event] = True
                    next_event += 1
            matched_total += int(matched.sum())
            episode_total += len(events)
            history = (
                eps.loc[eps.object_id.eq(obj), "start_ts"]
                .sort_values()
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            for event, hit in zip(events, matched, strict=True):
                at = times[np.searchsorted(times, event, side="right") - 1]
                # Exactly matches the strict70min observation delay, not the
                # actual event time that would be unavailable at forecast time.
                previous = np.searchsorted(history, at - 70 * 60 * 1_000_000_000, side="left") - 1
                if previous < 0:
                    regime = "no_observed_prior_episode"
                else:
                    age_hours = (at - history[previous]) / HOUR_NS
                    regime = (
                        "within_1d"
                        if age_hours <= 24
                        else "1_to_7d"
                        if age_hours <= 168
                        else "7_to_30d"
                        if age_hours <= 720
                        else "over_30d"
                    )
                rows.append(
                    {
                        "kind": kind,
                        "fold": fold,
                        "object_id": obj,
                        "regime": regime,
                        "episodes": 1,
                        "matched": int(hit),
                        "missed": int(not hit),
                    }
                )
        assert matched_total == metrics["true_alerts"] and episode_total == metrics["eligible_episodes"]
table = pd.DataFrame(rows)
summary = table.groupby(["kind", "regime"])[["episodes", "matched", "missed"]].sum().reset_index()
summary["recall"] = summary.matched / summary.episodes
summary["episode_share_within_kind"] = summary.episodes / summary.groupby("kind").episodes.transform("sum")
report = {
    "scope": "Descriptive recurrence audit on already-used historical folds, not a new test or a predictability bound.",
    "reference": "V9 count24 family after90/90 policy tuning: v12 uncorrected(access), v13(fire/fault). Not serving classifiers or original access policy.",
    "history": "At the latest eligible forecast at/before each episode, only starts strictly earlier than forecast minus70min can count as known history. No observed prior means none in the supplied history, not necessarily none ever.",
    "episode_source_sha256": sha256(episode_path),
    "summary": summary.to_dict("records"),
    "per_period": table.groupby(["kind", "fold", "regime"])[["episodes", "matched", "missed"]]
    .sum()
    .reset_index()
    .to_dict("records"),
    "limits": "A quiet event history does not rule out raw-sensor precursors. Every episode remains in the denominator; no relabeling or exclusion.",
}
write_json(Path("artifacts/recurrence_regime_audit.json"), report)
print(summary.to_string(index=False))
