"""Event-level warning metrics shared by training, research and the threshold preview.

Kept free of scikit-learn so the serving image can evaluate a proposed threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def alert_metrics(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    threshold: float,
    cooldown_hours: int = 24,
    with_matches: bool = False,
):
    """One alert can match one new event; one event can match one alert.

    Evaluate only episodes for which at least one eligible forecast opportunity
    exists. Report the excluded population separately instead of calling it safe.
    """
    matches = []
    total_events = 0
    alert_count = 0
    tp = 0
    opportunity_days = len(predictions) * 3 / 24
    for obj, group in predictions.groupby("object_id"):
        group = group.sort_values("as_of")
        opportunities = group.as_of.to_numpy(dtype="datetime64[ns]")
        ep = episodes[episodes.object_id.eq(int(obj))].sort_values("start_ts")
        event_times = ep.start_ts.to_numpy(dtype="datetime64[ns]")
        keep = []
        for i, event_time in enumerate(event_times):
            ix = np.searchsorted(opportunities, event_time, side="right") - 1
            if ix >= 0 and event_time - opportunities[ix] < np.timedelta64(24, "h"):
                keep.append(i)
        ep = ep.iloc[keep]
        event_times = ep.start_ts.to_numpy(dtype="datetime64[ns]")
        total_events += len(ep)
        used = set()
        previous = None
        for row in group[group.probability.ge(threshold)].itertuples():
            if previous is not None and (row.as_of - previous).total_seconds() < cooldown_hours * 3600:
                continue
            previous = row.as_of
            alert_count += 1
            t = np.datetime64(row.as_of.to_datetime64())
            idx = int(np.searchsorted(event_times, t))
            while idx < len(event_times) and idx in used:
                idx += 1
            hit = idx < len(event_times) and event_times[idx] - t < np.timedelta64(24, "h")
            lead = None
            episode_id = None
            if hit:
                used.add(idx)
                tp += 1
                lead = float((event_times[idx] - t) / np.timedelta64(1, "h"))
                episode_id = str(ep.iloc[idx].episode_id)
            matches.append(
                {
                    "object_id": int(obj),
                    "as_of": row.as_of,
                    "probability": float(row.probability),
                    "hit": bool(hit),
                    "episode_id": episode_id,
                    "lead_hours": lead,
                }
            )
    precision = tp / alert_count if alert_count else 0.0
    recall = tp / total_events if total_events else 0.0
    leads = [r["lead_hours"] for r in matches if r["hit"]]
    result = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_alerts": tp,
        "false_alerts": alert_count - tp,
        "alerts": alert_count,
        "eligible_episodes": total_events,
        "missed_episodes": total_events - tp,
        "false_alerts_per_object_day": (alert_count - tp) / opportunity_days if opportunity_days else None,
        "median_lead_hours": float(np.median(leads)) if leads else None,
        "p25_lead_hours": float(np.quantile(leads, 0.25)) if leads else None,
        "cooldown_hours": cooldown_hours,
        "matching": "one_to_one",
        "forecast_opportunities": len(predictions),
    }
    return (result, matches) if with_matches else result
