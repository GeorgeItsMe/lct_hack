"""The 24-hour warning rule: at most one warning per object and kind per day.

A forecast answers whether an episode of a kind starts on an object within the next
24 hours. One warning already covers that day, so another warning for the same object and
kind is issued no earlier than 24 hours after the previous one. The dispatcher queue, the
monitoring stream, threshold selection and every reported metric use this one rule.

Metrics count incident days: episodes of one object and kind that start within 24 hours of
the first one form one case. A case is caught when a warning was issued at most 24 hours
before its first episode. A warning is correct when any episode starts within 24 hours after
it; with the 24-hour pause this equals one-to-one matching. Strict recall over single
episodes is reported next to it for comparison.

Kept free of scikit-learn so the serving image can evaluate a proposed threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COOLDOWN_HOURS = 24
HORIZON_HOURS = 24
HOUR_NS = 3_600_000_000_000
TARGET_PRECISION = 0.7
MIN_ALERTS = 10


def _ns(values) -> np.ndarray:
    return pd.to_datetime(pd.Series(values)).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def issue_times(times_ns: np.ndarray, above: np.ndarray, cooldown_hours: int = COOLDOWN_HOURS) -> np.ndarray:
    """Warnings for one object and kind; ``times_ns`` must be sorted."""
    issued = []
    last = None
    for t in times_ns[above]:
        if last is None or t - last >= cooldown_hours * HOUR_NS:
            issued.append(t)
            last = t
    return np.array(issued, dtype=np.int64)


def schedule(predictions: pd.DataFrame, threshold: float) -> dict[int, np.ndarray]:
    """Issue times per object for one kind: columns object_id, as_of, probability."""
    result = {}
    for obj, part in predictions.groupby("object_id", sort=False):
        part = part.sort_values("as_of")
        result[int(obj)] = issue_times(_ns(part.as_of), part.probability.to_numpy() >= threshold)
    return result


def metrics(predictions: pd.DataFrame, episodes: pd.DataFrame, threshold: float) -> dict:
    """Warnings and incident days for one kind over a scored period.

    Only episodes with a forecast opportunity in the preceding 24 hours are counted,
    as in the event evaluator used for model selection.
    """
    alerts = correct = strict = caught = cases = events = 0
    leads = []
    for obj, part in predictions.groupby("object_id", sort=False):
        part = part.sort_values("as_of")
        times = _ns(part.as_of)
        ev = np.sort(_ns(episodes.loc[episodes.object_id.eq(int(obj)), "start_ts"]))
        previous = np.searchsorted(times, ev, side="right") - 1
        ev = ev[(previous >= 0) & ((ev - times[np.maximum(previous, 0)]) < HORIZON_HOURS * HOUR_NS)]
        events += len(ev)
        starts = []
        for e in ev:
            if not starts or e - starts[-1] >= HORIZON_HOURS * HOUR_NS:
                starts.append(e)
        cases += len(starts)
        issued = issue_times(times, part.probability.to_numpy() >= threshold)
        alerts += len(issued)
        pointer = 0
        for t in issued:
            k = int(np.searchsorted(ev, t))
            if k < len(ev) and ev[k] - t < HORIZON_HOURS * HOUR_NS:
                correct += 1
                leads.append((ev[k] - t) / HOUR_NS)
            pointer = max(pointer, k)
            if pointer < len(ev) and ev[pointer] - t < HORIZON_HOURS * HOUR_NS:
                strict += 1
                pointer += 1
        for s in starts:
            j = int(np.searchsorted(issued, s, side="right")) - 1
            if j >= 0 and s - issued[j] < HORIZON_HOURS * HOUR_NS:
                caught += 1
    days = max(1, pd.to_datetime(predictions.as_of).dt.normalize().nunique())
    precision = correct / alerts if alerts else 0.0
    recall = caught / cases if cases else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "recall_strict": strict / events if events else 0.0,
        "alerts": alerts,
        "true_alerts": correct,
        "false_alerts": alerts - correct,
        "incident_days": cases,
        "caught_incident_days": caught,
        "eligible_episodes": events,
        "days": int(days),
        "alerts_per_day": alerts / days,
        "false_alerts_per_day": (alerts - correct) / days,
        "median_lead_hours": float(np.median(leads)) if leads else None,
        "cooldown_hours": COOLDOWN_HOURS,
        "recall_unit": "incident_days",
    }


def select_threshold(predictions: pd.DataFrame, episodes: pd.DataFrame, points: int = 400) -> dict:
    """Highest incident-day recall with precision >= 0.7; if unreachable, the best F1.

    Candidate thresholds are quantiles of the period's own probabilities.
    """
    grid = np.unique(np.quantile(predictions.probability, np.linspace(0.5, 0.9995, points)))
    results = [metrics(predictions, episodes, t) for t in grid]
    results = [r for r in results if r["alerts"] >= MIN_ALERTS]
    if not results:
        raise ValueError("No threshold gives enough warnings on the selection period")
    reaching = [r for r in results if r["precision"] >= TARGET_PRECISION]
    if reaching:
        best = max(reaching, key=lambda r: (r["recall"], r["precision"]))
        best["rule"] = "max_recall_at_precision_0.7"
    else:
        best = max(results, key=lambda r: (r["f1"], r["precision"]))
        best["rule"] = "best_f1_precision_0.7_unreachable"
    return best
