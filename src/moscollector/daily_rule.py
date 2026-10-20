"""The 24-hour warning rule: at most one warning per object and kind per day.

A forecast answers whether an episode of a kind starts on an object within the next
24 hours. One warning already covers that day, so another warning for the same object and
kind is issued no earlier than 24 hours after the previous one. The dispatcher queue, the
monitoring stream, threshold selection and every reported metric use this one rule.

Metrics count incident days: episodes of one object and kind that start within 24 hours of
the first one form one case. A case is caught when a warning was issued at most 24 hours
before its first episode. Precision uses the same unit: a warning is correct when an incident
day starts within 24 hours after it. A warning issued while an incident day is already running
is correct only if the next one starts within its 24 hours; further episodes of the running day
do not make it correct, since they forecast nothing new. With the 24-hour pause each warning
matches at most one incident day. The older episode-level precision (any episode within 24
hours) and strict one-to-one recall over single episodes are reported next to it.

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
    """Warnings for one object and kind; ``times_ns`` must be sorted.

    Jumps from one warning to the first candidate at least ``cooldown_hours`` later, so the
    cost grows with the number of warnings, not with the number of hours.
    """
    candidates = times_ns[above]
    step = cooldown_hours * HOUR_NS
    issued = []
    i = 0
    while i < len(candidates):
        issued.append(candidates[i])
        i = int(np.searchsorted(candidates, candidates[i] + step, side="left"))
    return np.array(issued, dtype=np.int64)


def _followed(points: np.ndarray, issued: np.ndarray) -> np.ndarray:
    """Whether a sorted point lies within 24 hours at or after each warning."""
    if not len(points) or not len(issued):
        return np.zeros(len(issued), dtype=bool)
    k = np.searchsorted(points, issued)
    return (k < len(points)) & (points[np.minimum(k, len(points) - 1)] - issued < HORIZON_HOURS * HOUR_NS)


def schedule(predictions: pd.DataFrame, threshold: float) -> dict[int, np.ndarray]:
    """Issue times per object for one kind: columns object_id, as_of, probability."""
    result = {}
    for obj, part in predictions.groupby("object_id", sort=False):
        part = part.sort_values("as_of")
        result[int(obj)] = issue_times(_ns(part.as_of), part.probability.to_numpy() >= threshold)
    return result


class Evaluator:
    """Warnings and incident days for one kind over a scored period, for many thresholds.

    Only episodes with a forecast opportunity in the preceding 24 hours are counted,
    as in the event evaluator used for model selection.
    """

    def __init__(self, predictions: pd.DataFrame, episodes: pd.DataFrame):
        by_object = {int(k): np.sort(_ns(g.start_ts)) for k, g in episodes.groupby("object_id")}
        empty = np.array([], dtype=np.int64)
        self.groups = []
        self.objects = []
        self.events = self.cases = 0
        for obj, part in predictions.groupby("object_id", sort=False):
            part = part.sort_values("as_of")
            times = _ns(part.as_of)
            ev = by_object.get(int(obj), empty)
            previous = np.searchsorted(times, ev, side="right") - 1
            ev = ev[(previous >= 0) & ((ev - times[np.maximum(previous, 0)]) < HORIZON_HOURS * HOUR_NS)]
            starts = []
            for e in ev:
                if not starts or e - starts[-1] >= HORIZON_HOURS * HOUR_NS:
                    starts.append(e)
            self.groups.append((times, part.probability.to_numpy(), ev, np.array(starts, dtype=np.int64)))
            self.objects.append(int(obj))
            self.events += len(ev)
            self.cases += len(starts)
        self.days = max(1, pd.to_datetime(predictions.as_of).dt.normalize().nunique())

    def warnings(self, threshold: float) -> pd.DataFrame:
        """Every warning at this threshold: object, time, probability when issued, whether an
        incident day started within its 24 hours and how long before that start it came."""
        rows = []
        for obj, (times, probability, _, starts) in zip(self.objects, self.groups, strict=True):
            issued = issue_times(times, probability >= threshold)
            at = np.searchsorted(times, issued)
            hit = _followed(starts, issued)
            lead = np.full(len(issued), np.nan)
            if len(starts) and hit.any():
                k = np.searchsorted(starts, issued[hit])
                lead[hit] = (starts[k] - issued[hit]) / HOUR_NS
            rows.append(
                pd.DataFrame(
                    {
                        "object_id": obj,
                        "as_of": pd.to_datetime(issued),
                        "probability": probability[at],
                        "hit": hit,
                        "lead_hours": lead,
                    }
                )
            )
        columns = ["object_id", "as_of", "probability", "hit", "lead_hours"]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)

    def evaluate(self, threshold: float) -> dict:
        alerts = correct = any_episode = running = strict = caught = 0
        leads = []
        for times, probability, ev, starts in self.groups:
            issued = issue_times(times, probability >= threshold)
            alerts += len(issued)
            hit = _followed(starts, issued)
            correct += int(hit.sum())
            any_episode += int(_followed(ev, issued).sum())
            if len(starts) and len(issued):
                k = np.searchsorted(starts, issued)
                leads.extend(((starts[k[hit]] - issued[hit]) / HOUR_NS).tolist())
                j = np.searchsorted(starts, issued, side="left") - 1
                running += int(
                    ((j >= 0) & (issued - starts[np.maximum(j, 0)] < HORIZON_HOURS * HOUR_NS)).sum()
                )
            pointer = 0
            for t in issued:
                k = int(np.searchsorted(ev, t))
                pointer = max(pointer, k)
                if pointer < len(ev) and ev[pointer] - t < HORIZON_HOURS * HOUR_NS:
                    strict += 1
                    pointer += 1
            if len(starts) and len(issued):
                j = np.searchsorted(issued, starts, side="right") - 1
                caught += int(
                    ((j >= 0) & (starts - issued[np.maximum(j, 0)] < HORIZON_HOURS * HOUR_NS)).sum()
                )
        precision = correct / alerts if alerts else 0.0
        recall = caught / self.cases if self.cases else 0.0
        return {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "recall_strict": strict / self.events if self.events else 0.0,
            "precision_any_episode": any_episode / alerts if alerts else 0.0,
            "alerts_in_running_incident": running,
            "alerts": alerts,
            "true_alerts": correct,
            "false_alerts": alerts - correct,
            "incident_days": self.cases,
            "caught_incident_days": caught,
            "eligible_episodes": self.events,
            "days": int(self.days),
            "alerts_per_day": alerts / self.days,
            "false_alerts_per_day": (alerts - correct) / self.days,
            "median_lead_hours": float(np.median(leads)) if leads else None,
            "cooldown_hours": COOLDOWN_HOURS,
            "recall_unit": "incident_days",
            "precision_unit": "incident_days",
        }


def metrics(predictions: pd.DataFrame, episodes: pd.DataFrame, threshold: float) -> dict:
    """Warnings and incident days for one kind at one threshold."""
    return Evaluator(predictions, episodes).evaluate(threshold)


def select_threshold(predictions: pd.DataFrame, episodes: pd.DataFrame, points: int = 400) -> dict:
    """Highest incident-day recall with precision >= 0.7; if unreachable, the best F1.

    Candidate thresholds are quantiles of the period's own probabilities.
    """
    grid = np.unique(np.quantile(predictions.probability, np.linspace(0.5, 0.9995, points)))
    evaluator = Evaluator(predictions, episodes)
    results = [evaluator.evaluate(t) for t in grid]
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
