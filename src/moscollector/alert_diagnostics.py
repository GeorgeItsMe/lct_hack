"""Fast, legacy-compatible event evaluation and honest precision/recall diagnostics.

This module does not change episode definitions or the frozen release's scores.
Frontiers on test labels are descriptive diagnostics, never deployable thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from moscollector.paths import PROCESSED
from moscollector.prepare import write_json

HOUR_NS = 3_600_000_000_000


class EventEvaluator:
    """Cache opportunities/episodes; match the existing [t, t+24h) protocol."""

    def __init__(self, predictions: pd.DataFrame, episodes: pd.DataFrame, cadence_hours: int = 3):
        if predictions.duplicated(["object_id", "as_of"]).any():
            raise ValueError("Duplicate forecast opportunities")
        self.n = len(predictions)
        if cadence_hours <= 0:
            raise ValueError("Cadence must be positive")
        self.opportunity_days = self.n * cadence_hours / 24
        self.groups = []
        self.events = 0
        frame = predictions[["object_id", "as_of"]].reset_index(drop=True)
        for obj, rows in frame.groupby("object_id", sort=True):
            rows = rows.sort_values("as_of")
            times = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            event_times = (
                episodes.loc[episodes.object_id.eq(int(obj)), "start_ts"]
                .sort_values()
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            previous = np.searchsorted(times, event_times, side="right") - 1
            keep = (previous >= 0) & ((event_times - times[np.maximum(previous, 0)]) < 24 * HOUR_NS)
            event_times = event_times[keep]
            self.groups.append((int(obj), rows.index.to_numpy(), times, event_times))
            self.events += len(event_times)

    def evaluate(self, probabilities, threshold: float, cooldown_hours: int = 24):
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (self.n,) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid prediction vector")
        alerts = tp = 0
        leads = []
        for _, ids, times, events in self.groups:
            previous = None
            next_event = 0
            for t in times[probabilities[ids] >= threshold]:
                if previous is not None and t - previous < cooldown_hours * HOUR_NS:
                    continue
                previous = t
                alerts += 1
                next_event = max(next_event, int(np.searchsorted(events, t)))
                if next_event < len(events) and events[next_event] - t < 24 * HOUR_NS:
                    tp += 1
                    leads.append((events[next_event] - t) / HOUR_NS)
                    next_event += 1
        p = tp / alerts if alerts else 0.0
        r = tp / self.events if self.events else 0.0
        return {
            "precision": p,
            "recall": r,
            "f1": 2 * tp / (alerts + self.events) if alerts + self.events else 0.0,
            "true_alerts": tp,
            "false_alerts": alerts - tp,
            "alerts": alerts,
            "eligible_episodes": self.events,
            "missed_episodes": self.events - tp,
            "false_alerts_per_object_day": (alerts - tp) / self.opportunity_days if self.n else None,
            "median_lead_hours": float(np.median(leads)) if leads else None,
            "cooldown_hours": cooldown_hours,
            "matching": "one_to_one",
            "forecast_opportunities": self.n,
        }

    def clairvoyant_schedule(self, cooldown_hours=24):
        """Diagnostic using future event times: NOT a usable predictive model.

        Emit the earliest available warning for the earliest unmatched event.
        This reports an achievable oracle schedule, not a proven upper bound.
        """
        scores = np.zeros(self.n)
        for _, ids, times, events in self.groups:
            previous = None
            next_event = 0
            for i, t in zip(ids, times, strict=True):
                if previous is not None and t - previous < cooldown_hours * HOUR_NS:
                    continue
                next_event = max(next_event, int(np.searchsorted(events, t)))
                if next_event < len(events) and events[next_event] - t < 24 * HOUR_NS:
                    scores[i] = 1
                    previous = t
                    next_event += 1
        return self.evaluate(scores, 0.5, cooldown_hours)

    def frontier(self, probabilities, cooldown_hours=24):
        thresholds = np.unique(
            np.r_[np.linspace(0.01, 0.99, 99), np.quantile(probabilities, np.linspace(0.5, 1, 61)), 1.01]
        )
        return [
            {"threshold": float(t), **self.evaluate(probabilities, t, cooldown_hours)} for t in thresholds
        ]


def row_frontier(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "prevalence": float(np.mean(y)),
        "descriptive_recall_at_precision": {
            str(goal): float(np.max(recall[precision >= goal], initial=0)) for goal in (0.5, 0.7, 0.75, 0.8)
        },
    }


def diagnose(output: Path):
    frame = pd.read_parquet(
        PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
    )
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    report = {
        "scope": "diagnostics_on_previously_used_retrospective_periods",
        "warning": "Test-label frontiers and clairvoyant schedules must not select production thresholds.",
        "models": {},
    }
    for fold in ("screen_1", "screen_2", "confirmation"):
        for kind in ("fault", "fire", "access"):
            pred = pd.read_parquet(
                Path("artifacts/research-v4") / fold / "reference" / f"{kind}-predictions.parquet"
            )
            y = pred.merge(
                frame[["object_id", "as_of", f"target_{kind}"]],
                on=["object_id", "as_of"],
                validate="one_to_one",
            )[f"target_{kind}"]
            evaluator = EventEvaluator(pred, episodes[episodes.kind.eq(kind)])
            item = {"row": row_frontier(y, pred.probability), "event": {}}
            for cooldown in (3, 6, 12, 24):
                curve = evaluator.frontier(pred.probability, cooldown)
                item["event"][str(cooldown)] = {
                    "clairvoyant_schedule": evaluator.clairvoyant_schedule(cooldown),
                    "descriptive_best_f1": max(curve, key=lambda x: x["f1"]),
                    "descriptive_recall_at_precision": {
                        str(goal): max((r["recall"] for r in curve if r["precision"] >= goal), default=0)
                        for goal in (0.5, 0.7, 0.75, 0.8)
                    },
                }
            report["models"][f"{fold}/{kind}"] = item
            write_json(output, report)
            print(
                fold,
                kind,
                "row R@P.75",
                round(item["row"]["descriptive_recall_at_precision"]["0.75"], 3),
                "oracle R 24h",
                round(item["event"]["24"]["clairvoyant_schedule"]["recall"], 3),
                "event R@P.75 24h/3h",
                [round(item["event"][str(c)]["descriptive_recall_at_precision"]["0.75"], 3) for c in (24, 3)],
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metric_diagnostics.json"))
    diagnose(parser.parse_args().output)
