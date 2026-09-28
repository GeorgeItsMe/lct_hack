"""Describe frozen v35 scores before studying a different warning mechanism."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.count_research import episode_counts
from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v35")
destination = Path("artifacts/research-v36/motivation-audit.json")
source = PROCESSED / "episodes.parquet"
episodes = pd.read_parquet(source, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
sources = {source, root / "plan.json", root / "report.json"}
report = read(root / "report.json")
expected = {
    (kind, fold)
    for kind, outcome in report.items()
    for fold in (
        ["screen_1", "screen_2"]
        + (["confirmation", "stress_1", "stress_2"] if outcome["selection"]["passed_screen"] else [])
    )
}
paths = sorted(root.glob("*/*/result.json"))
assert {(read(p)["kind"], read(p)["fold"]) for p in paths} == expected
records = []
for path in paths:
    result = read(path)
    eps = episodes.loc[episodes.kind.eq(result["kind"])]
    sources.add(path)
    for name in ("binary_candidate", "count_direct"):
        arm = result["arms"][name]
        threshold, pause = (arm["policy"][k] for k in ("threshold", "cooldown_hours"))
        for part in ("policy", "test"):
            forecast = path.parent / f"{name}-{part}.parquet"
            sources.add(forecast)
            pred = pd.read_parquet(forecast)
            p, y = pred.probability.to_numpy(), episode_counts(pred, eps) > 0
            evaluator = evaluator_for(pred, eps, 1 / 60, result["exposure"][part])
            scores = evaluator.evaluate(pred.alert, 0.5, pause)
            saved = arm["scores"] if part == "test" else arm["policy"]
            assert all(v == saved[k] for k, v in scores.items())
            available = 0
            for _, ids, times, events in evaluator.groups:
                high = times[p[ids] >= threshold]
                if not len(high):
                    continue
                previous = np.searchsorted(high, events, side="right") - 1
                available += int(
                    np.sum((previous >= 0) & (events - high[np.maximum(previous, 0)] < 24 * HOUR_NS))
                )
            assert scores["true_alerts"] <= available <= evaluator.events
            records.append(
                {
                    "kind": result["kind"],
                    "fold": result["fold"],
                    "arm": name,
                    "part": part,
                    "rows": len(pred),
                    "positive_rows": int(y.sum()),
                    "prevalence": float(y.mean()),
                    "row_average_precision": float(average_precision_score(y, p)),
                    "row_roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                    "row_brier": float(brier_score_loss(y, p)),
                    "mean_probability": float(p.mean()),
                    "threshold": threshold,
                    "cooldown_hours": pause,
                    "above_threshold_rows": int(np.sum(p >= threshold)),
                    "events_with_any_above_threshold_past_opportunity": available,
                    "true_alerts": scores["true_alerts"],
                    "eligible_episodes": evaluator.events,
                }
            )
audit = {
    "scope": "Descriptive reused historical v35 scores only. Row AP/AUC are not event P/R or success substitutes. Threshold/pause remain exactly the frozen policy choices, with no test-period reoptimization. An event with a high prior score is NOT necessarily discoverable jointly with others: overlapping windows may share the same opportunity and one warning matches only one event. This count is neither an attainable recall nor an upper bound for other models. No attribution of every miss to cooldown alone. All full event denominators are retained; no June reads.",
    "periods": records,
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(episode_counts.__code__.co_filename),
            Path(EventEvaluator.__init__.__code__.co_filename),
            Path(evaluator_for.__code__.co_filename),
        )
    },
}
write_json(destination, audit)
for row in records:
    if row["arm"] == "binary_candidate" and row["part"] == "test":
        print(row, flush=True)
