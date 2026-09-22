"""Direct probability thresholds on the fixed minute opportunity grid."""

import numpy as np

from moscollector.fine_cadence_research import evaluator_for
from moscollector.goal90_research import primary_score

COOLDOWNS = (1 / 60, 0.25, 1, 2, 3, 6, 12, 24)
QUANTILES = (0, 0.5, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 1)
FIXED = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.01)


def select_threshold(pred, episodes, exposure):
    p = pred.probability.to_numpy()
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid direct probability inputs")
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("Invalid direct policy exposure")
    evaluator = evaluator_for(pred, episodes, 1 / 60, exposure)
    thresholds = np.unique(np.r_[FIXED, np.quantile(p, QUANTILES)])
    options = [{"threshold": float(t), **evaluator.evaluate(p, t, c)} for c in COOLDOWNS for t in thresholds]
    budget = [r for r in options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    chosen = dict(
        max(supported or budget, key=lambda r: (primary_score(r), r["f1"], r["recall"], r["precision"]))
    )
    chosen["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    return chosen, options
