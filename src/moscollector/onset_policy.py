"""Frozen v24 gate search with explicit cadence for irregular minute opportunities."""

import numpy as np

from moscollector.binary_gate_research import FLOORS
from moscollector.fine_cadence_research import FinePendingSimulator, evaluator_for
from moscollector.goal90_research import primary_score


def select_policy(pred, episodes, exposure, cadence=1 / 60):
    if cadence not in (1 / 60, 0.25):
        raise ValueError("Unsupported onset policy cadence")
    probability = pred.probability.to_numpy()
    expected = pred.expected_count.to_numpy()
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(expected).all()
        or np.any((probability < 0) | (probability > 1))
        or np.any(expected < 0)
    ):
        raise ValueError("Invalid calibrated probabilities or counts")
    simulator = FinePendingSimulator(pred, episodes)
    evaluator = evaluator_for(pred, episodes, cadence, exposure)
    masks = {floor: probability >= floor for floor in FLOORS}
    keys = {floor: np.packbits(value).tobytes() for floor, value in masks.items()}
    options, cache = [], {}
    for multiplier in (0.5, 1, 1.5, 2):
        for margin in (0.25, 0.5, 1, 2, 3, 5):
            for floor in FLOORS:
                key = (multiplier, margin, keys[floor])
                if key not in cache:
                    alerts = simulator.alerts(expected * multiplier, masks[floor].astype(float), margin, 0.5)
                    cache[key] = evaluator.evaluate(alerts, 0.5, cadence)
                options.append({"capacity": multiplier, "margin": margin, "floor": floor, **cache[key]})
    budget = [r for r in options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    fallback = {
        "capacity": 0,
        "margin": 1,
        "floor": 0,
        **evaluator.evaluate(np.zeros(len(pred)), 0.5, cadence),
    }
    chosen = dict(
        max(
            supported or budget or [fallback],
            key=lambda r: (primary_score(r), r["f1"], r["recall"], r["precision"]),
        )
    )
    chosen["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    chosen["gate_disabled"] = chosen["floor"] == 0
    chosen["gated_policy_rows"] = int(np.sum(probability < chosen["floor"]))
    return chosen, options
