"""Keep expired warning ownership until all in-window confirmations are known.

Research implementation; the frozen serving policy is not changed here.
"""

import numpy as np

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.experiments.binary_gate_research import FLOORS
from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.goal90_research import PendingSimulator, primary_score

LIFETIME = 24 * HOUR_NS
DELAY = 70 * 60 * 10**9
RETAIN_FOR = LIFETIME + DELAY + HOUR_NS


class RetainedPendingSimulator(PendingSimulator):
    """Expired records can own late events but never consume active capacity.

    Only ticks that could issue a warning need to be visited. Between these
    ticks no warning is added, so processing every newly released confirmation
    BEFORE pruning the ledger gives the same state as visiting every tick.
    In particular a long gap must not prune an old owner before resolving it.
    """

    def alerts(self, capacity, probability, margin, floor):
        capacity, probability = np.asarray(capacity), np.asarray(probability)
        if (
            any(x.shape != (self.n,) or not np.isfinite(x).all() for x in (capacity, probability))
            or np.any(capacity < 0)
            or np.any((probability < 0) | (probability > 1))
        ):
            raise ValueError("Invalid capacity/probability vector")
        if not np.isfinite(margin) or margin <= 0 or not np.isfinite(floor) or not 0 <= floor <= 1:
            raise ValueError("Invalid policy")
        result = np.zeros(self.n)
        possible = (capacity >= margin) & (probability >= floor)
        for ids, times, events in self.groups:
            release = ((events + DELAY) // HOUR_NS + 1) * HOUR_NS
            selected = np.flatnonzero(possible[ids])
            retained, next_event = [], 0
            for pos in selected:
                i, now = ids[pos], times[pos]
                stop = int(np.searchsorted(release, now, side="right"))
                for observed in events[next_event:stop]:
                    for j, issued in enumerate(retained):
                        if issued <= observed < issued + LIFETIME:
                            retained.pop(j)
                            break
                next_event = stop
                # Resolve first, even if a real gap spans the retention bound.
                retained = [issued for issued in retained if now - issued < RETAIN_FOR]
                active = sum(now - issued < LIFETIME for issued in retained)
                if capacity[i] >= active + margin:
                    result[i] = 1
                    retained.append(now)
        return result


def retained_alerts(pred, episodes, policy):
    return RetainedPendingSimulator(pred, episodes).alerts(
        pred.expected_count.to_numpy() * float(policy["capacity"]),
        pred.probability.to_numpy(),
        policy["margin"],
        policy["floor"],
    )


def select_retained(pred, episodes, cadence, exposure):
    if cadence not in (0.25, 1):
        raise ValueError("Unsupported original cadence")
    simulator = RetainedPendingSimulator(pred, episodes)
    evaluator = evaluator_for(pred, episodes, cadence, exposure)
    probability, expected = pred.probability.to_numpy(), pred.expected_count.to_numpy()
    # Validate the original values before replacing probabilities with masks.
    simulator.alerts(expected, probability, 1, 1)
    floors = FLOORS if cadence == 0.25 else (0, 0.5, 0.75, 0.9, 0.95)
    masks = {f: probability >= f for f in floors}
    keys = {f: np.packbits(m).tobytes() for f, m in masks.items()}
    cache, options = {}, []
    for multiplier in (0.5, 1, 1.5, 2):
        for margin in (0.25, 0.5, 1, 2, 3, 5):
            for floor in floors:
                key = (multiplier, margin, keys[floor])
                if key not in cache:
                    flags = simulator.alerts(expected * multiplier, masks[floor].astype(float), margin, 0.5)
                    cache[key] = evaluator.evaluate(flags, 0.5, cadence)
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
    if cadence == 0.25:
        chosen["gate_disabled"] = chosen["floor"] == 0
        chosen["gated_policy_rows"] = int(np.sum(probability < chosen["floor"]))
    return chosen, options
