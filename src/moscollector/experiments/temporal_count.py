"""Four-bin count targets and a causal deadline-aware forecast-mass budget.

The budget is a deterministic fluid approximation. It is NOT the probability
that a warning will match an event or a fitted point-process likelihood.
"""

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.experiments.binary_gate_research import FLOORS
from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.goal90_research import PendingSimulator, primary_score
from moscollector.experiments.near_term_count_research import horizon_counts
from moscollector.train import model_input

EDGES = (0, 6, 12, 18, 24)
RATES = [f"bin_count_{i}" for i in range(4)]
LEAD = "forecast_bin"


def bin_counts(rows, episodes):
    cumulative = np.column_stack(
        [np.zeros(len(rows), dtype=np.int32)] + [horizon_counts(rows, episodes, h) for h in EDGES[1:]]
    )
    result = np.diff(cumulative, axis=1)
    if np.any(result < 0):
        raise ValueError("Nonmonotone temporal count targets")
    return result


def training_table(rows, columns, targets):
    """Bin-major expansion retains every original anchor and total weight one."""
    targets = np.asarray(targets)
    if (
        LEAD in rows
        or LEAD in columns
        or targets.shape != (len(rows), 4)
        or len(columns) != len(set(columns))
        or any(c.startswith("target_") or c in ("as_of", "eligible", "count_target") for c in columns)
    ):
        raise ValueError("Invalid temporal training schema")
    if not np.isfinite(targets).all() or np.any(targets < 0):
        raise ValueError("Invalid temporal count targets")
    x = model_input(rows, columns)
    expanded = pd.concat([x.assign(**{LEAD: i}) for i in range(4)], ignore_index=True)
    return expanded, targets.T.reshape(-1), np.full(len(expanded), 0.25)


def predict_bins(model, rows, columns):
    x = model_input(rows, columns)
    raw = np.column_stack(
        [
            model.predict(x.assign(**{LEAD: i}), prediction_type="RawFormulaVal", thread_count=4)
            for i in range(4)
        ]
    )
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite temporal forecasts")
    return np.exp(np.clip(raw, -20, 20))


def reserved_mass(profile, deadlines_hours):
    """Greedily allocate forecast mass to oldest outstanding deadlines first."""
    rates = np.asarray(profile, dtype=float)
    deadlines = np.asarray(deadlines_hours, dtype=float)
    if (
        rates.shape != (4,)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
        or not np.isfinite(deadlines).all()
        or np.any((deadlines < 0) | (deadlines > 24))
    ):
        raise ValueError("Invalid profile/deadlines")
    spent = 0.0
    cumulative = np.r_[0.0, np.cumsum(rates)]
    for deadline in np.sort(deadlines):
        i = min(int(deadline // 6), 3)
        mass = cumulative[i] + rates[i] * (deadline - 6 * i) / 6
        spent = min(spent + 1, mass)
    return spent


class DeadlineSimulator(PendingSimulator):
    """Same forecast ticks and confirmation delay as FinePendingSimulator.

    Skip only ticks that cannot emit or release confirmations. Expiry before
    skipped ticks is reconstructed using the preceding ORIGINAL forecast tick;
    confirmations at the current tick still run before its expiry cleanup.
    """

    def alerts(self, profiles, probability, multiplier, margin, floor):
        profiles, probability = np.asarray(profiles, dtype=float), np.asarray(probability, dtype=float)
        if (
            profiles.shape != (self.n, 4)
            or probability.shape != (self.n,)
            or not np.isfinite(profiles).all()
            or np.any(profiles < 0)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or not np.isfinite([multiplier, margin, floor]).all()
            or multiplier < 0
            or margin <= 0
            or not 0 <= floor <= 1
        ):
            raise ValueError("Invalid deadline policy/forecasts")
        result = np.zeros(self.n)
        total = profiles.sum(axis=1) * multiplier
        can_emit = (probability >= floor) & (total >= margin)
        delay = 70 * 60 * 1_000_000_000
        lifetime = 24 * HOUR_NS
        for ids, times, events in self.groups:
            available = ((events + delay) // HOUR_NS + 1) * HOUR_NS
            releases = np.searchsorted(times, available, side="left")
            selected = np.union1d(np.flatnonzero(can_emit[ids]), releases[releases < len(times)])
            pending, next_event = [], 0
            for pos in selected:
                now, index = times[pos], ids[pos]
                if pos:
                    previous_tick = times[pos - 1]
                    pending = [issued for issued in pending if previous_tick - issued < lifetime]
                while next_event < len(events) and available[next_event] <= now:
                    observed = events[next_event]
                    next_event += 1
                    for j, issued in enumerate(pending):
                        if issued <= observed < issued + lifetime:
                            pending.pop(j)
                            break
                pending = [issued for issued in pending if now - issued < lifetime]
                if not can_emit[index]:
                    continue
                # Avoid allocating small NumPy arrays in the inner loop.
                rates = profiles[index] * multiplier
                cumulative = (0, rates[0], rates[0] + rates[1], rates[0] + rates[1] + rates[2])
                spent = 0.0
                for issued in pending:
                    remaining = (issued + lifetime - now) / HOUR_NS
                    b = min(int(remaining // 6), 3)
                    mass = cumulative[b] + rates[b] * (remaining - 6 * b) / 6
                    spent = min(spent + 1, mass)
                if total[index] - spent >= margin:
                    result[index] = 1
                    pending.append(now)
        return result


def deadline_alerts(pred, episodes, policy):
    return DeadlineSimulator(pred, episodes).alerts(
        pred[RATES].to_numpy(),
        pred.probability.to_numpy(),
        policy["capacity"],
        policy["margin"],
        policy["floor"],
    )


def select_deadline(pred, episodes, cadence, exposure):
    simulator = DeadlineSimulator(pred, episodes)
    evaluator = evaluator_for(pred, episodes, cadence, exposure)
    floors = (0, 0.5, 0.75, 0.9, 0.95) if cadence == 1 else FLOORS
    probability = pred.probability.to_numpy()
    profiles = pred[RATES].to_numpy()
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid calibrated probabilities")
    masks = {floor: probability >= floor for floor in floors}
    keys = {floor: np.packbits(value).tobytes() for floor, value in masks.items()}
    options, cache = [], {}
    for multiplier in (0.5, 1, 1.5, 2):
        for margin in (0.25, 0.5, 1, 2, 3, 5):
            for floor in floors:
                key = (multiplier, margin, keys[floor])
                if key not in cache:
                    values = simulator.alerts(profiles, masks[floor].astype(float), multiplier, margin, 0.5)
                    cache[key] = evaluator.evaluate(values, 0.5, cadence)
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
    return chosen, options
