"""Serializable, causal state for one object's count-based warning policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

HOUR_NS = 3_600_000_000_000
CONFIRMATION_NS = 70 * 60 * 1_000_000_000


@dataclass
class PendingWarningState:
    """One instance per object/type; prediction snapshots remain separate.

    Integers are naive source-clock nanoseconds, matching the feature pipeline.
    A confirmed episode may resolve only one warning, including across restarts.
    Same/older snapshots cannot issue a second warning or roll the state back.
    """

    pending: list[int] = field(default_factory=list)
    matched_events: list[int] = field(default_factory=list)
    last_as_of: int | None = None
    last_issued: int | None = None

    def to_dict(self):
        # Strings prevent nanosecond precision loss in JSON readers using doubles.
        return {
            key: [str(v) for v in value]
            if isinstance(value, list)
            else str(value)
            if value is not None
            else None
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            **{
                key: [int(v) for v in value]
                if isinstance(value, list)
                else int(value)
                if value is not None
                else None
                for key, value in data.items()
            }
        )

    def step(self, as_of, probability, expected_count, confirmed_events, margin, probability_floor):
        now = pd.Timestamp(as_of).value
        if not np.isfinite([probability, expected_count, margin, probability_floor]).all():
            raise ValueError("Non-finite warning input")
        if expected_count < 0 or margin <= 0 or not 0 <= probability <= 1:
            raise ValueError("Invalid warning input")
        if self.last_as_of is not None and now <= self.last_as_of:
            return False
        self.last_as_of = now
        matched = {event for event in self.matched_events if now - event < 48 * HOUR_NS}
        for value in sorted(set(confirmed_events)):
            event = int(value)
            if event in matched or event + CONFIRMATION_NS >= now:
                continue
            for i, issued in enumerate(self.pending):
                if issued <= event < issued + 24 * HOUR_NS:
                    self.pending.pop(i)
                    matched.add(event)
                    break
        self.matched_events = sorted(matched)
        # Resolve first: an event inside a warning's horizon can be confirmed
        # just after its expiry. It must not then also resolve a later warning.
        self.pending = [issued for issued in self.pending if now - issued < 24 * HOUR_NS]
        cadence_ok = self.last_issued is None or now - self.last_issued >= HOUR_NS
        issue = (
            cadence_ok and probability >= probability_floor and expected_count >= len(self.pending) + margin
        )
        if issue:
            self.pending.append(now)
            self.last_issued = now
        return bool(issue)


def replay_pending(predictions, episodes, margin, probability_floor, states=None):
    """Replay chronologically and optionally continue from serialized states."""
    states = {} if states is None else states
    frame = predictions.reset_index(drop=True)
    result = np.zeros(len(frame), dtype=np.float64)
    for obj, rows in frame.groupby("object_id"):
        state = states.setdefault(int(obj), PendingWarningState())
        events = (
            episodes.loc[episodes.object_id.eq(obj), "start_ts"]
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        for row in rows.sort_values("as_of").itertuples():
            now = pd.Timestamp(row.as_of).value
            left, right = np.searchsorted(events, [now - 24 * HOUR_NS, now - CONFIRMATION_NS], side="left")
            result[row.Index] = state.step(
                row.as_of, row.probability, row.expected_count, events[left:right], margin, probability_floor
            )
    return result, states
