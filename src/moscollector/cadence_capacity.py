"""Exact matching capacity of fixed slots, using future outcomes diagnostically."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from moscollector.alert_diagnostics import HOUR_NS


def subdivide_evaluation_slots(predictions, minutes=15):
    """Densify only gaps between adjacent valid hourly evaluation points.

    This uses the offline eligibility mask to preserve the original evaluation
    cohort. It is NOT a runtime availability decision or a forecasting feature.
    No points are added after the last slot or inside gaps of missing coverage.
    """
    import pandas as pd

    if minutes <= 0 or minutes >= 60 or 60 % minutes:
        raise ValueError("Minutes must be a positive proper divisor of60")
    rows = []
    for obj, group in predictions.groupby("object_id"):
        at = np.sort(group.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        if np.any(np.diff(at) < HOUR_NS):
            raise ValueError("Expected distinct hourly evaluation slots")
        inside = at[:-1][np.diff(at) == HOUR_NS]
        extra = [inside + k * minutes * HOUR_NS // 60 for k in range(1, 60 // minutes)]
        times = np.sort(np.concatenate([at, *extra]))
        rows.append(pd.DataFrame({"object_id": obj, "as_of": pd.to_datetime(times)}))
    if not rows:
        return predictions[["object_id", "as_of"]].copy()
    return pd.concat(rows, ignore_index=True)


def matched_slots(times, events):
    """Max-cardinality event/slot matching for [t,t+24h), one warning per slot.

    This has access to future outcomes and is NOT a forecasting model. Slots
    must already respect the minimum warning interval; no extra cooldown is
    optimized here. Return a boolean mask of slots used by one optimum.
    """
    times, events = np.asarray(times, dtype=np.int64), np.asarray(events, dtype=np.int64)
    if times.ndim != 1 or events.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("Need one-dimensional sorted distinct forecast slots")
    if not len(events) or not len(times):
        return np.zeros(len(times), dtype=bool)
    low = np.searchsorted(times, events - 24 * HOUR_NS, side="right")
    high = np.searchsorted(times, events, side="right")
    lengths = high - low
    indptr = np.r_[0, np.cumsum(lengths)]
    indices = np.concatenate([np.arange(a, b) for a, b in zip(low, high, strict=True)])
    graph = csr_matrix(
        (np.ones(len(indices), dtype=np.int8), indices, indptr), shape=(len(events), len(times))
    )
    return maximum_bipartite_matching(graph, perm_type="row") >= 0
