"""Apply the necessary P(N>0)<=E[N] bound as a fixed capacity projection.

This is not joint calibration or a predictive quality guarantee. In particular,
the necessary inequality alone does not characterize an entire distribution.
"""

import numpy as np


def bounded_count(expected, probability):
    expected, probability = np.asarray(expected, dtype=float), np.asarray(probability, dtype=float)
    if (
        expected.ndim != 1
        or probability.shape != expected.shape
        or not np.isfinite(expected).all()
        or not np.isfinite(probability).all()
        or np.any(expected < 0)
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Invalid mean count or probability")
    return np.maximum(expected, probability)


def project(table):
    if "source_expected_count" in table:
        raise ValueError("Projection provenance already present")
    source = table.expected_count.to_numpy()
    output = bounded_count(source, table.probability.to_numpy())
    result = table.copy()
    result["source_expected_count"] = source
    result["expected_count"] = output
    difference = output - source
    stats = {
        "rows": len(table),
        "raised_rows": int(np.sum(difference > 0)),
        "maximum_raise": float(difference.max()) if len(difference) else 0.0,
        "source_count_sum": float(source.sum()),
        "projected_count_sum": float(output.sum()),
        "scope": "Necessary lower bound, not preserved marginal mean calibration or an event-quality claim.",
    }
    return result, stats
