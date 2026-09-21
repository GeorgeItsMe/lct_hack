"""Post-test sensitivity analysis of frozen predictions; never fits or selects models.

Resample complete object histories, and separately complete parent groups, using
the same draws for model and baseline. Overlapping forecast windows are not IID.
These conditional percentile ranges are not a guarantee for a future month.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.prepare import write_json
from moscollector.train import alert_metrics

COUNT_COLUMNS = ["model_tp", "model_alerts", "episodes", "baseline_tp", "baseline_alerts"]


def ratios(counts):
    """Keep undefined ratios as NaN rather than inventing zero-valued evidence."""
    tp, alerts, events, base_tp, base_alerts = np.asarray(counts, dtype=float).T

    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)

    f1 = divide(2 * tp, alerts + events)
    baseline_f1 = divide(2 * base_tp, base_alerts + events)
    return {
        "precision": divide(tp, alerts),
        "recall": divide(tp, events),
        "f1": f1,
        "baseline_f1": baseline_f1,
        "f1_difference": f1 - baseline_f1,
    }


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (None, None)
    return {"low": low, "high": high, "valid_replicates": len(finite)}


def bootstrap(counts, replicates=5000, seed=20260921):
    """Each row is one complete cluster; model/baseline counts stay paired."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 5 or len(counts) < 2:
        raise ValueError("At least two clusters with five count columns are required")
    if replicates < 100 or (counts < 0).any():
        raise ValueError("Need at least 100 replicates and nonnegative counts")
    if (counts[:, 0] > np.minimum(counts[:, 1], counts[:, 2])).any() or (
        counts[:, 3] > np.minimum(counts[:, 4], counts[:, 2])
    ).any():
        raise ValueError("Matched episodes cannot exceed alerts or eligible episodes")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(replicates, len(counts)))
    sampled = ratios(counts[draws].sum(axis=1))
    leave_one_out = ratios(counts.sum(axis=0) - counts)["f1_difference"]
    finite = leave_one_out[np.isfinite(leave_one_out)]
    return {
        "clusters": len(counts),
        "clusters_with_episodes": int((counts[:, 2] > 0).sum()),
        "clusters_with_true_alerts": int((counts[:, 0] > 0).sum()),
        "replicates": replicates,
        "seed": seed,
        "percentile_95": {name: interval(values) for name, values in sampled.items()},
        "leave_one_cluster_out_f1_difference": {
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "valid_replicates": len(finite),
        },
    }


def object_counts(predictions, episodes, meta):
    rows = []
    for object_id, group in predictions.groupby("object_id", sort=True):
        events = episodes[episodes.object_id.eq(object_id)]
        model = alert_metrics(group, events, meta["threshold"])
        baseline = group.copy()
        baseline["probability"] = meta["baseline"]["object_rates"].get(
            str(object_id), meta["baseline"]["global_rate"]
        )
        base = alert_metrics(baseline, events, meta["baseline"]["threshold"])
        rows.append(
            {
                "object_id": int(object_id),
                "model_tp": model["true_alerts"],
                "model_alerts": model["alerts"],
                "episodes": model["eligible_episodes"],
                "baseline_tp": base["true_alerts"],
                "baseline_alerts": base["alerts"],
            }
        )
    return pd.DataFrame(rows)


def verify_totals(counts, frozen):
    actual = counts[COUNT_COLUMNS].sum().tolist()
    model, base = frozen["alerts"], frozen["baseline_alerts"]
    expected = [
        model["true_alerts"],
        model["alerts"],
        model["eligible_episodes"],
        base["true_alerts"],
        base["alerts"],
    ]
    if actual != expected:
        raise ValueError(f"Counts differ from frozen test: {actual} != {expected}")
    point = ratios(np.asarray([actual]))
    for key, value in [("f1", model["f1"]), ("baseline_f1", base["f1"])]:
        if not np.isclose(point[key][0], value, atol=1e-12, rtol=0):
            raise ValueError(f"{key} differs from frozen report")


def run(replicates=5000, seed=20260921):
    predictions_path = ARTIFACTS / "predictions/all.parquet"
    report_path = ARTIFACTS / "evaluation_report.json"
    report = json.loads(report_path.read_text())
    selection = json.loads((ARTIFACTS / "model_selection.json").read_text())
    predictions = pd.read_parquet(predictions_path, filters=[("split", "=", "test")])
    episodes = pd.read_parquet(PROCESSED / "episodes.parquet")
    objects = pd.read_parquet(PROCESSED / "objects.parquet")
    if not report.get("test_opened") or predictions.empty:
        raise ValueError("Requires an already evaluated frozen test")
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": "paired_cluster_percentile_bootstrap",
        "status": "post_test_sensitivity_not_new_validation",
        "confidence_level": 0.95,
        "period_start": str(predictions.as_of.min()),
        "period_end": str(predictions.as_of.max()),
        "input_sha256": {
            "evaluation_report": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "predictions": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        },
        "limitations": [
            "Models, calibration and thresholds fixed; no model selection or refitting.",
            "Complete June histories are resampled, not overlapping forecast windows.",
            "Percentile ranges assume independent clusters; cross-parent dependence may remain.",
            "Only 16 parent groups and one test month; coverage is approximate, especially for rare events.",
            "Does not measure training uncertainty, new-object transfer or future temporal drift.",
            "Replicates with undefined denominators are excluded per metric and counted explicitly.",
            "Disabled targets have no intervals: absence of warnings is not demonstrated predictive quality.",
        ],
        "method_reference": "https://www.stata.com/manuals/rbootstrap.pdf",
        "models": {},
    }
    for kind in sorted(report["models"]):
        model_file = ARTIFACTS / "models" / f"{kind}.cbm"
        if hashlib.sha256(model_file.read_bytes()).hexdigest() != selection["models"][kind]["sha256"]:
            raise ValueError(f"Frozen model hash differs for {kind}")
        meta = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text())
        frozen = report["models"][kind]
        if meta["threshold"] != frozen["threshold"]:
            raise ValueError(f"Frozen threshold differs for {kind}")
        counts = object_counts(predictions[predictions.kind.eq(kind)], episodes[episodes.kind.eq(kind)], meta)
        verify_totals(counts, frozen["test"])
        counts = counts.merge(objects[["object_id", "parent_id"]], validate="one_to_one")
        if counts.parent_id.isna().any() or len(counts) != predictions.object_id.nunique():
            raise ValueError("Missing object or parent mapping")
        entry = {
            "status": "disabled_insufficient_policy_events" if meta["threshold"] > 1 else "evaluated",
            "counts_match_frozen_report": True,
            "object_counts": counts.to_dict("records"),
        }
        if meta["threshold"] <= 1:
            entry["by_object"] = bootstrap(counts[COUNT_COLUMNS].to_numpy(), replicates, seed)
            parents = counts.groupby("parent_id", sort=True)[COUNT_COLUMNS].sum()
            entry["by_parent"] = bootstrap(parents.to_numpy(), replicates, seed)
        result["models"][kind] = entry
    write_json(ARTIFACTS / "uncertainty_report.json", result)
    for kind, entry in result["models"].items():
        print(kind, entry.get("by_parent", {}).get("percentile_95", entry["status"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    run(args.replicates, args.seed)
