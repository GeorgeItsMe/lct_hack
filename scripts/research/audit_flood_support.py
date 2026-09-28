"""Measure pre-June flood support without fitting or changing any episode labels."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.count_research import episode_counts
from moscollector.experiments.goal90_research import STRESS
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask

features = PROCESSED / "features-channel-novelty.parquet"
episode_path = PROCESSED / "episodes.parquet"
before = pd.Timestamp("2026-06-01")
frame = pd.read_parquet(
    features,
    columns=["object_id", "as_of", "eligible", "target_flood"],
    filters=[("as_of", "<", before)],
)
episodes = pd.read_parquet(episode_path, filters=[("start_ts", "<", before), ("kind", "==", "flood")])
assert frame.as_of.lt(before).all() and episodes.start_ts.lt(before).all()
assert episodes.episode_id.is_unique
assert np.array_equal(episode_counts(frame, episodes) > 0, frame.target_flood.to_numpy().astype(bool))
records = []
for fold, value in {**FOLDS, **STRESS}.items():
    test = pd.Timestamp(value)
    for months, validation_months in ((1, 1), (3, 1), (3, 3), (6, 1), (6, 6)):
        policy = test - pd.DateOffset(months=months)
        calibration = policy - pd.DateOffset(months=months)
        validation = calibration - pd.DateOffset(months=validation_months)
        periods = {
            "train": (pd.Timestamp("2022-01-01"), validation),
            "validation": (validation, calibration),
            "calibration": (calibration, policy),
            "policy": (policy, test),
            "test": (test, test + pd.DateOffset(months=1)),
        }
        support = {}
        for name, dates in periods.items():
            rows = frame.loc[mask(frame, *dates)]
            counts = episode_counts(rows, episodes)
            evaluator = EventEvaluator(rows, episodes, 3)
            support[name] = {
                "dates": [str(t) for t in dates],
                "rows": len(rows),
                "positive_rows": int(np.sum(counts > 0)),
                "positive_row_fraction": float(np.mean(counts > 0)) if len(rows) else None,
                "eligible_episodes": evaluator.events,
                "observed_objects": int(rows.object_id.nunique()),
                "positive_query_objects": int(rows.loc[counts > 0, "object_id"].nunique()),
            }
        records.append(
            {
                "fold": fold,
                "validation_months": validation_months,
                "calibration_and_policy_months_each": months,
                "parts": support,
            }
        )
monthly = [
    {"month": str(month), "episodes": len(group), "objects": int(group.object_id.nunique())}
    for month, group in episodes.groupby(episodes.start_ts.dt.to_period("M"))
]
result = {
    "scope": "Descriptive support audit for the remaining flood head. No fitting, policy selection, June reads, label changes, event exclusion or forecast-quality claim. These are proxy sensor episodes, not confirmed physical floods.",
    "period_design": "Fixed train start2022, separate calibration/policy windows of1/3/6months each, same existing five test months, original3h eligible opportunities and25h purge. Initial one-month validation has only1-2events in several cases, so also audit validation extended to the same3/6month length. Window lengths inspect event support only, not prediction outcomes. Not a new blind evaluation.",
    "episodes": len(episodes),
    "objects": int(episodes.object_id.nunique()),
    "monthly_counts": monthly,
    "duration_at_most_60_seconds": int(episodes.duration_seconds.le(60).sum()),
    "right_censored": int(episodes.right_censored.sum()),
    "duration_quantiles_seconds": {
        str(q): float(episodes.duration_seconds.quantile(q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 1)
    },
    "windows": records,
    "original_binary_targets_exact_parity": True,
    "limitations": "Longer calibration does not create new events or prove90/90. Short pulses are retained, not labelled false without confirmation. The count of eligible test events is not the number of independent physical incidents. The currently disabled production head is unchanged.",
    "source_hashes": {str(p): sha256(p) for p in (features, episode_path)},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(EventEvaluator.__init__.__code__.co_filename),
            Path(episode_counts.__code__.co_filename),
            Path(mask.__code__.co_filename),
        )
    },
}
write_json(Path("artifacts/flood_support_audit.json"), result)
print("Flood support", result["episodes"], "episodes; short pulses", result["duration_at_most_60_seconds"])
for row in records:
    print(
        row["fold"],
        row["validation_months"],
        row["calibration_and_policy_months_each"],
        {k: v["eligible_episodes"] for k, v in row["parts"].items()},
    )
