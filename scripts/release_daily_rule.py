"""Release the 24-hour warning rule for every kind and activate it.

The frozen archive classifiers already predict "an episode starts on this object within
24 hours". Under the 24-hour rule (one warning per object and kind per day) they are scored
at the service cadence of one hour:

1. for every kind two candidates are scored on 15-31 May: the archive classifier and the
   count head of the submitted bundle ``op-7f225ca12c58``. Each gets its threshold by the
   same rule (the highest incident-day recall with precision >= 0.7, otherwise the best F1,
   ``daily_rule.select_threshold``), and the candidate that reaches 0.7 with the higher
   recall wins, otherwise the one with the higher F1;
2. June, held out from training, calibration and selection, is scored once for the winner;
3. the chosen models, thresholds and the rule are written as a new immutable operational
   bundle, which then serves both the archive queue and new data.

Precision and recall share one unit, incident days (see ``daily_rule``).

The previous bundle is re-scored with its own hourly rule (repeat warnings after 1 hour)
so the report shows both rules on the same June. Needs the full prepared 2026 data;
hourly features are built with the production builder when missing.

    python scripts/release_daily_rule.py [--no-activate]
"""

import argparse
import hashlib
import json
import shutil

import pandas as pd
from catboost import CatBoostClassifier

from moscollector import daily_rule
from moscollector.alert_diagnostics import EventEvaluator
from moscollector.count_research import pending_alerts
from moscollector.features import build_dataset
from moscollector.inference import calibrated, model_input
from moscollector.model_registry import activate, active_version, load_bundle, manifest_version
from moscollector.paths import ARTIFACTS, PROCESSED, ROOT
from moscollector.prepare import write_json
from moscollector.research import mask

KINDS = ("fault", "fire", "access")
COUNT_SOURCE = "op-7f225ca12c58"  # the submitted bundle with the count heads
POLICY = (pd.Timestamp("2026-05-15"), pd.Timestamp("2026-06-01"))
TEST = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))
HOURLY = PROCESSED / "features-hourly-2026.parquet"
EPISODES = PROCESSED / "episodes-all-2026.parquet"
SUMMARY = (
    "precision",
    "precision_any_episode",
    "recall",
    "recall_strict",
    "alerts_per_day",
    "false_alerts_per_day",
    "median_lead_hours",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hourly_features():
    if not HOURLY.exists() or not EPISODES.exists():
        build_dataset(
            [2026],
            step_hours=1,
            output_path=HOURLY,
            episode_output_path=EPISODES,
            audit_output_path=PROCESSED / "feature-audit-hourly.json",
        )
    return pd.read_parquet(HOURLY), pd.read_parquet(EPISODES)


def previous_rule(kind, frame, eps, heads):
    """The replaced rule on June: count heads repeat warnings after 1 hour."""
    rows = frame[["object_id", "as_of"]]
    evaluator = EventEvaluator(rows, eps, cadence_hours=1)
    if kind not in heads or "alert_policy" not in heads[kind].meta:
        return None
    head = heads[kind]
    pred = rows.copy()
    pred["probability"] = head.probability(frame)
    pred["expected_count"] = head.expected_count(frame)
    policy = head.meta["alert_policy"]
    scores = pending_alerts(pred, eps, policy["margin"], policy["probability_floor"])
    result = evaluator.evaluate(scores, 0.5, 1)
    return {k: result[k] for k in ("precision", "recall", "alerts", "true_alerts", "eligible_episodes")} | {
        "alerts_per_day": result["alerts"] / frame.as_of.dt.normalize().nunique(),
        "repeat_after_hours": 1,
        "recall_unit": "episodes_one_to_one",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-activate", action="store_true")
    args = parser.parse_args()
    replaced = active_version()
    count_folder = ARTIFACTS / "operational" / COUNT_SOURCE
    old_heads = load_bundle(COUNT_SOURCE) if count_folder.exists() else {}
    hourly, episodes = hourly_features()
    manifest = {"created_from": replaced, "rule": "daily", "models": {}}
    report = {
        "rule": "one warning per object and kind per 24 hours, checked every hour",
        "replaced_version": replaced,
        "selection_period": list(map(str, POLICY)),
        "test_period": list(map(str, TEST)),
        "kinds": {},
    }
    staged = {}
    for kind in KINDS:
        source = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text(encoding="utf-8"))
        model = CatBoostClassifier()
        model.load_model(str(ARTIFACTS / "models" / f"{kind}.cbm"))
        candidates = {
            "archive_classifier": {
                "variant": "classifier_daily",
                "features": source["features"],
                "calibration": source["calibration"],
                "score": lambda frame, m=model, src=source: calibrated(
                    m.predict(model_input(frame, src["features"]), prediction_type="RawFormulaVal"),
                    src["calibration"],
                ),
                "file": ARTIFACTS / "models" / f"{kind}.cbm",
                "member": f"{kind}.cbm",
                "source_model": f"models/{kind}.cbm",
            }
        }
        if kind in old_heads and "rate_scale" in old_heads[kind].meta:
            head = old_heads[kind]
            member = head.meta["members"][0]["file"]
            candidates["count_head"] = {
                "variant": "count_daily",
                "features": head.meta["features"],
                "calibration": head.meta["calibration"],
                "score": head.probability,
                "file": count_folder / member,
                "member": member,
                "source_model": f"operational/{COUNT_SOURCE}/{member}",
            }
        eps = episodes[episodes.kind.eq(kind)]
        selection = hourly[mask(hourly, *POLICY)].reset_index(drop=True)
        test = hourly[mask(hourly, *TEST)].reset_index(drop=True)
        trials = {}
        for name, candidate in candidates.items():
            pred = selection[["object_id", "as_of"]].copy()
            pred["probability"] = candidate["score"](selection)
            trials[name] = (daily_rule.select_threshold(pred, eps), pred)
        name = max(
            trials,
            key=lambda n: (
                trials[n][0]["precision"] >= daily_rule.TARGET_PRECISION,
                trials[n][0]["recall"]
                if trials[n][0]["precision"] >= daily_rule.TARGET_PRECISION
                else trials[n][0]["f1"],
                trials[n][0]["precision"],
            ),
        )
        candidate, (chosen, selected) = candidates[name], trials[name]
        scored = test[["object_id", "as_of"]].copy()
        scored["probability"] = candidate["score"](test)
        june = daily_rule.metrics(scored, eps, chosen["threshold"])
        report["kinds"][kind] = {
            "model": name,
            "threshold": chosen["threshold"],
            "selection_rule": chosen["rule"],
            "selection": chosen,
            "selection_candidates": {n: trials[n][0] for n in trials},
            "june": june,
            "previous_rule_june": previous_rule(kind, test, eps, old_heads),
        }
        manifest["models"][kind] = {
            "variant": candidate["variant"],
            "features": candidate["features"],
            "calibration": candidate["calibration"],
            "threshold": chosen["threshold"],
            "alert_policy": {"type": "daily", "cooldown_hours": 24, "cadence_hours": 1, "horizon_hours": 24},
            "selection": {k: chosen[k] for k in ("rule", *SUMMARY)},
            "test": {k: june[k] for k in SUMMARY},
            "periods": {"selection": list(map(str, POLICY)), "test": list(map(str, TEST))},
            "source_model": candidate["source_model"],
            "members": [{"file": candidate["member"], "weight": 1.0, "sha256": sha256(candidate["file"])}],
        }
        staged[kind] = (candidate, selected)
        print(
            kind,
            name,
            {k: round(v, 3) if isinstance(v, float) else v for k, v in june.items() if k in SUMMARY},
            flush=True,
        )
    version = manifest_version(manifest)
    manifest["version"] = version
    folder = ARTIFACTS / "operational" / version
    folder.mkdir(parents=True, exist_ok=True)
    for kind, (candidate, pred) in staged.items():
        shutil.copy2(candidate["file"], folder / candidate["member"])
        # The analyst's threshold preview replays exactly these selection-period scores.
        pred.to_parquet(folder / f"{kind}_policy.parquet", index=False)
    write_json(folder / "manifest.json", manifest)
    load_bundle(version)
    report["version"] = version
    write_json(ROOT / "artifacts" / "daily_rule_evaluation.json", report)
    if not args.no_activate:
        activate(version)
    print("released", version, "active" if not args.no_activate else "not activated")


if __name__ == "__main__":
    main()
