"""Release the 24-hour warning rule for every kind and activate it.

The frozen archive classifiers already predict "an episode starts on this object within
24 hours". Under the 24-hour rule (one warning per object and kind per day) they are scored
at the service cadence of one hour:

1. thresholds are chosen on 15-31 May: the highest incident-day recall with precision
   >= 0.7, otherwise the best F1 (``daily_rule.select_threshold``);
2. June, held out from training, calibration and selection, is scored once;
3. the classifiers, thresholds and the rule are written as a new immutable operational
   bundle, which then serves both the archive queue and new data.

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
POLICY = (pd.Timestamp("2026-05-15"), pd.Timestamp("2026-06-01"))
TEST = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))
HOURLY = PROCESSED / "features-hourly-2026.parquet"
EPISODES = PROCESSED / "episodes-all-2026.parquet"
SUMMARY = (
    "precision",
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
    old_heads = load_bundle(replaced) if replaced != "legacy" else {}
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
        eps = episodes[episodes.kind.eq(kind)]
        scored = {}
        for name, (begin, end) in (("selection", POLICY), ("test", TEST)):
            frame = hourly[mask(hourly, begin, end)].reset_index(drop=True)
            raw = model.predict(model_input(frame, source["features"]), prediction_type="RawFormulaVal")
            pred = frame[["object_id", "as_of"]].copy()
            pred["probability"] = calibrated(raw, source["calibration"])
            scored[name] = (frame, pred)
        chosen = daily_rule.select_threshold(scored["selection"][1], eps)
        june = daily_rule.metrics(scored["test"][1], eps, chosen["threshold"])
        report["kinds"][kind] = {
            "threshold": chosen["threshold"],
            "selection_rule": chosen["rule"],
            "selection": chosen,
            "june": june,
            "previous_rule_june": previous_rule(kind, scored["test"][0], eps, old_heads),
        }
        manifest["models"][kind] = {
            "variant": "classifier_daily",
            "features": source["features"],
            "calibration": source["calibration"],
            "threshold": chosen["threshold"],
            "alert_policy": {"type": "daily", "cooldown_hours": 24, "cadence_hours": 1, "horizon_hours": 24},
            "selection": {k: chosen[k] for k in ("rule", *SUMMARY)},
            "test": {k: june[k] for k in SUMMARY},
            "periods": {"selection": list(map(str, POLICY)), "test": list(map(str, TEST))},
            "source_model": f"models/{kind}.cbm",
            "members": [
                {"file": f"{kind}.cbm", "weight": 1.0, "sha256": sha256(ARTIFACTS / "models" / f"{kind}.cbm")}
            ],
        }
        staged[kind] = scored["selection"][1]
        print(
            kind,
            {k: round(v, 3) if isinstance(v, float) else v for k, v in june.items() if k in SUMMARY},
            flush=True,
        )
    version = manifest_version(manifest)
    manifest["version"] = version
    folder = ARTIFACTS / "operational" / version
    folder.mkdir(parents=True, exist_ok=True)
    for kind, pred in staged.items():
        shutil.copy2(ARTIFACTS / "models" / f"{kind}.cbm", folder / f"{kind}.cbm")
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
