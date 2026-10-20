"""Every June warning of the released 24-hour rule next to what the journal recorded.

The analyst's model check shows this table ("predicted vs actual"). It is rebuilt from the
released bundle named in ``artifacts/daily_rule_evaluation.json``: each kind is scored every
hour on June with its released head and threshold, warnings follow the 24-hour rule, and each
is matched to the incident days of the journal (``daily_rule``). Totals must equal the June
figures of the release report, otherwise nothing is written.

Needs the hourly 2026 features and episodes prepared by ``scripts/release_daily_rule.py``.

    python scripts/publish_daily_matches.py
"""

import json

import pandas as pd

from moscollector import daily_rule
from moscollector.model_registry import load_bundle
from moscollector.paths import BUNDLE, PROCESSED, ROOT
from moscollector.prepare import write_json
from moscollector.research import mask

REPORT = ROOT / "artifacts" / "daily_rule_evaluation.json"
TARGETS = (ROOT / "artifacts" / "daily_rule_matches.json", BUNDLE / "artifacts" / "daily_rule_matches.json")


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    heads = load_bundle(report["version"])
    begin, end = (pd.Timestamp(x) for x in report["test_period"])
    hourly = pd.read_parquet(PROCESSED / "features-hourly-2026.parquet")
    episodes = pd.read_parquet(PROCESSED / "episodes-all-2026.parquet")
    test = hourly[mask(hourly, begin, end)].reset_index(drop=True)
    result = {
        "version": report["version"],
        "rule": report["rule"],
        "test_period": report["test_period"],
        "kinds": {},
    }
    for kind, item in report["kinds"].items():
        scored = test[["object_id", "as_of"]].copy()
        scored["probability"] = heads[kind].probability(test)
        warnings = daily_rule.Evaluator(scored, episodes[episodes.kind.eq(kind)]).warnings(item["threshold"])
        june = item["june"]
        if len(warnings) != june["alerts"] or int(warnings.hit.sum()) != june["true_alerts"]:
            raise SystemExit(
                f"{kind}: {len(warnings)} warnings / {int(warnings.hit.sum())} correct, "
                f"report has {june['alerts']} / {june['true_alerts']}"
            )
        warnings = warnings.sort_values(["as_of", "object_id"]).reset_index(drop=True)
        result["kinds"][kind] = {
            "model": item["model"],
            "threshold": item["threshold"],
            "rows": [
                {
                    "object_id": int(r.object_id),
                    "as_of": r.as_of.isoformat(),
                    "probability": round(float(r.probability), 4),
                    "hit": bool(r.hit),
                    "lead_hours": None if pd.isna(r.lead_hours) else round(float(r.lead_hours), 2),
                }
                for r in warnings.itertuples()
            ],
        }
        print(kind, len(warnings), "warnings,", int(warnings.hit.sum()), "correct", flush=True)
    for target in TARGETS:
        write_json(target, result)
        print("written", target)


if __name__ == "__main__":
    main()
