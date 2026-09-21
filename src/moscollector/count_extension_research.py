"""Predeclared v10: regularize rare-event count forecasts; never auto-promote."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moscollector.count_research import fit_one, pooled
from moscollector.paths import PROCESSED
from moscollector.precision_research import goal_score, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask

KINDS = ("fault", "fire")
CONFIGS = {
    "regularized": dict(feature_config="recent_reference", depth=4, l2_leaf_reg=30, half_life_days=180),
    "sequence_regularized": dict(feature_config="sequence", depth=4, l2_leaf_reg=30, half_life_days=180),
}
STRESS = {"stress_1": "2025-12-01", "stress_2": "2026-03-01"}
PLAN = {
    "scope": "adaptive_retrospective_count_extension_not_blind_test",
    "implementation_revision": "v10b: add missing Dec/Mar hourly features and validate opportunity coverage before fitting. v10 screen/May fits are reusable after exact common-feature parity; experiment choices and gates unchanged.",
    "hypotheses": [
        "Depth4, l2=30 and 180-day weight decay may stabilize rare-event Poisson predictions.",
        "Adding causal multi-resolution and recurrence features may help predict event multiplicity.",
    ],
    "configs": CONFIGS,
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "reference": "v9 Poisson recent_reference with pending-warning policy, identical opportunities and labels.",
    "unchanged": "v9 time splits, calibration, pending-policy grid and selection; same 24h labels and 70min outcome delay; no June reads.",
    "selection": "Choose largest pooled goal_score then F1 on Nov/Feb. Require >5% goal_score improvement and no F1 loss against v9; only that candidate reaches May.",
    "confirmation": "May must improve goal_score and not lose >5% F1 against v9. Count false positives and support explicitly. No threshold chosen from test labels.",
    "stress": "For screen-selected candidates, additionally compare with v9 on Dec2025/Mar2026; same selected config, refit at each origin with preceding-only data. No choosing a replacement candidate after stress. Require >5% pooled goal_score gain, no F1 loss, and no >10% per-month F1 loss for promotion eligibility.",
    "limitations": "Dates occurred in earlier training/calibration and are retrospective even when not used as an evaluation fold before. Feature history may capture recurrent sensor faults rather than physical accidents.",
    "promotion": "Research eligibility only; production requires comparison to current deployed family, feature parity and runtime checks. Never activate automatically.",
    "sources": [
        "https://catboost.ai/docs/en/concepts/loss-functions-regression",
        "https://proceedings.mlr.press/v119/li20p.html",
        "https://proceedings.mlr.press/v54/wang17f.html",
    ],
}


def read(path):
    return json.loads(path.read_text())


def improvement(candidate, reference, relative=1.0):
    return goal_score(candidate) > relative * goal_score(reference) and candidate["f1"] >= reference["f1"]


def validate_opportunities(frame, dense, periods):
    for period in ("policy", "test"):
        reference = frame.loc[mask(frame, *periods[period]), ["object_id", "as_of"]]
        available = dense.loc[mask(dense, *periods[period]), ["object_id", "as_of"]]
        missing = pd.MultiIndex.from_frame(reference).difference(pd.MultiIndex.from_frame(available))
        if len(missing):
            raise ValueError(
                f"Missing {len(missing)} reference opportunities in {period}; build complete hourly features"
            )


def run(output, stage):
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        PROCESSED / n
        for n in ("features-sequence.parquet", "features-dense-round4.parquet", "episodes.parquet")
    ]
    hashes = {str(p): sha256(p) for p in sources}
    plan_path = output / "plan.json"
    if not plan_path.exists():
        write_json(
            plan_path,
            {
                **PLAN,
                "created_at": datetime.now(UTC).isoformat(),
                "source_hashes": hashes,
                "code_hashes": {
                    str(p): sha256(p) for p in (Path(__file__), Path(fit_one.__code__.co_filename))
                },
            },
        )
    else:
        saved_plan = read(plan_path)
        if saved_plan["source_hashes"] != hashes:
            raise ValueError("Research sources changed")
        expected_plan = json.loads(json.dumps(PLAN))
        if any(saved_plan.get(k) != v for k, v in expected_plan.items()):
            raise ValueError("Research plan changed; use a new study directory")
        if any(sha256(Path(p)) != digest for p, digest in saved_plan["code_hashes"].items()):
            raise ValueError("Research implementation changed; use a new study directory")
    frame = pd.read_parquet(sources[0], filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
    dense = pd.read_parquet(sources[1], filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
    episodes = pd.read_parquet(sources[2], filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])

    def fit(fold, kind, config):
        begin = {**FOLDS, **STRESS}[fold]
        validate_opportunities(frame, dense, periods_for(begin, kind))
        if config == "reference":
            # Reuse immutable v9 fits where available. Missing rare-kind May is fitted separately.
            old = Path("artifacts/research-v9") / fold / f"{kind}.json"
            if old.exists():
                return read(old)
            return fit_one(frame, dense, episodes, output / fold / config, kind, begin)
        return fit_one(frame, dense, episodes, output / fold / config, kind, begin, **CONFIGS[config])

    if stage == "screen":
        selection = {}
        for kind in KINDS:
            scores = {
                config: pooled([fit(fold, kind, config) for fold in ("screen_1", "screen_2")], "pending")
                for config in ("reference", *CONFIGS)
            }
            chosen = max(CONFIGS, key=lambda c: (goal_score(scores[c]), scores[c]["f1"]))
            selection[kind] = {
                "candidate": chosen,
                "scores": scores,
                "passes": improvement(scores[chosen], scores["reference"], 1.05),
            }
            print("SELECT", kind, selection[kind], flush=True)
            write_json(output / "selection.json", selection)
    else:
        selection = read(output / "selection.json")
        if set(selection) != set(KINDS):
            raise ValueError("Finish screening first")
        result = {}
        for kind, selected in selection.items():
            if not selected["passes"]:
                result[kind] = {"passes": False, "status": "screening_failed"}
                continue
            folds = ("confirmation",) if stage == "confirm" else tuple(STRESS)
            candidates, references, comparisons = [], [], {}
            for fold in folds:
                candidate, reference = fit(fold, kind, selected["candidate"]), fit(fold, kind, "reference")
                candidates.append(candidate)
                references.append(reference)
                c, r = candidate["scores"]["pending"], reference["scores"]["pending"]
                assert c["eligible_episodes"] == r["eligible_episodes"]
                comparisons[fold] = {"candidate": c, "reference": r}
            c, r = pooled(candidates, "pending"), pooled(references, "pending")
            passed = (
                (goal_score(c) > goal_score(r) and c["f1"] >= 0.95 * r["f1"])
                if stage == "confirm"
                else (
                    improvement(c, r, 1.05)
                    and all(v["candidate"]["f1"] >= 0.9 * v["reference"]["f1"] for v in comparisons.values())
                )
            )
            result[kind] = {
                "candidate": selected["candidate"],
                "folds": comparisons,
                "scores": c,
                "reference": r,
                "passes": bool(passed),
                "target_met": c["precision"] >= 0.75 and c["recall"] >= 0.5,
            }
            print(stage.upper(), kind, result[kind], flush=True)
        write_json(output / ("confirmation.json" if stage == "confirm" else "stress.json"), result)
    if {str(p): sha256(p) for p in sources} != hashes:
        raise ValueError("Research modified its sources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v10b"))
    parser.add_argument("--stage", choices=("screen", "confirm", "stress"), default="screen")
    args = parser.parse_args()
    run(args.output, args.stage)
