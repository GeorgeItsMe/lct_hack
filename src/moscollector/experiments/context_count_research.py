"""v14: isolate the original 28 causal episode/recency features for access counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.count_research import fit_one
from moscollector.experiments.count_extension_research import validate_opportunities
from moscollector.experiments.goal90_research import (
    STRESS,
    apply_policy,
    pooled,
    primary_score,
    read,
    select_policy,
)
from moscollector.paths import PROCESSED
from moscollector.precision_research import columns_for, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

PLAN = {
    "scope": "adaptive_retrospective_context_count_research_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kind": "access; not proof for other kinds",
    "single_factor": "Extend v9 Poisson recent_reference66 with the28 original causal past-episode and report recency/burst features, to94 features. No new feature generation, richer sequence table, calendar input, tree hyperparameters or labels. No assumption that earlier binary-classifier experiments establish the answer for count regression.",
    "fit": "Same v9 Poisson depth6,l2=8,lr=.04,max1000,seed42,CPU4,earlystop100. Same time splits and25h purge. Count calibration and binary calibration unchanged. fit_one also records legacy75/50 policy as a secondary result; primary evaluation separately uses the locked v11 mean_retarget90/90 policy grid.",
    "reference": "Unchanged v9 counts with the same90/90 policy selection, archived as v12/uncorrected.json. Original operational-family policy also retained in earlier reports. This isolates added features from the change in policy objective.",
    "screen": "Fixed candidate, no alternative configuration. Pooled Nov/Feb must improve primary score>5% and not loseF1. Only then evaluate May and Dec/Mar; no replacement candidate after their outcomes.",
    "confirmation": "May improves primary score and preserves>=95%F1 vs same-policy reference; pooled Dec/Mar primary score improves>5%, F1 not lower, each monthF1>=90%reference. Research gates, no automatic activation.90/90 success reported separately.",
    "unchanged": "Same episodes,24h horizon,hourly opportunities,70min outcome delay and one-to-one matching; no June reads.",
}


def evaluate(root, fold, frame, dense, episodes):
    folder = root / fold
    path = folder / "result.json"
    if path.exists():
        return read(path)
    validate_opportunities(frame, dense, periods_for({**FOLDS, **STRESS}[fold], "access"))
    fit_one(
        frame, dense, episodes, folder, "access", {**FOLDS, **STRESS}[fold], feature_config="episode_context"
    )
    policy_pred = pd.read_parquet(folder / "access-policy.parquet")
    policy, options, diagnostic = select_policy(policy_pred, episodes, "mean_retarget")
    write_json(
        folder / "primary_policy.json", {"selected": policy, "options": options, "diagnostic": diagnostic}
    )
    test = pd.read_parquet(folder / "access-test.parquet")
    test["candidate_alert"] = apply_policy(test, episodes, "mean_retarget", policy)
    scores = EventEvaluator(test, episodes, 1).evaluate(test.candidate_alert, 0.5, 1)
    reference = read(Path("artifacts/research-v12") / fold / "uncorrected.json")["scores"]
    if scores["eligible_episodes"] != reference["eligible_episodes"]:
        raise ValueError("Changed episode denominator")
    test.to_parquet(folder / "evaluated.parquet", index=False)
    result = {"fold": fold, "policy": policy, "scores": scores, "reference": reference}
    write_json(path, result)
    print("PRIMARY", fold, scores, flush=True)
    return result


def run(root, stage):
    sources = [
        PROCESSED / n for n in ("features.parquet", "features-dense-round4.parquet", "episodes.parquet")
    ]
    sources.extend(Path("artifacts/research-v12") / fold / "uncorrected.json" for fold in (*FOLDS, *STRESS))
    code = [
        Path(__file__),
        Path(fit_one.__code__.co_filename),
        Path(columns_for.__code__.co_filename),
        Path(select_policy.__code__.co_filename),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(validate_opportunities.__code__.co_filename),
    ]
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in code},
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Changed research inputs or implementation; use a new directory")
    else:
        write_json(path, plan)
    frame = pd.read_parquet(sources[0], filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
    dense = pd.read_parquet(sources[1], columns=frame.columns.tolist())
    episodes = pd.read_parquet(
        sources[2], filters=[("start_ts", "<", pd.Timestamp("2026-06-01")), ("kind", "==", "access")]
    )
    base = columns_for(frame, "recent_reference", "access")
    extended = columns_for(frame, "episode_context", "access")
    extras = [c for c in extended if c not in base]
    if (
        len(base) != 66
        or len(extended) != 94
        or len(extras) != 28
        or any(not (c.startswith("past_") or c.endswith(("_recency_h", "_burst"))) for c in extras)
    ):
        raise ValueError("Expected exactly the original28 causal context features")
    if stage == "screen":
        rows = [evaluate(root, fold, frame, dense, episodes) for fold in ("screen_1", "screen_2")]
        c, r = (pooled([item[key] for item in rows]) for key in ("scores", "reference"))
        selected = {
            "candidate": c,
            "reference": r,
            "passed_screen": primary_score(c) > 1.05 * primary_score(r) and c["f1"] >= r["f1"],
            "added_features": extras,
        }
        write_json(root / "selection.json", selected)
        print("SELECTION", selected, flush=True)
        return
    selected = read(root / "selection.json")
    if not selected["passed_screen"]:
        print("Rejected at screen", flush=True)
        return
    rows = [evaluate(root, fold, frame, dense, episodes) for fold in ("confirmation", *STRESS)]
    may = rows[0]
    c, r = (pooled([item[key] for item in rows[1:]]) for key in ("scores", "reference"))
    passed_may = (
        primary_score(may["scores"]) > primary_score(may["reference"])
        and may["scores"]["f1"] >= 0.95 * may["reference"]["f1"]
    )
    passed_stress = (
        primary_score(c) > 1.05 * primary_score(r)
        and c["f1"] >= r["f1"]
        and all(item["scores"]["f1"] >= 0.9 * item["reference"]["f1"] for item in rows[1:])
    )
    all_rows = [read(root / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
    report = {
        "selection": selected,
        "periods": all_rows,
        "five_period_pooled": pooled([item["scores"] for item in all_rows]),
        "five_period_reference": pooled([item["reference"] for item in all_rows]),
        "passed_may": passed_may,
        "passed_stress": passed_stress,
        "research_eligible": bool(passed_may and passed_stress),
        "automatic_activation": False,
    }
    write_json(root / "report.json", report)
    print("REPORT", {k: v for k, v in report.items() if k != "periods"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v14"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
