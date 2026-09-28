"""v12: past-only, shrunk per-object correction of existing count forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.count_research import episode_counts
from moscollector.experiments.goal90_research import (
    STRESS,
    apply_policy,
    base_directory,
    pooled,
    primary_score,
    read,
    select_policy,
)
from moscollector.paths import PROCESSED
from moscollector.precision_research import periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import model_input

STRENGTHS = (5, 20, 80)
PLAN = {
    "scope": "adaptive_retrospective_object_calibration_not_blind_test",
    "primary_goal": {"precision": 0.9, "recall": 0.9},
    "kind": "access; not proof for all incident types",
    "hypothesis": "Global mean-count calibration may undercount some objects and overcount others. Correct expected counts with a shrunk object-specific ratio from the separate preceding calibration period, preserving weights and the probability gate.",
    "formula": "For each object sum observed and predicted24h counts over eligible3h calibration rows, multiply each sum by3/24 to reduce overlapping-window scale. Factor=(observed_sum+strength)/(predicted_sum+strength), clipped[.25,4]. Strengths5,20,80; unseen objects factor1. This is an empirical shrinkage rule, not an independence assumption or Bayesian coverage guarantee.",
    "policy": "After corrections, same mean_retarget grid and90/90 criterion as v11, fitted exclusively on each preceding policy period. Calibration outcomes do not include policy/test periods;25h purge preserved.",
    "selection": "Choose strength by pooled Nov/Feb primary score then F1. Require >5% relative primary-score improvement and no F1 loss against the v11 mean_retarget policy-only reference. Original v9 also reported. Only selected strength reaches May, Dec, Mar, with no replacement after results.",
    "confirmation": "May must improve primary score and preserve>=95%F1 vs uncorrected v11. Dec/Mar pooled must improve primary score>5% and not loseF1; each extra month preserves>=90%F1. Research gates only, never automatic deployment.",
    "unchanged": "No new weights, features, labels, forecast horizon, episode cohort, outcome confirmation delay or June reads. New comparisons remain retrospective.",
}


def correction_factors(object_ids, observed, predicted, strength):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if strength <= 0 or observed.shape != predicted.shape or observed.shape != (len(object_ids),):
        raise ValueError("Invalid correction inputs")
    if any(not np.isfinite(x).all() or (x < 0).any() for x in (observed, predicted)):
        raise ValueError("Counts must be finite and nonnegative")
    sums = (
        pd.DataFrame(
            {"object_id": object_ids, "observed": observed * 3 / 24, "predicted": predicted * 3 / 24}
        )
        .groupby("object_id")
        .sum()
    )
    factors = ((sums.observed + strength) / (sums.predicted + strength)).clip(0.25, 4)
    return {int(k): float(v) for k, v in factors.items()}


def calibrate_object_counts(frame, episodes, fold, strength):
    base = base_directory(fold)
    meta = read(base / "access.json")
    periods = periods_for({**FOLDS, **STRESS}[fold], "access")
    rows = frame.loc[mask(frame, *periods["calibration"])].reset_index(drop=True)
    observed = episode_counts(rows, episodes)
    if not np.array_equal(observed > 0, rows.target_access.to_numpy().astype(bool)):
        raise ValueError("Changed count labels")
    model = CatBoostRegressor()
    model.load_model(str(base / "access.cbm"))
    raw = model.predict(model_input(rows, meta["features"]), prediction_type="RawFormulaVal", thread_count=2)
    predicted = np.exp(np.clip(raw, -20, 20)) * meta["rate_scale"]
    return correction_factors(rows.object_id, observed, predicted, strength)


def evaluate(output, fold, strength, frame, episodes):
    directory = output / fold / f"strength_{strength}"
    path = directory / "result.json"
    if path.exists():
        return read(path)
    directory.mkdir(parents=True, exist_ok=True)
    factors = calibrate_object_counts(frame, episodes, fold, strength)
    write_json(
        directory / "calibration.json",
        {
            "strength": strength,
            "factors": factors,
            "period": list(map(str, periods_for({**FOLDS, **STRESS}[fold], "access")["calibration"])),
        },
    )
    base = base_directory(fold)
    pred = pd.read_parquet(base / "access-policy.parquet")
    pred["expected_count"] *= pred.object_id.map(factors).fillna(1)
    policy, options, diagnostic = select_policy(pred, episodes, "mean_retarget")
    write_json(directory / "policy.json", {"selected": policy, "options": options, "diagnostic": diagnostic})
    test = pd.read_parquet(base / "access-test.parquet")
    test["expected_count"] *= test.object_id.map(factors).fillna(1)
    test["candidate_alert"] = apply_policy(test, episodes, "mean_retarget", policy)
    scores = EventEvaluator(test, episodes, 1).evaluate(test.candidate_alert, 0.5, 1)
    test.to_parquet(directory / "evaluated.parquet", index=False)
    original = read(base / "access.json")["scores"]["pending"]
    if scores["eligible_episodes"] != original["eligible_episodes"]:
        raise ValueError("Changed episode denominator")
    reference_path = output / fold / "uncorrected.json"
    if reference_path.exists():
        reference = read(reference_path)
    else:
        reference_pred = pd.read_parquet(base / "access-policy.parquet")
        reference_policy, _, _ = select_policy(reference_pred, episodes, "mean_retarget")
        reference_test = pd.read_parquet(base / "access-test.parquet")
        alerts = apply_policy(reference_test, episodes, "mean_retarget", reference_policy)
        reference = {
            "policy": reference_policy,
            "scores": EventEvaluator(reference_test, episodes, 1).evaluate(alerts, 0.5, 1),
        }
        write_json(reference_path, reference)
    result = {
        "fold": fold,
        "strength": strength,
        "policy": policy,
        "scores": scores,
        "reference": reference["scores"],
        "original": original,
    }
    write_json(path, result)
    print("DONE", fold, strength, scores, flush=True)
    return result


def run(output, stage):
    sources = [PROCESSED / "features.parquet", PROCESSED / "episodes.parquet"]
    for fold in (*FOLDS, *STRESS):
        sources.extend(
            base_directory(fold) / n
            for n in ("access.json", "access.cbm", "access-policy.parquet", "access-test.parquet")
        )
    code = [
        Path(__file__),
        Path(select_policy.__code__.co_filename),
        Path(episode_counts.__code__.co_filename),
        Path(periods_for.__code__.co_filename),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(model_input.__code__.co_filename),
        Path(mask.__code__.co_filename),
    ]
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in code},
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Changed study inputs or implementation; use a new output directory")
    else:
        write_json(path, plan)
    frame = pd.read_parquet(
        PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
    )
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet",
        filters=[("kind", "==", "access"), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
    )
    if stage == "screen":
        for fold in ("screen_1", "screen_2"):
            for strength in STRENGTHS:
                evaluate(output, fold, strength, frame, episodes)
        return
    screen = {
        s: [read(output / f / f"strength_{s}" / "result.json") for f in ("screen_1", "screen_2")]
        for s in STRENGTHS
    }
    candidates = {s: pooled([r["scores"] for r in rows]) for s, rows in screen.items()}
    selected = max(candidates, key=lambda s: (primary_score(candidates[s]), candidates[s]["f1"]))
    reference = pooled([r["reference"] for r in screen[selected]])
    passed = (
        primary_score(candidates[selected]) > 1.05 * primary_score(reference)
        and candidates[selected]["f1"] >= reference["f1"]
    )
    selection = {
        "selected_strength": selected,
        "candidates": candidates,
        "reference": reference,
        "passed_screen": passed,
    }
    write_json(output / "selection.json", selection)
    print("SELECTION", selection, flush=True)
    if not passed:
        return
    results = [evaluate(output, f, selected, frame, episodes) for f in ("confirmation", *STRESS)]
    may = results[0]
    candidate_extra = pooled([r["scores"] for r in results[1:]])
    reference_extra = pooled([r["reference"] for r in results[1:]])
    passed_may = (
        primary_score(may["scores"]) > primary_score(may["reference"])
        and may["scores"]["f1"] >= 0.95 * may["reference"]["f1"]
    )
    passed_extra = (
        primary_score(candidate_extra) > 1.05 * primary_score(reference_extra)
        and candidate_extra["f1"] >= reference_extra["f1"]
        and all(r["scores"]["f1"] >= 0.9 * r["reference"]["f1"] for r in results[1:])
    )
    all_rows = screen[selected] + results
    report = {
        "selection": selection,
        "periods": all_rows,
        "five_period_pooled": pooled([r["scores"] for r in all_rows]),
        "five_period_reference": pooled([r["reference"] for r in all_rows]),
        "passed_may": passed_may,
        "passed_stress": passed_extra,
        "research_eligible": bool(passed_may and passed_extra),
        "automatic_activation": False,
        "new_blind_test": False,
    }
    write_json(output / "report.json", report)
    print("REPORT", {k: v for k, v in report.items() if k != "periods"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v12"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
