"""v17: ordered categorical channel changes added to the same24h count model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort
from moscollector.count_research import episode_counts
from moscollector.experiments.count_extension_research import validate_opportunities
from moscollector.experiments.goal90_research import (
    STRESS,
    apply_policy,
    pooled,
    primary_score,
    read,
    select_policy,
)
from moscollector.experiments.ordered_state_features import AGES, CATS, transform
from moscollector.experiments.waiting_time_research import KINDS, base_directory, reference_result
from moscollector.paths import PROCESSED
from moscollector.precision_research import columns_for, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

PLAN = {
    "scope": "adaptive_retrospective_ordered_sensor_state_study_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": list(KINDS),
    "hypothesis": "Dynamic channel identity and the ordered sequence of state changes contain predictive information lost by object-level numeric summaries.",
    "features": "Original v9 features (66access,94others) plus16categorical channel/transition tokens and8age values for the last4own and last4other-sibling state changes. Strict ts<as_of,30day expiry, cross-year carry, first2022observation marked. Numeric readings share one token but transitions into/out of faults or conflicts persist. No episode labels or durations enter new features. At whole-hour forecasts, keeping last4changes per object/hour preserves exact last4own/sibling history. Simultaneous records use channel-ID order; no within-second temporal-order claim.",
    "fit": "Same v9 original24h Poisson count target, depth6,l2=8,lr=.04,max1000,seed42,CPU4,earlystop100, training windows and25h purges. Only added features change. Separate calibration count scaling and raw->original24h binary sigmoid.",
    "policy": "Same v11 mean_retarget grid and90/90 selection on preceding policy interval; same24h pending expiry,70min event confirmation, one alert/hour. No new thresholds chosen from test labels.",
    "reference": "V9 count24 family with same90/90 policy from v12 uncorrected(access) and v13(fire/fault). Same hourly opportunities and episode cohort. Not a comparison to deployed fire/fault classifiers.",
    "screen": "Single candidate per kind; pooled Nov/Feb must improve primary score>5% and not loseF1. Only passing kinds proceed to May and Dec/Mar.",
    "confirmation": "May improves primary score withF1>=95%reference. Pooled Dec/Mar improves primary score>5%, F1 not lower and each monthF1>=90%reference. No automatic activation;90/90 target achievement reported separately.",
    "june": "Never used for evaluation or selection; all dates are adaptively reused historical periods, not a new blind test.",
    "source": "https://catboost.ai/docs/en/concepts/loss-functions-regression#Poisson",
}


def ordered_input(frame, columns):
    base = [c for c in columns if c not in CATS]
    x = model_input(frame, base)
    for name in CATS:
        if frame[name].isna().any():
            raise ValueError("Missing categorical transition")
        x[name] = frame[name].astype(str)
    return x[columns]


def fit(frame, dense, episodes, folder, kind, fold):
    path = folder / "fit.json"
    if path.exists():
        meta = read(path)
        if sha256(folder / "model.cbm") != meta["model_sha256"]:
            raise ValueError("Cached ordered-state weights changed")
        return meta
    periods = periods_for({**FOLDS, **STRESS}[fold], kind)
    validate_opportunities(frame, dense, periods)
    used = np.logical_or.reduce([mask(frame, *periods[k]) for k in ("train", "validation", "calibration")])
    training = frame.loc[used].reset_index(drop=True)
    masks = {k: mask(training, *periods[k]) for k in ("train", "validation", "calibration")}
    counts = episode_counts(training, episodes)
    y = training[f"target_{kind}"].to_numpy()
    if not np.array_equal(counts > 0, y.astype(bool)):
        raise ValueError("Original24h label parity failed")
    base = training.drop(columns=CATS + AGES)
    columns = columns_for(base, "recent_reference", kind) + CATS + AGES
    x = ordered_input(training, columns)
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        l2_leaf_reg=8,
        learning_rate=0.04,
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        thread_count=4,
        cat_features=CATEGORICAL + CATS,
        allow_writing_files=False,
        early_stopping_rounds=100,
        verbose=200,
    )
    print("START ordered24", kind, fold, int(masks["train"].sum()), flush=True)
    model.fit(
        x[masks["train"]],
        counts[masks["train"]],
        eval_set=(x[masks["validation"]], counts[masks["validation"]]),
    )
    raw = model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite ordered24 calibration")
    calibration = calibrate(raw, y[masks["calibration"]])
    rate_scale = float(counts[masks["calibration"]].sum() / np.exp(np.clip(raw, -20, 20)).sum())
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(str(folder / "model.cbm"))
    for period in ("policy", "test"):
        reference = frame.loc[mask(frame, *periods[period]), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *periods[period])], reference)
        pred = rows[["object_id", "as_of"]].copy()
        raw = model.predict(ordered_input(rows, columns), prediction_type="RawFormulaVal")
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite ordered24 forecast")
        pred["probability"] = calibrated(raw, calibration)
        pred["expected_count"] = np.exp(np.clip(raw, -20, 20)) * rate_scale
        base = pd.read_parquet(base_directory(kind, fold) / f"{kind}-{period}.parquet")
        keys = ["object_id", "as_of"]
        pd.testing.assert_frame_equal(
            pred[keys].sort_values(keys).reset_index(drop=True),
            base[keys].sort_values(keys).reset_index(drop=True),
        )
        assert_same_episode_cohort(pred, reference, episodes)
        pred.to_parquet(folder / f"{period}.parquet", index=False)
    meta = {
        "kind": kind,
        "fold": fold,
        "count_horizon_hours": 24,
        "categorical_features": CATEGORICAL + CATS,
        "probability_horizon_hours": 24,
        "evaluation_horizon_hours": 24,
        "features": columns,
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "calibration": calibration,
        "rate_scale": rate_scale,
        "training_rows": int(masks["train"].sum()),
        "best_iteration": model.best_iteration_,
        "ordered_history_is_past_only": True,
        "original_target_parity": True,
        "model_sha256": sha256(folder / "model.cbm"),
    }
    write_json(path, meta)
    return meta


def evaluate(root, kind, fold, frame, dense, episodes):
    folder = root / kind / fold
    path = folder / "result.json"
    if path.exists():
        return read(path)
    fit(frame, dense, episodes, folder, kind, fold)
    pred = pd.read_parquet(folder / "policy.parquet")
    policy, options, diagnostic = select_policy(pred, episodes, "mean_retarget")
    write_json(folder / "policy.json", {"selected": policy, "options": options, "diagnostic": diagnostic})
    test = pd.read_parquet(folder / "test.parquet")
    test["candidate_alert"] = apply_policy(test, episodes, "mean_retarget", policy)
    scores = EventEvaluator(test, episodes, 1).evaluate(test.candidate_alert, 0.5, 1)
    reference = reference_result(kind, fold)
    assert scores["eligible_episodes"] == reference["eligible_episodes"]
    test.to_parquet(folder / "evaluated.parquet", index=False)
    result = {
        "kind": kind,
        "fold": fold,
        "policy": policy,
        "scores": scores,
        "reference": reference,
        "evaluation_horizon_hours": 24,
    }
    write_json(path, result)
    print("DONE ordered24", kind, fold, scores, flush=True)
    return result


def run(root, stage):
    sources = [
        PROCESSED / n
        for n in ("features-ordered.parquet", "features-dense-ordered.parquet", "episodes.parquet")
    ]
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            sources.extend(
                base_directory(kind, fold) / f"{kind}-{period}.parquet" for period in ("policy", "test")
            )
            sources.append(
                Path("artifacts/research-v12") / fold / "uncorrected.json"
                if kind == "access"
                else Path("artifacts/research-v13-policy") / f"{kind}-{fold}.json"
            )
    sources.append(PROCESSED / "ordered-states-v17" / "build.json")
    code = [
        Path(transform.__code__.co_filename),
        Path(__file__),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(align_opportunities.__code__.co_filename),
        Path(columns_for.__code__.co_filename),
        Path(select_policy.__code__.co_filename),
        Path(calibrate.__code__.co_filename),
        Path(mask.__code__.co_filename),
        Path(validate_opportunities.__code__.co_filename),
        Path(reference_result.__code__.co_filename),
        Path(episode_counts.__code__.co_filename),
    ]
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in code},
    }
    plan = json.loads(json.dumps(plan))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Study inputs or implementation changed; use another directory")
    else:
        write_json(path, plan)
    frame = pd.read_parquet(sources[0], filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
    dense = pd.read_parquet(sources[1], columns=frame.columns.tolist())
    all_episodes = pd.read_parquet(sources[2], filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            eps = all_episodes[all_episodes.kind.eq(kind)]
            rows = [evaluate(root, kind, fold, frame, dense, eps) for fold in ("screen_1", "screen_2")]
            c, r = (pooled([item[key] for item in rows]) for key in ("scores", "reference"))
            selection[kind] = {
                "candidate": c,
                "reference": r,
                "passed_screen": primary_score(c) > 1.05 * primary_score(r) and c["f1"] >= r["f1"],
            }
            write_json(root / "selection.json", selection)
            print("SELECT ordered24", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Finish screening all kinds first")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        eps = all_episodes[all_episodes.kind.eq(kind)]
        rows = [evaluate(root, kind, fold, frame, dense, eps) for fold in ("confirmation", *STRESS)]
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
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": pooled([item["scores"] for item in rows]),
            "five_period_reference": pooled([item["reference"] for item in rows]),
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT ordered24", kind, {k: v for k, v in report[kind].items() if k != "periods"}, flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v17"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
