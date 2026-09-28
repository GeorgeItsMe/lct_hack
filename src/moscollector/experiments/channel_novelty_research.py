"""v18: isolate channel-relative features in two rare-event classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort
from moscollector.count_research import episode_counts
from moscollector.experiments.channel_novelty_features import COLUMNS, transform
from moscollector.experiments.count_extension_research import validate_opportunities
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.waiting_time_research import (
    base_directory,
    reference_result,
    threshold_alerts,
    threshold_policy,
)
from moscollector.paths import PROCESSED
from moscollector.precision_research import columns_for, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

KINDS = ("fault", "fire")
CONFIGS = ("reference", "channel_novelty")
PLAN = {
    "scope": "Adaptive retrospective feature study for fault/fire; success in these heads does not establish the full solution goal. Original episodes and24h target unchanged. No June evaluation or selection.",
    "goal": {"precision": 0.9, "recall": 0.9},
    "hypothesis": "Unusual behavior of a particular channel is diluted by aggregate object counters. Compare each channel to its own disjoint past baseline before taking object-level summaries.",
    "features": "50numeric additions: per-channel onsets of fault/unknown/power/fire/access/conflict states; recent6/24h channel counts, newly appearing channels vs rest of30d, maximum counts/excess/smoothed log rate ratios and concentration, plus counts of observed/established channels. Established means first observed>=30d ago, not continuous coverage. No episode outcome, duration or end is used in new features. Initial observed state is not an onset. Cross-year state carry.",
    "one_factor": "Compare identical freshly fitted classifiers with original94 features vs144 including the50new numeric features. Same train rows, splits, loss, calibration and policy. Dynamic channel ID is not a categorical predictor in this study.",
    "fit": "v6 recent-reference CatBoostClassifier: Logloss, early stop PRAUC patience100, max1000,depth6,l2=8,lr=.04,seed42,CPU4. Train14mo ending test-2mo; next1mo validation, next14d calibration, remainder policy;25h purge each boundary. Same original binary24h target, verify against episode counts.",
    "policy": "Same v15 threshold_policy grid under90/90: preceding-policy selection only, cooldown1/2/3/6/12/24h, FP budget.25/object/day, >=10alerts/episodes support flags. Hourly opportunities, original one-to-one matching and denominator.",
    "anchor": "Additionally compare to existing count24 family with90/90 policy (v13). This is a historical research anchor, not the deployed classifiers. Improving a weak binary reference alone is insufficient.",
    "screen": "Pool Nov/Feb. Candidate primary>1.05*binary reference and F1>=binary reference; primary>count anchor and F1>=count anchor. Only passing kinds proceed to May/Dec/Mar.",
    "confirmation": "May primary exceeds both references and F1>=95%each. Pool Dec/Mar: primary>1.05*each reference and F1>=each; each monthF1>=90%each reference. No automatic activation or goal completion from intermediate gates.",
    "external_data": "None. Provided telemetry only. Existing v17 canonical last-state caches supply cross-year seeds; full provenance verified.",
}


def fit(frame, dense, episodes, folder, kind, fold, config):
    meta_path = folder / "fit.json"
    if meta_path.exists():
        meta = read(meta_path)
        if sha256(folder / "model.cbm") != meta["model_sha256"]:
            raise ValueError("Cached v18 weights changed")
        return meta
    periods = periods_for({**FOLDS, **STRESS}[fold], kind)
    validate_opportunities(frame, dense, periods)
    used = np.logical_or.reduce([mask(frame, *periods[k]) for k in ("train", "validation", "calibration")])
    training = frame.loc[used].reset_index(drop=True)
    masks = {k: mask(training, *periods[k]) for k in ("train", "validation", "calibration")}
    columns = columns_for(training.drop(columns=COLUMNS), "recent_reference", kind)
    if config == "channel_novelty":
        columns += COLUMNS
    y = training[f"target_{kind}"].to_numpy()
    if not np.array_equal(episode_counts(training, episodes) > 0, y.astype(bool)):
        raise ValueError("Original label parity failed")
    x = model_input(training, columns)
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        l2_leaf_reg=8,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=42,
        thread_count=4,
        cat_features=CATEGORICAL,
        allow_writing_files=False,
        early_stopping_rounds=100,
        verbose=200,
    )
    print("START channel novelty", kind, fold, config, int(masks["train"].sum()), len(columns), flush=True)
    model.fit(x[masks["train"]], y[masks["train"]], eval_set=(x[masks["validation"]], y[masks["validation"]]))
    raw = model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite calibration values")
    calibration = calibrate(raw, y[masks["calibration"]])
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(str(folder / "model.cbm"))
    for period in ("policy", "test"):
        original = frame.loc[mask(frame, *periods[period]), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *periods[period])], original)
        pred = rows[["object_id", "as_of"]].copy()
        raw = model.predict(model_input(rows, columns), prediction_type="RawFormulaVal")
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite predictions")
        pred["probability"] = calibrated(raw, calibration)
        anchor = pd.read_parquet(base_directory(kind, fold) / f"{kind}-{period}.parquet")
        keys = ["object_id", "as_of"]
        pd.testing.assert_frame_equal(
            pred[keys].sort_values(keys).reset_index(drop=True),
            anchor[keys].sort_values(keys).reset_index(drop=True),
        )
        assert_same_episode_cohort(pred, original, episodes)
        pred.to_parquet(folder / f"{period}.parquet", index=False)
    meta = {
        "kind": kind,
        "fold": fold,
        "config": config,
        "features": columns,
        "training_rows": int(masks["train"].sum()),
        "best_iteration": model.best_iteration_,
        "calibration": calibration,
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "original_target_parity": True,
        "model_sha256": sha256(folder / "model.cbm"),
    }
    write_json(meta_path, meta)
    return meta


def evaluate(root, kind, fold, frame, dense, episodes):
    results = {}
    for config in CONFIGS:
        folder = root / kind / fold / config
        path = folder / "result.json"
        if path.exists():
            results[config] = read(path)
            continue
        fit(frame, dense, episodes, folder, kind, fold, config)
        policy_pred = pd.read_parquet(folder / "policy.parquet")
        policy, options = threshold_policy(policy_pred, episodes)
        write_json(folder / "policy.json", {"selected": policy, "options": options})
        test = pd.read_parquet(folder / "test.parquet")
        test["candidate_alert"] = threshold_alerts(test, policy)
        scores = EventEvaluator(test, episodes, 1).evaluate(test.candidate_alert, 0.5, 1)
        anchor = reference_result(kind, fold)
        assert scores["eligible_episodes"] == anchor["eligible_episodes"]
        test.to_parquet(folder / "evaluated.parquet", index=False)
        results[config] = {
            "kind": kind,
            "fold": fold,
            "config": config,
            "policy": policy,
            "scores": scores,
            "count_anchor": anchor,
        }
        write_json(path, results[config])
        print("DONE channel novelty", kind, fold, config, scores, flush=True)
    return results


def summarize(rows):
    return {
        "candidate": pooled([r["channel_novelty"]["scores"] for r in rows]),
        "reference": pooled([r["reference"]["scores"] for r in rows]),
        "count_anchor": pooled([r["reference"]["count_anchor"] for r in rows]),
    }


def run(root, stage):
    sources = [
        PROCESSED / n
        for n in (
            "features-channel-novelty.parquet",
            "features-dense-channel-novelty.parquet",
            "episodes.parquet",
        )
    ]
    sources.append(PROCESSED / "channel-novelty-v18" / "build.json")
    sources.extend(sorted((PROCESSED / "channel-novelty-v18").glob("onsets-*.json")))
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            sources.append(Path("artifacts/research-v13-policy") / f"{kind}-{fold}.json")
            sources.extend(base_directory(kind, fold) / f"{kind}-{p}.parquet" for p in ("policy", "test"))
    code = [
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (
                transform,
                columns_for,
                calibrate,
                mask,
                threshold_policy,
                threshold_alerts,
                align_opportunities,
                validate_opportunities,
                episode_counts,
                EventEvaluator.__init__,
            )
        ),
    ]
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "source_hashes": {str(p): sha256(p) for p in sources},
                "code_hashes": {str(p): sha256(p) for p in code},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Frozen v18 plan changed; use another study directory")
    if not path.exists():
        write_json(path, plan)
    frame = pd.read_parquet(sources[0], filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
    dense = pd.read_parquet(sources[1], columns=frame.columns.tolist())
    all_eps = pd.read_parquet(sources[2], filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            eps = all_eps[all_eps.kind.eq(kind)]
            rows = [evaluate(root, kind, fold, frame, dense, eps) for fold in ("screen_1", "screen_2")]
            summary = summarize(rows)
            c, b, a = (summary[k] for k in ("candidate", "reference", "count_anchor"))
            summary["passed_screen"] = (
                primary_score(c) > 1.05 * primary_score(b)
                and c["f1"] >= b["f1"]
                and primary_score(c) > primary_score(a)
                and c["f1"] >= a["f1"]
            )
            selection[kind] = summary
            write_json(root / "selection.json", selection)
            print("SELECT channel novelty", kind, summary, flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete screening both kinds first")
    report = {}
    for kind, chosen in selection.items():
        if not chosen["passed_screen"]:
            report[kind] = {"selection": chosen, "status": "screen_failed", "research_eligible": False}
            continue
        eps = all_eps[all_eps.kind.eq(kind)]
        rows = [evaluate(root, kind, fold, frame, dense, eps) for fold in ("confirmation", *STRESS)]
        may, stress = summarize(rows[:1]), summarize(rows[1:])
        passed_may = all(
            primary_score(may["candidate"]) > primary_score(may[k])
            and may["candidate"]["f1"] >= 0.95 * may[k]["f1"]
            for k in ("reference", "count_anchor")
        )
        passed_stress = all(
            primary_score(stress["candidate"]) > 1.05 * primary_score(stress[k])
            and stress["candidate"]["f1"] >= stress[k]["f1"]
            for k in ("reference", "count_anchor")
        )
        for row in rows[1:]:
            c = row["channel_novelty"]["scores"]
            passed_stress &= all(
                c["f1"] >= 0.9 * b["f1"]
                for b in (row["reference"]["scores"], row["reference"]["count_anchor"])
            )
        first = [
            {config: read(root / kind / fold / config / "result.json") for config in CONFIGS}
            for fold in ("screen_1", "screen_2")
        ]
        report[kind] = {
            "selection": chosen,
            "periods": first + rows,
            "may": may,
            "stress": stress,
            "five_period_pooled": summarize(first + rows),
            "passed_may": bool(passed_may),
            "passed_stress": bool(passed_stress),
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print(
            "REPORT channel novelty",
            kind,
            {k: v for k, v in report[kind].items() if k != "periods"},
            flush=True,
        )
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v18"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
