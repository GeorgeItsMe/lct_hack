"""v33: channel-aware count models at causal telemetry-triggered minute times."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.binary_gate_research import KINDS, select_gate_policy
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.count_extension_research import validate_opportunities
from moscollector.experiments.fine_cadence_research import (
    FinePendingSimulator,
    cohort,
    evaluator_for,
    policy_alerts,
)
from moscollector.experiments.fresh_counts_research import anchor, source_files
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.onset_channel_features import CATS, COLUMNS, FOLDER, KEYS, channel_context
from moscollector.experiments.onset_policy import select_policy
from moscollector.experiments.onset_training_data import (
    attach,
    augmented_slots,
    model_input,
    snapshot_weights,
)
from moscollector.experiments.peer_context_research import FIT, control_source
from moscollector.experiments.quarter_count_features import carry_features
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated

ARMS = ("onset_candidate", "minute_control", "quarter_control")
REFERENCES = (*ARMS[1:], "reference")
PLAN = {
    "scope": "adaptive_retrospective_channel_onset_count_study_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "hypothesis": "v32 raw fire-trigger rules increase access to short fault precursors but remain inaccurate. Test learned channel-specific context for all six signal families and all three supported prediction heads, on next-minute onset opportunities. No change to target episodes or evaluation horizon.",
    "features": "Original66/94/144 inputs held at source snapshot; add60 strictly-past onset context columns and source age. Per fault/unknown/power/fire/access/conflict: latest channelID and catalog sensor type (12categoricals), age, its preceding same-signal gap, its1/24h counts, and object's1min/1/6/24h counts. Latest channel valid30d; missing is explicit. No episode endpoints/membership or future values. Tied IDs use deterministic ordering, not physical order; no onset is not proof of healthy coverage.",
    "training": FIT,
    "training_protocol": "One candidate per kind and fold. Retain original eligible3h train/validation anchors and25h purge. Add all next-strict-minute onset times only between adjacent eligible anchors, no gap or final-interval extension. Same original24h episode-count target recomputed at each time, original binary target parity asserted. Each original anchor group retains weight1: anchor .5 and companions share .5, or anchor1 if no companions. Weighted Poisson validation earlystop; no new hyperparameter search. Correlated augmented windows are not additional independent incidents; weighted mass does not freeze tree borders or categorical statistics.",
    "inference": "Retain original quarter-hour grid and add next-strict-minute triggers between adjacent eligible hourly snapshots. Same exact episode identities and original hourly object-day exposure. Candidate uses fresh context at query and held original features. Matched minute control uses frozen old CatBoost logcounts held from the same hourly source. Quarter control replays original quarter grid and old weights. Minimum cooldown1min on minute grid,15min on quarter grid; one alert per opportunity.",
    "calibration": "Each arm independently fits sigmoid and mean scale on actual24h labels/counts at its own preceding calibration times. Calibration is per opportunity, not an estimate of independent incidents. Same original periods, no threshold or calibration fit on test periods.",
    "policy": "Identical456 gate/capacity/margin choices and objective asv24. Original hourly exposure budget. Causal outcome availability floor(start+70min to hour)+1hour; one confirmation clears at most one pending warning,24h expiry. Compare candidate with BOTH matched minute-grid control and old quarter-grid control, plus historical anchor. No attribution of grid effects to learned weights.",
    "screen": "Nov/Feb pooled candidate primary min(P/.9,R/.9,1) improves>5% over all three references with noF1 loss. Only passing kinds reach additional periods; no substitutions.",
    "confirmation": "May improves primary withF1>=95% of each reference. Dec/Mar pooled primary improves>5%, pooledF1 no worse, each month'sF1>=90% of each reference. Research gates are not full90/90 achievement or automatic activation.",
    "limits": "No June selection/evaluation and no new labels. Historical months adaptively reused. Grouped sensor episodes are proxy labels, not verified physical incidents. Flood unsupported. This study alone cannot establish future deployment quality.",
}


def fit(root, kind, fold, frame, dense, context, triggers, episodes):
    directory = root / kind / fold / "onset"
    weights, target = directory / "model.cbm", directory / "fit.json"
    old_weights, old_metadata, _ = source_files(kind, fold)
    old = read(old_metadata)
    columns = [*old["features"], *COLUMNS, "evt_source_age_h"]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate onset feature")
    signature = {
        "plan_sha256": sha256(root / "plan.json"),
        "kind": kind,
        "fold": fold,
        "features": columns,
        "periods": old["periods"],
        "fit": FIT,
    }
    if target.exists():
        meta = read(target)
        if meta["signature"] != signature or sha256(weights) != meta["model_sha256"]:
            raise ValueError("Onset model inputs or weights changed")
        return meta
    periods = {name: tuple(map(pd.Timestamp, dates)) for name, dates in old["periods"].items()}
    validate_opportunities(frame, dense, periods)
    pools, sizes = {}, {}
    for name in ("train", "validation"):
        base = frame.loc[mask(frame, *periods[name])]
        original_counts = episode_counts(base, episodes)
        if not np.array_equal(original_counts > 0, base[f"target_{kind}"].to_numpy().astype(bool)):
            raise ValueError("Original onset target differs from archived binary label")
        slots = augmented_slots(base, triggers)
        if not (slots.as_of + pd.Timedelta(hours=25)).lt(periods[name][1]).all():
            raise ValueError("Augmented target crosses split purge")
        rows = attach(base, slots, context, old["features"])
        counts, weight = episode_counts(rows, episodes), snapshot_weights(slots)
        if cohort(rows, episodes, 1 / 60) != cohort(base, episodes, 3):
            raise ValueError("Training augmentation changes event cohort")
        np.testing.assert_allclose(weight.sum(), len(base), rtol=0, atol=1e-7)
        pools[name] = Pool(
            model_input(rows, columns), counts, cat_features=[*CATEGORICAL, *CATS], weight=weight
        )
        sizes[name] = {
            "rows": len(rows),
            "original_rows": len(base),
            "weight_sum": float(weight.sum()),
            "positive_rows": int(np.sum(counts > 0)),
            "weighted_positive_rows": float(weight[counts > 0].sum()),
            "eligible_episodes": EventEvaluator(base, episodes, 3).events,
        }
        del rows, base, slots
        gc.collect()
    directory.mkdir(parents=True, exist_ok=True)
    model = CatBoostRegressor(**FIT, verbose=200)
    print("START onset fit", kind, fold, len(columns), sizes, flush=True)
    model.fit(pools["train"], eval_set=pools["validation"])
    temporary = weights.with_suffix(".cbm.tmp")
    model.save_model(str(temporary))
    temporary.replace(weights)
    meta = {
        "signature": signature,
        "features": columns,
        "periods": old["periods"],
        "sizes": sizes,
        "best_iteration": model.best_iteration_,
        "model_sha256": sha256(weights),
        "original_model_sha256": sha256(old_weights),
        "original_target_parity": True,
        "same_training_episode_cohort": True,
        "added_columns": [*COLUMNS, "evt_source_age_h"],
    }
    write_json(target, meta)
    print("DONE onset fit", kind, fold, "best", model.best_iteration_, flush=True)
    return meta


def verify_quarter_control(source, policy, scores, predictions):
    if not (source / "result.json").exists():
        return False
    before = read(source / "result.json")["arms"]["global_control"]
    if policy != before["policy"] or scores != before["scores"]:
        raise ValueError("Archived quarter control metrics changed")
    for period in ("policy", "test"):
        saved = pd.read_parquet(source / f"global_control-{period}.parquet")
        current = predictions[period]
        pd.testing.assert_frame_equal(current[KEYS], saved[KEYS])
        for column in ("probability", "expected_count"):
            np.testing.assert_allclose(current[column], saved[column], rtol=1e-10, atol=1e-10)
        if period == "test":
            np.testing.assert_array_equal(current.alert, saved.alert)
    return True


def evaluate(root, kind, fold, frame, dense, context, triggers, episodes):
    directory = root / kind / fold
    target = directory / "result.json"
    meta = fit(root, kind, fold, frame, dense, context, triggers, episodes)
    if target.exists():
        result = read(target)
        if result["fit"] != meta:
            raise ValueError("Cached onset result differs from verified fit")
        return result
    old_weights, old_metadata, _ = source_files(kind, fold)
    old = read(old_metadata)
    candidate, control = CatBoostRegressor(), CatBoostRegressor()
    candidate.load_model(str(directory / "onset/model.cbm"))
    control.load_model(str(old_weights))
    tables, exposure, sizes = {name: {} for name in ARMS}, {}, {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, old["periods"][period]))
        reference = frame.loc[mask(frame, *dates), KEYS]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        slots = augmented_slots(rows, triggers, spacing_hours=1, retain_quarters=True)
        features = attach(rows, slots, context, old["features"])
        hourly = rows[KEYS].copy()
        hourly["raw"] = control.predict(
            model_input(rows, old["features"]), prediction_type="RawFormulaVal", thread_count=2
        )
        new = slots[KEYS].copy()
        new["raw"] = candidate.predict(
            model_input(features, meta["features"]), prediction_type="RawFormulaVal", thread_count=2
        )
        tables["onset_candidate"][period] = new
        tables["minute_control"][period] = carry_features(hourly, slots[KEYS], ["raw"])
        tables["quarter_control"][period] = carry_features(
            hourly, subdivide_evaluation_slots(hourly), ["raw"]
        )
        for name in ARMS:
            pred = tables[name][period]
            cadence = 0.25 if name == "quarter_control" else 1 / 60
            if not np.isfinite(pred.raw).all():
                raise ValueError("Nonfinite onset forecast")
            if cohort(pred, episodes, cadence) != cohort(hourly, episodes, 1):
                raise ValueError("Onset opportunities change episode cohort")
        pd.testing.assert_frame_equal(
            tables["onset_candidate"][period][KEYS], tables["minute_control"][period][KEYS]
        )
        exposure[period] = len(hourly) / 24
        sizes[period] = {
            "hourly": len(hourly),
            "minute": len(slots),
            "quarter": len(tables["quarter_control"][period]),
        }
        del features, rows
        gc.collect()
    arms, archived_parity = {}, False
    for name in ARMS:
        cadence = 0.25 if name == "quarter_control" else 1 / 60
        cal = tables[name]["calibration"]
        counts = episode_counts(cal, episodes)
        calibration = calibrate(cal.raw.to_numpy(), counts > 0)
        scale = float(counts.sum() / np.exp(np.clip(cal.raw, -20, 20)).sum())
        predictions = {
            period: pred.assign(
                probability=calibrated(pred.raw, calibration),
                expected_count=np.exp(np.clip(pred.raw, -20, 20)) * scale,
            )
            for period, pred in tables[name].items()
        }
        print("START onset policy", kind, fold, name, sizes, flush=True)
        policy, frontier = select_policy(predictions["policy"], episodes, exposure["policy"], cadence)
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, policy)
        scores = evaluator_for(test, episodes, cadence, exposure["test"]).evaluate(test.alert, 0.5, cadence)
        if name == "quarter_control":
            archived_parity = verify_quarter_control(control_source(kind, fold), policy, scores, predictions)
        for period, pred in predictions.items():
            pred.to_parquet(directory / f"{name}-{period}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {
            "scores": scores,
            "policy": policy,
            "calibration": calibration,
            "rate_scale": scale,
            "cadence_hours": cadence,
        }
        print("DONE onset policy", kind, fold, name, scores, flush=True)
    reference = anchor(kind, fold)
    if not all(arm["scores"]["eligible_episodes"] == reference["eligible_episodes"] for arm in arms.values()):
        raise ValueError("Reference cohort denominator changed")
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "reference": reference,
        "fit": meta,
        "same_episode_cohort": True,
        "original_hourly_exposure": exposure,
        "opportunities": sizes,
        "archived_quarter_control_parity": archived_parity,
    }
    write_json(target, result)
    return result


def lock_plan(root):
    build = read(FOLDER / "build.json")
    for category in ("inputs", "outputs", "code_hashes"):
        for source, digest in build[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed onset feature build: {source}")
    sources = {
        FOLDER / "build.json",
        PROCESSED / "episodes.parquet",
        Path("artifacts/research-v31/report.json"),
        Path("artifacts/research-v32/report.json"),
        *(Path(p) for category in ("inputs", "outputs") for p in build[category]),
    }
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            weights, metadata, saved = source_files(kind, fold)
            sources.update((weights, metadata, *saved.values()))
            source = control_source(kind, fold)
            if (source / "result.json").exists():
                sources.update(
                    (
                        source / "result.json",
                        *(source / f"global_control-{period}.parquet" for period in ("policy", "test")),
                    )
                )
            sources.add(
                Path("artifacts/research-v20/access") / fold / "result.json"
                if kind == "access"
                else Path("artifacts/research-v19/fault") / fold / "result.json"
                if kind == "fault"
                else Path("artifacts/research-v13-policy") / f"fire-{fold}.json"
            )
    code = {
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (
                channel_context,
                attach,
                augmented_slots,
                model_input,
                snapshot_weights,
                select_policy,
                EventEvaluator.__init__,
                FinePendingSimulator.__init__,
                FinePendingSimulator.alerts,
                select_gate_policy,
                evaluator_for,
                policy_alerts,
                source_files,
                anchor,
                control_source,
                align_opportunities,
                episode_counts,
                mask,
                calibrate,
                calibrated,
                carry_features,
                subdivide_evaluation_slots,
                validate_opportunities,
                primary_score,
            )
        ),
    }
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
                "code_hashes": {str(p): sha256(p) for p in sorted(code)},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    if target.exists() and read(target) != plan:
        raise ValueError("Onset study inputs/code changed; use a new output directory")
    if not target.exists():
        write_json(target, plan)


def metric(row, name):
    return row["reference"] if name == "reference" else row["arms"][name]["scores"]


def run(root, stage):
    lock_plan(root)
    frame, dense = (
        pd.read_parquet(PROCESSED / name)
        for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    context = pd.read_parquet(FOLDER / "context.parquet", read_dictionary=CATS)
    triggers = pd.read_parquet(FOLDER / "triggers.parquet")
    if not all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense, context, triggers)):
        raise ValueError("Post-May onset study inputs")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            rows = [
                evaluate(
                    root, kind, fold, frame, dense, context, triggers, episodes.loc[episodes.kind.eq(kind)]
                )
                for fold in ("screen_1", "screen_2")
            ]
            scores = {name: pooled([metric(row, name) for row in rows]) for name in (*ARMS, "reference")}
            c = scores["onset_candidate"]
            selection[kind] = {
                **scores,
                "passed_screen": all(
                    primary_score(c) > 1.05 * primary_score(scores[name]) and c["f1"] >= scores[name]["f1"]
                    for name in REFERENCES
                ),
            }
            write_json(root / "selection.json", selection)
            print("SELECT onset", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete all onset screening before confirmation")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        rows = [
            evaluate(root, kind, fold, frame, dense, context, triggers, episodes.loc[episodes.kind.eq(kind)])
            for fold in ("confirmation", *STRESS)
        ]
        may, stress = (
            metric(rows[0], "onset_candidate"),
            pooled([metric(row, "onset_candidate") for row in rows[1:]]),
        )
        passed_may = all(
            primary_score(may) > primary_score(metric(rows[0], name))
            and may["f1"] >= 0.95 * metric(rows[0], name)["f1"]
            for name in REFERENCES
        )
        passed_stress = all(
            primary_score(stress) > 1.05 * primary_score(pooled([metric(row, name) for row in rows[1:]]))
            and stress["f1"] >= pooled([metric(row, name) for row in rows[1:]])["f1"]
            and all(metric(row, "onset_candidate")["f1"] >= 0.9 * metric(row, name)["f1"] for row in rows[1:])
            for name in REFERENCES
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": {
                name: pooled([metric(row, name) for row in rows]) for name in (*ARMS, "reference")
            },
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT onset", kind, report[kind]["five_period_pooled"], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v33"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
