"""v26: fixed catalog object-kind experts with a matched calibration control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.binary_gate_research import KINDS, select_gate_policy
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.experiments.fresh_counts_research import anchor, source_files
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.quarter_count_features import carry_features
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

GROUPS = ("controlHouse", "guardObject")
REFERENCES = ("global_control", "calibration_control", "reference")
PLAN = {
    "scope": "adaptive_retrospective_fixed_object_kind_experts_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "groups": GROUPS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "motivation": "Audit of v20 access finds118/923 events in controlHouse versus5294/6652 in guardObject. These are already-known catalog types present in original inputs; neither group may be excluded. Test separate count regressors to avoid learning being dominated by recurrent guard-object events.",
    "fit": "Original3h rows, same features and periods as v9 access/fire or v19 fault. One Poisson CatBoost per fixed object kind, depth6,l2=8,lr=.04,max1000,earlystop100,seed42,CPU4. Verify count>0 against original binary labels. Require>=20 distinct eligible training episodes and>=5 validation episodes plus nonconstant training counts; otherwise use frozen global count weights for that group and explicitly report fallback. No group chosen by test outcome.",
    "arms": "Global control: frozen original weights and global calibration. Calibration control: same global weights but separate calibration for each catalog kind. Candidate: separate group weights and the same group-calibration procedure. All use a single shared policy per incident kind, not separate tuned policies per group.",
    "calibration": "Original independent calibration period,25h purge, actual24h count labels at quarter times. Each arm fits global sigmoid(raw) and sum(count)/sum(exp(raw)) scaling. Group calibration requires>=5 distinct eligible episodes and both binary classes; otherwise falls back to that arm's global calibration. No cap or test-based correction. Correlated calibration rows are not independent incidents.",
    "inference": "Held whole-hour inputs on v20 quarter grid, no v21 fresh inputs or v23 augmentation. Unknown/untrained object kinds use original global weights. All original observations/events retained. Same confirmation,24h matching/pending expiry and original hourly exposure.",
    "policy": "Identical v24 expanded456-policy grid and min(P/.9,R/.9,1), thenF1,R,P; preceding policy period only,FP<=.25/objectday,support>=10alerts/episodes. Global control must exactly replay v24 screening predictions, policies and scores. Report subgroup counts descriptively, never substitute them for full-scope performance.",
    "screen": "Nov/Feb pooled candidate primary>1.05*each of global control, calibration control and historical anchor, with noF1 loss against any. Only passing kinds continue; no substitution of a winning control as the candidate.",
    "confirmation": "May improves primary withF1>=95% each reference. Pooled Dec/Mar primary improves>5%, noF1 loss, each month'sF1>=90% each. These research gates neither establish90/90 nor authorize automatic activation.",
    "june": "No June labels/selection/evaluation. Adaptive reuse of earlier months, not independent future validation. Flood remains unsupported and in full solution scope.",
}


def route_raw(global_raw, groups, overrides):
    result, groups = np.asarray(global_raw, dtype=float).copy(), np.asarray(groups)
    if result.ndim != 1 or groups.shape != result.shape or not np.isfinite(result).all():
        raise ValueError("Invalid global predictions or group vector")
    for group, values in overrides.items():
        selected = groups == group
        values = np.asarray(values, dtype=float)
        if values.shape != (int(selected.sum()),) or not np.isfinite(values).all():
            raise ValueError("Group predictions must preserve positional row coverage")
        result[selected] = values
    return result


def calibration_pair(raw, counts):
    raw, counts = np.asarray(raw, dtype=float), np.asarray(counts, dtype=float)
    if (
        raw.ndim != 1
        or raw.shape != counts.shape
        or not len(raw)
        or not np.isfinite(raw).all()
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
    ):
        raise ValueError("Invalid calibration data")
    return {
        "binary": calibrate(raw, counts > 0),
        "scale": float(counts.sum() / np.exp(np.clip(raw, -20, 20)).sum()),
    }


def apply_calibration(raw, groups, global_calibration, group_calibrations):
    raw, groups = np.asarray(raw, dtype=float), np.asarray(groups)
    if raw.ndim != 1 or raw.shape != groups.shape or not np.isfinite(raw).all():
        raise ValueError("Invalid forecast data")
    probability = calibrated(raw, global_calibration["binary"])
    expected = np.exp(np.clip(raw, -20, 20)) * global_calibration["scale"]
    for group, cal in group_calibrations.items():
        selected = groups == group
        probability[selected] = calibrated(raw[selected], cal["binary"])
        expected[selected] = np.exp(np.clip(raw[selected], -20, 20)) * cal["scale"]
    if not np.isfinite(probability).all() or not np.isfinite(expected).all() or np.any(expected < 0):
        raise ValueError("Invalid calibrated forecasts")
    return probability, expected


def fit_group(directory, frame, episodes, kind, fold, group):
    output = directory / "fit.json"
    if output.exists():
        meta = read(output)
        if sha256(Path(meta["weights_file"])) != meta["model_sha256"]:
            raise ValueError("Group weights changed")
        return meta
    weights, metadata, _ = source_files(kind, fold)
    old = read(metadata)
    sets, sizes = {}, {}
    for name in ("train", "validation"):
        rows = frame.loc[mask(frame, *map(pd.Timestamp, old["periods"][name])) & frame.object_kind.eq(group)]
        counts = episode_counts(rows, episodes)
        if not np.array_equal(counts > 0, rows[f"target_{kind}"].to_numpy().astype(bool)):
            raise ValueError("Original binary target changed")
        sets[name] = (model_input(rows, old["features"]), counts)
        sizes[name] = {
            "rows": len(rows),
            "positive_rows": int(np.sum(counts > 0)),
            "eligible_episodes": EventEvaluator(rows, episodes, 3).events,
        }
    enough = (
        sizes["train"]["eligible_episodes"] >= 20
        and sizes["validation"]["eligible_episodes"] >= 5
        and len(np.unique(sets["train"][1])) > 1
    )
    meta = {
        "kind": kind,
        "fold": fold,
        "group": group,
        "features": old["features"],
        "periods": old["periods"],
        "sizes": sizes,
        "mode": "specialist" if enough else "global_fallback",
        "original_target_parity": True,
    }
    directory.mkdir(parents=True, exist_ok=True)
    if enough:
        model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            l2_leaf_reg=8,
            learning_rate=0.04,
            loss_function="Poisson",
            eval_metric="Poisson",
            random_seed=42,
            thread_count=4,
            cat_features=CATEGORICAL,
            allow_writing_files=False,
            early_stopping_rounds=100,
            verbose=200,
        )
        print("START object kind fit", kind, fold, group, sizes, flush=True)
        model.fit(*sets["train"], eval_set=sets["validation"])
        weights = directory / "model.cbm"
        model.save_model(str(weights))
        meta["best_iteration"] = model.best_iteration_
    meta.update(weights_file=str(weights), model_sha256=sha256(weights))
    write_json(output, meta)
    print("DONE object kind fit", kind, fold, group, meta["mode"], flush=True)
    return meta


def evaluate(root, kind, fold, frame, dense, episodes):
    directory, target = root / kind / fold, root / kind / fold / "result.json"
    if target.exists():
        return read(target)
    weights, metadata, _ = source_files(kind, fold)
    old = read(metadata)
    original = CatBoostRegressor()
    original.load_model(str(weights))
    models, fits = {}, {}
    for group in GROUPS:
        fits[group] = fit_group(directory / "groups" / group, frame, episodes, kind, fold, group)
        if fits[group]["mode"] == "specialist":
            model = CatBoostRegressor()
            model.load_model(fits[group]["weights_file"])
            models[group] = model
    tables, exposure, group_exposure = {}, {}, {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, old["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        x = model_input(rows, old["features"])
        hourly = rows[["object_id", "as_of", "object_kind"]].copy()
        hourly["global_raw"] = original.predict(x, prediction_type="RawFormulaVal", thread_count=2)
        overrides = {
            group: model.predict(
                x.loc[rows.object_kind.eq(group)], prediction_type="RawFormulaVal", thread_count=2
            )
            for group, model in models.items()
            if rows.object_kind.eq(group).any()
        }
        hourly["specialist_raw"] = route_raw(hourly.global_raw, hourly.object_kind, overrides)
        pred = carry_features(
            hourly, subdivide_evaluation_slots(hourly), ["object_kind", "global_raw", "specialist_raw"]
        )
        assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
        tables[period], exposure[period] = pred, len(hourly) / 24
        group_exposure[period] = (hourly.groupby("object_kind").size() / 24).to_dict()
    cal = tables["calibration"]
    counts = episode_counts(cal, episodes)
    arms = {}
    for name, raw_column, grouped in (
        ("global_control", "global_raw", False),
        ("calibration_control", "global_raw", True),
        ("specialist_candidate", "specialist_raw", True),
    ):
        global_cal = calibration_pair(cal[raw_column], counts)
        group_cals, support = {}, {}
        if grouped:
            for group in GROUPS:
                selected = cal.object_kind.eq(group).to_numpy()
                n_events = EventEvaluator(cal.loc[selected], episodes, 0.25).events
                supported = n_events >= 5 and len(np.unique(counts[selected] > 0)) == 2
                group_cals[group] = (
                    calibration_pair(cal.loc[selected, raw_column], counts[selected])
                    if supported
                    else global_cal
                )
                support[group] = {
                    "eligible_episodes": n_events,
                    "rows": int(selected.sum()),
                    "status": "group_calibration" if supported else "global_calibration_fallback",
                }
        predictions = {}
        for period, table in tables.items():
            pred = table.copy()
            pred["probability"], pred["expected_count"] = apply_calibration(
                pred[raw_column], pred.object_kind, global_cal, group_cals
            )
            predictions[period] = pred
        print("START object kind policy", kind, fold, name, flush=True)
        policy, frontier = select_gate_policy(predictions["policy"], episodes, exposure["policy"])
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, policy)
        scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
        if name == "global_control" and fold.startswith("screen_"):
            old_directory = Path("artifacts/research-v24") / kind / fold
            saved_result = read(old_directory / "result.json")["arms"]["count_probability_control"]
            assert policy == saved_result["policy"] and scores == saved_result["scores"]
            for period in ("policy", "test"):
                saved = pd.read_parquet(old_directory / f"count_probability_control-{period}.parquet")
                shared = predictions[period].merge(
                    saved, on=["object_id", "as_of"], suffixes=("_new", "_old"), validate="one_to_one"
                )
                assert len(shared) == len(saved) == len(predictions[period])
                for column in ("probability", "expected_count"):
                    np.testing.assert_allclose(
                        shared[f"{column}_new"], shared[f"{column}_old"], atol=1e-10, rtol=1e-10
                    )
        subgroup_scores = {}
        for group, view in test.groupby("object_kind"):
            view = view.reset_index(drop=True)
            subgroup_scores[group] = evaluator_for(
                view, episodes, 0.25, group_exposure["test"][group]
            ).evaluate(view.alert, 0.5, 0.25)
        subgroup_totals = pooled(list(subgroup_scores.values()))
        assert all(
            subgroup_totals[key] == scores[key] for key in ("true_alerts", "alerts", "eligible_episodes")
        )
        for period in ("policy", "test"):
            predictions[period].to_parquet(directory / f"{name}-{period}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {
            "scores": scores,
            "policy": policy,
            "global_calibration": global_cal,
            "group_calibrations": group_cals,
            "calibration_support": support,
            "subgroup_scores": subgroup_scores,
        }
        print("DONE object kind", kind, fold, name, scores, flush=True)
    result = {
        "kind": kind,
        "fold": fold,
        "scores": arms["specialist_candidate"]["scores"],
        "global_control": arms["global_control"]["scores"],
        "calibration_control": arms["calibration_control"]["scores"],
        "reference": anchor(kind, fold),
        "arms": arms,
        "fits": fits,
        "identical_episode_cohort": True,
        "global_control_exact_replay": fold.startswith("screen_"),
        "exposure_days": exposure["test"],
    }
    assert all(
        result["scores"]["eligible_episodes"] == result[key]["eligible_episodes"] for key in REFERENCES
    )
    write_json(target, result)
    return result


def lock_plan(root):
    sources = [
        PROCESSED / "features-channel-novelty.parquet",
        PROCESSED / "features-dense-channel-novelty.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/object_kind_error_audit.json"),
        Path("artifacts/research-v25/report.json"),
    ]
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            weights, metadata, saved = source_files(kind, fold)
            sources.extend([weights, metadata, *saved.values()])
            if fold.startswith("screen_"):
                base = Path("artifacts/research-v24") / kind / fold
                sources.extend(
                    [
                        base / "result.json",
                        *(base / f"count_probability_control-{p}.parquet" for p in ("policy", "test")),
                    ]
                )
            if kind == "access":
                sources.append(Path("artifacts/research-v20/access") / fold / "result.json")
            elif kind == "fault":
                sources.append(Path("artifacts/research-v19/fault") / fold / "result.json")
            else:
                sources.append(Path("artifacts/research-v13-policy") / f"fire-{fold}.json")
    code = {
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (
                EventEvaluator.__init__,
                select_gate_policy,
                evaluator_for,
                policy_alerts,
                episode_counts,
                source_files,
                model_input,
                carry_features,
                align_opportunities,
                subdivide_evaluation_slots,
                primary_score,
                mask,
            )
        ),
    }
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "source_hashes": {str(p): sha256(p) for p in sources},
                "code_hashes": {str(p): sha256(p) for p in sorted(code)},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Study inputs/code changed; use a new study directory")
    else:
        write_json(path, plan)


def run(root, stage):
    lock_plan(root)
    frame = pd.read_parquet(PROCESSED / "features-channel-novelty.parquet")
    dense = pd.read_parquet(PROCESSED / "features-dense-channel-novelty.parquet")
    assert all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense))
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            rows = [
                evaluate(root, kind, fold, frame, dense, episodes[episodes.kind.eq(kind)])
                for fold in ("screen_1", "screen_2")
            ]
            totals = {key: pooled([row[key] for row in rows]) for key in ("scores", *REFERENCES)}
            candidate = totals["scores"]
            selection[kind] = {
                **totals,
                "passed_screen": all(
                    primary_score(candidate) > 1.05 * primary_score(totals[key])
                    and candidate["f1"] >= totals[key]["f1"]
                    for key in REFERENCES
                ),
            }
            write_json(root / "selection.json", selection)
            print("SELECT object kind", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Finish all screening first")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        rows = [
            evaluate(root, kind, fold, frame, dense, episodes[episodes.kind.eq(kind)])
            for fold in ("confirmation", *STRESS)
        ]
        may, stress = rows[0], pooled([row["scores"] for row in rows[1:]])
        passed_may = all(
            primary_score(may["scores"]) > primary_score(may[key])
            and may["scores"]["f1"] >= 0.95 * may[key]["f1"]
            for key in REFERENCES
        )
        passed_stress = all(
            primary_score(stress) > 1.05 * primary_score(pooled([row[key] for row in rows[1:]]))
            and stress["f1"] >= pooled([row[key] for row in rows[1:]])["f1"]
            and all(row["scores"]["f1"] >= 0.9 * row[key]["f1"] for row in rows[1:])
            for key in REFERENCES
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": {
                key: pooled([row[key] for row in rows]) for key in ("scores", *REFERENCES)
            },
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print(
            "REPORT object kind",
            kind,
            {key: value for key, value in report[kind].items() if key != "periods"},
            flush=True,
        )
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v26"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
