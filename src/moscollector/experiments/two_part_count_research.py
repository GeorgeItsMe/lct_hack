"""v25: factor the unconditional count mean into occurrence and recurrence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.binary_gate_research import (
    KINDS,
    classifier_cache,
    select_gate_policy,
)
from moscollector.experiments.binary_gate_research import (
    evaluate as evaluate_gate,
)
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.experiments.fresh_counts_research import anchor, source_files
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.quarter_count_features import carry_features
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrated, model_input

REFERENCES = ("binary_control", "count_control", "reference")
PLAN = {
    "scope": "adaptive_retrospective_two_part_conditional_mean_study_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "hypothesis": "V24 separates occurrence probability but retains the old unconditional mean. Factor expected24h count as P(N>0|x)*(1+E[N-1|N>0,x]) to separate absence from recurrence variability.",
    "fit": "Same original3h rows, features and train/validation periods as count anchor. Fit Poisson regression of N-1 on positive-count rows only, depth6,l2=8,lr=.04,max1000,earlystop100,seed42,CPU4. Verify N>0 equals original binary labels. Positive conditioning applies only to training/validation, never to serving or evaluation eligibility. Constant training target uses exact constant; no positive training rows is an error. If validation has no positive rows, fixed500 iterations without early stopping.",
    "probability": "Reuse the exact calibrated binary classifier and15min held-hourly predictions from v24. No new probability fit or calibration. Frozen weight hashes checked. All original evaluation rows retained.",
    "calibration": "On independent original calibration period with25h purge, at actual quarter times: extra_scale=sum(N-1 on N>0)/sum(predicted_extra on N>0). Conditional mean=1+scale*nonnegative_extra, so it stays>=1. No positive calibration rows or zero predicted denominator with positive observations uses unscaled extra and explicit unsupported status; all observed extra zero gives scale0. No unconditional second scaling.",
    "inference": "Predict extra count on every original eligible hourly row, carry to15min points with conservative v20 observation availability. Same24h events, exact cohort, original hourly exposure,25h purge, pending warning expiration and confirmation. No fresh raw inputs/phase augmentation.",
    "loss_limits": "Poisson is only a nonnegative regression loss for extra count. This is factorization of a conditional mean, not an implemented zero-truncated joint likelihood or assertion of a Poisson data process.",
    "controls": "Exact v24 binary-gate and count-probability predictions/policies with its identical456-policy expanded grid, plus historical anchor. Candidate uses the same grid. Rejected v24 is a matched diagnostic control, not a promoted model. Missing additional-month controls are computed under this declared v25 study.",
    "screen": "Nov/Feb pooled min(P/.9,R/.9,1) improves>5% with noF1 loss against ALL three controls. Only passing kinds continue, no variant substitution.",
    "confirmation": "May primary improves withF1>=95% against all. Pooled Dec/Mar primary improves>5%, noF1 loss, each month'sF1>=90% against all. Gates do not prove90/90 or activate production.",
    "june": "Excluded from labels/selection/evaluation. Prior months adaptively reused. Flood unsupported; a single head's gain is not full-solution success.",
}


def positive_extra_targets(counts):
    counts = np.asarray(counts, dtype=float)
    if (
        counts.ndim != 1
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError("Counts must be finite nonnegative integers")
    positive = counts > 0
    return positive, counts[positive] - 1


def extra_calibration(counts, extra):
    positive, target = positive_extra_targets(counts)
    extra = np.asarray(extra, dtype=float)
    if extra.shape != positive.shape or not np.isfinite(extra).all() or np.any(extra < 0):
        raise ValueError("Invalid predicted extra counts")
    observed, predicted = float(target.sum()), float(extra[positive].sum())
    if not positive.any():
        scale, status = 1.0, "unsupported_no_positive_calibration_rows"
    elif observed == 0:
        scale, status = 0.0, "no_observed_recurrence"
    elif predicted == 0:
        scale, status = 1.0, "unsupported_zero_prediction_denominator"
    else:
        scale, status = observed / predicted, "mean_matched_on_positive_rows"
    return {
        "scale": scale,
        "status": status,
        "positive_rows": int(positive.sum()),
        "observed_extra_sum": observed,
        "predicted_extra_sum": predicted,
    }


def factor_mean(probability, extra, scale):
    probability, extra = np.asarray(probability, dtype=float), np.asarray(extra, dtype=float)
    if (
        probability.shape != extra.shape
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or not np.isfinite(extra).all()
        or np.any(extra < 0)
        or not np.isfinite(scale)
        or scale < 0
    ):
        raise ValueError("Invalid two-part mean inputs")
    result = probability * (1 + scale * extra)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite two-part mean")
    return result


def fit(directory, frame, episodes, kind, fold):
    path = directory / "fit.json"
    if path.exists():
        meta = read(path)
        if meta["model_sha256"] is not None and sha256(directory / "model.cbm") != meta["model_sha256"]:
            raise ValueError("Conditional count weights changed")
        return meta
    old = read(source_files(kind, fold)[1])
    sets, sizes = {}, {}
    for name in ("train", "validation"):
        rows = frame.loc[mask(frame, *map(pd.Timestamp, old["periods"][name]))]
        counts = episode_counts(rows, episodes)
        positive, target = positive_extra_targets(counts)
        if not np.array_equal(positive, rows[f"target_{kind}"].to_numpy().astype(bool)):
            raise ValueError("Original target parity failed")
        sets[name] = (model_input(rows.loc[positive], old["features"]), target)
        sizes[name] = {
            "all_rows": len(rows),
            "positive_rows": int(positive.sum()),
            "extra_sum": float(target.sum()),
        }
    if len(sets["train"][1]) == 0:
        raise ValueError("No positive training observations")
    meta = {
        "kind": kind,
        "fold": fold,
        "features": old["features"],
        "periods": old["periods"],
        "sizes": sizes,
        "original_target_parity": True,
        "model_sha256": None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    target = sets["train"][1]
    if np.all(target == target[0]):
        meta.update(mode="constant", constant_extra=float(target[0]), best_iteration=None)
    else:
        has_validation = len(sets["validation"][1]) > 0
        model = CatBoostRegressor(
            iterations=1000 if has_validation else 500,
            depth=6,
            l2_leaf_reg=8,
            learning_rate=0.04,
            loss_function="Poisson",
            eval_metric="Poisson",
            random_seed=42,
            thread_count=4,
            cat_features=CATEGORICAL,
            allow_writing_files=False,
            early_stopping_rounds=100 if has_validation else None,
            verbose=200,
        )
        print("START conditional count fit", kind, fold, sizes, flush=True)
        model.fit(*sets["train"], eval_set=sets["validation"] if has_validation else None)
        model.save_model(str(directory / "model.cbm"))
        meta.update(
            mode="catboost",
            best_iteration=model.best_iteration_,
            model_sha256=sha256(directory / "model.cbm"),
        )
    write_json(path, meta)
    print("DONE conditional count fit", kind, fold, meta["mode"], meta["best_iteration"], flush=True)
    return meta


def evaluate(root, kind, fold, frame, dense, episodes):
    directory = root / kind / fold
    output = directory / "result.json"
    if output.exists():
        return read(output)
    control_root = Path("artifacts/research-v24") if fold.startswith("screen_") else root / "controls"
    if fold.startswith("screen_"):
        control = read(control_root / kind / fold / "result.json")
    else:
        control = evaluate_gate(control_root, kind, fold, frame, dense, episodes)
    meta = fit(directory / "conditional", frame, episodes, kind, fold)
    conditional = None
    if meta["mode"] == "catboost":
        conditional = CatBoostRegressor()
        conditional.load_model(str(directory / "conditional/model.cbm"))
    binary_meta = control["binary_model"]
    if sha256(Path(binary_meta["weights_file"])) != binary_meta["model_sha256"]:
        raise ValueError("Binary weights changed")
    binary = CatBoostClassifier()
    binary.load_model(binary_meta["weights_file"])
    binary_cal = control["arms"]["binary_gate_candidate"]["calibration"]
    tables, exposure = {}, {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, meta["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        x = model_input(rows, meta["features"])
        hourly = rows[["object_id", "as_of"]].copy()
        hourly["binary_raw"] = binary.predict(x, prediction_type="RawFormulaVal", thread_count=2)
        hourly["extra"] = (
            np.exp(np.clip(conditional.predict(x, prediction_type="RawFormulaVal", thread_count=2), -20, 20))
            if conditional is not None
            else meta["constant_extra"]
        )
        pred = carry_features(hourly, subdivide_evaluation_slots(hourly), ["binary_raw", "extra"])
        pred["probability"] = calibrated(pred.binary_raw, binary_cal)
        assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
        if period in ("policy", "test"):
            old = pd.read_parquet(control_root / kind / fold / f"binary_gate_candidate-{period}.parquet")
            shared = pred.merge(
                old, on=["object_id", "as_of"], suffixes=("_new", "_old"), validate="one_to_one"
            )
            assert len(shared) == len(pred) == len(old)
            for column in ("probability", "binary_raw"):
                np.testing.assert_allclose(
                    shared[f"{column}_new"], shared[f"{column}_old"], atol=1e-10, rtol=1e-10
                )
        tables[period], exposure[period] = pred, len(hourly) / 24
    cal = tables["calibration"]
    calibration = extra_calibration(episode_counts(cal, episodes), cal.extra)
    for pred in tables.values():
        pred["expected_count"] = factor_mean(pred.probability, pred.extra, calibration["scale"])
    print("START conditional count policy", kind, fold, calibration, flush=True)
    chosen, frontier = select_gate_policy(tables["policy"], episodes, exposure["policy"])
    test = tables["test"]
    test["alert"] = policy_alerts(test, episodes, chosen)
    scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
    for period in ("policy", "test"):
        tables[period].to_parquet(directory / f"candidate-{period}.parquet", index=False)
    write_json(directory / "frontier.json", frontier)
    result = {
        "kind": kind,
        "fold": fold,
        "scores": scores,
        "binary_control": control["scores"],
        "count_control": control["control"],
        "reference": anchor(kind, fold),
        "policy": chosen,
        "conditional_calibration": calibration,
        "binary_calibration": binary_cal,
        "binary_model": binary_meta,
        "model": meta,
        "identical_episode_cohort": True,
        "binary_prediction_parity": True,
        "exposure_days": exposure["test"],
        "gated_test_rows": int(np.sum(test.probability < chosen["floor"])),
    }
    assert all(scores["eligible_episodes"] == result[key]["eligible_episodes"] for key in REFERENCES)
    write_json(output, result)
    print("DONE conditional count", kind, fold, scores, flush=True)
    return result


def lock_plan(root):
    sources = [
        PROCESSED / "features-channel-novelty.parquet",
        PROCESSED / "features-dense-channel-novelty.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/research-v24/plan.json"),
        Path("artifacts/research-v24/report.json"),
    ]
    sources.extend(sorted(Path("artifacts/research-v24").glob("*/screen_*/*.json")))
    sources.extend(sorted(Path("artifacts/research-v24").glob("*/screen_*/*.parquet")))
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            weights, metadata, saved = source_files(kind, fold)
            sources.extend([weights, metadata, *saved.values()])
            weights, metadata, saved = classifier_cache(kind, fold)
            if metadata.exists():
                sources.extend([weights, metadata, *saved.values()])
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
                evaluate_gate,
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
            pooled_scores = {key: pooled([row[key] for row in rows]) for key in ("scores", *REFERENCES)}
            c = pooled_scores["scores"]
            selected = {
                **pooled_scores,
                "passed_screen": all(
                    primary_score(c) > 1.05 * primary_score(pooled_scores[key])
                    and c["f1"] >= pooled_scores[key]["f1"]
                    for key in REFERENCES
                ),
            }
            selection[kind] = selected
            write_json(root / "selection.json", selection)
            print("SELECT conditional count", kind, selected, flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Finish screening all kinds first")
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
            "REPORT conditional count",
            kind,
            {k: v for k, v in report[kind].items() if k != "periods"},
            flush=True,
        )
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v25"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
