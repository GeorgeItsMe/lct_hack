"""v23: fit the count models to causal within-hour companion snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts, select
from moscollector.experiments.fresh_counts_research import anchor, source_files
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.phase_augmentation import paired_weights, shifted_snapshots
from moscollector.experiments.quarter_count_features import carry_features, refresh_counts
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

KINDS = ("access", "fire", "fault")
FEATURE_FOLDER = PROCESSED / "phase-augmentation-v23"
PLAN = {
    "scope": "adaptive_retrospective_training_augmentation_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "hypothesis": "V21 fresh raw counts modestly improved screening but old models were fitted only to3h whole-hour snapshots. Refit with causal within-hour companion snapshots and exact targets to adapt to the new input distribution.",
    "training_rows": "Retain every eligible original training row. Add at most one companion at15/30/45min, determined without outcomes by((unix_hour//3+object_id)%3+1)*15. Require following eligible3h source point for offline target coverage. Drop old target columns; recompute original24h episode counts at new time. Reapply25h purge within each split before pairing.",
    "weighting": "Original+companion each weight.5; unpaired original weight1. Total weight equals original row count, separately for training and validation. These correlated rows are not additional independent events. CatBoost's discretization/CTR/optimization can still change with the new rows; total weight preservation does not prove identical effective regularization.",
    "features": "Same66 access/94 fire inputs as v9; same144 fault inputs as previously selected v19. Only44 raw count windows and six available burst ratios refresh inside the hour. All other features remain at the same latest whole hour. No new feature columns or future confirmation information.",
    "fit": "CatBoost Poisson, weighted validationPoisson, depth6,l2=8,lr=.04,max1000,earlystop100,seed42,CPU4. Same original train/validation dates. Training and validation use the paired augmentation. No loss/hyperparameter search.",
    "inference": "Both new model and frozen old-weight control use exactly v21 fresh15min inputs and separate calibration on the original calibration dates. Same raw->binary sigmoid and count mean scaling,24h target,25h purge,warning policy and whole-hour confirmation availability as v20/v21.",
    "reference": "Both matched old-weight fresh15min control and prior anchor(access v20 quarter,fire v13 mean90,fault v19 hourly). Screen controls must exactly reproduce v21 fresh predictions/policies. If a new model passes, old-weight controls on additional months are part of this predeclared v23 comparison, not a promotion or continuation of the rejected v21 candidate.",
    "policy": "Frozen v20 grid, joint90/90 objective, policy period only,>=10alerts/events supported,FP<=.25/object-day. Identical original hourly exposure and exact eligible event identities. No opportunity or event exclusions to improve scores.",
    "screen": "Nov/Feb pooled candidate primary min(P/.9,R/.9,1) improves>5% against BOTH controls, without pooledF1 loss. Only passing kinds continue.",
    "confirmation": "May improves primary withF1>=95% against both; pooled Dec/Mar primary improves>5%, no pooledF1 loss, each month'sF1>=90% its controls. No substitution after outcomes. Gates do not establish90/90 or automatically activate production.",
    "june": "Excluded from labels/evaluation/selection. Prior test months are adaptively reused; no independent final-weight validation is claimed. Flood remains unsupported, not silently declared successful.",
}


def assemble(frame, shifted, episodes, columns, dates, kind):
    base = frame.loc[mask(frame, *dates)].copy()
    extra = shifted.loc[mask(shifted, *dates)].copy()
    a, b = paired_weights(base, extra)
    counts = episode_counts(base, episodes)
    if not np.array_equal(counts > 0, base[f"target_{kind}"].to_numpy().astype(bool)):
        raise ValueError("Original episode target changed")
    base = base[[*columns, "as_of"]].assign(training_count=counts, training_weight=a)
    extra = extra[[*columns, "as_of"]].assign(
        training_count=episode_counts(extra, episodes), training_weight=b
    )
    combined = (
        pd.concat([base, extra], ignore_index=True).sort_values(["as_of", "object_id"]).reset_index(drop=True)
    )
    summary = {
        "original_rows": len(base),
        "companion_rows": len(extra),
        "total_rows": len(combined),
        "weight_sum": float(combined.training_weight.sum()),
    }
    if summary["weight_sum"] != len(base):
        raise ValueError("Per-source training weight changed")
    return combined, summary


def fit(directory, frame, shifted, episodes, kind, fold):
    output = directory / "fit.json"
    if output.exists():
        meta = read(output)
        if sha256(directory / "model.cbm") != meta["model_sha256"]:
            raise ValueError("Augmented model weights changed")
        return meta
    old_weights, old_metadata, _ = source_files(kind, fold)
    old = read(old_metadata)
    columns = old["features"]
    sets, sizes = {}, {}
    for name in ("train", "validation"):
        view, sizes[name] = assemble(
            frame, shifted, episodes, columns, tuple(map(pd.Timestamp, old["periods"][name])), kind
        )
        sets[name] = Pool(
            model_input(view, columns),
            label=view.training_count,
            weight=view.training_weight,
            cat_features=CATEGORICAL,
        )
        del view
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        l2_leaf_reg=8,
        learning_rate=0.04,
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        thread_count=4,
        allow_writing_files=False,
        early_stopping_rounds=100,
        verbose=200,
    )
    print("START phase fit", kind, fold, sizes, flush=True)
    model.fit(sets["train"], eval_set=sets["validation"])
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(directory / "model.cbm"))
    result = {
        "kind": kind,
        "fold": fold,
        "features": columns,
        "periods": old["periods"],
        "sizes": sizes,
        "best_iteration": model.best_iteration_,
        "model_sha256": sha256(directory / "model.cbm"),
        "original_model_sha256": sha256(old_weights),
        "target_horizon_hours": 24,
        "old_target_parity": True,
        "sample_weights_sum_to_original_rows": True,
    }
    write_json(output, result)
    print("DONE phase fit", kind, fold, "best_iteration", model.best_iteration_, flush=True)
    return result


def evaluate(root, kind, fold, frame, dense, shifted, episodes, counts):
    directory = root / kind / fold
    output = directory / "result.json"
    if output.exists():
        return read(output)
    meta = fit(directory / "augmented", frame, shifted, episodes, kind, fold)
    candidate = CatBoostRegressor()
    candidate.load_model(str(directory / "augmented/model.cbm"))
    old_weights, _, _ = source_files(kind, fold)
    control = CatBoostRegressor()
    control.load_model(str(old_weights))
    tables = {"augmented_candidate": {}, "old_weight_control": {}}
    exposure = {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, meta["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        hourly = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        slots = subdivide_evaluation_slots(hourly)
        held = carry_features(hourly, slots, meta["features"])
        fresh = refresh_counts(held, counts)
        whole = fresh.as_of.eq(fresh.source_time)
        pd.testing.assert_frame_equal(held.loc[whole, meta["features"]], fresh.loc[whole, meta["features"]])
        assert cohort(fresh, episodes, 0.25) == cohort(hourly, episodes, 1)
        x = model_input(fresh, meta["features"])
        for name, model in (("augmented_candidate", candidate), ("old_weight_control", control)):
            pred = fresh[["object_id", "as_of", "source_time"]].copy()
            pred["raw"] = model.predict(x, prediction_type="RawFormulaVal", thread_count=2)
            if not np.isfinite(pred.raw).all():
                raise ValueError("Nonfinite forecast")
            tables[name][period] = pred
        exposure[period] = len(hourly) / 24
    arms = {}
    for name, predictions in tables.items():
        cal = predictions["calibration"]
        y = episode_counts(cal, episodes)
        calibration = calibrate(cal.raw.to_numpy(), y > 0)
        scale = float(y.sum() / np.exp(np.clip(cal.raw.to_numpy(), -20, 20)).sum())
        for pred in predictions.values():
            pred["probability"] = calibrated(pred.raw, calibration)
            pred["expected_count"] = np.exp(np.clip(pred.raw, -20, 20)) * scale
        print("START phase policy", kind, fold, name, flush=True)
        chosen, frontier = select(predictions["policy"], episodes, 0.25, exposure["policy"])
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, chosen)
        scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
        for period in ("policy", "test"):
            predictions[period].to_parquet(directory / f"{name}-{period}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {
            "scores": scores,
            "policy": chosen,
            "calibration": calibration,
            "rate_scale": scale,
            "calibration_rows": len(cal),
            "exposure_days": exposure["test"],
        }
        print("DONE phase policy", kind, fold, name, scores, flush=True)
    prior_path = Path("artifacts/research-v21") / kind / fold / "result.json"
    if prior_path.exists():
        prior = read(prior_path)
        actual = arms["old_weight_control"]
        assert actual["policy"] == prior["arms"]["fresh_candidate"]["policy"]
        assert actual["scores"] == prior["scores"]
        for period in ("policy", "test"):
            old = pd.read_parquet(prior_path.parent / f"fresh_candidate-{period}.parquet")
            keys = ["object_id", "as_of"]
            new = tables["old_weight_control"][period]
            joined = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
            assert len(joined) == len(new) == len(old)
            for name in ("raw", "probability", "expected_count"):
                np.testing.assert_allclose(
                    joined[f"{name}_old"], joined[f"{name}_new"], atol=1e-10, rtol=1e-10
                )
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "scores": arms["augmented_candidate"]["scores"],
        "control": arms["old_weight_control"]["scores"],
        "reference": anchor(kind, fold),
        "identical_episode_cohort": True,
        "model_sha256": meta["model_sha256"],
        "fit": meta,
    }
    for other in (result["control"], result["reference"]):
        assert result["scores"]["eligible_episodes"] == other["eligible_episodes"]
    write_json(output, result)
    return result


def lock_plan(root):
    sources = [
        PROCESSED / "features-channel-novelty.parquet",
        PROCESSED / "features-dense-channel-novelty.parquet",
        PROCESSED / "episodes.parquet",
        FEATURE_FOLDER / "shifted-features.parquet",
        FEATURE_FOLDER / "build.json",
        Path("artifacts/research-v21/report.json"),
        Path("artifacts/research-v22/report.json"),
    ]
    manifests = [
        FEATURE_FOLDER / "build.json",
        *(FEATURE_FOLDER / f"counts-{year}.json" for year in range(2022, 2026)),
        PROCESSED / "quarter-counts-v21/counts-2026.json",
    ]
    for manifest in manifests:
        sources.append(manifest)
        for category in ("inputs", "outputs"):
            for p, digest in read(manifest)[category].items():
                if sha256(Path(p)) != digest:
                    raise ValueError(f"Augmentation provenance changed: {p}")
    for year in range(2022, 2027):
        sources.append(
            (FEATURE_FOLDER if year < 2026 else PROCESSED / "quarter-counts-v21") / f"counts-{year}.parquet"
        )
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            weights, metadata, saved = source_files(kind, fold)
            sources.extend([weights, metadata, *saved.values()])
            if kind == "access":
                sources.append(Path("artifacts/research-v20/access") / fold / "result.json")
            elif kind == "fault":
                sources.append(Path("artifacts/research-v19/fault") / fold / "result.json")
            else:
                sources.append(Path("artifacts/research-v13-policy") / f"fire-{fold}.json")
            if fold in ("screen_1", "screen_2"):
                prior = Path("artifacts/research-v21") / kind / fold
                sources.extend(prior / f"fresh_candidate-{period}.parquet" for period in ("policy", "test"))
                sources.append(prior / "result.json")
    code = {
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (
                paired_weights,
                shifted_snapshots,
                refresh_counts,
                select,
                cohort,
                episode_counts,
                model_input,
                mask,
                subdivide_evaluation_slots,
                align_opportunities,
                source_files,
                primary_score,
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
    shifted = pd.read_parquet(FEATURE_FOLDER / "shifted-features.parquet")
    dense = pd.read_parquet(PROCESSED / "features-dense-channel-novelty.parquet")
    assert all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, shifted, dense))
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    # Inference periods need only the v21 suffix. Older count buckets were used
    # to materialize training features and are verified, not retained in RAM.
    counts = pd.concat(
        [
            pd.read_parquet(PROCESSED / "quarter-counts-v21" / f"counts-{year}.parquet")
            for year in (2025, 2026)
        ],
        ignore_index=True,
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            eps = episodes[episodes.kind.eq(kind)]
            rows = [
                evaluate(root, kind, fold, frame, dense, shifted, eps, counts)
                for fold in ("screen_1", "screen_2")
            ]
            c, h, r = (pooled([item[key] for item in rows]) for key in ("scores", "control", "reference"))
            selection[kind] = {
                "candidate": c,
                "control": h,
                "reference": r,
                "passed_screen": all(
                    primary_score(c) > 1.05 * primary_score(other) and c["f1"] >= other["f1"]
                    for other in (h, r)
                ),
            }
            write_json(root / "selection.json", selection)
            print("SELECT phase augmentation", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete screening all kinds first")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        eps = episodes[episodes.kind.eq(kind)]
        rows = [
            evaluate(root, kind, fold, frame, dense, shifted, eps, counts)
            for fold in ("confirmation", *STRESS)
        ]
        may = rows[0]
        c = pooled([item["scores"] for item in rows[1:]])
        passed_may = all(
            primary_score(may["scores"]) > primary_score(may[key])
            and may["scores"]["f1"] >= 0.95 * may[key]["f1"]
            for key in ("control", "reference")
        )
        passed_stress = all(
            primary_score(c) > 1.05 * primary_score(pooled([item[key] for item in rows[1:]]))
            and c["f1"] >= pooled([item[key] for item in rows[1:]])["f1"]
            and all(item["scores"]["f1"] >= 0.9 * item[key]["f1"] for item in rows[1:])
            for key in ("control", "reference")
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": pooled([item["scores"] for item in rows]),
            "five_period_control": pooled([item["control"] for item in rows]),
            "five_period_reference": pooled([item["reference"] for item in rows]),
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print(
            "REPORT phase augmentation",
            kind,
            {k: v for k, v in report[kind].items() if k != "periods"},
            flush=True,
        )
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v23"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
