"""v30: short cross-object telemetry context for the frozen count-model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.binary_gate_research import KINDS, select_gate_policy
from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.cadence_research import align_opportunities
from moscollector.count_extension_research import validate_opportunities
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.fresh_counts_research import anchor, source_files
from moscollector.goal90_research import STRESS, pooled, primary_score, read
from moscollector.paths import PROCESSED
from moscollector.peer_context_features import COLUMNS, FOLDER, peer_context
from moscollector.prepare import sha256, write_json
from moscollector.quarter_count_features import carry_features
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

FIT = {
    "iterations": 1000,
    "depth": 6,
    "l2_leaf_reg": 8,
    "learning_rate": 0.04,
    "loss_function": "Poisson",
    "eval_metric": "Poisson",
    "random_seed": 42,
    "thread_count": 4,
    "allow_writing_files": False,
    "early_stopping_rounds": 100,
}
PLAN = {
    "scope": "adaptive_retrospective_cross_object_count_study_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "hypothesis": "Standalone neural models and fixed blends failed confirmation. Test short past cross-object report activity and shared telemetry shifts using original catalog groups. Earlier v4 classifier context used24/168h normalized same-parent summaries; v17 used ordered neighboring transitions. Neither proves benefit of this fixed64-feature addition to the current count heads.",
    "features": "Retain original66 access/94 fire/144 fault features, add64 numeric peer columns:8 raw signals at1/6/24h and active-object counts at6h, each for same-parent peers excluding self and for objects outside the entire parent. Compute before eligibility masking, only same-time causal historical summaries. No target, eligibility, future rows, fitted normalizers or future-inferred exposure denominator. Missing peer windows propagate as missing; zero logged reports are not healthy-state labels. Static catalog only, historical topology unknown.",
    "training": FIT,
    "training_protocol": "One candidate per kind. Original eligible3h training/validation rows, dates and25h purge. Same original24h episode-count target and exact binary-label parity. Poisson validation earlystop, no hyperparameter search or new sample weighting. Completed fits resume by verified metadata/weights; interrupted incomplete CatBoost fits are retrained.",
    "inference": "Same eligible whole-hour rows held on15min warning grid, exact event identities and original hourly exposure. Both candidate and old-count-weight control separately fit original mean scale and sigmoid on actual quarter-time24h targets in the preceding calibration period. Same456-policy grid, causal confirmation and pending expiry.",
    "reference": "Matched frozen CatBoost control must exactly replay v28 screening and v29 additional fault controls where available. Also compare historical anchor(access v20 quarter,fire v13 mean90,fault v19 hourly). These are count-model comparisons, not claims about deployed fire/fault classifiers.",
    "screen": "Pooled Nov/Feb candidate primary min(P/.9,R/.9,1)>1.05*both references, noF1 loss. Only passing kinds proceed; no alternative feature subset after outcomes.",
    "confirmation": "May primary improves withF1>=95% both. Pooled Dec/Mar primary>1.05*both,noF1 loss; each monthF1>=90%both. No automatic activation or success by subgroup exclusion.",
    "limits": "No June selection/evaluation or new labels. Historical months adaptively reused, not independent future evidence. Sensor episodes remain proxy labels; flood unsupported and full-scope90/90 unproved.",
}


def control_source(kind, fold):
    return (
        (
            Path("artifacts/research-v28")
            if fold.startswith("screen_")
            else Path("artifacts/research-v29/controls")
        )
        / kind
        / fold
    )


def fit(root, kind, fold, frame, dense, episodes):
    directory = root / kind / fold / "peer"
    weights, target = directory / "model.cbm", directory / "fit.json"
    old_weights, old_metadata, _ = source_files(kind, fold)
    old = read(old_metadata)
    columns = [*old["features"], *COLUMNS]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate peer feature column")
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
            raise ValueError("Peer model inputs or weights changed")
        return meta
    periods = {name: tuple(map(pd.Timestamp, dates)) for name, dates in old["periods"].items()}
    validate_opportunities(frame, dense, periods)
    pools, sizes = {}, {}
    for name in ("train", "validation"):
        rows = frame.loc[mask(frame, *periods[name])]
        counts = episode_counts(rows, episodes)
        if not np.array_equal(counts > 0, rows[f"target_{kind}"].to_numpy().astype(bool)):
            raise ValueError("Peer count target differs from original binary label")
        pools[name] = Pool(model_input(rows, columns), counts, cat_features=CATEGORICAL)
        sizes[name] = {
            "rows": len(rows),
            "positive_rows": int(np.sum(counts > 0)),
            "eligible_episodes": EventEvaluator(rows, episodes, 3).events,
        }
    directory.mkdir(parents=True, exist_ok=True)
    model = CatBoostRegressor(**FIT, verbose=200)
    print("START peer fit", kind, fold, len(columns), sizes, flush=True)
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
        "added_columns": COLUMNS,
    }
    write_json(target, meta)
    print("DONE peer fit", kind, fold, "best", model.best_iteration_, flush=True)
    return meta


def evaluate(root, kind, fold, frame, dense, episodes):
    directory = root / kind / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    meta = fit(root, kind, fold, frame, dense, episodes)
    old_weights, old_metadata, _ = source_files(kind, fold)
    old = read(old_metadata)
    candidate, control = CatBoostRegressor(), CatBoostRegressor()
    candidate.load_model(str(directory / "peer/model.cbm"))
    control.load_model(str(old_weights))
    tables, exposure = {}, {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, old["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        hourly = rows[["object_id", "as_of"]].copy()
        for name, model, columns in (
            ("peer_candidate", candidate, meta["features"]),
            ("global_control", control, old["features"]),
        ):
            hourly[name] = model.predict(
                model_input(rows, columns), prediction_type="RawFormulaVal", thread_count=2
            )
            if not np.isfinite(hourly[name]).all():
                raise ValueError("Nonfinite peer forecast")
        pred = carry_features(
            hourly, subdivide_evaluation_slots(hourly), ["peer_candidate", "global_control"]
        )
        assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
        tables[period], exposure[period] = pred, len(hourly) / 24
    counts = episode_counts(tables["calibration"], episodes)
    arms = {}
    for name in ("peer_candidate", "global_control"):
        calibration = calibrate(tables["calibration"][name].to_numpy(), counts > 0)
        scale = float(counts.sum() / np.exp(np.clip(tables["calibration"][name], -20, 20)).sum())
        predictions = {
            period: pred.assign(
                probability=calibrated(pred[name], calibration),
                expected_count=np.exp(np.clip(pred[name], -20, 20)) * scale,
            )
            for period, pred in tables.items()
        }
        print("START peer policy", kind, fold, name, flush=True)
        policy, frontier = select_gate_policy(predictions["policy"], episodes, exposure["policy"])
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, policy)
        scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
        source = control_source(kind, fold)
        if name == "global_control" and (source / "result.json").exists():
            before = read(source / "result.json")["arms"]["global_control"]
            assert policy == before["policy"] and scores == before["scores"]
            for period in ("policy", "test"):
                saved = pd.read_parquet(source / f"global_control-{period}.parquet")
                current = predictions[period]
                pd.testing.assert_frame_equal(current[["object_id", "as_of"]], saved[["object_id", "as_of"]])
                for column in ("probability", "expected_count"):
                    np.testing.assert_allclose(current[column], saved[column], rtol=1e-10, atol=1e-10)
                if period == "test":
                    np.testing.assert_array_equal(current.alert, saved.alert)
        for period, pred in predictions.items():
            pred.to_parquet(directory / f"{name}-{period}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {"scores": scores, "policy": policy, "calibration": calibration, "rate_scale": scale}
        print("DONE peer policy", kind, fold, name, scores, flush=True)
    reference = anchor(kind, fold)
    assert all(arm["scores"]["eligible_episodes"] == reference["eligible_episodes"] for arm in arms.values())
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "reference": reference,
        "fit": meta,
        "same_episode_cohort": True,
        "exposure_days": exposure["test"],
    }
    write_json(target, result)
    return result


def lock_plan(root):
    build = read(FOLDER / "build.json")
    for category in ("inputs", "outputs", "code_hashes"):
        for source, digest in build[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed peer feature build: {source}")
    sources = {
        FOLDER / "build.json",
        FOLDER / "original.parquet",
        FOLDER / "dense.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/research-v29/report.json"),
    }
    sources.update(Path(p) for p in build["inputs"])
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
                peer_context,
                EventEvaluator.__init__,
                select_gate_policy,
                evaluator_for,
                policy_alerts,
                source_files,
                align_opportunities,
                episode_counts,
                mask,
                model_input,
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
        raise ValueError("Peer study inputs/code changed; use a new output directory")
    if not target.exists():
        write_json(target, plan)


def metric(row, name):
    return row["reference"] if name == "reference" else row["arms"][name]["scores"]


def run(root, stage):
    lock_plan(root)
    frame, dense = (pd.read_parquet(FOLDER / f"{name}.parquet") for name in ("original", "dense"))
    if not all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense)):
        raise ValueError("Post-May peer inputs")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            rows = [
                evaluate(root, kind, fold, frame, dense, episodes.loc[episodes.kind.eq(kind)])
                for fold in ("screen_1", "screen_2")
            ]
            scores = {
                name: pooled([metric(row, name) for row in rows])
                for name in ("peer_candidate", "global_control", "reference")
            }
            c = scores["peer_candidate"]
            selection[kind] = {
                **scores,
                "passed_screen": all(
                    primary_score(c) > 1.05 * primary_score(scores[name]) and c["f1"] >= scores[name]["f1"]
                    for name in ("global_control", "reference")
                ),
            }
            write_json(root / "selection.json", selection)
            print("SELECT peer", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete all peer screening before confirmation")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        rows = [
            evaluate(root, kind, fold, frame, dense, episodes.loc[episodes.kind.eq(kind)])
            for fold in ("confirmation", *STRESS)
        ]
        may, stress = (
            metric(rows[0], "peer_candidate"),
            pooled([metric(row, "peer_candidate") for row in rows[1:]]),
        )
        passed_may = all(
            primary_score(may) > primary_score(metric(rows[0], name))
            and may["f1"] >= 0.95 * metric(rows[0], name)["f1"]
            for name in ("global_control", "reference")
        )
        passed_stress = all(
            primary_score(stress) > 1.05 * primary_score(pooled([metric(row, name) for row in rows[1:]]))
            and stress["f1"] >= pooled([metric(row, name) for row in rows[1:]])["f1"]
            and all(metric(row, "peer_candidate")["f1"] >= 0.9 * metric(row, name)["f1"] for row in rows[1:])
            for name in ("global_control", "reference")
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": {
                name: pooled([metric(row, name) for row in rows])
                for name in ("peer_candidate", "global_control", "reference")
            },
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT peer", kind, report[kind]["five_period_pooled"], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v30"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
