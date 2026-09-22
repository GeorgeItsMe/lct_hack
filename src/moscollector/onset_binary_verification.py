"""Recompute v35 model forecasts and direct warning decisions without promotion."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.fresh_counts_research import anchor, source_files
from moscollector.goal90_research import pooled, read
from moscollector.minute_cadence_research import forecasts as pending_forecasts
from moscollector.onset_binary_policy import select_threshold
from moscollector.onset_binary_research import (
    ARMS,
    BINARY_FIT,
    KINDS,
    confirmation,
    forecast_tables,
    metric,
    screening,
)
from moscollector.onset_channel_features import CATS, COLUMNS, FOLDER, KEYS
from moscollector.onset_training_data import augmented_slots, snapshot_weights
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256
from moscollector.research import mask
from moscollector.waiting_time_research import threshold_alerts


def verify(root=Path("artifacts/research-v35")):
    plan, report, selected = (read(root / name) for name in ("plan.json", "report.json", "selection.json"))
    for category in ("source_hashes", "code_hashes"):
        for path, digest in plan[category].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed binary onset provenance: {path}")
    if set(report) != set(KINDS) or set(selected) != set(KINDS):
        raise ValueError("Incomplete binary onset report")
    for name, prior in (
        ("count_controls", Path("artifacts/research-v33/plan.json")),
        ("pending_controls", Path("artifacts/research-v34/plan.json")),
    ):
        control_plan = read(root / name / "plan.json")
        if control_plan["parent_plan_sha256"] != sha256(root / "plan.json") or control_plan[
            "frozen_source_plan_sha256"
        ] != sha256(prior):
            raise ValueError("Binary control plan belongs to another experiment")
    frame, dense = (
        pd.read_parquet(PROCESSED / name)
        for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    context = pd.read_parquet(FOLDER / "context.parquet", read_dictionary=CATS)
    triggers = pd.read_parquet(FOLDER / "triggers.parquet")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    models, count_models, pending_models, files, periods = {}, {}, {}, {}, {}
    expected_fits, expected_counts, expected_pending = set(), set(), set()
    for kind in KINDS:
        folds = ["screen_1", "screen_2"] + (
            ["confirmation", "stress_1", "stress_2"] if selected[kind]["passed_screen"] else []
        )
        rows = []
        eps = episodes.loc[episodes.kind.eq(kind)]
        for fold in folds:
            directory = root / kind / fold
            fit_path = directory / "binary/fit.json"
            expected_fits.add(str(fit_path))
            result, meta = read(directory / "result.json"), read(fit_path)
            weights = directory / "binary/model.cbm"
            original_weights, original_metadata, _ = source_files(kind, fold)
            original = read(original_metadata)
            if (
                result["fit"] != meta
                or meta["signature"]["plan_sha256"] != sha256(root / "plan.json")
                or meta["signature"]["fit"] != BINARY_FIT
                or meta["model_sha256"] != sha256(weights)
                or meta["periods"] != original["periods"]
                or meta["original_model_sha256"] != sha256(original_weights)
                or meta["features"] != [*original["features"], *COLUMNS, "evt_source_age_h"]
            ):
                raise ValueError("Changed classifier weights or fit metadata")
            for part in ("train", "validation"):
                dates = tuple(map(pd.Timestamp, meta["periods"][part]))
                anchors = frame.loc[mask(frame, *dates)]
                slots = augmented_slots(anchors, triggers)
                weight, counts = snapshot_weights(slots), episode_counts(slots, eps)
                sizes = meta["sizes"][part]
                if (
                    sizes["rows"] != len(slots)
                    or sizes["original_rows"] != len(anchors)
                    or sizes["positive_rows"] != int(np.sum(counts > 0))
                    or sizes["eligible_episodes"] != EventEvaluator(anchors, eps, 3).events
                    or cohort(slots, eps, 1 / 60) != cohort(anchors, eps, 3)
                    or not (slots.as_of + pd.Timedelta(hours=25)).lt(dates[1]).all()
                    or not np.array_equal(
                        episode_counts(anchors, eps) > 0,
                        anchors[f"target_{kind}"].to_numpy().astype(bool),
                    )
                ):
                    raise ValueError("Binary training rows, target support or purge changed")
                np.testing.assert_allclose(
                    [weight.sum(), weight[counts > 0].sum()],
                    [sizes["weight_sum"], sizes["weighted_positive_rows"]],
                    rtol=0,
                    atol=1e-7,
                )
            classifier = CatBoostClassifier()
            classifier.load_model(str(weights))
            params = classifier.get_all_params()
            if (
                classifier.feature_names_ != meta["features"]
                or params["loss_function"] != "Logloss"
                or params["eval_metric"] != BINARY_FIT["eval_metric"]
                or classifier.tree_count_ != meta["best_iteration"] + 1
            ):
                raise ValueError("Classifier schema, loss or weighted validation changed")
            count = Path(result["count_control"])
            count_meta = read(count / "fit.json")
            if (
                count_meta != result["count_fit"]
                or count_meta["features"] != meta["features"]
                or count_meta["sizes"] != meta["sizes"]
                or sha256(count / "model.cbm") != count_meta["model_sha256"]
            ):
                raise ValueError("Matched count control changed")
            count_model = CatBoostRegressor()
            count_model.load_model(str(count / "model.cbm"))
            if (
                count_model.feature_names_ != meta["features"]
                or count_model.get_all_params()["loss_function"] != "Poisson"
            ):
                raise ValueError("Wrong matched count model")
            if not fold.startswith("screen_"):
                expected_counts.add(str(count / "fit.json"))
                if count_meta["signature"]["plan_sha256"] != sha256(root / "count_controls/plan.json"):
                    raise ValueError("New count control lacks parent provenance")
            count_models[str(count / "fit.json")] = count_meta
            pending = Path(result["pending_control"])
            pending_result = read(pending / "result.json")
            if (
                sha256(pending / "result.json") != result["pending_result_sha256"]
                or result["old_pending"] != pending_result["arms"]["minute_candidate"]["scores"]
                or result["reference"] != anchor(kind, fold)
            ):
                raise ValueError("Changed pending or historical reference")
            if pending.is_relative_to(root):
                expected_pending.add(str(pending / "result.json"))
                if pending_result["plan_sha256"] != sha256(root / "pending_controls/plan.json"):
                    raise ValueError("New pending control lacks parent provenance")
                pending_tables, pending_cal, pending_exposure, pending_sizes = pending_forecasts(
                    kind, fold, frame, dense, triggers, eps
                )
                if (
                    pending_exposure != pending_result["exposure"]
                    or pending_sizes != pending_result["opportunities"]
                ):
                    raise ValueError("New pending control exposure changed")
                for pending_arm, predictions in pending_tables.items():
                    record = pending_result["arms"][pending_arm]
                    if any(record[k] != value for k, value in pending_cal[pending_arm].items()):
                        raise ValueError("New pending calibration changed")
                    for part, actual in predictions.items():
                        path = pending / f"{pending_arm}-{part}.parquet"
                        saved = pd.read_parquet(path)
                        pd.testing.assert_frame_equal(actual[KEYS], saved[KEYS])
                        for column in ("raw", "probability", "expected_count"):
                            np.testing.assert_allclose(actual[column], saved[column], rtol=1e-10, atol=1e-10)
                        if part != "calibration":
                            alerts = policy_alerts(actual, eps, record["policy"])
                            np.testing.assert_array_equal(alerts, saved.alert)
                            scores = evaluator_for(
                                actual, eps, record["cadence_hours"], pending_exposure[part]
                            ).evaluate(alerts, 0.5, record["cadence_hours"])
                            expected = record["scores"] if part == "test" else record["policy"]
                            if any(value != expected[k] for k, value in scores.items()):
                                raise ValueError("New pending warning replay changed")
                        files[str(path)] = sha256(path)
            pending_models[str(pending / "result.json")] = pending_result
            tables, calibrations, exposure = forecast_tables(
                kind,
                fold,
                frame,
                dense,
                context,
                triggers,
                eps,
                weights,
                count / "model.cbm",
                meta["features"],
            )
            if exposure != result["exposure"] or exposure != pending_result["exposure"]:
                raise ValueError("Binary comparison changed exposure")
            for arm in ARMS:
                record = result["arms"][arm]
                if calibrations[arm] != record["calibration"]:
                    raise ValueError("Binary calibration changed")
                policy, frontier = select_threshold(tables[arm]["policy"], eps, exposure["policy"])
                frontier_path = directory / f"{arm}-frontier.json"
                if policy != record["policy"] or frontier != read(frontier_path):
                    raise ValueError("Binary policy selection or full frontier changed")
                for part, actual in tables[arm].items():
                    path = directory / f"{arm}-{part}.parquet"
                    saved = pd.read_parquet(path)
                    pd.testing.assert_frame_equal(actual[KEYS], saved[KEYS])
                    for column in ("raw", "probability"):
                        np.testing.assert_allclose(actual[column], saved[column], rtol=1e-10, atol=1e-10)
                    if arm == "count_direct" and fold.startswith("screen_"):
                        old = pd.read_parquet(
                            Path("artifacts/research-v33") / kind / fold / f"onset_candidate-{part}.parquet"
                        )
                        pd.testing.assert_frame_equal(actual[KEYS], old[KEYS])
                        for column in ("raw", "probability"):
                            np.testing.assert_allclose(actual[column], old[column], rtol=1e-10, atol=1e-10)
                    if part != "calibration":
                        alerts = threshold_alerts(actual, record["policy"])
                        np.testing.assert_array_equal(alerts, saved.alert)
                        scores = evaluator_for(actual, eps, 1 / 60, exposure[part]).evaluate(
                            alerts, 0.5, record["policy"]["cooldown_hours"]
                        )
                        expected = record["scores"] if part == "test" else record["policy"]
                        if any(value != expected[k] for k, value in scores.items()):
                            raise ValueError("Binary warning metric replay failed")
                    files[str(path)] = sha256(path)
                files[str(frontier_path)] = sha256(frontier_path)
            rows.append(result)
            periods[str(directory / "result.json")] = result
            models[str(fit_path)] = meta
        if screening(rows[:2]) != selected[kind] or report[kind]["selection"] != selected[kind]:
            raise ValueError("Binary screening decision changed")
        if selected[kind]["passed_screen"]:
            gates = confirmation(rows[2:])
            aggregates = {
                name: pooled([metric(row, name) for row in rows])
                for name in (*ARMS, "old_pending", "reference")
            }
            if (
                report[kind]["periods"] != rows
                or report[kind]["five_period_pooled"] != aggregates
                or any(report[kind][k] != v for k, v in gates.items())
                or report[kind]["automatic_activation"]
            ):
                raise ValueError("Binary confirmation decision changed")
        elif report[kind] != {
            "selection": selected[kind],
            "research_eligible": False,
            "status": "screen_failed",
        }:
            raise ValueError("Failed binary kind evaluated additional months")
    if (
        {str(p) for p in root.glob("*/*/binary/fit.json")} != expected_fits
        or {str(p) for p in root.glob("count_controls/*/*/onset/fit.json")} != expected_counts
        or {str(p) for p in root.glob("pending_controls/*/*/result.json")} != expected_pending
    ):
        raise ValueError("Unexpected binary model or matched control runs")
    return {
        "report": report,
        "models": models,
        "count_controls": count_models,
        "pending_controls": pending_models,
        "periods": periods,
        "prediction_and_frontier_hashes": files,
        "all_raw_scores_calibration_and_metrics_recomputed": True,
        "all_training_grids_targets_weights_and_cohorts_recomputed": True,
        "all_direct_policy_frontiers_and_selection_recomputed": True,
        "new_count_controls": len(expected_counts),
        "new_pending_controls": len(expected_pending),
    }
