"""Recompute v33 grids, held-source predictions, calibration and warning metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.experiments.fresh_counts_research import source_files
from moscollector.experiments.goal90_research import pooled, primary_score, read
from moscollector.experiments.onset_channel_features import CATS, COLUMNS, FOLDER, KEYS, channel_context
from moscollector.experiments.onset_count_research import ARMS, REFERENCES, metric, verify_quarter_control
from moscollector.experiments.onset_training_data import (
    attach,
    augmented_slots,
    model_input,
    snapshot_weights,
)
from moscollector.experiments.peer_context_research import control_source
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256
from moscollector.research import mask
from moscollector.train import calibrate, calibrated
from moscollector.train import model_input as legacy_input


def verify_context_sample():
    """Deterministic values audit on each stored object row group, not a quality test."""
    catalog = pd.read_parquet(PROCESSED / "channels.parquet")
    onsets = pd.concat(
        [
            pd.read_parquet(PROCESSED / "channel-novelty-v18" / f"onsets-{year}.parquet")
            for year in range(2022, 2027)
        ],
        ignore_index=True,
    )
    parquet = pq.ParquetFile(FOLDER / "context.parquet")
    checked = 0
    for i in range(parquet.num_row_groups):
        group = parquet.read_row_group(i).to_pandas()
        sample = group.iloc[
            np.unique(np.linspace(0, len(group) - 1, min(32, len(group)), dtype=int))
        ].reset_index(drop=True)
        obj = sample.object_id.unique()
        if len(obj) != 1:
            raise ValueError("Onset row group no longer represents one object")
        recomputed = channel_context(sample[KEYS], onsets.loc[onsets.object_id.eq(obj[0])], catalog)
        pd.testing.assert_frame_equal(sample[COLUMNS], recomputed)
        checked += len(sample)
    return {
        "rows": checked,
        "object_row_groups": parquet.num_row_groups,
        "sampling": "32 evenly spaced queries per object, including both endpoints; compare all60 columns to unpruned raw-onset recomputation. Other rows protected by full build hashes and synthetic causality tests.",
    }


def verify_legacy_input_parity():
    columns = list(
        dict.fromkeys(
            c
            for kind in ("access", "fire", "fault")
            for c in read(source_files(kind, "screen_1")[1])["features"]
        )
    )
    checked = {}
    for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet"):
        count = 0
        for batch in pq.ParquetFile(PROCESSED / name).iter_batches(batch_size=32768, columns=columns):
            frame = batch.to_pandas()
            pd.testing.assert_frame_equal(model_input(frame, columns), legacy_input(frame, columns))
            count += len(frame)
        checked[name] = count
    return checked


def verify(root=Path("artifacts/research-v33")):
    plan, outcome, selection = (read(root / name) for name in ("plan.json", "report.json", "selection.json"))
    for category in ("source_hashes", "code_hashes"):
        for path, digest in plan[category].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed onset study provenance: {path}")
    build = read(FOLDER / "build.json")
    for category in ("inputs", "outputs", "code_hashes"):
        for path, digest in build[category].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed onset feature build: {path}")
    if set(outcome) != set(plan["kinds"]) or set(selection) != set(plan["kinds"]):
        raise ValueError("Incomplete onset study")
    base = pd.read_parquet(PROCESSED / "features-channel-novelty.parquet")
    dense = pd.read_parquet(PROCESSED / "features-dense-channel-novelty.parquet")
    triggers = pd.read_parquet(FOLDER / "triggers.parquet")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    models, forecasts, periods, expected_fits = {}, {}, {}, set()
    for kind in plan["kinds"]:
        eps = episodes.loc[episodes.kind.eq(kind)]
        folds = ["screen_1", "screen_2"]
        if selection[kind]["passed_screen"]:
            folds += ["confirmation", "stress_1", "stress_2"]
        all_results = []
        for fold in folds:
            directory = root / kind / fold
            fit_path = directory / "onset/fit.json"
            expected_fits.add(str(fit_path))
            meta, result = read(fit_path), read(directory / "result.json")
            weights, old_meta_path, _ = source_files(kind, fold)
            old = read(old_meta_path)
            model_path = fit_path.parent / "model.cbm"
            if (
                result["fit"] != meta
                or meta["signature"]["plan_sha256"] != sha256(root / "plan.json")
                or meta["model_sha256"] != sha256(model_path)
                or meta["original_model_sha256"] != sha256(weights)
                or meta["periods"] != old["periods"]
                or meta["features"] != [*old["features"], *COLUMNS, "evt_source_age_h"]
            ):
                raise ValueError("Onset fit, inputs and result differ")
            candidate, control = CatBoostRegressor(), CatBoostRegressor()
            candidate.load_model(str(model_path))
            control.load_model(str(weights))
            if candidate.feature_names_ != meta["features"]:
                raise ValueError("Onset model column order changed")
            for part in ("train", "validation"):
                dates = tuple(map(pd.Timestamp, meta["periods"][part]))
                anchors = base.loc[mask(base, *dates)]
                slots = augmented_slots(anchors, triggers)
                weight = snapshot_weights(slots)
                counts = episode_counts(slots, eps)
                sizes = meta["sizes"][part]
                if (
                    sizes["rows"] != len(slots)
                    or sizes["original_rows"] != len(anchors)
                    or sizes["positive_rows"] != int(np.sum(counts > 0))
                    or sizes["eligible_episodes"] != EventEvaluator(anchors, eps, 3).events
                    or cohort(slots, eps, 1 / 60) != cohort(anchors, eps, 3)
                    or not (slots.as_of + pd.Timedelta(hours=25)).lt(dates[1]).all()
                ):
                    raise ValueError("Onset training rows, target support or purge changed")
                np.testing.assert_allclose(
                    [weight.sum(), weight[counts > 0].sum()],
                    [sizes["weight_sum"], sizes["weighted_positive_rows"]],
                    rtol=0,
                    atol=1e-7,
                )
                np.testing.assert_allclose(weight.sum(), len(anchors), rtol=0, atol=1e-7)
            predictions = {name: {} for name in ARMS}
            for part in ("calibration", "policy", "test"):
                dates = tuple(map(pd.Timestamp, meta["periods"][part]))
                reference = base.loc[mask(base, *dates), KEYS]
                hours = align_opportunities(dense.loc[mask(dense, *dates)], reference)
                slots = augmented_slots(hours, triggers, spacing_hours=1, retain_quarters=True)
                quarters = subdivide_evaluation_slots(hours)
                context = pd.read_parquet(
                    FOLDER / "context.parquet",
                    read_dictionary=CATS,
                    filters=[("as_of", ">=", dates[0]), ("as_of", "<", dates[1])],
                )
                features = attach(hours, slots, context, old["features"])
                new_raw = candidate.predict(
                    model_input(features, meta["features"]), prediction_type="RawFormulaVal", thread_count=2
                )
                old_raw = control.predict(
                    model_input(hours, old["features"]), prediction_type="RawFormulaVal", thread_count=2
                )
                hourly_raw = hours[KEYS].assign(raw=old_raw).rename(columns={"as_of": "source_time"})
                exposure = len(hours) / 24
                if result["original_hourly_exposure"][part] != exposure:
                    raise ValueError("Onset exposure changed")
                for name in ARMS:
                    path = directory / f"{name}-{part}.parquet"
                    saved = pd.read_parquet(path)
                    cadence = result["arms"][name]["cadence_hours"]
                    grid = quarters if name == "quarter_control" else slots[KEYS]
                    pd.testing.assert_frame_equal(grid[KEYS], saved[KEYS])
                    if name == "onset_candidate":
                        actual_raw = new_raw
                    else:
                        held = grid.assign(source_time=grid.as_of.dt.floor("h")).merge(
                            hourly_raw, on=["object_id", "source_time"], validate="many_to_one"
                        )
                        actual_raw = held.raw.to_numpy()
                    np.testing.assert_allclose(actual_raw, saved.raw, rtol=1e-10, atol=1e-10)
                    if cohort(saved, eps, cadence) != cohort(hours, eps, 1):
                        raise ValueError("Onset event identities changed")
                    arm = result["arms"][name]
                    if part == "calibration":
                        labels = episode_counts(saved, eps)
                        fitted = calibrate(saved.raw.to_numpy(), labels > 0)
                        if fitted != arm["calibration"]:
                            raise ValueError("Onset calibration changed")
                        np.testing.assert_allclose(
                            labels.sum() / np.exp(np.clip(saved.raw, -20, 20)).sum(),
                            arm["rate_scale"],
                            rtol=1e-12,
                        )
                    np.testing.assert_allclose(
                        calibrated(saved.raw, arm["calibration"]), saved.probability, rtol=1e-10, atol=1e-10
                    )
                    np.testing.assert_allclose(
                        np.exp(np.clip(saved.raw, -20, 20)) * arm["rate_scale"],
                        saved.expected_count,
                        rtol=1e-10,
                        atol=1e-10,
                    )
                    if part in ("policy", "test"):
                        alerts = policy_alerts(saved, eps, arm["policy"])
                        scores = evaluator_for(saved, eps, cadence, exposure).evaluate(alerts, 0.5, cadence)
                        expected = arm["policy"] if part == "policy" else arm["scores"]
                        if any(value != expected[k] for k, value in scores.items()):
                            raise ValueError("Onset warning metric replay failed")
                        if part == "test":
                            np.testing.assert_array_equal(alerts, saved.alert)
                    predictions[name][part] = saved
                    forecasts[str(path)] = sha256(path)
                del context, features
            quarter = result["arms"]["quarter_control"]
            parity = verify_quarter_control(
                control_source(kind, fold),
                quarter["policy"],
                quarter["scores"],
                predictions["quarter_control"],
            )
            if parity != result["archived_quarter_control_parity"]:
                raise ValueError("Quarter control verification changed")
            models[str(fit_path)] = meta
            periods[str(directory / "result.json")] = result
            all_results.append(result)
        screen = all_results[:2]
        scores = {name: pooled([metric(row, name) for row in screen]) for name in (*ARMS, "reference")}
        c = scores["onset_candidate"]
        passed = all(
            primary_score(c) > 1.05 * primary_score(scores[name]) and c["f1"] >= scores[name]["f1"]
            for name in REFERENCES
        )
        if (
            selection[kind] != {**scores, "passed_screen": passed}
            or outcome[kind]["selection"] != selection[kind]
        ):
            raise ValueError("Onset screening decision changed")
        if passed:
            if outcome[kind]["periods"] != all_results:
                raise ValueError("Onset confirmation periods differ")
            pooled_five = {
                name: pooled([metric(row, name) for row in all_results]) for name in (*ARMS, "reference")
            }
            may = all_results[2]
            stress = pooled([metric(row, "onset_candidate") for row in all_results[3:]])
            may_pass = all(
                primary_score(metric(may, "onset_candidate")) > primary_score(metric(may, name))
                and metric(may, "onset_candidate")["f1"] >= 0.95 * metric(may, name)["f1"]
                for name in REFERENCES
            )
            stress_pass = all(
                primary_score(stress)
                > 1.05 * primary_score(pooled([metric(row, name) for row in all_results[3:]]))
                and stress["f1"] >= pooled([metric(row, name) for row in all_results[3:]])["f1"]
                and all(
                    metric(row, "onset_candidate")["f1"] >= 0.9 * metric(row, name)["f1"]
                    for row in all_results[3:]
                )
                for name in REFERENCES
            )
            if (
                outcome[kind]["five_period_pooled"] != pooled_five
                or outcome[kind]["passed_may"] != may_pass
                or outcome[kind]["passed_stress"] != stress_pass
                or outcome[kind]["research_eligible"] != (may_pass and stress_pass)
            ):
                raise ValueError("Onset confirmation decision changed")
        elif outcome[kind] != {
            "selection": selection[kind],
            "research_eligible": False,
            "status": "screen_failed",
        }:
            raise ValueError("Failed onset kind evaluated additional periods")
    if {str(p) for p in root.glob("*/*/onset/fit.json")} != expected_fits:
        raise ValueError("Unexpected onset fits")
    sample_check = verify_context_sample()
    legacy_parity = verify_legacy_input_parity()
    errors = read(root / "error-audit.json")
    if not errors["completed_study"] or len(errors["periods"]) != 3 * len(periods):
        raise ValueError("Onset error audit is incomplete")
    for category in ("source_hashes", "code_hashes"):
        for path, digest in errors[category].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed onset error audit: {path}")
    for row in errors["periods"]:
        result = periods[str(root / row["kind"] / row["fold"] / "result.json")]
        scores = result["arms"][row["arm"]]["scores"]
        if (
            any(row[k] != scores[k] for k in ("true_alerts", "eligible_episodes", "missed_episodes"))
            or row["false_empty"] + row["false_redundant"] != scores["false_alerts"]
        ):
            raise ValueError("Onset error audit changes metric denominators")
    return {
        "report": outcome,
        "build": build,
        "models": models,
        "predictions": forecasts,
        "periods": periods,
        "recomputed_raw_scores_calibration_and_metrics": True,
        "context_value_sample": sample_check,
        "legacy_input_exact_parity": legacy_parity,
        "error_audit": errors,
    }
