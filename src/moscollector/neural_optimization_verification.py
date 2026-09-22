"""Torch-free verification of completed validation-only optimization evidence."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from moscollector.count_validation import profile_count_loss
from moscollector.event_sequence_verification import check_hashes
from moscollector.goal90_research import read
from moscollector.prepare import sha256

CASES = {
    (k, f, v)
    for k in ("access", "fire", "fault")
    for f in ("screen_1", "screen_2")
    for v in ("current_mlp", "event_gru")
}


def verify_scale(path=Path("artifacts/neural_count_scale_audit.json")):
    report = read(path)
    check_hashes(report)
    root = Path("artifacts/research-v39-diagnostic")
    if (
        report["plan_sha256"] != sha256(root / "plan.json")
        or report["models"] != 12
        or report["states_assessed"] != 36
        or len(report["rows"]) != 12
        or {(r["kind"], r["fold"], r["variant"]) for r in report["rows"]} != CASES
    ):
        raise ValueError("Incomplete count scale diagnostic")
    for row in report["rows"]:
        table_path = root / f"{row['kind']}-{row['fold']}-{row['variant']}.parquet"
        if str(table_path) not in report["source_hashes"]:
            raise ValueError("Unprotected validation scores")
        table = pd.read_parquet(table_path)
        if len(table) != row["validation_rows"] or not row["validation_arrays_and_original_losses_replayed"]:
            raise ValueError("Validation population mismatch")
        for name, column in zip(
            ("initial_constant", "selected_best", "final_epoch"),
            ("raw_initial", "raw_best", "raw_final"),
            strict=True,
        ):
            stats = {
                **profile_count_loss(table[column], table.target, table.weight),
                "weighted_row_average_precision": float(
                    average_precision_score(table.target > 0, table[column], sample_weight=table.weight)
                ),
            }
            if stats != row["states"][name]:
                raise ValueError("Validation count decomposition mismatch")
        states = row["states"]
        improves = states["final_epoch"]["profiled_loss"] < states["selected_best"]["profiled_loss"] - 1e-5
        useful = row["best_epoch"] == 0 and states["final_epoch"]["shape_gain_over_constant"] > 1e-5
        if (
            improves != row["profiled_final_improves_selected"]
            or useful != row["epoch0_selected_but_final_has_shape_skill"]
        ):
            raise ValueError("Count scale conclusion mismatch")
    if report["final_profile_improves_selected_models"] != sum(
        r["profiled_final_improves_selected"] for r in report["rows"]
    ) or report["epoch0_selected_but_final_has_shape_skill_models"] != sum(
        r["epoch0_selected_but_final_has_shape_skill"] for r in report["rows"]
    ):
        raise ValueError("Count scale aggregate mismatch")
    return report


def verify_prefix(path=Path("artifacts/event_prefix_audit.json")):
    report = read(path)
    check_hashes(report)
    root = Path("artifacts/research-v39-prefix-fixed")
    parent = Path("artifacts/research-v37-fixed")
    plan = read(root / "plan.json")
    numerical = read(root / "numerical-preflight.json")
    check_hashes(numerical)
    if (
        plan["tensor_tolerances"] != {"rtol": 1.3e-6, "atol": 1e-5}
        or plan["reuse_fits"] != numerical["reuse_fits"]
    ):
        raise ValueError("Changed numerical follow-up or fit reuse")
    if (
        report["plan_sha256"] != sha256(root / "plan.json")
        or report["models"] != 12
        or len(report["rows"]) != 12
        or {(r["kind"], r["fold"], r["variant"]) for r in report["rows"]} != CASES
    ):
        raise ValueError("Incomplete first-epoch diagnostic")
    if (
        report["event_precision_recall_evaluated"]
        or report["serving_changed"]
        or report["goal_achieved"]
        or report["trajectory_equivalence_claimed"]
    ):
        raise ValueError("Validation-only probe makes unsupported quality claims")
    for row in report["rows"]:
        relative = Path(row["kind"]) / row["fold"] / row["variant"]
        directory = root / relative
        fit_directory = Path(plan["reuse_fits"].get(str(relative), str(directory)))
        imported = fit_directory != directory
        if (
            row["fit_directory"] != str(fit_directory)
            or row["imported_unchanged"] != imported
            or row["tensor_tolerances"] != plan["tensor_tolerances"]
        ):
            raise ValueError("Changed first-epoch model source/tolerances")
        fit = read(fit_directory / "fit.json")
        old = read(parent / relative / "fit.json")
        required = {str(fit_directory / n) for n in ("fit.json", "model.pt", "last.pt")}
        if not required.issubset(report["source_hashes"]):
            raise ValueError("Unprotected first-epoch weights/metadata")
        if (
            row["fit_sha256"] != sha256(fit_directory / "fit.json")
            or row["weights_sha256"] != fit["model_sha256"]
            or fit["last_sha256"] != sha256(fit_directory / "last.pt")
        ):
            raise ValueError("Changed first-epoch evidence")
        sig = fit["signature"]
        training_plan = (
            sha256(Path("artifacts/research-v39-prefix/plan.json")) if imported else report["plan_sha256"]
        )
        for key in ("config", "settings", "data_hashes", "event_source_hashes", "device", "torch"):
            if sig[key] != old["signature"][key]:
                raise ValueError("Prefix training differs from original inputs/settings")
        if (
            sig["old_fit_sha256"] != sha256(parent / relative / "fit.json")
            or sig["plan_sha256"] != training_plan
            or sig["epochs_replayed"] != 1
        ):
            raise ValueError("Wrong prefix parent plan")
        step_count = math.ceil(old["signature"]["sizes"]["train"]["rows"] / sig["settings"]["batch_size"])
        interval = max(
            1,
            math.ceil(
                math.ceil(old["signature"]["sizes"]["train"]["original_rows"] / sig["settings"]["batch_size"])
                / 4
            ),
        )
        checks = sorted({0, *range(interval, step_count + 1, interval), step_count})
        if (
            interval != sig["every_steps"]
            or [h["step"] for h in fit["history"]] != checks
            or fit["completed_steps"] != step_count
            or row["checks"] != len(checks)
        ):
            raise ValueError("Wrong within-epoch check schedule")
        best = fit["history"][0]
        for h in fit["history"][1:]:
            if h["validation_loss"] < best["validation_loss"] - sig["settings"]["min_delta"]:
                best = h
            if h["best_step"] != best["step"]:
                raise ValueError("Wrong prefix selection history")
        if (
            best["step"] != row["best_step"]
            or fit["best_step"] != best["step"]
            or fit["best_validation_loss"] != best["validation_loss"]
        ):
            raise ValueError("Wrong selected prefix")
        np.testing.assert_allclose(
            row["best_prefix_validation_loss"], best["validation_loss"], rtol=1e-7, atol=1e-8
        )
        loss_match = bool(
            np.isclose(
                fit["first_epoch_validation_loss"], old["history"][1]["validation_loss"], rtol=1e-6, atol=1e-7
            )
            and np.isclose(
                fit["first_epoch_training_loss"], old["history"][1]["training_loss"], rtol=1e-6, atol=1e-7
            )
        )
        if (
            row["first_epoch_trajectory_losses_match"] != loss_match
            or not row["validation_arrays_and_selected_weights_replayed"]
        ):
            raise ValueError("Numerical guard failure concealed or selected weights unverified")
        if (
            row["end_validation_loss_difference"]
            != fit["first_epoch_validation_loss"] - old["history"][1]["validation_loss"]
            or row["end_training_loss_difference"]
            != fit["first_epoch_training_loss"] - old["history"][1]["training_loss"]
        ):
            raise ValueError("Changed numerical residuals")
        if (
            row["old_best_epoch"] != old["best_epoch"]
            or row["old_best_validation_loss"] != old["best_validation_loss"]
            or row["improves_old_selected"]
            != (
                row["best_prefix_validation_loss"]
                < old["best_validation_loss"] - sig["settings"]["min_delta"]
            )
        ):
            raise ValueError("Prefix comparison differs from original best")
    if report["improved_models"] != sum(r["improves_old_selected"] for r in report["rows"]):
        raise ValueError("Wrong prefix aggregate")
    if report["all_original_loss_guards_passed"] != all(
        r["first_epoch_trajectory_losses_match"] for r in report["rows"]
    ):
        raise ValueError("Wrong numerical guard aggregate")
    return report
