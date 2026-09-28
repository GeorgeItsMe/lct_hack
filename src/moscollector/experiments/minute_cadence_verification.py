"""Recompute v34 forecasts, exposure, warning metrics and continuation decisions."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.experiments.fine_cadence_research import evaluator_for, policy_alerts
from moscollector.experiments.fresh_counts_research import source_files
from moscollector.experiments.goal90_research import pooled, read
from moscollector.experiments.minute_cadence_research import (
    ARMS,
    KINDS,
    PRIOR,
    confirmation,
    forecasts,
    metric,
    screening,
)
from moscollector.experiments.onset_channel_features import FOLDER, KEYS
from moscollector.experiments.onset_count_research import verify_quarter_control
from moscollector.experiments.peer_context_research import control_source
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256


def verify(root=Path("artifacts/research-v34")):
    plan, report, selected = (read(root / name) for name in ("plan.json", "report.json", "selection.json"))
    for category in ("source_hashes", "code_hashes"):
        for path, digest in plan[category].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Changed minute cadence source: {path}")
    if set(report) != set(KINDS) or set(selected) != set(KINDS):
        raise ValueError("Incomplete cadence report")
    if list(root.rglob("*.cbm")):
        raise ValueError("Unexpected new cadence model weights")
    frame, dense = (
        pd.read_parquet(PROCESSED / name)
        for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    triggers = pd.read_parquet(FOLDER / "triggers.parquet")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    periods, files, expected = {}, {}, set()
    for kind in KINDS:
        folds = ["screen_1", "screen_2"] + (
            ["confirmation", "stress_1", "stress_2"] if selected[kind]["passed_screen"] else []
        )
        rows = []
        eps = episodes.loc[episodes.kind.eq(kind)]
        for fold in folds:
            directory = root / kind / fold
            path = directory / "result.json"
            expected.add(str(path))
            result = read(path)
            weights, metadata, _ = source_files(kind, fold)
            if (
                result["model_sha256"] != sha256(weights)
                or result["metadata_sha256"] != sha256(metadata)
                or result["plan_sha256"] != sha256(root / "plan.json")
                or result["new_weights"]
            ):
                raise ValueError("Changed cadence model provenance")
            tables, calibrations, exposure, sizes = forecasts(kind, fold, frame, dense, triggers, eps)
            if (
                exposure != result["exposure"]
                or sizes != result["opportunities"]
                or result["reused_known_screening"] != fold.startswith("screen_")
            ):
                raise ValueError("Cadence opportunities or reuse status changed")
            for arm in ARMS:
                record = result["arms"][arm]
                if any(record[k] != value for k, value in calibrations[arm].items()):
                    raise ValueError("Cadence calibration changed")
                saved_predictions = {}
                cadence = 1 / 60 if arm == "minute_candidate" else 0.25
                if record["cadence_hours"] != cadence:
                    raise ValueError("Changed cooldown")
                old_arm = "minute_control" if arm == "minute_candidate" else arm
                if fold.startswith("screen_"):
                    archived = read(PRIOR / kind / fold / "result.json")["arms"][old_arm]
                    if record["policy"] != archived["policy"] or record["scores"] != archived["scores"]:
                        raise ValueError("Known screening decision changed")
                for part, actual in tables[arm].items():
                    source = directory / f"{arm}-{part}.parquet"
                    saved = pd.read_parquet(source)
                    pd.testing.assert_frame_equal(actual[KEYS], saved[KEYS])
                    for column in ("raw", "probability", "expected_count"):
                        np.testing.assert_allclose(actual[column], saved[column], rtol=1e-10, atol=1e-10)
                    if part != "calibration":
                        alerts = policy_alerts(actual, eps, record["policy"])
                        np.testing.assert_array_equal(alerts, saved.alert)
                        metrics = evaluator_for(actual, eps, cadence, exposure[part]).evaluate(
                            alerts, 0.5, cadence
                        )
                        saved_metrics = record["scores"] if part == "test" else record["policy"]
                        if any(value != saved_metrics[k] for k, value in metrics.items()):
                            raise ValueError("Cadence warning metric replay failed")
                    files[str(source)] = sha256(source)
                    saved_predictions[part] = saved
                if arm == "quarter_control":
                    parity = verify_quarter_control(
                        control_source(kind, fold), record["policy"], record["scores"], saved_predictions
                    )
                    if parity != record["archived_quarter_parity"]:
                        raise ValueError("Archived quarter comparison changed")
                frontier = directory / f"{arm}-frontier.json"
                if len(read(frontier)) != 456:
                    raise ValueError("Changed cadence policy grid")
                files[str(frontier)] = sha256(frontier)
            rows.append(result)
            periods[str(path)] = result
        if screening(rows[:2]) != selected[kind] or report[kind]["selection"] != selected[kind]:
            raise ValueError("Cadence screening decision changed")
        if selected[kind]["passed_screen"]:
            gates = confirmation(rows[2:])
            aggregates = {name: pooled([metric(row, name) for row in rows]) for name in (*ARMS, "reference")}
            if (
                report[kind]["periods"] != rows
                or report[kind]["five_period_pooled"] != aggregates
                or any(report[kind][k] != value for k, value in gates.items())
                or report[kind]["automatic_activation"]
            ):
                raise ValueError("Cadence confirmation decision changed")
        elif report[kind] != {
            "selection": selected[kind],
            "research_eligible": False,
            "status": "screen_failed",
        }:
            raise ValueError("Failed cadence kind evaluated additional periods")
    if {str(p) for p in root.glob("*/*/result.json")} != expected:
        raise ValueError("Unexpected cadence periods")
    return {
        "report": report,
        "periods": periods,
        "prediction_and_frontier_hashes": files,
        "all_raw_scores_calibration_and_metrics_recomputed": True,
        "new_tree_weights": 0,
    }
