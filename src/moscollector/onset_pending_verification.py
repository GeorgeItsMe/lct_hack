"""Replay v36 compositions, requiring already recomputed underlying components."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.fresh_counts_research import anchor
from moscollector.goal90_research import pooled, read
from moscollector.onset_channel_features import KEYS
from moscollector.onset_pending_research import ARMS, KINDS, PRIOR, compose, confirmation, metric, screening
from moscollector.onset_policy import select_policy
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256


def verify(root, verified_components, verified_pending):
    if not (
        verified_components["all_raw_scores_calibration_and_metrics_recomputed"]
        and verified_pending["all_raw_scores_calibration_and_metrics_recomputed"]
    ):
        raise ValueError("Underlying classifier components require model replay first")
    plan, selected, report = (read(root / name) for name in ("plan.json", "selection.json", "report.json"))
    audit = read(root / "motivation-audit.json")
    for manifest in (plan, audit):
        for category in ("source_hashes", "code_hashes"):
            for path, digest in manifest[category].items():
                if sha256(Path(path)) != digest:
                    raise ValueError(f"Changed onset pending provenance: {path}")
    if set(selected) != set(KINDS) or set(report) != set(KINDS):
        raise ValueError("Incomplete onset pending report")
    child = root / "component_controls"
    for destination, parent, old in (
        (child / "plan.json", root / "plan.json", PRIOR / "plan.json"),
        (child / "count_controls/plan.json", child / "plan.json", Path("artifacts/research-v33/plan.json")),
        (child / "pending_controls/plan.json", child / "plan.json", Path("artifacts/research-v34/plan.json")),
    ):
        record = read(destination)
        if record["parent_plan_sha256"] != sha256(parent) or record["frozen_source_plan_sha256"] != sha256(
            old
        ):
            raise ValueError("Component plan linkage changed")
    prior_files = {
        **verified_pending["prediction_and_frontier_hashes"],
        **verified_components["prediction_and_frontier_hashes"],
    }
    known_fits = {**verified_components["models"], **verified_components["count_controls"]}
    # Newly trained components, if any, cannot pass by metadata alone. The caller
    # must supply a model-replayed component audit covering their exact paths.
    for path in child.rglob("fit.json"):
        if str(path) not in known_fits or read(path) != known_fits[str(path)]:
            raise ValueError("New component has not undergone classifier/count model replay")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    files, periods, expected_results = {}, {}, set()
    for kind in KINDS:
        folds = ["screen_1", "screen_2"] + (
            ["confirmation", "stress_1", "stress_2"] if selected[kind]["passed_screen"] else []
        )
        eps = episodes.loc[episodes.kind.eq(kind)]
        rows = []
        for fold in folds:
            directory = root / kind / fold
            target = directory / "result.json"
            expected_results.add(str(target))
            result = read(target)
            source, pending = Path(result["component"]), Path(result["pending"])
            component, old = read(source / "result.json"), read(pending / "result.json")
            component_key, pending_key = str(source / "result.json"), str(pending / "result.json")
            if (
                component_key not in verified_components["periods"]
                or component != verified_components["periods"][component_key]
                or pending_key not in verified_components["pending_controls"]
                or old != verified_components["pending_controls"][pending_key]
                or result["plan_sha256"] != sha256(root / "plan.json")
                or result["component_result_sha256"] != sha256(source / "result.json")
                or result["pending_result_sha256"] != sha256(pending / "result.json")
                or result["exposure"] != component["exposure"]
                or result["exposure"] != old["exposure"]
                or result["direct_control"] != component["arms"]["binary_candidate"]["scores"]
                or result["reference"] != anchor(kind, fold)
            ):
                raise ValueError("Composition references are changed or lack full model replay")
            tables = {name: {} for name in ARMS}
            for part in ("calibration", "policy", "test"):
                binary_path = source / f"binary_candidate-{part}.parquet"
                count_path = pending / f"minute_candidate-{part}.parquet"
                for path in (binary_path, count_path):
                    if str(path) not in prior_files or sha256(path) != prior_files[str(path)]:
                        raise ValueError("Composition source lacks recomputed forecast provenance")
                binary, count = pd.read_parquet(binary_path), pd.read_parquet(count_path)
                pd.testing.assert_frame_equal(binary[KEYS], count[KEYS])
                if cohort(binary, eps, 1 / 60) != cohort(count, eps, 1 / 60):
                    raise ValueError("Composition changed event identities")
                tables["pending_candidate"][part] = compose(binary, count)
                tables["count_control"][part] = count.drop(columns="alert", errors="ignore")
            for arm in ARMS:
                record = result["arms"][arm]
                policy, frontier = select_policy(tables[arm]["policy"], eps, result["exposure"]["policy"])
                path = directory / f"{arm}-frontier.json"
                if policy != record["policy"] or frontier != read(path):
                    raise ValueError("Pending composition frontier or choice changed")
                files[str(path)] = sha256(path)
                for part, actual in tables[arm].items():
                    path = directory / f"{arm}-{part}.parquet"
                    saved = pd.read_parquet(path)
                    pd.testing.assert_frame_equal(actual, saved.drop(columns="alert", errors="ignore"))
                    if part != "calibration":
                        alerts = policy_alerts(actual, eps, policy)
                        np.testing.assert_array_equal(alerts, saved.alert)
                        scores = evaluator_for(actual, eps, 1 / 60, result["exposure"][part]).evaluate(
                            alerts, 0.5, 1 / 60
                        )
                        expected = record["scores"] if part == "test" else policy
                        if any(v != expected[k] for k, v in scores.items()):
                            raise ValueError("Pending composition warning metrics changed")
                    if arm == "count_control":
                        pd.testing.assert_frame_equal(
                            saved, pd.read_parquet(pending / f"minute_candidate-{part}.parquet")
                        )
                    files[str(path)] = sha256(path)
                if arm == "count_control":
                    if (
                        policy != old["arms"]["minute_candidate"]["policy"]
                        or record["scores"] != old["arms"]["minute_candidate"]["scores"]
                    ):
                        raise ValueError("Archived pending control parity failed")
                    if frontier != read(pending / "minute_candidate-frontier.json"):
                        raise ValueError("Archived pending frontier parity failed")
            rows.append(result)
            periods[str(target)] = result
        if screening(rows[:2]) != selected[kind] or report[kind]["selection"] != selected[kind]:
            raise ValueError("Pending composition screening changed")
        if selected[kind]["passed_screen"]:
            gates = confirmation(rows[2:])
            totals = {
                name: pooled([metric(r, name) for r in rows])
                for name in (*ARMS, "direct_control", "reference")
            }
            if (
                report[kind]["periods"] != rows
                or report[kind]["five_period_pooled"] != totals
                or any(report[kind][k] != v for k, v in gates.items())
                or report[kind]["automatic_activation"]
            ):
                raise ValueError("Pending composition confirmation changed")
        elif report[kind] != {
            "selection": selected[kind],
            "research_eligible": False,
            "status": "screen_failed",
        }:
            raise ValueError("Failed composition opened additional months")
    if {str(p) for p in root.glob("*/*/result.json")} != expected_results:
        raise ValueError("Unexpected composition periods")
    return {
        "report": report,
        "motivation_audit": audit,
        "periods": periods,
        "prediction_and_frontier_hashes": files,
        "all_component_forecasts_verified_by_model_replay": True,
        "all_compositions_policies_and_metrics_recomputed": True,
        "all_count_controls_exact_parity": True,
        "new_component_models": len(list(child.rglob("fit.json"))),
    }
