"""Rebuild V38 inputs and replay saved weights, calibration and every policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.fine_cadence_research import policy_alerts, select
from moscollector.flood_research import (
    ARMS,
    FIVE,
    ROOT,
    baseline_raw,
    fit_baseline,
    load_data,
    make_report,
    part_signature,
    parts_for,
    periods_for,
    prepare,
    validate_hashes,
)
from moscollector.goal90_research import read
from moscollector.prepare import sha256, write_json
from moscollector.train import calibrate, calibrated, model_input


def verified_evidence(root=ROOT):
    proof = read(root / "weight-replay.json")
    if (
        proof.get("status") != "exact_replay_passed"
        or proof.get("folds") != list(FIVE)
        or proof.get("models") != 5
        or proof.get("forecast_and_frontier_files") != 40
        or proof.get("plan_sha256") != sha256(root / "plan.json")
    ):
        raise ValueError("Incomplete flood weight replay")
    validate_hashes(proof)
    # Ensure a proof cannot silently omit any completed study artifact.
    expected = {
        str(p)
        for p in root.rglob("*")
        if p.suffix in (".json", ".parquet", ".cbm") and p.name != "weight-replay.json"
    }
    if not expected.issubset(proof["source_hashes"]):
        raise ValueError("Missing flood proof artifact coverage")
    plan = read(root / "plan.json")
    for category in ("source_hashes", "code_hashes"):
        if not set(plan[category].items()).issubset(proof[category].items()):
            raise ValueError("Missing frozen flood inputs")
    return proof


def verify(root=ROOT):
    plan = prepare(root)
    frame, dense, episodes, columns = load_data()
    results, table_count = [], 0
    for fold, begin in FIVE.items():
        directory = root / fold
        parts = parts_for(frame, dense, episodes, begin)
        signatures = {
            p: part_signature(r, episodes, 3 if p in ("train", "validation") else 1) for p, r in parts.items()
        }
        fit = read(directory / "fit.json")
        if fit["parts"] != signatures or signatures != plan["preflight"][fold]["parts"]:
            raise ValueError("Training/validation/calibration/policy/test inputs differ")
        if fit["plan_sha256"] != sha256(root / "plan.json") or fit["features"] != columns:
            raise ValueError("Wrong model plan/schema")
        if fit["model_sha256"] != sha256(directory / "model.cbm"):
            raise ValueError("Changed weights")
        model = CatBoostRegressor()
        model.load_model(str(directory / "model.cbm"))
        if model.tree_count_ != fit["tree_count"] or model.tree_count_ != fit["best_iteration"] + 1:
            raise ValueError("Wrong early-stopping model")
        baseline = fit_baseline(parts["train"], parts["train"].count_target)
        if baseline != read(directory / "baseline.json"):
            raise ValueError("Baseline does not come from the training part")
        saved = read(directory / "result.json")
        expected_periods = {k: list(map(str, v)) for k, v in periods_for(begin).items()}
        if (
            saved["plan_sha256"] != sha256(root / "plan.json")
            or saved["periods"] != expected_periods
            or saved["fold"] != fold
        ):
            raise ValueError("Result fold/period mismatch")
        for arm in ARMS:
            tables = {}
            for part in ("calibration", "policy", "test"):
                rows = parts[part]
                table = rows[["object_id", "as_of"]].copy()
                table["raw"] = (
                    model.predict(model_input(rows, columns), prediction_type="RawFormulaVal", thread_count=4)
                    if arm == "poisson"
                    else baseline_raw(rows, baseline)
                )
                table["count_target"] = rows.count_target.to_numpy()
                tables[part] = table
            cal = tables["calibration"]
            calibration = calibrate(cal.raw.to_numpy(), cal.count_target.to_numpy() > 0)
            scale = float(cal.count_target.sum() / np.exp(np.clip(cal.raw.to_numpy(), -20, 20)).sum())
            if saved["arms"][arm]["calibration"] != calibration or saved["arms"][arm]["rate_scale"] != scale:
                raise ValueError("Calibration replay mismatch")
            for table in tables.values():
                table["probability"] = calibrated(table.raw, calibration)
                table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
            policy, frontier = select(tables["policy"], episodes, 1, len(tables["policy"]) / 24)
            if policy != saved["arms"][arm]["policy"] or frontier != read(directory / f"{arm}-frontier.json"):
                raise ValueError("Full policy search replay mismatch")
            table_count += 1
            for part, table in tables.items():
                if part != "calibration":
                    table["alert"] = policy_alerts(table, episodes, policy)
                    scores = EventEvaluator(table, episodes, 1).evaluate(table.alert, 0.5, 1)
                    if scores != saved["arms"][arm]["scores"][part]:
                        raise ValueError("Event metrics replay mismatch")
                pd.testing.assert_frame_equal(
                    table, pd.read_parquet(directory / f"{arm}-{part}.parquet"), check_exact=True
                )
                table_count += 1
        zero = EventEvaluator(parts["test"], episodes, 1).evaluate(np.zeros(len(parts["test"])), 0.5, 1)
        if zero != saved["disabled_head"]:
            raise ValueError("Disabled baseline changed")
        results.append(saved)
        print("VERIFIED FLOOD", fold, flush=True)
    if make_report(root, results) != read(root / "report.json"):
        raise ValueError("Final pooled report differs from fold results")
    sources = dict(plan["source_hashes"])
    sources.update(
        {
            str(p): sha256(p)
            for p in root.rglob("*")
            if p.suffix in (".json", ".parquet", ".cbm") and p.name != "weight-replay.json"
        }
    )
    proof = {
        "status": "exact_replay_passed",
        "plan_sha256": sha256(root / "plan.json"),
        "folds": list(FIVE),
        "models": 5,
        "forecast_and_frontier_files": table_count,
        "source_hashes": sources,
        "code_hashes": plan["code_hashes"],
        "checks": "Rebuilt every training/evaluation input, count target and train-only baseline; loaded5weights; exact30forecast tables,10complete120-rulefrontiers, calibrations, policies, alerts and pooled event metrics. No fitting or June evaluation.",
    }
    validate_hashes(proof)
    write_json(root / "weight-replay.json", proof)
    return verified_evidence(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    print(verify(parser.parse_args().output)["status"])
