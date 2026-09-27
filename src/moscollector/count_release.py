"""Refit only count-model heads that passed both research gates; never read June."""

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts, pending_alerts
from moscollector.model_registry import active_version, load_bundle, manifest_version
from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.precision_research import columns_for, goal_score, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input


def run():
    research = ARTIFACTS / "research-v9"
    selection, confirmation = (
        json.loads((research / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("selection", "confirmation")
    )
    if set(selection) != {"fault", "fire", "access"} or set(confirmation) != set(selection):
        raise ValueError("Finish all count experiments before preparing a release")
    selected = [kind for kind in selection if selection[kind]["passes"] and confirmation[kind]["passes"]]
    if not selected or any(selection[k]["candidate"] != "pending" for k in selected):
        raise ValueError("This release builder requires a gated pending-count policy")
    parent = active_version()
    load_bundle(parent)
    parent_folder = ARTIFACTS / "operational" / parent
    parent_manifest = json.loads((parent_folder / "manifest.json").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp("2026-06-01")
    frame = pd.read_parquet(PROCESSED / "features.parquet", filters=[("as_of", "<", cutoff)])
    dense = pd.read_parquet(
        PROCESSED / "features-dense-rich.parquet",
        columns=frame.columns.tolist(),
        filters=[("as_of", "<", cutoff)],
    )
    episodes = pd.read_parquet(PROCESSED / "episodes.parquet", filters=[("start_ts", "<", cutoff)])
    destination = ARTIFACTS / "operational"
    with tempfile.TemporaryDirectory(prefix="count-build-", dir=destination) as temporary:
        folder = Path(temporary)
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "parent_version": parent,
            "scope": "new_batches_only_historical_replay_unchanged",
            "assessment": "Adaptive retrospective Nov/Feb/May evidence; refitted weights have no new blind test. Episode-level targets are not guaranteed on future months.",
            "selection_sha256": sha256(research / "selection.json"),
            "confirmation_sha256": sha256(research / "confirmation.json"),
            "models": parent_manifest["models"],
        }
        for meta in manifest["models"].values():
            for member in meta["members"]:
                shutil.copy2(parent_folder / member["file"], folder / member["file"])
        for kind in selected:
            periods = periods_for(cutoff, kind)
            columns = columns_for(frame, "recent_reference", kind)
            masks = {k: mask(frame, *periods[k]) for k in ("train", "validation", "calibration")}
            eps = episodes[episodes.kind.eq(kind)]
            counts = episode_counts(frame, eps)
            used = np.logical_or.reduce(list(masks.values()))
            assert np.array_equal(counts[used] > 0, frame.loc[used, f"target_{kind}"].astype(bool))
            x = model_input(frame, columns)
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
            model.fit(
                x[masks["train"]],
                counts[masks["train"]],
                eval_set=(x[masks["validation"]], counts[masks["validation"]]),
            )
            raw = model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal")
            cal = calibrate(raw, frame.loc[masks["calibration"], f"target_{kind}"].to_numpy())
            rate_scale = float(counts[masks["calibration"]].sum() / np.exp(np.clip(raw, -20, 20)).sum())
            reference = frame.loc[mask(frame, *periods["policy"]), ["object_id", "as_of"]]
            rows = align_opportunities(dense.loc[mask(dense, *periods["policy"])], reference)
            pred = rows[["object_id", "as_of"]].copy()
            raw = model.predict(model_input(rows, columns), prediction_type="RawFormulaVal")
            pred["probability"] = calibrated(raw, cal)
            pred["expected_count"] = np.exp(np.clip(raw, -20, 20)) * rate_scale
            evaluator = EventEvaluator(pred, eps, cadence_hours=1)
            policies = []
            for margin in (0.25, 0.5, 0.75, 1, 1.5, 2, 3):
                for floor in (0, 0.25, 0.5, 0.75):
                    scores = pending_alerts(pred, eps, margin, floor)
                    policies.append(
                        {"margin": margin, "probability_floor": floor, **evaluator.evaluate(scores, 0.5, 1)}
                    )
            supported = [
                p
                for p in policies
                if p["alerts"] >= 10
                and p["eligible_episodes"] >= 10
                and p["false_alerts_per_object_day"] <= 0.25
            ]
            if not supported:
                raise ValueError(f"No supported final policy for {kind}")
            policy = max(supported, key=lambda p: (goal_score(p), p["recall"], p["precision"], p["f1"]))
            name = f"{kind}_count.cbm"
            model.save_model(str(folder / name))
            manifest["models"][kind] = {
                "variant": "poisson_pending",
                "features": columns,
                "calibration": cal,
                "rate_scale": rate_scale,
                "threshold": policy["probability_floor"],
                "alert_policy": {
                    "type": "pending_count",
                    "margin": policy["margin"],
                    "probability_floor": policy["probability_floor"],
                    "cadence_hours": 1,
                    "confirmation_minutes": 70,
                },
                "policy_period_metrics": policy,
                "periods": {k: list(map(str, v)) for k, v in periods.items() if k != "test"},
                "training_rows": int(masks["train"].sum()),
                "best_iteration": model.best_iteration_,
                "members": [{"file": name, "weight": 1.0, "sha256": sha256(folder / name)}],
            }
            pred.to_parquet(folder / f"{kind}_policy.parquet", index=False)
        manifest["version"] = manifest_version(manifest)
        write_json(folder / "manifest.json", manifest)
        shutil.copytree(folder, destination / manifest["version"])
    load_bundle(manifest["version"])
    write_json(ARTIFACTS / "candidate_count_release.json", manifest)
    print(manifest["version"], selected, flush=True)


if __name__ == "__main__":
    run()
