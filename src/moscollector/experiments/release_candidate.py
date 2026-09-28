"""Refit only research-gated heads; no June reads and no implicit activation."""

import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from moscollector.model_registry import load_bundle, manifest_version
from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.prepare import write_json
from moscollector.train import CATEGORICAL, calibrate, calibrated, choose_policy, model_input, split_mask


def run():
    comparison_path = ARTIFACTS / "research-v5" / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if set(comparison) != {"fault", "fire", "access"}:
        raise ValueError("Finish all predeclared experiments before refitting")
    chosen = {k: v["selected"] for k, v in comparison.items() if v["selected"] != "reference"}
    if not chosen:
        raise ValueError("No candidate passed the stability gate")
    selection = json.loads((ARTIFACTS / "model_selection.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(
        PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
    )
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    destination = ARTIFACTS / "operational"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=destination) as directory:
        from pathlib import Path

        folder = Path(directory)
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "scope": "new_batches_only_historical_replay_unchanged",
            "assessment": "Adaptive retrospective validation; newly refitted weights have no new blind test.",
            "comparison_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
            "original_report_sha256": hashlib.sha256(
                (ARTIFACTS / "evaluation_report.json").read_bytes()
            ).hexdigest(),
            "models": {},
        }
        for kind, variant in chosen.items():
            if (
                variant not in ("regularized", "blend")
                or not comparison[kind]["variants"][variant]["passes_gate"]
            ):
                raise ValueError("Candidate has not passed the declared gate")
            legacy_file = ARTIFACTS / "models" / f"{kind}.cbm"
            if hashlib.sha256(legacy_file.read_bytes()).hexdigest() != selection["models"][kind]["sha256"]:
                raise ValueError("Original model changed")
            original = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text(encoding="utf-8"))
            columns = original["features"]
            x, y = model_input(frame, columns), frame[f"target_{kind}"].to_numpy()
            train = split_mask(frame, "train") & frame.as_of.ge(original["training_range"][0])
            validation, calibration_mask, policy_mask = (
                split_mask(frame, p) for p in ("validation", "calibration", "policy")
            )
            age = (
                pd.Timestamp("2026-03-01") - frame.loc[train, "as_of"]
            ).dt.total_seconds().to_numpy() / 86400
            weights = np.exp2(-age / 180)
            weights /= weights.mean()
            model = CatBoostClassifier(
                iterations=1000,
                depth=4,
                learning_rate=0.04,
                l2_leaf_reg=30,
                random_seed=42,
                thread_count=4,
                cat_features=CATEGORICAL,
                loss_function="Logloss",
                eval_metric="PRAUC",
                early_stopping_rounds=80,
                allow_writing_files=False,
                verbose=100,
            )
            model.fit(x[train], y[train], sample_weight=weights, eval_set=(x[validation], y[validation]))
            name = f"{kind}_regularized.cbm"
            model.save_model(str(folder / name))
            members = [(name, model, 1.0 if variant == "regularized" else 0.5)]
            if variant == "blend":
                name = f"{kind}_reference.cbm"
                shutil.copy2(legacy_file, folder / name)
                reference = CatBoostClassifier()
                reference.load_model(str(folder / name))
                members.append((name, reference, 0.5))
            raw_calibration = sum(
                w * m.predict(x[calibration_mask], prediction_type="RawFormulaVal") for _, m, w in members
            )
            calibration = calibrate(raw_calibration, y[calibration_mask])
            pred = frame.loc[policy_mask, ["object_id", "as_of"]].copy()
            pred["probability"] = calibrated(
                sum(w * m.predict(x[policy_mask], prediction_type="RawFormulaVal") for _, m, w in members),
                calibration,
            )
            policy, _ = choose_policy(pred, episodes[episodes.kind.eq(kind)])
            manifest["models"][kind] = {
                "variant": variant,
                "features": columns,
                "calibration": calibration,
                "threshold": policy["threshold"],
                "policy_period": policy,
                "training_range": original["training_range"],
                "training_rows": int(train.sum()),
                "best_iteration": model.best_iteration_,
                "members": [
                    {
                        "file": name,
                        "weight": w,
                        "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
                    }
                    for name, _, w in members
                ],
            }
        manifest["version"] = manifest_version(manifest)
        write_json(folder / "manifest.json", manifest)
        shutil.copytree(folder, destination / manifest["version"])
    load_bundle(manifest["version"])
    write_json(ARTIFACTS / "candidate_release.json", manifest)
    print(manifest["version"], flush=True)


if __name__ == "__main__":
    run()
