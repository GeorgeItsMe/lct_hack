"""Independent v37 weight replay, plus a Torch-free verified-evidence reader."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.goal90_research import read
from moscollector.prepare import sha256, write_json


def check_hashes(manifest):
    for category in ("source_hashes", "code_hashes"):
        for name, digest in manifest[category].items():
            if sha256(Path(name)) != digest:
                raise ValueError(f"Changed raw-onset replay evidence: {name}")


def verified_evidence(root):
    """Base environment can validate completed GPU replay without importing Torch."""
    proof = read(root / "weight-replay.json")
    check_hashes(proof)
    plan = read(root / "plan.json")
    check_hashes(plan)
    if proof["device"] != plan["device"] or proof["torch"] != plan["torch"]:
        raise ValueError("Raw-onset replay backend differs from training")
    if not all(
        proof[key]
        for key in (
            "training_arrays_and_vocabularies_recomputed",
            "raw_scores_calibration_policies_and_metrics_recomputed",
            "all_archived_controls_exact_parity",
        )
    ):
        raise ValueError("Incomplete raw-onset model replay")
    actual = {str(p) for p in root.glob("*/*/result.json")}
    if actual != set(proof["periods"]):
        raise ValueError("Raw-onset replay does not cover all periods")
    actual = {str(p) for p in root.glob("*/*/*/fit.json")}
    if actual != set(proof["models"]):
        raise ValueError("Raw-onset replay does not cover all neural weights")
    required = {root / "plan.json", root / "report.json", root / "selection.json"}
    required.update(Path(p) for p in proof["periods"])
    required.update(Path(p) for p in proof["prediction_and_frontier_hashes"])
    for path in proof["models"]:
        required.update({Path(path), Path(path).parent / "model.pt"})
    if not {str(p) for p in required}.issubset(proof["source_hashes"]):
        raise ValueError("Raw-onset replay omits required evidence hashes")
    if proof["report"] != read(root / "report.json"):
        raise ValueError("Raw-onset replay report differs from completed research")
    return proof


def replay(root, device):
    """Run only after fitting, in the optional neural environment on its saved device."""
    import copy
    import gc

    import torch
    from catboost import CatBoostRegressor

    from moscollector.count_research import episode_counts
    from moscollector.event_sequence_data import FOLDER as EVENT_FOLDER
    from moscollector.event_sequence_data import encode_events, fit_event_codec
    from moscollector.event_sequence_inputs import input_arrays, period_rows, training_part
    from moscollector.event_sequence_model import EventCountNetwork
    from moscollector.event_sequence_research import (
        COUNT_CONTROLS,
        KINDS,
        REFERENCES,
        VARIANTS,
        confirmation,
        metric,
        screening,
    )
    from moscollector.event_sequence_training import array_digest, predict, validation_loss
    from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
    from moscollector.fresh_counts_research import anchor, source_files
    from moscollector.goal90_research import pooled
    from moscollector.minute_cadence_research import forecasts
    from moscollector.neural_count_research import TRAINING
    from moscollector.neural_sequence_data import fit_codec
    from moscollector.onset_channel_features import CATS, KEYS
    from moscollector.onset_channel_features import FOLDER as ONSET_FOLDER
    from moscollector.onset_policy import select_policy
    from moscollector.onset_training_data import attach, model_input
    from moscollector.paths import PROCESSED
    from moscollector.research import mask
    from moscollector.train import CATEGORICAL, calibrate, calibrated

    plan, selected, report = (read(root / name) for name in ("plan.json", "selection.json", "report.json"))
    check_hashes(plan)
    if device != plan["device"] or str(torch.__version__) != plan["torch"]:
        raise ValueError("Replay requires the saved device and Torch version")
    torch.set_num_threads(TRAINING["cpu_threads"])
    if set(selected) != set(KINDS) or set(report) != set(KINDS):
        raise ValueError("Incomplete raw-onset research report")
    sources = set(root.glob("*.json")) - {root / "weight-replay.json"}
    for name, version in (("fresh_controls", "v33"), ("old_controls", "v34")):
        path = root / name / "plan.json"
        child = read(path)
        if child["parent_plan_sha256"] != sha256(root / "plan.json") or child[
            "frozen_source_plan_sha256"
        ] != sha256(Path(f"artifacts/research-{version}/plan.json")):
            raise ValueError("Raw-onset control parent plan changed")
        sources.add(path)
    frame, dense = (
        pd.read_parquet(PROCESSED / name)
        for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    events = pd.read_parquet(EVENT_FOLDER / "events.parquet", read_dictionary=["signal", "sensor_type"])
    triggers = pd.read_parquet(ONSET_FOLDER / "triggers.parquet")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    periods_verified, models_verified, files = {}, {}, {}
    for kind in KINDS:
        folds = ["screen_1", "screen_2"] + (
            ["confirmation", "stress_1", "stress_2"] if selected[kind]["passed_screen"] else []
        )
        eps, rows = episodes.loc[episodes.kind.eq(kind)], []
        for fold in folds:
            directory = root / kind / fold
            result = read(directory / "result.json")
            original_path = source_files(kind, fold)[1]
            original = read(original_path)
            periods = original["periods"]
            anchors = frame.loc[mask(frame, *map(pd.Timestamp, periods["train"]))]
            codec = fit_codec(anchors, original["features"])
            if codec != read(directory / "current-codec.json"):
                raise ValueError("Current codec was not fitted to original training anchors")
            data, sizes = {}, {}
            for part in ("train", "validation"):
                base, slots = period_rows(frame, dense, triggers, periods, part)
                data[part], sizes[part] = training_part(base, slots, events, codec, eps, kind)
            event_codec = fit_event_codec(events, data["train"]["bounds"])
            if event_codec != read(directory / "event-codec.json"):
                raise ValueError("Event vocabulary differs from retained training-only history")
            source = {
                "encoded": encode_events(events, event_codec),
                "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            }
            provenance = {
                "plan_sha256": sha256(root / "plan.json"),
                "kind": kind,
                "fold": fold,
                "sizes": sizes,
                "codec_sha256": sha256(directory / "current-codec.json"),
                "event_codec_sha256": sha256(directory / "event-codec.json"),
            }
            if read(directory / "data.json") != {
                "periods": periods,
                "features": original["features"],
                "sizes": sizes,
                "provenance": provenance,
            }:
                raise ValueError("Raw-onset training observations changed")
            config = {
                "numeric_dim": data["train"]["numeric"].shape[1],
                "category_sizes": [len(codec["vocabulary"][c]) + 1 for c in CATEGORICAL],
                "event_category_sizes": [
                    len(event_codec["vocabulary"][c]) + 1 for c in ("channel_id", "sensor_type")
                ]
                + [7],
                "initial_log_mean": float(
                    np.log(
                        max(float(np.average(data["train"]["target"], weights=data["train"]["weight"])), 1e-6)
                    )
                ),
            }
            data_hashes = {
                part: {name: array_digest(value) for name, value in values.items()}
                for part, values in data.items()
            }
            event_hashes = {name: array_digest(value) for name, value in source.items()}
            models = {}
            for variant in VARIANTS:
                fit_path = directory / variant / "fit.json"
                fit = read(fit_path)
                if "reused_from" in fit:
                    origin = fit["reused_from"]
                    original_fit = Path(origin["fit_file"])
                    original_weights = Path(origin["weights_file"])
                    if (
                        sha256(original_fit) != origin["fit_sha256"]
                        or sha256(original_weights) != origin["model_sha256"]
                    ):
                        raise ValueError("Changed pre-fix neural weights")
                    derived = copy.deepcopy(read(original_fit))
                    if derived["signature"]["plan_sha256"] != origin["original_plan_sha256"]:
                        raise ValueError("Changed pre-fix neural plan")
                    derived["signature"]["plan_sha256"] = sha256(root / "plan.json")
                    derived["weights_file"] = str(directory / variant / "model.pt")
                    derived["reused_from"] = origin
                    if derived != fit or origin["model_sha256"] != fit["model_sha256"]:
                        raise ValueError("Schema fix changed trained model metadata")
                network_config = {**config, "use_history": variant == "event_gru"}
                expected = {
                    **provenance,
                    "config": network_config,
                    "settings": TRAINING,
                    "device": device,
                    "torch": str(torch.__version__),
                    "data_hashes": data_hashes,
                    "event_source_hashes": event_hashes,
                }
                weights = directory / variant / "model.pt"
                if (
                    fit["signature"] != expected
                    or fit["config"] != network_config
                    or fit != result["fits"][variant]
                    or fit["model_sha256"] != sha256(weights)
                    or fit["device"] != device
                ):
                    raise ValueError("Raw-onset model provenance changed")
                model = EventCountNetwork(**network_config).to(device)
                model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
                val = validation_loss(model, data["validation"], source, device)
                np.testing.assert_allclose(val, fit["best_validation_loss"], rtol=1e-7, atol=1e-8)
                best, bad = fit["history"][0], 0
                for row in fit["history"][1:]:
                    if row["validation_loss"] < best["validation_loss"] - TRAINING["min_delta"]:
                        best, bad = row, 0
                    else:
                        bad += 1
                    if row["best_epoch"] != best["epoch"]:
                        raise ValueError("Changed best-epoch history")
                if (
                    best["epoch"] != fit["best_epoch"]
                    or best["validation_loss"] != fit["best_validation_loss"]
                    or [row["epoch"] for row in fit["history"]] != list(range(fit["completed_epochs"] + 1))
                    or not (fit["completed_epochs"] == TRAINING["epochs"] or bad == TRAINING["patience"])
                    or sum(p.numel() for p in model.parameters()) != fit["parameter_count"]
                ):
                    raise ValueError("Changed early stopping or model architecture")
                models[variant] = model
                models_verified[str(fit_path)] = fit
                sources.update({fit_path, weights})
            del data, anchors
            gc.collect()
            fresh, old = Path(result["fresh_control"]), Path(result["old_control"])
            fresh_meta, old_result = read(fresh / "fit.json"), read(old / "result.json")
            if (
                result["plan_sha256"] != sha256(root / "plan.json")
                or result["fresh_fit_sha256"] != sha256(fresh / "fit.json")
                or result["old_result_sha256"] != sha256(old / "result.json")
                or fresh_meta["model_sha256"] != sha256(fresh / "model.cbm")
                or not result["same_episode_cohort"]
                or not result["same_training_observations"]
            ):
                raise ValueError("Changed raw-onset control provenance")
            for part, size in fresh_meta["sizes"].items():
                if any(sizes[part][k] != v for k, v in size.items()):
                    raise ValueError("Count/neural training observations differ")
            count_model = CatBoostRegressor()
            count_model.load_model(str(fresh / "model.cbm"))
            context = pd.read_parquet(
                ONSET_FOLDER / "context.parquet",
                filters=[
                    ("as_of", ">=", pd.Timestamp(periods["calibration"][0])),
                    ("as_of", "<", pd.Timestamp(periods["test"][1])),
                ],
                read_dictionary=CATS,
            )
            old_tables, old_cal, exposure, _ = forecasts(kind, fold, frame, dense, triggers, eps)
            if exposure != result["exposure"] or exposure != old_result["exposure"]:
                raise ValueError("Raw-onset exposure changed")
            tables = {name: {} for name in (*VARIANTS, *COUNT_CONTROLS)}
            for part in ("calibration", "policy", "test"):
                base, slots = period_rows(frame, dense, triggers, periods, part)
                if exposure[part] != len(base) / 24 or cohort(slots, eps, 1 / 60) != cohort(base, eps, 1):
                    raise ValueError("Raw-onset opportunity/event identities changed")
                inputs = input_arrays(base, slots, events, codec)
                for variant in VARIANTS:
                    tables[variant][part] = (
                        slots[KEYS].copy().assign(raw=predict(models[variant], inputs, source, device))
                    )
                values = attach(base, slots, context, original["features"])
                tables["fresh_count"][part] = (
                    slots[KEYS]
                    .copy()
                    .assign(
                        raw=count_model.predict(
                            model_input(values, fresh_meta["features"]),
                            prediction_type="RawFormulaVal",
                            thread_count=2,
                        )
                    )
                )
                tables["old_count"][part] = old_tables["minute_candidate"][part]
            for arm, predictions in tables.items():
                record = result["arms"][arm]
                if arm == "old_count":
                    calibration, scale = (
                        old_cal["minute_candidate"][k] for k in ("calibration", "rate_scale")
                    )
                else:
                    cal = predictions["calibration"]
                    counts = episode_counts(cal, eps)
                    calibration = calibrate(cal.raw.to_numpy(), counts > 0)
                    scale = float(counts.sum() / np.exp(np.clip(cal.raw, -20, 20)).sum())
                    for pred in predictions.values():
                        pred["probability"] = calibrated(pred.raw, calibration)
                        pred["expected_count"] = np.exp(np.clip(pred.raw, -20, 20)) * scale
                if record["calibration"] != calibration or record["rate_scale"] != scale:
                    raise ValueError("Raw-onset calibration changed on weight replay")
                policy, frontier = select_policy(predictions["policy"], eps, exposure["policy"])
                frontier_path = directory / f"{arm}-frontier.json"
                if policy != record["policy"] or frontier != read(frontier_path):
                    raise ValueError("Raw-onset policy search changed")
                archived = (
                    old
                    if arm == "old_count"
                    else (
                        Path("artifacts/research-v33") / kind / fold
                        if arm == "fresh_count" and fold.startswith("screen_")
                        else None
                    )
                )
                archived_arm = "minute_candidate" if arm == "old_count" else "onset_candidate"
                if archived:
                    archived_record = read(archived / "result.json")["arms"][archived_arm]
                    if any(archived_record[k] != v for k, v in record.items()) or frontier != read(
                        archived / f"{archived_arm}-frontier.json"
                    ):
                        raise ValueError("Archived count control changed")
                files[str(frontier_path)] = sha256(frontier_path)
                for part, prediction in predictions.items():
                    if part != "calibration":
                        prediction["alert"] = policy_alerts(prediction, eps, policy)
                        scores = evaluator_for(prediction, eps, 1 / 60, exposure[part]).evaluate(
                            prediction.alert, 0.5, 1 / 60
                        )
                        expected_scores = record["scores"] if part == "test" else policy
                        if any(expected_scores[k] != v for k, v in scores.items()):
                            raise ValueError("Raw-onset event metrics changed")
                    path = directory / f"{arm}-{part}.parquet"
                    pd.testing.assert_frame_equal(prediction, pd.read_parquet(path), check_exact=True)
                    if archived:
                        archived_prediction = pd.read_parquet(archived / f"{archived_arm}-{part}.parquet")
                        if arm == "fresh_count" and part == "policy":
                            archived_prediction["alert"] = policy_alerts(
                                archived_prediction, eps, archived_record["policy"]
                            )
                        pd.testing.assert_frame_equal(
                            prediction,
                            archived_prediction,
                            check_exact=True,
                        )
                    files[str(path)] = sha256(path)
            if result["reference"] != anchor(kind, fold) or any(
                record["scores"]["eligible_episodes"] != result["reference"]["eligible_episodes"]
                for record in result["arms"].values()
            ):
                raise ValueError("Raw-onset historical comparison changed")
            sources.update({original_path, fresh / "fit.json", fresh / "model.cbm", old / "result.json"})
            sources.update(directory.glob("*.json"))
            periods_verified[str(directory / "result.json")] = result
            rows.append(result)
            del models, model, source, tables, inputs, context, old_tables
            gc.collect()
            print("VERIFIED event network", kind, fold, "training, weights, all4arms exact", flush=True)
        if screening(rows[:2]) != selected[kind] or report[kind]["selection"] != selected[kind]:
            raise ValueError("Raw-onset screening changed")
        if selected[kind]["passed_screen"]:
            chosen = selected[kind]["selected_variant"]
            gates = confirmation(rows[2:], chosen)
            totals = {name: pooled([metric(row, name) for row in rows]) for name in (*VARIANTS, *REFERENCES)}
            if (
                report[kind]["selected_variant"] != chosen
                or report[kind]["periods"] != rows
                or report[kind]["five_period_pooled"] != totals
                or any(report[kind][k] != v for k, v in gates.items())
                or report[kind]["automatic_activation"]
            ):
                raise ValueError("Raw-onset confirmation or chosen variant changed")
        elif report[kind] != {
            "selection": selected[kind],
            "research_eligible": False,
            "status": "screen_failed",
        }:
            raise ValueError("Screen-failing raw-onset head opened extra months")
    if set(periods_verified) != {str(p) for p in root.glob("*/*/result.json")} or set(models_verified) != {
        str(p) for p in root.glob("*/*/*/fit.json")
    }:
        raise ValueError("Unexpected raw-onset neural periods or models")
    sources.update(Path(p) for p in files)
    proof = {
        "device": device,
        "torch": str(torch.__version__),
        "report": report,
        "periods": periods_verified,
        "models": models_verified,
        "prediction_and_frontier_hashes": files,
        "training_arrays_and_vocabularies_recomputed": True,
        "raw_scores_calibration_policies_and_metrics_recomputed": True,
        "all_archived_controls_exact_parity": True,
        "source_hashes": {**plan["source_hashes"], **{str(p): sha256(p) for p in sorted(sources)}},
        "code_hashes": {**plan["code_hashes"], str(Path(__file__)): sha256(Path(__file__))},
    }
    write_json(root / "weight-replay.json", proof)
    return verified_evidence(root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/research-v37-fixed"))
    parser.add_argument("--device", default="mps", choices=("cpu", "mps"))
    args = parser.parse_args()
    result = replay(args.root, args.device)
    print("VERIFIED", len(result["models"]), "networks and", len(result["periods"]), "periods")
