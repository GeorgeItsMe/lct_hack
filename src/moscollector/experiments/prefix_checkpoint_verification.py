"""Replay new early-checkpoint forecasts; base runtime can verify its proof."""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.count_research import episode_counts
from moscollector.experiments.event_sequence_data import FOLDER as EVENT_FOLDER
from moscollector.experiments.event_sequence_data import encode_events
from moscollector.experiments.event_sequence_inputs import input_arrays, period_rows
from moscollector.experiments.event_sequence_verification import check_hashes
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.experiments.goal90_research import read
from moscollector.experiments.onset_channel_features import FOLDER as ONSET_FOLDER
from moscollector.experiments.onset_policy import select_policy
from moscollector.experiments.prefix_checkpoint_research import (
    FOLDS,
    KINDS,
    PARENT,
    REFERENCES,
    ROOT,
    VARIANTS,
    screening,
)
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.train import calibrate, calibrated


def verified_evidence(root=ROOT):
    proof = read(root / "weight-replay.json")
    check_hashes(proof)
    if (
        proof["status"] != "exact_screen_replay_passed"
        or proof["plan_sha256"] != sha256(root / "plan.json")
        or proof["periods"] != 6
        or proof["forecast_and_frontier_files"] != 48
    ):
        raise ValueError("Incomplete checkpoint screen replay")
    required = {
        str(p)
        for p in root.rglob("*")
        if p.suffix in (".json", ".parquet") and p.name != "weight-replay.json"
    }
    if not required.issubset(proof["source_hashes"]) or proof["report"] != read(root / "report.json"):
        raise ValueError("Checkpoint screen proof omits or differs from result files")
    return proof


def verify(root=ROOT):
    import torch

    from moscollector.experiments.event_sequence_model import EventCountNetwork
    from moscollector.experiments.event_sequence_training import predict

    plan = read(root / "plan.json")
    check_hashes(plan)
    if (
        plan["torch"] != str(torch.__version__)
        or plan["device"] != "mps"
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("Need original checkpoint inference backend")
    torch.set_num_threads(4)
    report = read(root / "report.json")
    if set(report) != set(KINDS):
        raise ValueError("Missing incident direction")
    frame, dense = (
        pd.read_parquet(PROCESSED / n)
        for n in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    events = pd.read_parquet(EVENT_FOLDER / "events.parquet", read_dictionary=["signal", "sensor_type"])
    triggers = pd.read_parquet(ONSET_FOLDER / "triggers.parquet")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    files, changed = 0, 0
    for kind in KINDS:
        eps = episodes.loc[episodes.kind.eq(kind)]
        results = []
        for fold in FOLDS:
            directory = root / kind / fold
            parent = PARENT / kind / fold
            result = read(directory / "result.json")
            original = read(parent / "result.json")
            preparation = read(parent / "data.json")
            codec, event_codec = (read(parent / n) for n in ("current-codec.json", "event-codec.json"))
            source = {
                "encoded": encode_events(events, event_codec),
                "times": events.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            }
            if (
                result["kind"] != kind
                or result["fold"] != fold
                or result["exposure"] != original["exposure"]
                or result["plan_sha256"] != sha256(root / "plan.json")
            ):
                raise ValueError("Changed checkpoint period/provenance")
            for variant in VARIANTS:
                item = plan["selections"][f"{kind}/{fold}/{variant}"]
                if (
                    item != result["selected_checkpoints"][variant]
                    or sha256(Path(item["weights"])) != item["weights_sha256"]
                ):
                    raise ValueError("Wrong selected checkpoint")
                model = None
                if item["uses_prefix"]:
                    model = EventCountNetwork(**item["config"]).to("mps")
                    model.load_state_dict(torch.load(item["weights"], map_location="mps", weights_only=True))
                    changed += 1
                tables = {}
                for part in ("calibration", "policy", "test"):
                    previous = pd.read_parquet(parent / f"{variant}-{part}.parquet")
                    table = previous[["object_id", "as_of", "raw"]].copy()
                    if model is not None:
                        base, slots = period_rows(frame, dense, triggers, preparation["periods"], part)
                        pd.testing.assert_frame_equal(
                            table[["object_id", "as_of"]], slots[["object_id", "as_of"]]
                        )
                        if (
                            cohort(slots, eps, 1 / 60) != cohort(base, eps, 1)
                            or len(base) / 24 != result["exposure"][part]
                        ):
                            raise ValueError("Checkpoint replay changed opportunities/cohort")
                        data = input_arrays(base, slots, events, codec)
                        table["raw"] = predict(model, data, source, "mps")
                        del data
                    tables[part] = table
                cal = tables["calibration"]
                counts = episode_counts(cal, eps)
                calibration = calibrate(cal.raw.to_numpy(), counts > 0)
                scale = float(counts.sum() / np.exp(np.clip(cal.raw, -20, 20)).sum())
                for table in tables.values():
                    table["probability"] = calibrated(table.raw, calibration)
                    table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
                policy, frontier = select_policy(tables["policy"], eps, result["exposure"]["policy"])
                if frontier != read(directory / f"{variant}-frontier.json"):
                    raise ValueError("Changed full checkpoint policy search")
                files += 1
                for part in ("policy", "test"):
                    tables[part]["alert"] = policy_alerts(tables[part], eps, policy)
                scores = evaluator_for(tables["test"], eps, 1 / 60, result["exposure"]["test"]).evaluate(
                    tables["test"].alert, 0.5, 1 / 60
                )
                if {
                    "scores": scores,
                    "policy": policy,
                    "calibration": calibration,
                    "rate_scale": scale,
                } != result["arms"][variant]:
                    raise ValueError("Checkpoint event score/calibration/policy replay mismatch")
                for part, table in tables.items():
                    pd.testing.assert_frame_equal(
                        table, pd.read_parquet(directory / f"{variant}-{part}.parquet"), check_exact=True
                    )
                    if model is None:
                        pd.testing.assert_frame_equal(
                            table, pd.read_parquet(parent / f"{variant}-{part}.parquet"), check_exact=True
                        )
                    files += 1
                if result["old_neural"][variant] != original["arms"][variant]["scores"]:
                    raise ValueError("Changed original neural comparison")
                del model, tables
                gc.collect()
            controls = {
                r: original["reference"] if r == "reference" else original["arms"][r]["scores"]
                for r in REFERENCES
            }
            if result["controls"] != controls:
                raise ValueError("Changed stronger references")
            results.append(result)
            print("VERIFIED CHECKPOINT", kind, fold, flush=True)
        selection = screening(results)
        expected = {
            "selection": selection,
            "periods": results,
            "status": "requires_linked_confirmation" if selection["passed_screen"] else "screen_rejected",
            "goal_achieved": False,
            "serving_changed": False,
        }
        if report[kind] != expected:
            raise ValueError("Changed selection or unsupported completion claim")
    sources = dict(plan["source_hashes"])
    sources.update(
        {
            str(p): sha256(p)
            for p in root.rglob("*")
            if p.suffix in (".json", ".parquet") and p.name != "weight-replay.json"
        }
    )
    proof = {
        "status": "exact_screen_replay_passed",
        "plan_sha256": sha256(root / "plan.json"),
        "periods": 6,
        "forecast_and_frontier_files": files,
        "new_checkpoint_inference_models": changed,
        "unchanged_archived_models": 12 - changed,
        "source_hashes": sources,
        "code_hashes": plan["code_hashes"],
        "report": report,
    }
    check_hashes(proof)
    write_json(root / "weight-replay.json", proof)
    return verified_evidence(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    print(verify(parser.parse_args().output)["status"])
