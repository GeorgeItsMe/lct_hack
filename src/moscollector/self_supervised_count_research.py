"""v31: telemetry-forecast pretraining before unchanged GRU count fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.binary_gate_research import KINDS, select_gate_policy
from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.fresh_counts_research import source_files
from moscollector.goal90_research import STRESS, pooled, primary_score, read
from moscollector.neural_count_model import CountNetwork
from moscollector.neural_count_research import TRAINING, build_data, predict
from moscollector.neural_count_research import evaluate as evaluate_original
from moscollector.neural_sequence_data import HISTORY_HOURS, encode, fit_codec
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.quarter_count_features import carry_features
from moscollector.research import FOLDS, mask
from moscollector.self_supervised_model import (
    HistoryPredictor,
    pretext_baselines,
    telemetry_indices,
    transfer_history,
)
from moscollector.self_supervised_training import PRETRAIN, fit_stage
from moscollector.train import CATEGORICAL, calibrate, calibrated

REFERENCES = ("scratch_gru", "global_control", "reference")
PLAN = {
    "scope": "adaptive_retrospective_telemetry_pretraining_study_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "hypothesis": "V28 GRU count fitting often stops within a few epochs, with sparse incident support. Test whether a history encoder first trained to forecast dynamic telemetry yields a better initialization. Same rows and inference architecture; extra optimization is explicit and not a compute-matched causal isolation of pretext semantics.",
    "pretext": "Predict the current normalized dynamic telemetry columns solely from the same16 strictly-earlier snapshots within48h. Targets exclude calendar, static inventory and all past_* episode summaries; inputs remain original causal histories. Current numeric/categorical inputs and24h episode targets never enter this loss. Missing current values and rows with no history are masked. Target cells use SmoothL1 beta1, averaged per minibatch; validation averaged over all observed cells. Same eligible train/validation rows and25h purges as v28, no extra or external data.",
    "pretext_network": "Copy only history_projection and GRU from a seed42 initialized original CountNetwork. Decoder32->64SiLU->target_dim, zero final weights/bias, seed43. Best validation telemetry loss including epoch0; max8epochs,patience2. Record mean-zero and last-snapshot baselines on exactly the same validation target cells; missing last values use encoded training mean0. Reconstruction quality is not event quality.",
    "pretrain_config": PRETRAIN,
    "fine_tune": "Fresh seed42 CountNetwork identical to v28 GRU. Transfer only history projection and recurrent weights; verify every current encoder, embedding and count head weight is unchanged by transfer. Discard decoder, reset AdamW and train original24h Poisson objective using unchanged20epoch/patience4 v28 procedure. All count-network parameters trainable; same parameter count and inference cost as scratch GRU.",
    "count_config": TRAINING,
    "selection": "One pretrained candidate per kind. Original whole-hour scores held on15min slots, same quarter-time calibration,456-policy choice on previous policy period, same confirmed-event timing and original exposure. Nov/Feb pooled primary min(P/.9,R/.9,1)>1.05*scratchGRU,CatBoost and historical anchor, noF1 loss against any. All reference alerts/metrics replay exactly.",
    "confirmation": "Only passing kinds: May improves primary withF1>=95%every reference; pooled Dec/Mar primary>1.05*all,noF1 loss,each monthF1>=90%all. Reuse frozen v29 scratch fault controls; missing access/fire controls use exact v28 procedure in a separate controls directory. No feature, architecture or objective substitution after screen.",
    "provenance": "Frozen original66/94/144 lists and training-only codecs. MPS and PyTorch version fixed from v28. Atomic checkpoints for both stages with model/optimizer/best weights. Pretext and count weight hashes separately recorded. No source modification or activation.",
    "limits": "No June labels, adaptive historical reuse rather than independent future validation. Overlapping windows/telemetry cells are not new independent incidents. Labels remain sensor proxies; flood unsupported. Neither good telemetry reconstruction nor a subgroup gain proves full90/90.",
    "sources": ["https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html"],
}


def existing_control(kind, fold):
    return (
        (
            Path("artifacts/research-v28")
            if fold.startswith("screen_")
            else Path("artifacts/research-v29/controls")
        )
        / kind
        / fold
    )


def control_source(root, kind, fold, frame, dense, episodes, device):
    source = existing_control(kind, fold)
    if (source / "result.json").exists():
        return source
    target = root / "controls"
    evaluate_original(target, kind, fold, frame, dense, episodes, device)
    return target / kind / fold


def fit_candidate(root, directory, source, kind, fold, frame, episodes, device):
    old = read(source_files(kind, fold)[1])
    before = read(source / "result.json")
    codec = read(source / "codec.json")
    periods = {name: tuple(map(pd.Timestamp, dates)) for name, dates in old["periods"].items()}
    train_rows = frame.loc[mask(frame, *periods["train"])]
    if fit_codec(train_rows, old["features"]) != codec:
        raise ValueError("Codec differs from frozen training-only preprocessing")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "codec.json", codec)
    assert sha256(directory / "codec.json") == sha256(source / "codec.json")
    history = frame.loc[
        frame.as_of.ge(periods["train"][0] - pd.Timedelta(hours=HISTORY_HOURS))
        & frame.as_of.lt(periods["test"][1])
    ].reset_index(drop=True)
    history_values, _ = encode(history, codec)
    data, sizes = {}, {}
    for period in ("train", "validation"):
        rows = train_rows if period == "train" else frame.loc[mask(frame, *periods[period])]
        data[period] = build_data(rows, history, codec, episodes, kind)
        sizes[period] = {
            "rows": len(rows),
            "positive_rows": int(np.sum(data[period]["target"] > 0)),
            "eligible_episodes": EventEvaluator(rows, episodes, 3).events,
        }
    config = {
        "numeric_dim": data["train"]["numeric"].shape[1],
        "category_sizes": [len(codec["vocabulary"][c]) + 1 for c in CATEGORICAL],
        "use_history": True,
        "initial_log_mean": float(np.log(max(float(data["train"]["target"].mean()), 1e-6))),
    }
    assert config == before["fits"]["gru"]["config"] and sizes == before["fits"]["gru"]["sizes"]
    indices = telemetry_indices(codec)
    signature = {
        "plan_sha256": sha256(root / "plan.json"),
        "kind": kind,
        "fold": fold,
        "count_config": config,
        "sizes": sizes,
        "codec_sha256": sha256(directory / "codec.json"),
        "telemetry_columns": [codec["numeric"][i] for i in indices],
    }
    torch.manual_seed(TRAINING["seed"])
    initial = CountNetwork(**config)
    torch.manual_seed(PRETRAIN["seed"])
    pretext = HistoryPredictor(initial, len(indices))
    baselines = pretext_baselines(data["validation"], history_values, indices)
    pretext, pre_meta = fit_stage(
        directory / "models/pretext", pretext, data, history_values, device, "pretext", signature, indices
    )
    assert abs(pre_meta["history"][0]["validation_loss"] - baselines["zero_training_mean"]) < 1e-5
    torch.manual_seed(TRAINING["seed"])
    model = CountNetwork(**config)
    transfer_history(pretext, model)
    del pretext, initial
    count_signature = {
        **signature,
        "pretext_model_sha256": pre_meta["model_sha256"],
        "pretext_fit_sha256": sha256(directory / "models/pretext/fit.json"),
    }
    model, count_meta = fit_stage(
        directory / "models/count", model, data, history_values, device, "count", count_signature
    )
    assert count_meta["parameter_count"] == before["fits"]["gru"]["parameter_count"]
    assert (
        count_meta["history"][0]["validation_loss"] == before["fits"]["gru"]["history"][0]["validation_loss"]
    )
    fit = {
        "pretext": pre_meta,
        "count": count_meta,
        "pretext_validation_baselines": baselines,
        "codec_sha256": signature["codec_sha256"],
        "same_count_architecture": True,
        "same_rows_and_codec_as_control": True,
        "only_history_weights_transferred": True,
    }
    write_json(directory / "fit.json", fit)
    return model, codec, history, history_values, old, before, fit


def evaluate(root, kind, fold, frame, dense, episodes, device):
    directory = root / kind / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    source = control_source(root, kind, fold, frame, dense, episodes, device)
    model, codec, history, history_values, old, before, fit = fit_candidate(
        root, directory, source, kind, fold, frame, episodes, device
    )
    tables, exposure = {}, {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, old["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        data = build_data(rows, history, codec)
        hourly = rows[["object_id", "as_of"]].copy()
        hourly["raw"] = predict(model, data, history_values, device)
        pred = carry_features(hourly, subdivide_evaluation_slots(hourly), ["raw"])
        assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
        tables[period], exposure[period] = pred, len(hourly) / 24
    counts = episode_counts(tables["calibration"], episodes)
    calibration = calibrate(tables["calibration"].raw.to_numpy(), counts > 0)
    scale = float(counts.sum() / np.exp(np.clip(tables["calibration"].raw, -20, 20)).sum())
    predictions = {
        period: pred.assign(
            probability=calibrated(pred.raw, calibration),
            expected_count=np.exp(np.clip(pred.raw, -20, 20)) * scale,
        )
        for period, pred in tables.items()
    }
    print("START selfsup policy", kind, fold, flush=True)
    policy, frontier = select_gate_policy(predictions["policy"], episodes, exposure["policy"])
    test = predictions["test"]
    test["alert"] = policy_alerts(test, episodes, policy)
    scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
    controls = {}
    for name, stored in (("scratch_gru", "gru"), ("global_control", "global_control")):
        for period in ("policy", "test"):
            saved = pd.read_parquet(source / f"{stored}-{period}.parquet")
            pd.testing.assert_frame_equal(
                predictions[period][["object_id", "as_of"]], saved[["object_id", "as_of"]]
            )
            alerts = policy_alerts(saved, episodes, before["arms"][stored]["policy"])
            actual = evaluator_for(saved, episodes, 0.25, exposure[period]).evaluate(alerts, 0.5, 0.25)
            if period == "test":
                np.testing.assert_array_equal(alerts, saved.alert)
                assert actual == before["arms"][stored]["scores"]
            else:
                assert all(before["arms"][stored]["policy"][key] == value for key, value in actual.items())
        controls[name] = before["arms"][stored]["scores"]
    controls["reference"] = before["reference"]
    assert all(scores["eligible_episodes"] == control["eligible_episodes"] for control in controls.values())
    for period in ("policy", "test"):
        predictions[period].to_parquet(directory / f"pretrained-{period}.parquet", index=False)
    write_json(directory / "frontier.json", frontier)
    result = {
        "kind": kind,
        "fold": fold,
        "scores": scores,
        "controls": controls,
        "policy": policy,
        "calibration": calibration,
        "rate_scale": scale,
        "fit": fit,
        "source_control_directory": str(source),
        "same_episode_cohort": True,
        "control_alert_and_metric_parity": True,
        "exposure_days": exposure["test"],
    }
    write_json(target, result)
    print("DONE selfsup", kind, fold, scores, flush=True)
    return result


def lock_plan(root):
    old_path = Path("artifacts/research-v28/plan.json")
    old = read(old_path)
    if str(torch.__version__) != old["torch_version"]:
        raise ValueError("Frozen PyTorch version changed")
    sources = {
        old_path,
        Path("artifacts/research-v28/report.json"),
        Path("artifacts/research-v29/report.json"),
        Path("artifacts/research-v30/report.json"),
    }
    sources.update(Path(p) for p in old["source_hashes"])
    for category in ("source_hashes", "code_hashes"):
        for source, digest in old[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed neural prerequisite: {source}")
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            directory = existing_control(kind, fold)
            if (directory / "result.json").exists():
                sources.update(
                    (
                        directory / "result.json",
                        directory / "codec.json",
                        directory / "models/gru/model.pt",
                        directory / "models/gru/fit.json",
                    )
                )
                sources.update(
                    directory / f"{variant}-{period}.parquet"
                    for variant in ("gru", "global_control")
                    for period in ("policy", "test")
                )
    code = {Path(p) for p in old["code_hashes"]}
    code.update(
        (
            Path(__file__),
            Path(HistoryPredictor.__init__.__code__.co_filename),
            Path(fit_stage.__code__.co_filename),
        )
    )
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "device": old["device"],
                "torch_version": old["torch_version"],
                "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
                "code_hashes": {str(p): sha256(p) for p in sorted(code)},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    if target.exists() and read(target) != plan:
        raise ValueError("Self-supervised inputs/code changed; use another output directory")
    if not target.exists():
        write_json(target, plan)
    controls = root / "controls/plan.json"
    control_plan = {
        "parent_plan_sha256": sha256(target),
        "frozen_training_source_plan_sha256": sha256(old_path),
        "scope": "Frozen v28 scratch controls needed only for v31 passing additional periods",
    }
    if controls.exists() and read(controls) != control_plan:
        raise ValueError("Self-supervised control plan changed")
    write_json(controls, control_plan)
    return plan


def metric(row, name):
    return row["scores"] if name == "pretrained" else row["controls"][name]


def run(root, stage):
    device = read(Path("artifacts/research-v28/plan.json"))["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Frozen MPS device unavailable; no silent fallback")
    torch.set_num_threads(TRAINING["cpu_threads"])
    lock_plan(root)
    frame = pd.read_parquet(PROCESSED / "features-channel-novelty.parquet")
    dense = pd.read_parquet(PROCESSED / "features-dense-channel-novelty.parquet")
    assert all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense))
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            rows = [
                evaluate(root, kind, fold, frame, dense, episodes.loc[episodes.kind.eq(kind)], device)
                for fold in ("screen_1", "screen_2")
            ]
            scores = {
                name: pooled([metric(row, name) for row in rows]) for name in ("pretrained", *REFERENCES)
            }
            c = scores["pretrained"]
            selection[kind] = {
                **scores,
                "passed_screen": all(
                    primary_score(c) > 1.05 * primary_score(scores[name]) and c["f1"] >= scores[name]["f1"]
                    for name in REFERENCES
                ),
            }
            write_json(root / "selection.json", selection)
            print("SELECT selfsup", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete all self-supervised screening before confirmation")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        rows = [
            evaluate(root, kind, fold, frame, dense, episodes.loc[episodes.kind.eq(kind)], device)
            for fold in ("confirmation", *STRESS)
        ]
        may, stress = metric(rows[0], "pretrained"), pooled([metric(row, "pretrained") for row in rows[1:]])
        passed_may = all(
            primary_score(may) > primary_score(metric(rows[0], name))
            and may["f1"] >= 0.95 * metric(rows[0], name)["f1"]
            for name in REFERENCES
        )
        passed_stress = all(
            primary_score(stress) > 1.05 * primary_score(pooled([metric(row, name) for row in rows[1:]]))
            and stress["f1"] >= pooled([metric(row, name) for row in rows[1:]])["f1"]
            and all(metric(row, "pretrained")["f1"] >= 0.9 * metric(row, name)["f1"] for row in rows[1:])
            for name in REFERENCES
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": {
                name: pooled([metric(row, name) for row in rows]) for name in ("pretrained", *REFERENCES)
            },
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT selfsup", kind, report[kind]["five_period_pooled"], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v31"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
