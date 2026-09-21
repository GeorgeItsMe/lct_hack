"""v28: current-snapshot MLP versus causal 48-hour GRU count forecasting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.binary_gate_research import KINDS, select_gate_policy
from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.fresh_counts_research import anchor, source_files
from moscollector.goal90_research import STRESS, pooled, primary_score, read
from moscollector.neural_count_model import CountNetwork
from moscollector.neural_sequence_data import (
    HISTORY_HOURS,
    encode,
    fit_codec,
    history_batch,
    history_positions,
)
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.quarter_count_features import carry_features
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

VARIANTS = ("mlp", "gru")
TRAINING = {
    "epochs": 20,
    "patience": 4,
    "min_delta": 1e-5,
    "batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "gradient_clip": 5,
    "seed": 42,
    "cpu_threads": 4,
}
PLAN = {
    "scope": "adaptive_retrospective_neural_count_study_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "variants": VARIANTS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "hypothesis": "Repeated aggregate/tree/policy variants have not reached90/90. Test a learned recurrent summary of past snapshots against a current-only neural control and frozen CatBoost count controls. Extra architecture/parameters mean any GRU gain alone does not isolate temporal order causally.",
    "features": "Same66 access/94 fire/144 fault original feature lists. Numeric signed_log1p, training-only mean/std,clip8,explicit missing indicators. Categorical vocabularies from eligible training rows only, embeddings8/4/2 with fixed zero unknown padding. Targets/as_of/eligibility never model features. One codec shared by both networks per fold.",
    "history": "At most16 original3h snapshots of the same object in[t-48h,t), oldest first, explicit ages and right padding. Present/future rows excluded, irregular gaps preserved through ages, missing history stays missing. No use of historical eligibility or future targets to choose inputs. Current hourly features update as in the baseline; history is drawn from original3h snapshots, never interpolated from future rows.",
    "networks": "Common current encoder numeric+embeddings->64SiLU->32SiLU. MLP head32->32SiLU->1. GRU: history numeric+normalized age->32tanh,one unidirectional GRU32; concatenate last valid recurrent state with current32,head64->32SiLU->1. All-missing history contributes zero. No dropout, output log count clipped[-12,8], final weight0/bias log training mean. Parameter counts recorded; not capacity-matched.",
    "training": TRAINING,
    "optimizer": "AdamW, float32, PoissonNLL(log_input=True,full=False). Full original eligible training rows and same validation dates,25h purge. Epoch0 constant-rate network is a valid early-stop candidate. Best validation mean Poisson loss, min_delta1e-5,patience4,max20epochs. No hyperparameter search. Deterministic seed and per-epoch shuffle; GPU bitwise reproducibility across machines is not claimed. Atomic epoch checkpoints contain model/optimizer/best state and support exact configuration resumption.",
    "hardware": "Device fixed from synthetic preflight before any task outcomes. PyTorch2.14 in separate optional environment, base runtime packages unchanged. No task metric used in device selection.",
    "inference": "Predict at the same eligible whole-hour points, then hold these outputs on the original v20 quarter grid. Each arm independently calibrates binary sigmoid and mean-count scale on the original calibration period against actual quarter-time24h counts. Same456-policy grid and prior policy period, original hourly exposure, same event identities and causal confirmation. No June rows/labels or source changes.",
    "controls": "Frozen original count weights v9 access/fire and v19 fault with the same recalibration/expanded policy grid, plus historical anchor(access v20 quarter,fire v13 mean90,fault v19 hourly). Screening global controls must exactly replay v26; additional fault controls replay v27 where available.",
    "screen": "Evaluate both predeclared networks on Nov/Feb. Eligible if pooled primary min(P/.9,R/.9,1)>1.05*both controls with noF1 loss. Select highest primary thenF1,R,P among eligible only,first listed on tie. If MLP wins it remains selected; GRU is not assumed superior.",
    "confirmation": "Only passing kinds proceed; both networks may be fitted for a matched architecture comparison but the winner cannot change. Selected variant May primary improves withF1>=95% both controls; pooled Dec/Mar primary>1.05*both,noF1 loss,each month'sF1>=90%both. No automatic activation or subgroup-only success.",
    "limits": "Overlapping windows are not independent events; labels remain grouped sensor proxies, not verified physical incidents. Earlier months adaptively reused, no independent final-weight future test. Flood remains unsupported and part of the unmet full-solution objective.",
    "sources": [
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html",
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.PoissonNLLLoss.html",
        "https://docs.pytorch.org/docs/2.14/notes/mps.html",
    ],
}


def save_torch(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def build_data(rows, history, codec, episodes=None, kind=None):
    numeric, categorical = encode(rows, codec)
    positions, ages = history_positions(history, rows)
    result = {"numeric": numeric, "categorical": categorical, "positions": positions, "ages": ages}
    if episodes is not None:
        counts = episode_counts(rows, episodes)
        if kind is not None and not np.array_equal(
            counts > 0, rows[f"target_{kind}"].to_numpy().astype(bool)
        ):
            raise ValueError("Neural count target differs from original binary label")
        result["target"] = counts.astype(np.float32)
    return result


def forward_batch(model, data, ids, history_values, device):
    numeric = torch.from_numpy(data["numeric"][ids]).to(device)
    categorical = torch.from_numpy(data["categorical"][ids]).to(device)
    if model.use_history:
        tokens, lengths = history_batch(history_values, data["positions"][ids], data["ages"][ids])
        return model(
            numeric, categorical, torch.from_numpy(tokens).to(device), torch.from_numpy(lengths).to(device)
        )
    return model(numeric, categorical)


def predict(model, data, history_values, device):
    model.eval()
    output = np.empty(len(data["numeric"]), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(output), 1024):
            ids = np.arange(start, min(start + 1024, len(output)))
            output[ids] = forward_batch(model, data, ids, history_values, device).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite neural predictions")
    return output


def validation_loss(model, data, history_values, device):
    raw = predict(model, data, history_values, device)
    return float(np.mean(np.exp(raw) - data["target"] * raw))


def fit_network(directory, variant, data, history_values, codec, sizes, device, plan_hash):
    directory.mkdir(parents=True, exist_ok=True)
    meta_path, weights = directory / "fit.json", directory / "model.pt"
    config = {
        "numeric_dim": data["train"]["numeric"].shape[1],
        "category_sizes": [len(codec["vocabulary"][c]) + 1 for c in CATEGORICAL],
        "use_history": variant == "gru",
        "initial_log_mean": float(np.log(max(float(data["train"]["target"].mean()), 1e-6))),
    }
    signature = {
        "plan_sha256": plan_hash,
        "variant": variant,
        "config": config,
        "sizes": sizes,
        "device": device,
        "torch": str(torch.__version__),
        "codec_sha256": sha256(directory.parent.parent / "codec.json"),
    }
    torch.manual_seed(TRAINING["seed"])
    model = CountNetwork(**config).to(device)
    if meta_path.exists():
        meta = read(meta_path)
        if meta["signature"] != signature or sha256(weights) != meta["model_sha256"]:
            raise ValueError("Neural model/configuration changed")
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        return model, meta
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"]
    )
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Checkpoint belongs to a different neural experiment")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print("RESUME neural", directory, "epoch", state["epoch"], flush=True)
    else:
        loss = validation_loss(model, data["validation"], history_values, device)
        state = {
            "signature": signature,
            "epoch": 0,
            "best_epoch": 0,
            "best_loss": loss,
            "best_state": cpu_state(model),
            "bad_epochs": 0,
            "history": [{"epoch": 0, "validation_loss": loss}],
        }
    start_time = time.perf_counter()
    print("START neural", directory, "device", device, "rows", sizes, flush=True)
    for epoch in range(state["epoch"] + 1, TRAINING["epochs"] + 1):
        if state["bad_epochs"] >= TRAINING["patience"]:
            break
        model.train()
        order = np.random.default_rng(TRAINING["seed"] + epoch).permutation(len(data["train"]["target"]))
        total, samples = 0.0, 0
        for start in range(0, len(order), TRAINING["batch_size"]):
            ids = order[start : start + TRAINING["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            raw = forward_batch(model, data["train"], ids, history_values, device)
            target = torch.from_numpy(data["train"]["target"][ids]).to(device)
            loss = torch.nn.functional.poisson_nll_loss(raw, target, log_input=True, full=False)
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite neural training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), TRAINING["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
            samples += len(ids)
        val = validation_loss(model, data["validation"], history_values, device)
        improved = val < state["best_loss"] - TRAINING["min_delta"]
        if improved:
            state.update(best_loss=val, best_epoch=epoch, best_state=cpu_state(model), bad_epochs=0)
        else:
            state["bad_epochs"] += 1
        state.update(epoch=epoch, model=cpu_state(model), optimizer=optimizer.state_dict())
        row = {
            "epoch": epoch,
            "training_loss": total / samples,
            "validation_loss": val,
            "best_epoch": state["best_epoch"],
            "elapsed_this_process_seconds": time.perf_counter() - start_time,
        }
        state["history"].append(row)
        save_torch(checkpoint, state)
        write_json(
            directory / "progress.json",
            {
                "epoch": epoch,
                "best_epoch": state["best_epoch"],
                "validation_loss": val,
                "checkpoint_sha256": sha256(checkpoint),
            },
        )
        print("EPOCH neural", directory, row, flush=True)
    model.load_state_dict(state["best_state"])
    save_torch(weights, state["best_state"])
    meta = {
        "signature": signature,
        "variant": variant,
        "config": config,
        "sizes": sizes,
        "best_epoch": state["best_epoch"],
        "best_validation_loss": state["best_loss"],
        "completed_epochs": state["epoch"],
        "history": state["history"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_sha256": sha256(weights),
        "weights_file": str(weights),
        "device": device,
        "torch": torch.__version__,
    }
    write_json(meta_path, meta)
    print("DONE neural fit", directory, "best", meta["best_epoch"], flush=True)
    return model, meta


def evaluate(root, kind, fold, frame, dense, episodes, device):
    directory = root / kind / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    old_weights, old_metadata, _ = source_files(kind, fold)
    old = read(old_metadata)
    periods = {key: tuple(map(pd.Timestamp, dates)) for key, dates in old["periods"].items()}
    history = frame.loc[
        frame.as_of.ge(periods["train"][0] - pd.Timedelta(hours=HISTORY_HOURS))
        & frame.as_of.lt(periods["test"][1])
    ].reset_index(drop=True)
    train_rows = frame.loc[mask(frame, *periods["train"])]
    codec = fit_codec(train_rows, old["features"])
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "codec.json", codec)
    history_values, _ = encode(history, codec)
    training_data, sizes = {}, {}
    for name in ("train", "validation"):
        rows = train_rows if name == "train" else frame.loc[mask(frame, *periods[name])]
        training_data[name] = build_data(rows, history, codec, episodes, kind)
        sizes[name] = {
            "rows": len(rows),
            "positive_rows": int(np.sum(training_data[name]["target"] > 0)),
            "eligible_episodes": EventEvaluator(rows, episodes, 3).events,
        }
    models, fits = {}, {}
    for variant in VARIANTS:
        models[variant], fits[variant] = fit_network(
            directory / "models" / variant,
            variant,
            training_data,
            history_values,
            codec,
            sizes,
            device,
            sha256(root / "plan.json"),
        )
    del training_data, train_rows
    original = CatBoostRegressor()
    original.load_model(str(old_weights))
    tables, exposure = {}, {}
    for period in ("calibration", "policy", "test"):
        reference = frame.loc[mask(frame, *periods[period]), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *periods[period])], reference)
        data = build_data(rows, history, codec)
        hourly = rows[["object_id", "as_of"]].copy()
        for variant, model in models.items():
            hourly[variant] = predict(model, data, history_values, device)
        hourly["global_control"] = original.predict(
            model_input(rows, old["features"]), prediction_type="RawFormulaVal", thread_count=2
        )
        pred = carry_features(hourly, subdivide_evaluation_slots(hourly), [*VARIANTS, "global_control"])
        assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
        tables[period], exposure[period] = pred, len(hourly) / 24
    arms = {}
    cal = tables["calibration"]
    counts = episode_counts(cal, episodes)
    for variant in (*VARIANTS, "global_control"):
        calibration = calibrate(cal[variant].to_numpy(), counts > 0)
        scale = float(counts.sum() / np.exp(np.clip(cal[variant], -20, 20)).sum())
        predictions = {
            period: pred.assign(
                probability=calibrated(pred[variant], calibration),
                expected_count=np.exp(np.clip(pred[variant], -20, 20)) * scale,
            )
            for period, pred in tables.items()
        }
        print("START neural policy", kind, fold, variant, flush=True)
        policy, frontier = select_gate_policy(predictions["policy"], episodes, exposure["policy"])
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, policy)
        scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
        if variant == "global_control":
            source = Path("artifacts/research-v26") / kind / fold
            if not (source / "result.json").exists():
                source = Path("artifacts/research-v27/controls") / kind / fold
            if (source / "result.json").exists():
                saved = read(source / "result.json")["arms"]["global_control"]
                assert policy == saved["policy"] and scores == saved["scores"]
                for period in ("policy", "test"):
                    before = pd.read_parquet(source / f"global_control-{period}.parquet")
                    shared = predictions[period].merge(
                        before, on=["object_id", "as_of"], suffixes=("_new", "_old"), validate="one_to_one"
                    )
                    assert len(shared) == len(before) == len(predictions[period])
                    for column in ("probability", "expected_count"):
                        np.testing.assert_allclose(
                            shared[f"{column}_new"], shared[f"{column}_old"], atol=1e-10, rtol=1e-10
                        )
        for period in ("policy", "test"):
            predictions[period].to_parquet(directory / f"{variant}-{period}.parquet", index=False)
        write_json(directory / f"{variant}-frontier.json", frontier)
        arms[variant] = {"scores": scores, "policy": policy, "calibration": calibration, "rate_scale": scale}
        print("DONE neural policy", kind, fold, variant, scores, flush=True)
    reference = anchor(kind, fold)
    assert all(
        value["scores"]["eligible_episodes"] == reference["eligible_episodes"] for value in arms.values()
    )
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "reference": reference,
        "fits": fits,
        "codec_sha256": sha256(directory / "codec.json"),
        "same_episode_cohort": True,
        "exposure_days": exposure["test"],
    }
    write_json(target, result)
    return result


def lock_plan(root, device):
    sources = [
        PROCESSED / "features-channel-novelty.parquet",
        PROCESSED / "features-dense-channel-novelty.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/research-v27/report.json"),
        Path("artifacts/neural_compute_preflight.json"),
        Path("requirements.lock"),
        Path("requirements-neural.lock"),
    ]
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            weights, metadata, saved = source_files(kind, fold)
            sources.extend([weights, metadata, *saved.values()])
            base = Path("artifacts/research-v26") / kind / fold
            if not (base / "result.json").exists():
                base = Path("artifacts/research-v27/controls") / kind / fold
            if (base / "result.json").exists():
                sources.extend(
                    [
                        base / "result.json",
                        *(base / f"global_control-{p}.parquet" for p in ("policy", "test")),
                    ]
                )
            if kind == "access":
                sources.append(Path("artifacts/research-v20/access") / fold / "result.json")
            elif kind == "fault":
                sources.append(Path("artifacts/research-v19/fault") / fold / "result.json")
            else:
                sources.append(Path("artifacts/research-v13-policy") / f"fire-{fold}.json")
    code = {
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (
                CountNetwork.__init__,
                history_positions,
                EventEvaluator.__init__,
                select_gate_policy,
                evaluator_for,
                policy_alerts,
                episode_counts,
                source_files,
                model_input,
                carry_features,
                align_opportunities,
                subdivide_evaluation_slots,
                primary_score,
                mask,
            )
        ),
    }
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "device": device,
                "torch_version": str(torch.__version__),
                "source_hashes": {str(p): sha256(p) for p in sources},
                "code_hashes": {str(p): sha256(p) for p in sorted(code)},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    if target.exists():
        if read(target) != plan:
            raise ValueError("Neural study inputs/code/device changed; use a new study directory")
    else:
        write_json(target, plan)


def run(root, stage, device):
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Selected MPS is unavailable in this process; do not silently change backend")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Selected CUDA is unavailable")
    torch.set_num_threads(TRAINING["cpu_threads"])
    lock_plan(root, device)
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
                evaluate(root, kind, fold, frame, dense, episodes[episodes.kind.eq(kind)], device)
                for fold in ("screen_1", "screen_2")
            ]
            controls = {
                "global_control": pooled([row["arms"]["global_control"]["scores"] for row in rows]),
                "reference": pooled([row["reference"] for row in rows]),
            }
            variants = {}
            for variant in VARIANTS:
                scores = pooled([row["arms"][variant]["scores"] for row in rows])
                variants[variant] = {
                    "scores": scores,
                    "passed_screen": all(
                        primary_score(scores) > 1.05 * primary_score(ref) and scores["f1"] >= ref["f1"]
                        for ref in controls.values()
                    ),
                }
            eligible = [name for name in VARIANTS if variants[name]["passed_screen"]]
            winner = (
                max(
                    eligible,
                    key=lambda name: tuple(
                        variants[name]["scores"][key]
                        for key in ("primary_score", "f1", "recall", "precision")
                    ),
                )
                if eligible
                else None
            )
            selection[kind] = {
                "variants": variants,
                **controls,
                "selected_variant": winner,
                "passed_screen": winner is not None,
            }
            write_json(root / "selection.json", selection)
            print("SELECT neural", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Finish all neural screening before confirmation")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        variant = selected["selected_variant"]
        rows = [
            evaluate(root, kind, fold, frame, dense, episodes[episodes.kind.eq(kind)], device)
            for fold in ("confirmation", *STRESS)
        ]

        def metrics(row, key):
            return row["reference"] if key == "reference" else row["arms"][key]["scores"]

        may = rows[0]
        stress = pooled([metrics(row, variant) for row in rows[1:]])
        passed_may = all(
            primary_score(metrics(may, variant)) > primary_score(metrics(may, key))
            and metrics(may, variant)["f1"] >= 0.95 * metrics(may, key)["f1"]
            for key in ("global_control", "reference")
        )
        passed_stress = all(
            primary_score(stress) > 1.05 * primary_score(pooled([metrics(row, key) for row in rows[1:]]))
            and stress["f1"] >= pooled([metrics(row, key) for row in rows[1:]])["f1"]
            and all(metrics(row, variant)["f1"] >= 0.9 * metrics(row, key)["f1"] for row in rows[1:])
            for key in ("global_control", "reference")
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": {
                key: pooled([metrics(row, key) for row in rows])
                for key in (*VARIANTS, "global_control", "reference")
            },
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print(
            "REPORT neural",
            kind,
            {key: value for key, value in report[kind].items() if key != "periods"},
            flush=True,
        )
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v28"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    selected_device = args.device or read(Path("artifacts/neural_compute_preflight.json"))["selected_device"]
    run(args.output, args.stage, selected_device)
