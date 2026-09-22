"""v37: weighted raw-onset GRU versus a matched current-only neural count model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from moscollector.count_research import episode_counts
from moscollector.event_sequence_data import FOLDER as EVENT_FOLDER
from moscollector.event_sequence_data import (
    encode_events,
    fit_event_codec,
    history_bounds,
)
from moscollector.event_sequence_inputs import input_arrays, period_rows, training_part
from moscollector.event_sequence_model import EventCountNetwork
from moscollector.event_sequence_training import fit_network, predict
from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.fresh_counts_research import anchor, source_files
from moscollector.goal90_research import STRESS, pooled, primary_score, read
from moscollector.minute_cadence_research import evaluate as evaluate_old_count
from moscollector.neural_count_research import TRAINING
from moscollector.neural_sequence_data import fit_codec
from moscollector.onset_binary_research import KINDS
from moscollector.onset_channel_features import CATS, KEYS
from moscollector.onset_channel_features import FOLDER as ONSET_FOLDER
from moscollector.onset_count_research import fit as fit_fresh_count
from moscollector.onset_policy import select_policy
from moscollector.onset_training_data import attach, model_input
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated

VARIANTS = ("current_mlp", "event_gru")
COUNT_CONTROLS = ("fresh_count", "old_count")
REFERENCES = (*COUNT_CONTROLS, "reference")
PLAN = {
    "scope": "adaptive_retrospective_raw_onset_sequence_study_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "variants": VARIANTS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "motivation": "V35/v36 warning-policy combinations do not meet90/90. Earlier GRUs saw16 three-hour snapshots, andv33/v35 latest-channel features discard most per-channel ordering. Test a learned history of individual raw canonical onsets. Unlabelled oldest-screen training audit found flat latest64 truncates>60%queries; family-balanced retention extends median retained span from about5h to25h without test-label architecture selection.",
    "events": "Frozen5,893,413 canonical onsets of6families, same catalog ownership, no initial-observation onsets and no June. For each query retain at most16 latest strictly-past onsets of EACH family in[t-48h,t), merge oldest-first, right-pad to96. Ties ordered family/channel deterministically, not claimed physical order. Full48h family counts retained separately; ties/truncation and empty histories reported. Missing onsets do not mean healthy coverage.",
    "vocabularies": "Channel and sensor-type embeddings use dictionaries fitted ONLY to source rows actually retained in training-query histories. Unknown/padding0 with fixed zero embeddings;6 predefined signal semantics. Current categories and numeric statistics fit original eligible training anchors only, same original66/94/144 inputs as controls. Catalog types assumed static because historical versions are absent.",
    "current_inputs": "Hold original source snapshot unchanged at minute queries. Same signed_log1p/training normalization/missing indicators asv28. Append source age log1p/log4 and complete6 family48h counts log1p/log1001, fixed units. Both networks receive exactly these current inputs and categories, same rows/weights/targets. Only event_gru additionally receives retained event categories plus normalized log ages, gaps BETWEEN RETAINED tokens and previous-token indicator.",
    "networks": "Same current encoder and count head family asv28. Current MLP has no sequence. EventGRU embeddings channel12,type4,signal4; concat3 time fields,23->32tanh,unidirectionalGRU32; last valid state plus current32 into count head. Empty history exactly zero. Output logcount[-12,8], head0/bias weighted log train mean. Architectures/parameter counts differ, so a gain is not an isolated causal-order proof.",
    "training": TRAINING,
    "weighted_objective": "Samev33 augmented3h train/validation anchors with minute companions only between adjacent eligible anchors, no gap/final-interval extension, original snapshot mass and25h purge. Recompute24h counts at each timestamp, original labels/cohorts exactly match. Float32 weighted Poisson gradient uses minibatch mean(weight*loss)/GLOBAL train mean(weight), not random batch weight normalization. Validation is exact full weighted Poisson mean. Same20epochs,patience4,AdamW,seed42,epoch0 and atomic resumable model/optimizer/best-state checkpoints asv28. No outcome-based hyperparameter changes.",
    "hardware": "Local MPS fixed by separate synthetic preflight before task outcomes; CPU/MPS forward, gradient and interruption-resume tests pass. Same device for both architectures. Existing neural_compute_preflight.json remains frozen. No cluster or CUDA performance claimed.",
    "evaluation": "Same quarter-plus-next-minute opportunities asv33-v36, original hourly exposure, exact original episode identities and24h matching. Each arm uses own preceding-period sigmoid and count scale, then SAME456 pending policies and delayed confirmations. Raw history strictly predates each query; target/eligibility never model input. No June reread for model evaluation.",
    "controls": "Fresh_count uses frozenv33 onset Poisson weights with60 channel context fields and same training rows/weights. Old_count is exactv34 old-weight minute policy. All six fresh and six old screening control periods must reproduce all archived forecasts/calibration/policies/metrics. Additional fresh weights reusev35count-controls when present, otherwise only fit passing-kind missing periods with frozenv33code under parent-linked plan. Missing old-count policies use frozenv34code separately. Original completed studies never extended.",
    "screen": "For each network Nov/Feb pooled min(P/.9,R/.9,1)>1.05*ALL3references and noF1 loss. Pick highest primary,F1,R,P among eligible, current_mlp first on exact tie. Both are predeclared candidates; no assumption GRU wins. Only passing kinds continue.",
    "confirmation": "Keep selected network fixed; both networks may be trained in additional months for comparison. May primary improves withF1>=95%every reference. Dec/Mar primary>1.05*all controls with no pooledF1 loss and each month'sF1>=90%every reference. No promotion from these gates alone.",
    "limits": "All months adaptively reused historical proxy sensor episodes, not independently verified physical incidents. Catalog history and transport latency unknown. Flood remains unsupported and full goal unchanged. No automatic activation.",
    "sources": [
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html",
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html",
    ],
}


def metric(row, name):
    return row["reference"] if name == "reference" else row["arms"][name]["scores"]


def screening(rows):
    scores = {name: pooled([metric(row, name) for row in rows]) for name in (*VARIANTS, *REFERENCES)}
    eligible = [
        name
        for name in VARIANTS
        if all(
            primary_score(scores[name]) > 1.05 * primary_score(scores[r])
            and scores[name]["f1"] >= scores[r]["f1"]
            for r in REFERENCES
        )
    ]
    selected = (
        max(
            eligible,
            key=lambda name: (
                primary_score(scores[name]),
                scores[name]["f1"],
                scores[name]["recall"],
                scores[name]["precision"],
            ),
        )
        if eligible
        else None
    )
    return {
        "pooled": scores,
        "eligible_variants": eligible,
        "selected_variant": selected,
        "passed_screen": selected is not None,
    }


def confirmation(rows, variant):
    if variant not in VARIANTS:
        raise ValueError("No selected event-network variant")
    may, stress = metric(rows[0], variant), pooled([metric(r, variant) for r in rows[1:]])
    passed_may = all(
        primary_score(may) > primary_score(metric(rows[0], r))
        and may["f1"] >= 0.95 * metric(rows[0], r)["f1"]
        for r in REFERENCES
    )
    passed_stress = all(
        primary_score(stress) > 1.05 * primary_score(pooled([metric(row, r) for row in rows[1:]]))
        and stress["f1"] >= pooled([metric(row, r) for row in rows[1:]])["f1"]
        and all(metric(row, variant)["f1"] >= 0.9 * metric(row, r)["f1"] for row in rows[1:])
        for r in REFERENCES
    )
    return {
        "passed_may": passed_may,
        "passed_stress": passed_stress,
        "research_eligible": bool(passed_may and passed_stress),
    }


def saved_json(path, content):
    if path.exists() and read(path) != content:
        raise ValueError(f"Changed event-network prepared metadata: {path}")
    if not path.exists():
        write_json(path, content)


def fit_models(root, kind, fold, frame, dense, events, triggers, episodes, device):
    directory = root / kind / fold
    original = read(source_files(kind, fold)[1])
    periods = original["periods"]
    anchors = frame.loc[mask(frame, *map(pd.Timestamp, periods["train"]))]
    codec = fit_codec(anchors, original["features"])
    saved_json(directory / "current-codec.json", codec)
    data, sizes = {}, {}
    for part in ("train", "validation"):
        base, slots = period_rows(frame, dense, triggers, periods, part)
        data[part], sizes[part] = training_part(base, slots, events, codec, episodes, kind)
    event_codec = fit_event_codec(events, data["train"]["bounds"])
    saved_json(directory / "event-codec.json", event_codec)
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
    saved_json(
        directory / "data.json",
        {"periods": periods, "features": original["features"], "sizes": sizes, "provenance": provenance},
    )
    config = {
        "numeric_dim": data["train"]["numeric"].shape[1],
        "category_sizes": [len(codec["vocabulary"][c]) + 1 for c in CATEGORICAL],
        "event_category_sizes": [len(event_codec["vocabulary"][c]) + 1 for c in ("channel_id", "sensor_type")]
        + [7],
        "initial_log_mean": float(
            np.log(max(float(np.average(data["train"]["target"], weights=data["train"]["weight"])), 1e-6))
        ),
    }
    models, fits = {}, {}
    for variant in VARIANTS:
        models[variant], fits[variant] = fit_network(
            directory / variant,
            data,
            source,
            {**config, "use_history": variant == "event_gru"},
            provenance,
            device,
        )
    del data, anchors
    gc.collect()
    return models, fits, codec, source, original


def control_sources(root, kind, fold, frame, dense, context, triggers, episodes):
    fresh = Path("artifacts/research-v33") / kind / fold / "onset"
    if not (fresh / "fit.json").exists():
        fresh = Path("artifacts/research-v35/count_controls") / kind / fold / "onset"
    if not (fresh / "fit.json").exists():
        if fold.startswith("screen_"):
            raise ValueError("Missing predeclared fresh screening control")
        # Full training context is needed only for an actually missing fit.
        all_context = pd.read_parquet(ONSET_FOLDER / "context.parquet", read_dictionary=CATS)
        fit_fresh_count(root / "fresh_controls", kind, fold, frame, dense, all_context, triggers, episodes)
        fresh = root / "fresh_controls" / kind / fold / "onset"
    old = Path("artifacts/research-v34") / kind / fold
    if not (old / "result.json").exists():
        old = Path("artifacts/research-v35/pending_controls") / kind / fold
    if not (old / "result.json").exists():
        if fold.startswith("screen_"):
            raise ValueError("Missing predeclared old screening control")
        evaluate_old_count(root / "old_controls", kind, fold, frame, dense, triggers, episodes)
        old = root / "old_controls" / kind / fold
    return fresh, old


def evaluate(root, kind, fold, frame, dense, events, triggers, episodes, device):
    directory = root / kind / fold
    result_path = directory / "result.json"
    if result_path.exists():
        result = read(result_path)
        if result["plan_sha256"] != sha256(root / "plan.json"):
            raise ValueError("Changed event-network result provenance")
        for variant in VARIANTS:
            fit = read(directory / variant / "fit.json")
            if (
                fit != result["fits"][variant]
                or fit["signature"]["plan_sha256"] != sha256(root / "plan.json")
                or fit["device"] != device
                or fit["model_sha256"] != sha256(directory / variant / "model.pt")
            ):
                raise ValueError("Changed event-network cached weights")
        return result
    models, fits, codec, source, original = fit_models(
        root, kind, fold, frame, dense, events, triggers, episodes, device
    )
    periods = original["periods"]
    context = pd.read_parquet(
        ONSET_FOLDER / "context.parquet",
        filters=[
            ("as_of", ">=", pd.Timestamp(periods["calibration"][0])),
            ("as_of", "<", pd.Timestamp(periods["test"][1])),
        ],
        read_dictionary=CATS,
    )
    fresh, old = control_sources(root, kind, fold, frame, dense, context, triggers, episodes)
    fresh_meta, old_result = read(fresh / "fit.json"), read(old / "result.json")
    if fresh_meta["model_sha256"] != sha256(fresh / "model.cbm"):
        raise ValueError("Changed fresh count weights")
    # Identical augmented counts, weights and event support; neural input widths
    # and codecs differ intentionally, but the training observations do not.
    for part, size in fresh_meta["sizes"].items():
        if any(fits[VARIANTS[0]]["signature"]["sizes"][part][k] != v for k, v in size.items()):
            raise ValueError("Neural/count training observations differ")
    count_model = CatBoostRegressor()
    count_model.load_model(str(fresh / "model.cbm"))
    tables = {name: {} for name in (*VARIANTS, *COUNT_CONTROLS)}
    exposure = {}
    for part in ("calibration", "policy", "test"):
        base, slots = period_rows(frame, dense, triggers, periods, part)
        data = input_arrays(base, slots, events, codec)
        for variant in VARIANTS:
            tables[variant][part] = (
                slots[KEYS].copy().assign(raw=predict(models[variant], data, source, device))
            )
        values = attach(base, slots, context, original["features"])
        raw = count_model.predict(
            model_input(values, fresh_meta["features"]), prediction_type="RawFormulaVal", thread_count=2
        )
        tables["fresh_count"][part] = slots[KEYS].copy().assign(raw=raw)
        previous = pd.read_parquet(old / f"minute_candidate-{part}.parquet")
        pd.testing.assert_frame_equal(slots[KEYS], previous[KEYS])
        if cohort(slots, episodes, 1 / 60) != cohort(base, episodes, 1):
            raise ValueError("Event-network inference changed episode identities")
        tables["old_count"][part] = previous.drop(columns="alert", errors="ignore")
        exposure[part] = len(base) / 24
        del data, values
    if exposure != old_result["exposure"]:
        raise ValueError("Event-network comparison changed exposure")
    arms = {}
    for name, predictions in tables.items():
        if name == "old_count":
            calibration, scale = (
                old_result["arms"]["minute_candidate"][k] for k in ("calibration", "rate_scale")
            )
        else:
            cal = predictions["calibration"]
            counts = episode_counts(cal, episodes)
            calibration = calibrate(cal.raw.to_numpy(), counts > 0)
            scale = float(counts.sum() / np.exp(np.clip(cal.raw, -20, 20)).sum())
            for pred in predictions.values():
                pred["probability"] = calibrated(pred.raw, calibration)
                pred["expected_count"] = np.exp(np.clip(pred.raw, -20, 20)) * scale
        print("START event-network policy", kind, fold, name, flush=True)
        policy, frontier = select_policy(predictions["policy"], episodes, exposure["policy"])
        for part in ("policy", "test"):
            predictions[part]["alert"] = policy_alerts(predictions[part], episodes, policy)
        scores = evaluator_for(predictions["test"], episodes, 1 / 60, exposure["test"]).evaluate(
            predictions["test"].alert, 0.5, 1 / 60
        )
        record = {"scores": scores, "policy": policy, "calibration": calibration, "rate_scale": scale}
        archived = None
        if name == "old_count":
            archived, old_arm = old, "minute_candidate"
        elif name == "fresh_count" and fold.startswith("screen_"):
            archived, old_arm = Path("artifacts/research-v33") / kind / fold, "onset_candidate"
        if archived:
            original_arm = read(archived / "result.json")["arms"][old_arm]
            if any(original_arm[k] != v for k, v in record.items()) or frontier != read(
                archived / f"{old_arm}-frontier.json"
            ):
                raise ValueError("Archived event-network count control policy changed")
            for part, pred in predictions.items():
                archived_pred = pd.read_parquet(archived / f"{old_arm}-{part}.parquet")
                if name == "fresh_count" and part == "policy":
                    # V33 archives this split before warning flags are attached.
                    # Verify its unchanged forecast fields and recompute the
                    # missing flags from the same archived selected policy.
                    archived_pred["alert"] = policy_alerts(archived_pred, episodes, original_arm["policy"])
                pd.testing.assert_frame_equal(pred, archived_pred, check_exact=True)
        for part, pred in predictions.items():
            pred.to_parquet(directory / f"{name}-{part}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = record
        print("DONE event-network policy", kind, fold, name, scores, flush=True)
    historical = anchor(kind, fold)
    if any(arm["scores"]["eligible_episodes"] != historical["eligible_episodes"] for arm in arms.values()):
        raise ValueError("Historical event-network denominator changed")
    result = {
        "kind": kind,
        "fold": fold,
        "plan_sha256": sha256(root / "plan.json"),
        "arms": arms,
        "fits": fits,
        "reference": historical,
        "exposure": exposure,
        "fresh_control": str(fresh),
        "fresh_fit_sha256": sha256(fresh / "fit.json"),
        "old_control": str(old),
        "old_result_sha256": sha256(old / "result.json"),
        "same_episode_cohort": True,
        "same_training_observations": True,
    }
    write_json(result_path, result)
    return result


def lock_plan(root, device):
    prior_path = Path("artifacts/research-v36/plan.json")
    prior = read(prior_path)
    build = read(EVENT_FOLDER / "build.json")
    preflight, audit = read(root / "compute-preflight.json"), read(root / "input-audit.json")
    controls = read(root / "control-preflight.json")
    fix_path = root / "schema-fix.json"
    fix = read(fix_path) if fix_path.exists() else {"source_hashes": {}, "code_hashes": {}}
    if device != preflight["chosen_device"] or str(torch.__version__) != preflight["torch"]:
        raise ValueError("Event-network device/version differs from preflight")
    if (
        controls["torch"] != preflight["torch"]
        or {(row["kind"], row["fold"]) for row in controls["periods"]}
        != {(kind, fold) for kind in KINDS for fold in ("screen_1", "screen_2")}
        or len(controls["periods"]) != 6
        or not all(row["fresh_and_old_exact_parity"] for row in controls["periods"])
    ):
        raise ValueError("Incomplete event-network control preflight")
    for manifest, categories in (
        (prior, ("source_hashes", "code_hashes")),
        (build, ("inputs", "outputs", "code_hashes")),
        (preflight, ("code_hashes",)),
        (audit, ("source_hashes", "code_hashes")),
        (controls, ("source_hashes", "code_hashes")),
        (fix, ("source_hashes", "code_hashes")),
    ):
        for category in categories:
            for path, digest in manifest[category].items():
                if sha256(Path(path)) != digest:
                    raise ValueError(f"Changed event-network input: {path}")
    sources = {
        prior_path,
        EVENT_FOLDER / "build.json",
        EVENT_FOLDER / "events.parquet",
        root / "compute-preflight.json",
        root / "input-audit.json",
        root / "control-preflight.json",
        Path("requirements.lock"),
        Path("requirements-neural.lock"),
    }
    sources.update(Path("artifacts/research-v36").glob("*.json"))
    if fix_path.exists():
        sources.add(fix_path)
    sources.update(Path("artifacts/research-v36").glob("*/*/*.json"))
    code = {
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (history_bounds, input_arrays, EventCountNetwork.__init__, fit_network, fit_codec)
        ),
    }
    code.update(Path("tests").glob("test_event_sequence*.py"))
    code.add(Path("src/moscollector/neural_count_research.py"))
    code.add(Path("src/moscollector/neural_count_model.py"))
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "device": device,
                "torch": str(torch.__version__),
                "source_hashes": {
                    **prior["source_hashes"],
                    **build["inputs"],
                    **build["outputs"],
                    **audit["source_hashes"],
                    **controls["source_hashes"],
                    **fix["source_hashes"],
                    **{str(p): sha256(p) for p in sorted(sources)},
                },
                "code_hashes": {
                    **prior["code_hashes"],
                    **preflight["code_hashes"],
                    **audit["code_hashes"],
                    **controls["code_hashes"],
                    **fix["code_hashes"],
                    **{str(p): sha256(p) for p in sorted(code)},
                },
            }
        )
    )
    saved_json(root / "plan.json", plan)
    for name, old in (
        ("fresh_controls", Path("artifacts/research-v33/plan.json")),
        ("old_controls", Path("artifacts/research-v34/plan.json")),
    ):
        saved_json(
            root / name / "plan.json",
            {
                "parent_plan_sha256": sha256(root / "plan.json"),
                "frozen_source_plan_sha256": sha256(old),
                "purpose": "Only missing controls for screening-passing v37 kinds; prior studies unchanged.",
            },
        )


def run(root, stage, device):
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Selected MPS unavailable; no silent device substitution")
    lock_plan(root, device)
    frame, dense = (
        pd.read_parquet(PROCESSED / name)
        for name in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    events = pd.read_parquet(EVENT_FOLDER / "events.parquet", read_dictionary=["signal", "sensor_type"])
    triggers = pd.read_parquet(ONSET_FOLDER / "triggers.parquet")
    if (
        not all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense, triggers))
        or not events.ts.lt(pd.Timestamp("2026-06-01")).all()
    ):
        raise ValueError("Post-May event-network source")
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selected = {}
        for kind in KINDS:
            rows = [
                evaluate(
                    root,
                    kind,
                    fold,
                    frame,
                    dense,
                    events,
                    triggers,
                    episodes.loc[episodes.kind.eq(kind)],
                    device,
                )
                for fold in ("screen_1", "screen_2")
            ]
            selected[kind] = screening(rows)
            write_json(root / "selection.json", selected)
            print("SELECT event network", kind, selected[kind], flush=True)
        return
    selected, report = read(root / "selection.json"), {}
    if set(selected) != set(KINDS):
        raise ValueError("Complete all event-network screening first")
    for kind in KINDS:
        choice = selected[kind]["selected_variant"]
        if choice is None:
            report[kind] = {
                "selection": selected[kind],
                "research_eligible": False,
                "status": "screen_failed",
            }
            continue
        rows = [
            evaluate(
                root, kind, fold, frame, dense, events, triggers, episodes.loc[episodes.kind.eq(kind)], device
            )
            for fold in ("confirmation", *STRESS)
        ]
        gates = confirmation(rows, choice)
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected[kind],
            "selected_variant": choice,
            "periods": rows,
            "five_period_pooled": {
                name: pooled([metric(r, name) for r in rows]) for name in (*VARIANTS, *REFERENCES)
            },
            **gates,
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT event network", kind, gates, report[kind]["five_period_pooled"], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v37-fixed"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    parser.add_argument("--device", default="mps", choices=("cpu", "mps"))
    args = parser.parse_args()
    run(args.output, args.stage, args.device)
