"""v24: separate any-episode probability from mean-count warning capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.fine_cadence_research import (
    FinePendingSimulator,
    cohort,
    evaluator_for,
    policy_alerts,
)
from moscollector.experiments.fresh_counts_research import anchor, source_files
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.quarter_count_features import carry_features
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

KINDS = ("access", "fire", "fault")
FLOORS = (0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1)
PLAN = {
    "scope": "adaptive_retrospective_probability_gate_study_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "motivation": "V20 error audit found1466 false warnings with no future event and704 redundant warnings. Mean count alone need not determine probability of any event when recurrence variability differs. Test a separately trained binary probability gate while retaining count capacity.",
    "unchanged": "Same frozen v9 mean-count weights for access/fire and v19 for fault, held whole-hour inputs on15min grid, same24h count capacity calibration, original event identities, forecast horizon, exposure and causal confirmation from v20. No adoption of fresh-count or phase-augmentation variants.",
    "binary_models": "Matched feature lists and periods: access v6b recent_reference66; fire v18 reference94; fault v18 channel_novelty144. Reuse existing classifiers after exact metadata and archived-prediction parity checks. For missing additional-month classifiers only, fit the same Logloss/PRAUC earlystop100,depth6,l2=8,lr=.04,max1000,seed42,CPU4 on original3h rows. Verify binary labels against original24h episode counts.",
    "calibration": "On original independent calibration period with25h purge, calibrate each arm's own raw score against actual any-episode labels in[new15min_time,new_time+24h). Binary and count scores are both lagged whole-hour inputs. Shared mean capacity is unchanged and mean-scaled using the same calibration counts.",
    "arms": "count_probability_control uses sigmoid of raw log mean count; binary_gate_candidate uses sigmoid of independent classifier logit. Both use the exact same expected_count values. This is a probability-gate comparison, not a fitted joint hurdle likelihood.",
    "policy": "Both arms use identical expanded fixed floor grid plus original capacity multipliers .5,1,1.5,2 and margins .25,.5,1,2,3,5. Policy-period only, primary min(P/.9,R/.9,1) thenF1,R,P. >=10alerts/events support, FP<=.25/original object-day. Identical threshold masks are cached without changing any grid result. Floor0 disables the probability gate and is explicitly reported.",
    "floor_grid": FLOORS,
    "reference": "Compare to matched count-probability arm with the same expanded grid AND historical anchor(access v20 quarter, fire v13 mean90, fault v19 hourly). Expanded-grid gains cannot be attributed to the independent classifier.",
    "screen": "Nov/Feb pooled candidate primary improves>5% over both references with noF1 loss. Only passing kinds continue; no variant substitution.",
    "confirmation": "May improves primary withF1>=95% against both. Pooled Dec/Mar primary improves>5%, no pooledF1 loss, each month'sF1>=90%each. Research gates do not establish90/90 or automatically activate production.",
    "june": "No June selection/evaluation/labels. Earlier months are adaptively reused. Flood remains unsupported; an individual head's result is not success for the full solution.",
}


def select_gate_policy(pred, episodes, exposure):
    probability = pred.probability.to_numpy()
    expected = pred.expected_count.to_numpy()
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(expected).all()
        or np.any((probability < 0) | (probability > 1))
        or np.any(expected < 0)
    ):
        raise ValueError("Invalid calibrated probabilities or counts")
    simulator, evaluator = FinePendingSimulator(pred, episodes), evaluator_for(pred, episodes, 0.25, exposure)
    masks = {floor: probability >= floor for floor in FLOORS}
    keys = {floor: np.packbits(value).tobytes() for floor, value in masks.items()}
    options, cache = [], {}
    for multiplier in (0.5, 1, 1.5, 2):
        for margin in (0.25, 0.5, 1, 2, 3, 5):
            for floor in FLOORS:
                key = (multiplier, margin, keys[floor])
                if key not in cache:
                    alerts = simulator.alerts(expected * multiplier, masks[floor].astype(float), margin, 0.5)
                    cache[key] = evaluator.evaluate(alerts, 0.5, 0.25)
                options.append({"capacity": multiplier, "margin": margin, "floor": floor, **cache[key]})
    budget = [r for r in options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    fallback = {"capacity": 0, "margin": 1, "floor": 0, **evaluator.evaluate(np.zeros(len(pred)), 0.5, 0.25)}
    chosen = dict(
        max(
            supported or budget or [fallback],
            key=lambda r: (primary_score(r), r["f1"], r["recall"], r["precision"]),
        )
    )
    chosen["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    chosen["gate_disabled"] = chosen["floor"] == 0
    chosen["gated_policy_rows"] = int(np.sum(probability < chosen["floor"]))
    return chosen, options


def classifier_cache(kind, fold):
    if kind == "access":
        base = Path("artifacts/research-v6b") / fold / "recent_reference"
        return (
            base / "access.cbm",
            base / "access.json",
            {p: base / f"access-{p}.parquet" for p in ("policy", "test")},
        )
    base = (
        Path("artifacts/research-v18") / kind / fold / ("reference" if kind == "fire" else "channel_novelty")
    )
    return base / "model.cbm", base / "fit.json", {p: base / f"{p}.parquet" for p in ("policy", "test")}


def get_classifier(directory, frame, episodes, kind, fold, count_meta):
    saved = directory / "fit.json"
    if saved.exists():
        meta = read(saved)
        if sha256(Path(meta["weights_file"])) != meta["model_sha256"]:
            raise ValueError("Binary-gate classifier changed")
        return meta
    weights, metadata, archived = classifier_cache(kind, fold)
    columns, periods = count_meta["features"], count_meta["periods"]
    if metadata.exists():
        old = read(metadata)
        if old["features"] != columns or old["periods"] != periods:
            raise ValueError("Cached classifier does not match count features/splits")
        meta = {
            "kind": kind,
            "fold": fold,
            "features": columns,
            "periods": periods,
            "weights_file": str(weights),
            "model_sha256": sha256(weights),
            "newly_fitted": False,
            "original_metadata": str(metadata),
            "archived_predictions": {p: str(path) for p, path in archived.items()},
        }
    else:
        train = frame.loc[mask(frame, *map(pd.Timestamp, periods["train"]))]
        validation = frame.loc[mask(frame, *map(pd.Timestamp, periods["validation"]))]
        labels = []
        for rows in (train, validation):
            y = rows[f"target_{kind}"].to_numpy()
            if not np.array_equal(episode_counts(rows, episodes) > 0, y.astype(bool)):
                raise ValueError("Binary classifier target differs from original episodes")
            labels.append(y)
        model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            l2_leaf_reg=8,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=42,
            thread_count=4,
            cat_features=CATEGORICAL,
            allow_writing_files=False,
            early_stopping_rounds=100,
            verbose=200,
        )
        print("START binary gate fit", kind, fold, len(train), flush=True)
        model.fit(
            model_input(train, columns), labels[0], eval_set=(model_input(validation, columns), labels[1])
        )
        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / "model.cbm"
        model.save_model(str(weights))
        meta = {
            "kind": kind,
            "fold": fold,
            "features": columns,
            "periods": periods,
            "weights_file": str(weights),
            "model_sha256": sha256(weights),
            "newly_fitted": True,
            "training_rows": len(train),
            "best_iteration": model.best_iteration_,
            "original_target_parity": True,
            "archived_predictions": {},
        }
    write_json(saved, meta)
    return meta


def evaluate(root, kind, fold, frame, dense, episodes):
    directory = root / kind / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    weights, metadata, _ = source_files(kind, fold)
    count_meta = read(metadata)
    binary_meta = get_classifier(directory / "binary", frame, episodes, kind, fold, count_meta)
    count_model, binary_model = CatBoostRegressor(), CatBoostClassifier()
    count_model.load_model(str(weights))
    binary_model.load_model(binary_meta["weights_file"])
    tables, exposure = {}, {}
    parity = {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, count_meta["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        x = model_input(rows, count_meta["features"])
        hourly = rows[["object_id", "as_of"]].copy()
        hourly["count_raw"] = count_model.predict(x, prediction_type="RawFormulaVal", thread_count=2)
        hourly["binary_raw"] = binary_model.predict(x, prediction_type="RawFormulaVal", thread_count=2)
        if not np.isfinite(hourly[["count_raw", "binary_raw"]].to_numpy()).all():
            raise ValueError("Nonfinite forecasts")
        if period in binary_meta["archived_predictions"]:
            original = read(Path(binary_meta["original_metadata"]))
            saved = pd.read_parquet(binary_meta["archived_predictions"][period])
            shared = hourly.merge(saved, on=["object_id", "as_of"], validate="one_to_one")
            assert len(shared) == len(saved)
            np.testing.assert_allclose(
                calibrated(shared.binary_raw, original["calibration"]),
                shared.probability,
                atol=1e-10,
                rtol=1e-10,
            )
            parity[period] = len(shared)
        pred = carry_features(hourly, subdivide_evaluation_slots(hourly), ["count_raw", "binary_raw"])
        assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
        tables[period] = pred
        exposure[period] = len(hourly) / 24
    cal = tables["calibration"]
    labels = episode_counts(cal, episodes)
    scale = float(labels.sum() / np.exp(np.clip(cal.count_raw.to_numpy(), -20, 20)).sum())
    for pred in tables.values():
        pred["expected_count"] = np.exp(np.clip(pred.count_raw, -20, 20)) * scale
    arms = {}
    for name, raw_column in (
        ("count_probability_control", "count_raw"),
        ("binary_gate_candidate", "binary_raw"),
    ):
        calibration = calibrate(cal[raw_column].to_numpy(), labels > 0)
        predictions = {
            p: pred.assign(probability=calibrated(pred[raw_column], calibration))
            for p, pred in tables.items()
        }
        if name == "count_probability_control":
            for period in ("policy", "test"):
                old_path = (
                    Path("artifacts/research-v20") / kind / fold / f"quarter_candidate-{period}.parquet"
                )
                if kind == "fault":
                    old_path = Path("artifacts/research-v21/fault") / fold / f"held_control-{period}.parquet"
                if old_path.exists():
                    old, new = pd.read_parquet(old_path), predictions[period]
                    shared = new.merge(
                        old, on=["object_id", "as_of"], suffixes=("_new", "_old"), validate="one_to_one"
                    )
                    assert len(shared) == len(old) == len(new)
                    for column in ("probability", "expected_count"):
                        np.testing.assert_allclose(
                            shared[f"{column}_new"], shared[f"{column}_old"], atol=1e-10, rtol=1e-10
                        )
        print("START binary gate policy", kind, fold, name, flush=True)
        policy, frontier = select_gate_policy(predictions["policy"], episodes, exposure["policy"])
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, policy)
        scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
        for period in ("policy", "test"):
            predictions[period].to_parquet(directory / f"{name}-{period}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {
            "scores": scores,
            "policy": policy,
            "calibration": calibration,
            "rate_scale": scale,
            "exposure_days": exposure["test"],
            "gated_test_rows": int(np.sum(test.probability < policy["floor"])),
        }
        print("DONE binary gate policy", kind, fold, name, scores, "floor", policy["floor"], flush=True)
    result = {
        "kind": kind,
        "fold": fold,
        "scores": arms["binary_gate_candidate"]["scores"],
        "control": arms["count_probability_control"]["scores"],
        "reference": anchor(kind, fold),
        "arms": arms,
        "identical_episode_cohort": True,
        "shared_count_capacity": True,
        "archived_binary_prediction_parity_rows": parity,
        "binary_model": binary_meta,
        "count_model_sha256": sha256(weights),
    }
    for other in (result["control"], result["reference"]):
        assert result["scores"]["eligible_episodes"] == other["eligible_episodes"]
    write_json(target, result)
    return result


def lock_plan(root):
    sources = [
        PROCESSED / "features-channel-novelty.parquet",
        PROCESSED / "features-dense-channel-novelty.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/fine_cadence_error_audit.json"),
        Path("artifacts/research-v23/report.json"),
    ]
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            weights, metadata, saved = source_files(kind, fold)
            sources.extend([weights, metadata, *saved.values()])
            binary_weights, binary_metadata, binary_saved = classifier_cache(kind, fold)
            if binary_metadata.exists():
                sources.extend([binary_weights, binary_metadata, *binary_saved.values()])
            if kind == "access":
                sources.append(Path("artifacts/research-v20/access") / fold / "result.json")
            elif kind == "fault":
                sources.append(Path("artifacts/research-v19/fault") / fold / "result.json")
            else:
                sources.append(Path("artifacts/research-v13-policy") / f"fire-{fold}.json")
            for period in ("policy", "test"):
                old_path = (
                    Path("artifacts/research-v20") / kind / fold / f"quarter_candidate-{period}.parquet"
                )
                if kind == "fault":
                    old_path = Path("artifacts/research-v21/fault") / fold / f"held_control-{period}.parquet"
                if old_path.exists():
                    sources.append(old_path)
    code = {
        Path(__file__),
        Path("src/moscollector/experiments/channel_novelty_research.py"),
        Path("src/moscollector/precision_research.py"),
        *(
            Path(f.__code__.co_filename)
            for f in (
                FinePendingSimulator.__init__,
                evaluator_for,
                source_files,
                anchor,
                model_input,
                episode_counts,
                carry_features,
                align_opportunities,
                subdivide_evaluation_slots,
                primary_score,
            )
        ),
    }
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "source_hashes": {str(p): sha256(p) for p in sources},
                "code_hashes": {str(p): sha256(p) for p in sorted(code)},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Study inputs/code changed; use a new study directory")
    else:
        write_json(path, plan)


def run(root, stage):
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
            eps = episodes[episodes.kind.eq(kind)]
            rows = [evaluate(root, kind, fold, frame, dense, eps) for fold in ("screen_1", "screen_2")]
            c, h, r = (pooled([item[key] for item in rows]) for key in ("scores", "control", "reference"))
            selection[kind] = {
                "candidate": c,
                "control": h,
                "reference": r,
                "passed_screen": all(
                    primary_score(c) > 1.05 * primary_score(other) and c["f1"] >= other["f1"]
                    for other in (h, r)
                ),
            }
            write_json(root / "selection.json", selection)
            print("SELECT binary gate", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete screening all kinds first")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        eps = episodes[episodes.kind.eq(kind)]
        rows = [evaluate(root, kind, fold, frame, dense, eps) for fold in ("confirmation", *STRESS)]
        may = rows[0]
        c = pooled([item["scores"] for item in rows[1:]])
        passed_may = all(
            primary_score(may["scores"]) > primary_score(may[key])
            and may["scores"]["f1"] >= 0.95 * may[key]["f1"]
            for key in ("control", "reference")
        )
        passed_stress = all(
            primary_score(c) > 1.05 * primary_score(pooled([item[key] for item in rows[1:]]))
            and c["f1"] >= pooled([item[key] for item in rows[1:]])["f1"]
            and all(item["scores"]["f1"] >= 0.9 * item[key]["f1"] for item in rows[1:])
            for key in ("control", "reference")
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": pooled([item["scores"] for item in rows]),
            "five_period_control": pooled([item["control"] for item in rows]),
            "five_period_reference": pooled([item["reference"] for item in rows]),
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print(
            "REPORT binary gate", kind, {k: v for k, v in report[kind].items() if k != "periods"}, flush=True
        )
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v24"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
