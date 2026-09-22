"""v44: fixed Tweedie mean-count loss with matched corrected warning ownership."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort, evaluator_for
from moscollector.flood_research import fingerprint, validate_hashes
from moscollector.goal90_research import pooled, primary_score, read
from moscollector.peer_context_research import FIT
from moscollector.prepare import sha256, write_json
from moscollector.quarter_count_features import carry_features
from moscollector.retained_pending import retained_alerts, select_retained
from moscollector.retained_pending_research import PLAN as PARENT_PLAN
from moscollector.retained_pending_research import save_json
from moscollector.retained_pending_verification import verified_evidence as verify_parent
from moscollector.temporal_count_research import KEYS, KINDS, base_model, load_inputs, parts_for
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

ROOT = Path("artifacts/research-v44")
PARENT = Path("artifacts/research-v43")
LOSS = "Tweedie:variance_power=1.5"
FIT_TWEEDIE = {
    **FIT,
    "loss_function": LOSS,
    "eval_metric": LOSS,
    "leaf_estimation_method": "Newton",
    "leaf_estimation_iterations": 10,
}
PLAN = {
    "scope": "adaptive_retrospective_fixed_tweedie_mean_study_not_blind_test",
    "goal": "Both event precision and recall>=.90 across all four original incident types.",
    "motivation": "V43 corrected late ownership but did not improve eventF1. Existing original count heads use Poisson loss. Oldest-screen TRAIN counts show variance/mean8.61 access,9.29 fire,2.06 fault; this marginal heterogeneity is not proof of conditional overdispersion or a forecast limit. Test a different mean-regression loss, without new features or event exclusion.",
    "kinds": KINDS,
    "folds": PARENT_PLAN["folds"],
    "fit": FIT_TWEEDIE,
    "loss": "Single native Tweedie variance_power1.5, fixed before new task predictions. For log mean a, loss=2exp(a/2)+2yexp(-a/2); gradient=(exp(a)-y)/sqrt(exp(a)). This changes gradient weighting relative to Poisson but both target conditional mean, not a proven distribution of sensor events. No alpha/power grid, no negative-binomial candidate, no sample reweighting or target scaling.",
    "training": "Exact original66/94/144/94 inputs and3h anchors, all negatives,24h counts, original dates and25h purge from verified v42. CPU4,max1000,depth6,l2=8,lr=.04,seed42,earlystop100 on own Tweedie validation. Explicit Newton10 matches archived Poisson. All native training parameters except loss/eval metric must match the frozen baseline; floating metadata alone permits rtol1e-9,atol1e-12 for CBM decimal serialization. Raw predictions remain exact on replay. Nonnegative integral labels<=1000 required before fit; numerical failure is recorded, never silently retried with changed settings.",
    "evaluation": "Same frozen v42 opportunities,15min except hourly flood, identical hourly exposure and original event identities. Predict explicit RawFormulaVal as log mean, clip[-20,20] only at existing exp-count transform. Separate original calibration interval: sigmoid of raw log mean plus mean-count scaling. Same456/120policy choices, selected on preceding period only, FP<=.25/object-day. Candidate uses corrected v43 retained ledger and exact original delayed confirmations.",
    "controls": "Matched frozen Poisson with corrected v43 ledger, original legacy Poisson, original historical and stronger v33/v35/v34 references. Flood also compares already-known v42uniform_old9/28/40. Parent corrected flags/policies and all raw Poisson calibration/policy/test inputs must reproduce exactly. No candidate substitution after outcomes.",
    "selection": "Only Tweedie is candidate. Fixed eleven-period first stage: Nov/Feb for access/fire/fault, all five flood periods/all40events. Pooled primary min(P/.9,R/.9,1)>1.05*EVERY comparator and noF1 loss. Passing requires a separately frozen additional-month confirmation, not automatic release or a90/90claim.",
    "invariants": "No June evaluation, removed negatives, changed event labels/cohorts, serving change or overwriting completed studies. Reused historical months are not independent future evidence. Proxy episodes do not establish physical incident quality.",
    "source": "https://catboost.ai/docs/en/concepts/loss-functions-regression#Tweedie",
}


def count_support(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values != np.floor(values))
        or values.max() > 1000
    ):
        raise ValueError("Invalid or out-of-range unchanged count target")
    mean, variance = float(values.mean()), float(values.var())
    return {
        "rows": len(values),
        "positive_rows": int(np.sum(values > 0)),
        "count_mass": int(values.sum()),
        "mean": mean,
        "variance": variance,
        "variance_to_mean": variance / mean if mean else None,
        "maximum": int(values.max()),
    }


def training_pool(rows, columns):
    if len(columns) != len(set(columns)) or any(
        c.startswith("target_") or c in ("count_target", "eligible", "as_of") for c in columns
    ):
        raise ValueError("Duplicate or future target in model inputs")
    count_support(rows.count_target)
    return Pool(model_input(rows, columns), rows.count_target.to_numpy(), cat_features=CATEGORICAL)


def check_native_parameters(candidate, baseline):
    a, b = candidate.get_all_params(), baseline.get_all_params()

    def equal(x, y):
        if isinstance(x, float) and isinstance(y, float):
            return bool(np.isclose(x, y, rtol=1e-9, atol=1e-12))
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y, strict=True))
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        return x == y

    changed = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if not equal(a.get(k), b.get(k))}
    if (
        set(changed) != {"loss_function", "eval_metric"}
        or a["loss_function"] != LOSS
        or b["loss_function"] != "Poisson"
    ):
        raise ValueError(f"Unmatched native training settings: {changed}")
    return changed


def prepare(root=ROOT):
    spec = json.loads(json.dumps(PLAN))
    target = root / "plan.json"
    if target.exists():
        plan = read(target)
        if any(plan[k] != v for k, v in spec.items()):
            raise ValueError("Frozen Tweedie specification changed")
        validate_hashes(plan)
        return plan
    parent = verify_parent(PARENT)
    sources = {
        **parent["source_hashes"],
        str(PARENT / "policy-replay.json"): sha256(PARENT / "policy-replay.json"),
    }
    code = {Path(__file__), Path("tests/test_tweedie_count_research.py")}
    code.update(
        Path(m.__file__)
        for n, m in sys.modules.items()
        if n.startswith("moscollector.") and getattr(m, "__file__", "").endswith(".py")
    )
    code = {p.resolve().relative_to(Path.cwd()) for p in code}
    plan = {
        **spec,
        "source_hashes": sources,
        "code_hashes": {**parent["code_hashes"], **{str(p): sha256(p) for p in sorted(code)}},
    }
    validate_hashes(plan)
    write_json(target, plan)
    return read(target)


def comparison(rows):
    if not rows:
        raise ValueError("Missing Tweedie periods")
    required = {"tweedie", "poisson_corrected", "poisson_legacy"}
    for row in rows:
        refs = (
            {"object_weekday", "deadline_uniform_old"} if row["kind"] == "flood" else {"historical", "recent"}
        )
        if set(row["references"]) != refs:
            raise ValueError("Missing required Tweedie reference")
    metrics = [{**{a: v["scores"] for a, v in r["arms"].items()}, **r["references"]} for r in rows]
    if any(set(r["arms"]) != required for r in rows) or any(set(m) != set(metrics[0]) for m in metrics):
        raise ValueError("Missing Tweedie comparator")
    if any(len({v["eligible_episodes"] for v in m.values()}) != 1 for m in metrics):
        raise ValueError("Changed Tweedie episode cohort")
    scores = {a: pooled([m[a] for m in metrics]) for a in sorted(metrics[0])}
    candidate = scores["tweedie"]
    passed = all(
        primary_score(candidate) > 1.05 * primary_score(s) and candidate["f1"] >= s["f1"]
        for a, s in scores.items()
        if a != "tweedie"
    )
    return {"scores": scores, "passed_screen": passed}


def evaluate(kind, fold, inputs, root=ROOT, replay=False):
    directory = root / kind / fold
    parts, eps, columns, _ = parts_for(kind, fold, inputs)
    parent = read(PARENT / kind / fold / "result.json")
    original = read(Path("artifacts/research-v42") / kind / fold / "result.json")
    part_signatures = {p: {"rows": len(r), "fingerprint": fingerprint(r)} for p, r in parts.items()}
    if any(
        any(original["signature"]["parts"][p][k] != v for k, v in s.items())
        for p, s in part_signatures.items()
    ):
        raise ValueError("Tweedie data differs from original complete cohort")
    signature = {
        "plan_sha256": sha256(root / "plan.json"),
        "features": columns,
        "parts": part_signatures,
        "training_support": {p: count_support(parts[p].count_target) for p in ("train", "validation")},
    }
    model = CatBoostRegressor()
    old = CatBoostRegressor()
    old_weights, _ = base_model(kind, fold)
    old.load_model(str(old_weights))
    weights, meta_path = directory / "model.cbm", directory / "fit.json"
    if meta_path.exists():
        fit = read(meta_path)
        if fit["signature"] != signature or fit["model_sha256"] != sha256(weights):
            raise ValueError("Changed Tweedie fit or inputs")
        model.load_model(str(weights))
        if check_native_parameters(model, old) != fit["native_parameter_differences"]:
            raise ValueError("Changed native training settings")
    else:
        if replay:
            raise ValueError("Missing Tweedie model cannot be replayed")
        pools = {p: training_pool(parts[p], columns) for p in ("train", "validation")}
        print("TRAIN TWEEDIE", kind, fold, signature["training_support"], flush=True)
        model = CatBoostRegressor(**FIT_TWEEDIE, verbose=200)
        model.fit(pools["train"], eval_set=pools["validation"])
        differences = check_native_parameters(model, old)
        directory.mkdir(parents=True, exist_ok=True)
        model.save_model(str(weights))
        fit = {
            "signature": signature,
            "model_sha256": sha256(weights),
            "best_iteration": model.best_iteration_,
            "validation_history": model.evals_result_,
            "native_parameter_differences": differences,
        }
        write_json(meta_path, fit)
        del pools
    cadence = 1 if kind == "flood" else 0.25
    tables = {}
    # Recheck raw baseline predictions on each original calibration/policy/test
    # query. This proves the shared data pipeline, not only summary metrics.
    for part in ("calibration", "policy", "test"):
        rows = parts[part]
        hourly = rows[KEYS].assign(
            raw=model.predict(model_input(rows, columns), prediction_type="RawFormulaVal", thread_count=4),
            old_raw=old.predict(model_input(rows, columns), prediction_type="RawFormulaVal", thread_count=4),
        )
        table = (
            hourly
            if cadence == 1
            else carry_features(hourly, subdivide_evaluation_slots(hourly[KEYS]), ["raw", "old_raw"])
        )
        if not np.isfinite(table[["raw", "old_raw"]]).all().all() or cohort(table, eps, cadence) != cohort(
            rows, eps, 1
        ):
            raise ValueError("Invalid Tweedie raw values or changed episode identities")
        control = pd.read_parquet(
            Path("artifacts/research-v42") / kind / fold / f"count_control-{part}.parquet"
        )
        pd.testing.assert_frame_equal(table[KEYS], control[KEYS], check_exact=True)
        np.testing.assert_array_equal(table.old_raw.to_numpy(), control.raw.to_numpy())
        tables[part] = table.drop(columns="old_raw")
    counts = episode_counts(tables["calibration"], eps)
    calibration = calibrate(tables["calibration"].raw.to_numpy(), counts > 0)
    scale = float(counts.sum() / np.exp(np.clip(tables["calibration"].raw.to_numpy(), -20, 20)).sum())
    for table in tables.values():
        table["probability"] = calibrated(table.raw, calibration)
        table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
    exposure = {p: len(parts[p]) / 24 for p in ("calibration", "policy", "test")}
    policy, frontier = select_retained(tables["policy"], eps, cadence, exposure["policy"])
    outputs = {}
    for part in ("policy", "test"):
        tables[part]["alert"] = retained_alerts(tables[part], eps, policy)
    scores = evaluator_for(tables["test"], eps, cadence, exposure["test"]).evaluate(
        tables["test"].alert, 0.5, cadence
    )
    controls = {}
    for arm, name in (("poisson_corrected", "retained_selected"), ("poisson_legacy", "legacy_control")):
        controls[arm] = {"scores": parent["arms"][name]["scores"], "policy": parent["arms"][name]["policy"]}
        if arm == "poisson_corrected":
            source = pd.read_parquet(PARENT / kind / fold / f"{name}-policy.parquet")
            control_policy, control_frontier = select_retained(source, eps, cadence, exposure["policy"])
            if control_policy != controls[arm]["policy"] or control_frontier != read(
                PARENT / kind / fold / "retained-frontier.json"
            ):
                raise ValueError("Changed corrected Poisson policy/frontier")
            for part in ("policy", "test"):
                source = pd.read_parquet(PARENT / kind / fold / f"{name}-{part}.parquet")
                np.testing.assert_array_equal(retained_alerts(source, eps, control_policy), source.alert)
                pd.testing.assert_frame_equal(tables[part][KEYS], source[KEYS], check_exact=True)
    for part, table in tables.items():
        target = directory / f"tweedie-{part}.parquet"
        if target.exists():
            pd.testing.assert_frame_equal(table, pd.read_parquet(target), check_exact=True)
        elif replay:
            raise ValueError("Missing Tweedie forecast")
        else:
            table.to_parquet(target, index=False)
        outputs[str(target)] = sha256(target)
    target = directory / "tweedie-frontier.json"
    save_json(target, frontier, replay)
    outputs[str(target)] = sha256(target)
    result = {
        "kind": kind,
        "fold": fold,
        "signature": signature,
        "model_sha256": sha256(weights),
        "arms": {
            "tweedie": {"scores": scores, "policy": policy, "calibration": calibration, "rate_scale": scale},
            **controls,
        },
        "references": parent["references"],
        "outputs": outputs,
    }
    comparison([result])
    save_json(directory / "result.json", result, replay)
    print("VERIFIED" if replay else "TWEEDIE SCORE", kind, fold, scores, flush=True)
    return result


def run(root=ROOT, replay=False):
    plan = prepare(root)
    inputs = load_inputs()
    report = {}
    for kind in KINDS:
        rows = [evaluate(kind, fold, inputs, root, replay) for fold in PLAN["folds"][kind]]
        selection = comparison(rows)
        if kind == "flood" and selection["scores"]["tweedie"]["eligible_episodes"] != 40:
            raise ValueError("Changed Tweedie flood cohort")
        report[kind] = {
            "periods": rows,
            "selection": selection,
            "status": "requires_separate_confirmation" if selection["passed_screen"] else "screen_rejected",
            "serving_changed": False,
            "goal_achieved": False,
        }
        if not replay:
            write_json(root / "report.json", report)
        print("TWEEDIE RESULT", kind, selection, flush=True)
        gc.collect()
    save_json(root / "report.json", report, replay)
    if replay:
        sources = {
            **plan["source_hashes"],
            **{str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "weight-replay.json"},
        }
        proof = {
            "status": "exact_replay_passed",
            "source_hashes": sources,
            "code_hashes": plan["code_hashes"],
            "report": report,
        }
        validate_hashes(proof)
        write_json(root / "weight-replay.json", proof)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    prepare(args.output) if args.prepare_only else run(args.output, args.replay)
