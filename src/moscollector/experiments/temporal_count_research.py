"""Fixed four-bin count forecast with matched deadline-budget controls (v42)."""

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort
from moscollector.count_research import episode_counts
from moscollector.experiments.binary_gate_research import select_gate_policy
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.channel_tag_research import base_model, references
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts, select
from moscollector.experiments.flood_research import FIVE, fingerprint, validate_hashes
from moscollector.experiments.flood_research import load_data as flood_data
from moscollector.experiments.flood_research import parts_for as flood_parts
from moscollector.experiments.fresh_counts_research import source_files
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.peer_context_research import FIT, control_source
from moscollector.experiments.quarter_count_features import carry_features
from moscollector.experiments.temporal_count import (
    LEAD,
    RATES,
    bin_counts,
    deadline_alerts,
    predict_bins,
    select_deadline,
    training_table,
)
from moscollector.prepare import sha256, write_json
from moscollector.research import mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

ROOT = Path("artifacts/research-v42")
KINDS = ("access", "fire", "fault", "flood")
KEYS = ["object_id", "as_of"]
CUTOFF = pd.Timestamp("2026-06-01")
PLAN = {
    "scope": "adaptive_retrospective_temporal_profile_study_not_new_blind_test",
    "goal": "Both event precision and recall>=.90 across all four incident types. No row-loss substitute.",
    "motivation": "V41 grouped channels failed. Existing count policy reserves one full event per outstanding warning until expiry. Test whether the future count profile and known warning deadlines improve allocation. Earlier v15 AFT rank and v16 first6h scalar did not model four future bins or this deadline budget.",
    "kinds": KINDS,
    "folds": FIVE,
    "fit": FIT,
    "targets": "Original episode starts in four disjoint6h windows within[t,t+24h). Sum exactly equals original24h count. Same original3h anchors, all negatives, old66/94/144/94 input fields plus numeric forecast_bin0..3. Bin-major long expansion gives4 rows per anchor, weight.25 each. One shared Poisson CatBoost per kind/fold, original dates and25h purge. Same iterations/depth/earlystop, but4x rows: not a compute-matched comparison.",
    "calibration": "On independent original calibration period only, scale each bin by its observed total / predicted total. Independent sigmoid of log sum of unscaled bins against original24h occurrence. On quarter grid hold the last hourly bin prediction as a lagged feature; recompute each bin target at current quarter time for calibration. No interpolation of future scores; hourly flood unchanged.",
    "budget": "Deterministic fluid heuristic, NOT a calibrated matching probability or Poisson-process assertion. Piecewise-uniform forecast mass within each6h bin; earliest pending deadline first, reserve min(1, available cumulative mass minus previously reserved). Emit if unreserved total mass>=margin and probability>=floor. Same24h expiry and strict whole-hour release after70min; expiry/confirmation ordering exactly preserves original forecast ticks.",
    "arms": ["profile", "uniform_new", "uniform_old", "count_control"],
    "controls": "uniform_new uses identical new total/probability but flat time profile; uniform_old uses frozen old calibrated forecasts and flat profile; count_control replays the original full-one-event reservation. Each arm chooses its own policy on the same preceding interval, same456 options or120 for flood. Also historical and stronger recent v33 access/v35 fire/v34 fault references, with access recent absent outside Nov/Feb explicitly retained as limitation.",
    "selection": "Only profile is candidate, no post-result substitution. Nov/Feb primary min(P/.9,R/.9,1)>1.05*EVERY other arm/reference and no pooledF1 loss. Passing kinds only: May primary improves andF1>=95% each; Dec/Mar pooled same>5% and noF1 loss, each monthF1>=90%reference. Flood always all5months/all40events, no test-based choice.",
    "invariants": "Original event identities,24h horizon, same hourly exposure, same quarter warning cadence except hourly flood. No relabeling, exclusions, June evaluation or automatic activation. Known reused months cannot establish independent future quality.",
}


def load_inputs():
    folder = Path("data/processed")
    frame, dense = (
        pd.read_parquet(folder / n)
        for n in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    if not frame.as_of.lt(CUTOFF).all() or not dense.as_of.lt(CUTOFF).all():
        raise ValueError("Post-May temporal inputs")
    episodes = pd.read_parquet(folder / "episodes.parquet", filters=[("start_ts", "<", CUTOFF)])
    return frame, dense, episodes, flood_data()


def parts_for(kind, fold, inputs):
    frame, dense, episodes, flood = inputs
    eps = episodes.loc[episodes.kind.eq(kind)]
    _, meta = base_model(kind, fold)
    if kind == "flood":
        parts = flood_parts(flood[0], flood[1], flood[2], FIVE[fold])
    else:
        parts = {}
        for name, dates in meta["periods"].items():
            reference = frame.loc[mask(frame, *map(pd.Timestamp, dates))].reset_index(drop=True)
            rows = (
                reference
                if name in ("train", "validation")
                else align_opportunities(dense.loc[mask(dense, *map(pd.Timestamp, dates))], reference)
            )
            if name not in ("train", "validation"):
                assert_same_episode_cohort(rows, reference, eps)
            counts = episode_counts(rows, eps)
            if not np.array_equal(counts > 0, rows[f"target_{kind}"].astype(bool)):
                raise ValueError("Changed original temporal target")
            parts[name] = rows.assign(count_target=counts).reset_index(drop=True)
    targets = {p: bin_counts(r, eps) for p, r in parts.items()}
    if any(not np.array_equal(targets[p].sum(axis=1), r.count_target) for p, r in parts.items()):
        raise ValueError("Temporal targets do not partition original counts")
    return parts, eps, meta["features"], targets


def prepare(root=ROOT):
    target = root / "plan.json"
    if target.exists():
        plan = read(target)
        if any(plan[k] != (list(v) if isinstance(v, tuple) else v) for k, v in PLAN.items()):
            raise ValueError("Frozen temporal specification changed")
        validate_hashes(plan)
        return plan
    original = read(Path("artifacts/hourly_feature_parity.json"))["source_files_unchanged"]
    for path, value in original.items():
        if sha256(Path(path)) != value:
            raise ValueError("Changed original evidence")
    sources = {
        Path("data/processed") / n
        for n in (
            "features.parquet",
            "features-hourly-research.parquet",
            "features-channel-novelty.parquet",
            "features-dense-channel-novelty.parquet",
            "episodes.parquet",
        )
    }
    sources.update((Path("artifacts/research-v41/report.json"), Path("artifacts/hourly_feature_parity.json")))
    for kind in KINDS:
        for fold in FIVE:
            if kind == "flood":
                sources.update(p for p in (Path("artifacts/research-v38") / fold).iterdir() if p.is_file())
            else:
                w, m, tables = source_files(kind, fold)
                sources.update((w, m, *tables.values()))
                for version in ("v20", "v19", "v33", "v34", "v35"):
                    p = Path(f"artifacts/research-{version}") / kind / fold / "result.json"
                    if p.exists():
                        sources.add(p)
                if kind == "fire":
                    sources.add(Path("artifacts/research-v13-policy") / f"fire-{fold}.json")
                control = control_source(kind, fold)
                if (control / "result.json").exists():
                    sources.update(
                        (
                            control / "result.json",
                            control / "global_control-policy.parquet",
                            control / "global_control-test.parquet",
                        )
                    )
    code = {
        Path(__file__),
        Path("tests/test_temporal_count.py"),
        Path("tests/test_temporal_count_research.py"),
    }
    code.update(
        Path(m.__file__)
        for n, m in sys.modules.items()
        if n.startswith("moscollector.") and getattr(m, "__file__", "").endswith(".py")
    )
    code = {p.resolve().relative_to(Path.cwd()) for p in code}
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
        "code_hashes": {str(p): sha256(p) for p in sorted(code)},
    }
    write_json(target, plan)
    return read(target)


def shape_scores(table, episodes):
    counts = bin_counts(table, episodes)
    rates = table[RATES].to_numpy()
    positive_mass = int(counts.sum())
    share = np.divide(
        rates,
        rates.sum(axis=1, keepdims=True),
        out=np.full_like(rates, 0.25),
        where=rates.sum(axis=1, keepdims=True) > 0,
    )
    return {
        "count_mass": positive_mass,
        "mean_squared_bin_error": float(np.mean((counts - rates) ** 2)),
        "conditional_bin_log_loss": float(-np.sum(counts * np.log(np.clip(share, 1e-12, 1))) / positive_mass)
        if positive_mass
        else None,
        "uniform_log_loss": float(np.log(4)) if positive_mass else None,
        "scope": "Descriptive temporal count quality, not event precision/recall or selection criterion.",
    }


def evaluate(kind, fold, inputs, root=ROOT, replay=False):
    directory = root / kind / fold
    parts, eps, columns, targets = parts_for(kind, fold, inputs)
    signature = {
        "plan_sha256": sha256(root / "plan.json"),
        "features": [*columns, LEAD],
        "parts": {
            p: {
                "rows": len(r),
                "fingerprint": fingerprint(r),
                "bin_targets_sha256": hashlib.sha256(targets[p].tobytes()).hexdigest(),
                "bin_target_dtype": str(targets[p].dtype),
                "bin_count_mass": targets[p].sum(axis=0).tolist(),
            }
            for p, r in parts.items()
        },
    }
    model = CatBoostRegressor()
    weights, fit_path = directory / "model.cbm", directory / "fit.json"
    if fit_path.exists():
        fit = read(fit_path)
        if fit["signature"] != signature or sha256(weights) != fit["model_sha256"]:
            raise ValueError("Changed temporal training arrays/weights")
        model.load_model(str(weights))
    else:
        if replay:
            raise ValueError("Missing model cannot be verified")
        pools = {}
        for part in ("train", "validation"):
            x, y, weight = training_table(parts[part], columns, targets[part])
            pools[part] = Pool(x, y, cat_features=CATEGORICAL, weight=weight)
            del x, y, weight
        model = CatBoostRegressor(**FIT, verbose=200)
        print("TRAIN TEMPORAL", kind, fold, len(parts["train"]) * 4, flush=True)
        model.fit(pools["train"], eval_set=pools["validation"])
        del pools
        directory.mkdir(parents=True, exist_ok=True)
        model.save_model(str(weights))
        fit = {
            "signature": signature,
            "model_sha256": sha256(weights),
            "best_iteration": model.best_iteration_,
            "validation_history": model.evals_result_,
        }
        write_json(fit_path, fit)
    old_weights, meta = base_model(kind, fold)
    old_model = CatBoostRegressor()
    old_model.load_model(str(old_weights))
    cadence = 1 if kind == "flood" else 0.25
    new_tables, old_tables, exposure = {}, {}, {}
    for part in ("calibration", "policy", "test"):
        rows = parts[part]
        profile = rows[KEYS].copy()
        profile[RATES] = predict_bins(model, rows, columns)
        old = rows[KEYS].assign(
            raw=old_model.predict(
                model_input(rows, meta["features"]), prediction_type="RawFormulaVal", thread_count=4
            )
        )
        if cadence != 1:
            slots = subdivide_evaluation_slots(profile[KEYS])
            profile = carry_features(profile, slots, RATES)
            old = carry_features(old, slots, ["raw"])
        if cohort(profile, eps, cadence) != cohort(rows, eps, 1):
            raise ValueError("Changed temporal event identities")
        new_tables[part], old_tables[part], exposure[part] = profile, old, len(rows) / 24
    cal = new_tables["calibration"]
    actual_bins = bin_counts(cal, eps)
    scale = actual_bins.sum(axis=0) / cal[RATES].sum().to_numpy()
    new_cal = calibrate(np.log(cal[RATES].sum(axis=1)), actual_bins.sum(axis=1) > 0)
    old_counts = episode_counts(old_tables["calibration"], eps)
    old_cal = calibrate(old_tables["calibration"].raw.to_numpy(), old_counts > 0)
    old_scale = float(
        old_counts.sum() / np.exp(np.clip(old_tables["calibration"].raw.to_numpy(), -20, 20)).sum()
    )
    for part, table in new_tables.items():
        table["raw"] = np.log(table[RATES].sum(axis=1))
        table["probability"] = calibrated(table.raw, new_cal)
        table[RATES] = table[RATES].to_numpy() * scale
        table["expected_count"] = table[RATES].sum(axis=1)
        old = old_tables[part]
        old["probability"] = calibrated(old.raw, old_cal)
        old["expected_count"] = np.exp(np.clip(old.raw, -20, 20)) * old_scale
    arms, outputs = {}, {}
    for arm in PLAN["arms"]:
        tables = {
            p: t.copy() for p, t in (new_tables if arm in ("profile", "uniform_new") else old_tables).items()
        }
        if arm != "profile":
            for table in tables.values():
                table[RATES] = np.repeat(table.expected_count.to_numpy()[:, None] / 4, 4, axis=1)
        if arm == "count_control":
            policy, frontier = (
                select(tables["policy"], eps, cadence, exposure["policy"])
                if kind == "flood"
                else select_gate_policy(tables["policy"], eps, exposure["policy"])
            )
            alert_function = policy_alerts
        else:
            policy, frontier = select_deadline(tables["policy"], eps, cadence, exposure["policy"])
            alert_function = deadline_alerts
        for p in ("policy", "test"):
            tables[p]["alert"] = alert_function(tables[p], eps, policy)
        scores = evaluator_for(tables["test"], eps, cadence, exposure["test"]).evaluate(
            tables["test"].alert, 0.5, cadence
        )
        result = {
            "scores": scores,
            "policy": policy,
            "calibration": new_cal if arm in ("profile", "uniform_new") else old_cal,
            "rate_scale": scale.tolist() if arm in ("profile", "uniform_new") else old_scale,
            "temporal_scores": {p: shape_scores(tables[p], eps) for p in ("policy", "test")},
        }
        if arm == "count_control":
            source = Path("artifacts/research-v38") / fold if kind == "flood" else control_source(kind, fold)
            if (source / "result.json").exists():
                before = read(source / "result.json")["arms"][
                    "poisson" if kind == "flood" else "global_control"
                ]
                if (
                    scores != (before["scores"]["test"] if kind == "flood" else before["scores"])
                    or policy != before["policy"]
                    or old_cal != before["calibration"]
                    or old_scale != before["rate_scale"]
                ):
                    raise ValueError("Original temporal control differs from archive")
                for p in ("policy", "test"):
                    saved = pd.read_parquet(
                        source / f"{'poisson' if kind == 'flood' else 'global_control'}-{p}.parquet"
                    )
                    compare = [*KEYS, "probability", "expected_count"] + (["alert"] if p == "test" else [])
                    pd.testing.assert_frame_equal(tables[p][compare], saved[compare], check_exact=True)
        for p, table in tables.items():
            target = directory / f"{arm}-{p}.parquet"
            if replay:
                pd.testing.assert_frame_equal(table, pd.read_parquet(target), check_exact=True)
            else:
                table.to_parquet(target, index=False)
            outputs[str(target)] = sha256(target)
        target = directory / f"{arm}-frontier.json"
        if replay:
            if frontier != read(target):
                raise ValueError("Changed temporal policy frontier")
        else:
            write_json(target, frontier)
        outputs[str(target)] = sha256(target)
        arms[arm] = result
    refs = references(kind, fold)
    if any(r["eligible_episodes"] != arms["profile"]["scores"]["eligible_episodes"] for r in refs.values()):
        raise ValueError("Changed reference event cohort")
    result = {
        "kind": kind,
        "fold": fold,
        "signature": signature,
        "model_sha256": sha256(weights),
        "arms": arms,
        "references": refs,
        "outputs": outputs,
    }
    target = directory / "result.json"
    if replay:
        if result != read(target):
            raise ValueError("Changed temporal result")
    else:
        write_json(target, result)
    print("VERIFIED" if replay else "TEMPORAL SCORE", kind, fold, arms["profile"]["scores"], flush=True)
    return result


def metrics(row):
    return {**{k: v["scores"] for k, v in row["arms"].items()}, **row["references"]}


def screen(rows):
    if not rows:
        raise ValueError("No temporal comparison periods")
    if any(set(r["arms"]) != set(PLAN["arms"]) for r in rows):
        raise ValueError("Temporal comparison omits a required arm")
    keys = set(metrics(rows[0]))
    if any(set(metrics(r)) != keys for r in rows):
        raise ValueError("Temporal comparison omits a reference")
    scores = {k: pooled([metrics(r)[k] for r in rows]) for k in sorted(keys)}
    candidate = scores["profile"]
    passed = all(
        primary_score(candidate) > 1.05 * primary_score(v) and candidate["f1"] >= v["f1"]
        for k, v in scores.items()
        if k != "profile"
    )
    return {"scores": scores, "passed_screen": passed}


def run(root=ROOT, replay=False):
    plan = prepare(root)
    inputs = load_inputs()
    report, completed = {}, []
    for kind in KINDS:
        folds = tuple(FIVE) if kind == "flood" else ("screen_1", "screen_2")
        rows = [evaluate(kind, fold, inputs, root, replay) for fold in folds]
        selection = screen(rows)
        completed.extend((kind, fold) for fold in folds)
        row = {"selection": selection, "periods": rows, "serving_changed": False, "goal_achieved": False}
        if kind != "flood" and selection["passed_screen"]:
            extra = [evaluate(kind, fold, inputs, root, replay) for fold in ("confirmation", *STRESS)]
            completed.extend((kind, fold) for fold in ("confirmation", *STRESS))
            may, stress = metrics(extra[0]), screen(extra[1:])
            passed_may = all(
                primary_score(may["profile"]) > primary_score(v) and may["profile"]["f1"] >= 0.95 * v["f1"]
                for k, v in may.items()
                if k != "profile"
            )
            passed_stress = stress["passed_screen"] and all(
                metrics(r)["profile"]["f1"] >= 0.9 * v["f1"]
                for r in extra[1:]
                for k, v in metrics(r).items()
                if k != "profile"
            )
            row.update(
                periods=rows + extra,
                passed_may=passed_may,
                passed_stress=passed_stress,
                status="research_passed" if passed_may and passed_stress else "confirmation_rejected",
            )
        else:
            row["status"] = "all_five_flood_reported" if kind == "flood" else "screen_rejected"
        report[kind] = row
        if not replay:
            write_json(root / "report.json", report)
        print("TEMPORAL RESULT", kind, selection, row["status"], flush=True)
        gc.collect()
    if replay:
        if report != read(root / "report.json"):
            raise ValueError("Changed full temporal report")
        sources = dict(plan["source_hashes"])
        sources.update(
            {str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "weight-replay.json"}
        )
        proof = {
            "status": "exact_replay_passed",
            "periods": completed,
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
