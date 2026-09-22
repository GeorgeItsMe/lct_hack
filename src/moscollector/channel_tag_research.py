"""Fixed literal-tag group context plus raw water/pump onsets for all heads."""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.binary_gate_research import select_gate_policy
from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort
from moscollector.channel_tag_features import COLUMNS, CUTOFF, FOLDER, KEYS, build
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import cohort, evaluator_for, policy_alerts, select
from moscollector.flood_research import FIVE, fingerprint, validate_hashes
from moscollector.flood_research import load_data as flood_data
from moscollector.flood_research import parts_for as flood_parts
from moscollector.fresh_counts_research import anchor, source_files
from moscollector.goal90_research import STRESS, pooled, primary_score, read
from moscollector.peer_context_research import FIT, control_source
from moscollector.prepare import sha256, write_json
from moscollector.quarter_count_features import carry_features
from moscollector.research import mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

ROOT = Path("artifacts/research-v41")
KINDS = ("access", "fire", "fault", "flood")
PLAN = {
    "scope": "adaptive_retrospective_tag_channel_study_not_new_blind_test",
    "goal": "Both event precision and recall>=.90 for the full solution, all four incident types.",
    "hypothesis": "Within-object literal parent-tag co-activity may distinguish shared disturbances from isolated channel activity. Two new raw families (water and pump state entries) also support the rare flood head; grouping and those families are not ablated here.",
    "features": "64 fixed fields: eight summaries for each of eight raw onset families. Same-object parent tags after dropping exactly one final numeric segment. Unknown syntax becomes a distinct singleton per channel. Count active groups, maximum distinct channels and catalog fractions in6/24h; maximum other-signal-only channels and family diversity in active groups at6h. Strict[t-window,t), no future durations, labels or post-onset membership. Static catalog groups are literal strings, not proven wiring or historical topology. All original model columns and negative opportunities retained.",
    "data_basis": "V40 numeric-channel addition failed. Current catalog has structured unique tags and many mixed-type parent groups; predictive utility not yet established. Original six onset families are re-extracted and checked exactly; added flood entries follow original state predicates, pump entries exclude initial observations and alarm-only changes. No changes to sensor episode labels, and no June telemetry reads.",
    "kinds": KINDS,
    "folds": FIVE,
    "training": FIT,
    "protocol": "One candidate per kind/fold, no feature subset or hyperparameter search. Original3h train/validation, same24h counts and25h purge. Poisson early stopping on preceding validation. Access/fire/fault original archived date windows; flood uses frozen v38 independent3month validation/calibration/policy windows. All train/input fingerprints preserved.",
    "evaluation": "Access/fire/fault original hourly features carried to identical15min slots,456 pending policies. Flood hourly120 policies on all five folds irrespective of screen results (only5flood episodes on screen). Independent sigmoid and count scaling on calibration only, policy chosen on subsequent old policy interval only. Same delayed confirmation and event identities.",
    "references": "Matched original count weights plus historical anchors. Stronger recent reference: access v33 onset-count on Nov/Feb only; fire v35 binary and fault v34 minute on all available5folds. For access additional months compare matched/historical controls only; no claim of additional-month superiority over unevaluated v33. Flood compares frozen v38 Poisson and object-weekday baseline.",
    "screen": "For access/fire/fault pooled Nov/Feb primary min(P/.9,R/.9,1)>1.05*EVERY reference with noF1 loss. Only passing kinds continue; no substitution after results.",
    "confirm": "May primary improves withF1>=95% each applicable reference. Pooled Dec/Mar primary>1.05*each,noF1 loss,and each monthF1>=90%reference. Flood reports all40events and all5months; no test-based candidate choice.",
    "promotion": "No automatic serving changes. Historical adaptive comparisons are not independent future evidence. No relabeling, June evaluation, excluded cohorts, or global90/90 claim from one head.",
}


def attach(rows, context):
    if set(context) != set(KEYS + COLUMNS):
        raise ValueError("Unexpected tag context schema")
    if set(COLUMNS) & set(rows):
        raise ValueError("Tag features already present")
    out = rows.merge(context, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True)
    if not out._merge.eq("both").all() or len(out) != len(rows):
        raise ValueError("Missing tag query rows, including negative opportunities")
    pd.testing.assert_frame_equal(out[rows.columns], rows.reset_index(drop=True), check_exact=True)
    return out.drop(columns="_merge")


def base_model(kind, fold):
    if kind == "flood":
        parent = Path("artifacts/research-v38") / fold
        return parent / "model.cbm", read(parent / "fit.json")
    weights, meta, _ = source_files(kind, fold)
    return weights, read(meta)


def references(kind, fold):
    if kind == "flood":
        old = read(Path("artifacts/research-v38") / fold / "result.json")
        return {"object_weekday": old["arms"]["object_weekday"]["scores"]["test"]}
    result = {"historical": anchor(kind, fold)}
    extra = {
        "access": ("v33", "onset_candidate"),
        "fire": ("v35", "binary_candidate"),
        "fault": ("v34", "minute_candidate"),
    }[kind]
    path = Path(f"artifacts/research-{extra[0]}") / kind / fold / "result.json"
    if kind != "access" or fold.startswith("screen_"):
        result["recent"] = read(path)["arms"][extra[1]]["scores"]
    return result


def load_inputs():
    context = pd.read_parquet(FOLDER / "context.parquet")
    frame, dense = (
        pd.read_parquet(FOLDER.parent / n)
        for n in ("features-channel-novelty.parquet", "features-dense-channel-novelty.parquet")
    )
    if not frame.as_of.lt(CUTOFF).all() or not dense.as_of.lt(CUTOFF).all():
        raise ValueError("Post-May tag study rows")
    episodes = pd.read_parquet(FOLDER.parent / "episodes.parquet", filters=[("start_ts", "<", CUTOFF)])
    flood = flood_data()
    return context, frame, dense, episodes, flood


def parts_for(kind, fold, inputs):
    context, frame, dense, episodes, flood = inputs
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
                raise ValueError("Changed original count cohort")
            parts[name] = rows.assign(count_target=counts).reset_index(drop=True)
    parts = {p: attach(r.reset_index(drop=True), context) for p, r in parts.items()}
    return parts, eps, [*meta["features"], *COLUMNS]


def prepare(root=ROOT):
    target = root / "plan.json"
    if target.exists():
        plan = read(target)
        if any(plan[k] != (list(v) if isinstance(v, tuple) else v) for k, v in PLAN.items()):
            raise ValueError("Frozen tag study specification changed")
        validate_hashes(plan)
        return plan
    manifest = build()
    audit = read(Path("artifacts/channel_tag_feature_audit.json"))
    validate_hashes(audit)
    if (
        audit["status"] != "direct_onset_snapshots_passed"
        or sha256(Path("artifacts/research-v41-feature-check/direct-values.parquet"))
        != audit["direct_values_sha256"]
    ):
        raise ValueError("Tag features lack direct verification")
    sources = {Path(p) for p in manifest["inputs"]} | {FOLDER / "build.json", FOLDER / "context.parquet"}
    for path in manifest["extraction_manifests"]:
        sources.add(Path(path))
        sources.update(Path(p) for p in read(Path(path))["inputs"])
    sources.update(
        (
            Path("artifacts/channel_tag_support_audit.json"),
            FOLDER.parent / "episodes.parquet",
            Path("artifacts/research-v40/report.json"),
            Path("artifacts/channel_tag_feature_audit.json"),
            Path("artifacts/research-v41-feature-check/plan.json"),
            Path("artifacts/research-v41-feature-check/direct-values.parquet"),
        )
    )
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
    modules = (
        "channel_tag_features",
        "channel_tag_research",
        "binary_gate_research",
        "peer_context_research",
        "flood_research",
        "fresh_counts_research",
        "waiting_time_research",
        "cadence_capacity",
        "cadence_research",
        "count_research",
        "fine_cadence_research",
        "alert_diagnostics",
        "goal90_research",
        "quarter_count_features",
        "research",
        "train",
        "prepare",
    )
    code = {Path(f"src/moscollector/{m}.py") for m in modules}
    code.update((Path("tests/test_channel_tag_features.py"), Path("tests/test_channel_tag_research.py")))
    code.add(Path("scripts/audit_channel_tag_features.py"))
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
        "code_hashes": {str(p): sha256(p) for p in sorted(code)},
    }
    write_json(target, plan)
    return read(target)


def evaluate(kind, fold, inputs, root=ROOT, replay=False):
    directory = root / kind / fold
    target = directory / "result.json"
    parts, eps, columns = parts_for(kind, fold, inputs)
    signatures = {
        p: {
            "rows": len(r),
            "fingerprint": fingerprint(r),
            "events": EventEvaluator(r, eps, 3 if p in ("train", "validation") else 1).events,
        }
        for p, r in parts.items()
    }
    signature = {"plan_sha256": sha256(root / "plan.json"), "parts": signatures, "features": columns}
    fit_path, weights = directory / "fit.json", directory / "model.cbm"
    model = CatBoostRegressor()
    if fit_path.exists():
        fit = read(fit_path)
        if fit["signature"] != signature or sha256(weights) != fit["model_sha256"]:
            raise ValueError("Changed tag training arrays or weights")
        model.load_model(str(weights))
    else:
        if replay:
            raise ValueError("Cannot verify missing model")
        pools = {
            p: Pool(model_input(parts[p], columns), parts[p].count_target, cat_features=CATEGORICAL)
            for p in ("train", "validation")
        }
        model = CatBoostRegressor(**FIT, verbose=200)
        print("TRAIN TAG", kind, fold, len(columns), flush=True)
        model.fit(pools["train"], eval_set=pools["validation"])
        directory.mkdir(parents=True, exist_ok=True)
        model.save_model(str(weights))
        fit = {
            "signature": signature,
            "model_sha256": sha256(weights),
            "best_iteration": model.best_iteration_,
            "validation_history": model.evals_result_,
        }
        write_json(fit_path, fit)
    control_weights, meta = base_model(kind, fold)
    control = CatBoostRegressor()
    control.load_model(str(control_weights))
    cadence = 1 if kind == "flood" else 0.25
    arms = {}
    output_hashes = {}
    for name, estimator, features in (("tag", model, columns), ("control", control, meta["features"])):
        tables, exposure = {}, {}
        for part in ("calibration", "policy", "test"):
            r = parts[part]
            hourly = r[KEYS].assign(
                raw=estimator.predict(
                    model_input(r, features), prediction_type="RawFormulaVal", thread_count=4
                )
            )
            table = (
                hourly
                if cadence == 1
                else carry_features(hourly, subdivide_evaluation_slots(hourly), ["raw"])
            )
            if cohort(table, eps, cadence) != cohort(hourly, eps, 1):
                raise ValueError("Warning cadence changed episode identities")
            tables[part], exposure[part] = table, len(hourly) / 24
        actual = episode_counts(tables["calibration"], eps)
        calibration = calibrate(tables["calibration"].raw.to_numpy(), actual > 0)
        scale = float(actual.sum() / np.exp(np.clip(tables["calibration"].raw.to_numpy(), -20, 20)).sum())
        for table in tables.values():
            table["probability"] = calibrated(table.raw, calibration)
            table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
        policy, frontier = (
            select(tables["policy"], eps, cadence, exposure["policy"])
            if kind == "flood"
            else select_gate_policy(tables["policy"], eps, exposure["policy"])
        )
        for part in ("policy", "test"):
            tables[part]["alert"] = policy_alerts(tables[part], eps, policy)
        scores = evaluator_for(tables["test"], eps, cadence, exposure["test"]).evaluate(
            tables["test"].alert, 0.5, cadence
        )
        arm = {"scores": scores, "policy": policy, "calibration": calibration, "rate_scale": scale}
        if name == "control":
            source = Path("artifacts/research-v38") / fold if kind == "flood" else control_source(kind, fold)
            if (source / "result.json").exists():
                old = read(source / "result.json")["arms"]["poisson" if kind == "flood" else "global_control"]
                before = old["scores"]["test"] if kind == "flood" else old["scores"]
                if (
                    scores != before
                    or policy != old["policy"]
                    or calibration != old["calibration"]
                    or scale != old["rate_scale"]
                ):
                    raise ValueError("Matched tag control differs from original")
                for part in ("policy", "test"):
                    saved = pd.read_parquet(
                        source / f"{'poisson' if kind == 'flood' else 'global_control'}-{part}.parquet"
                    )
                    compare = [*KEYS, "probability", "expected_count"] + (["alert"] if part == "test" else [])
                    pd.testing.assert_frame_equal(tables[part][compare], saved[compare], check_exact=True)
        for part, table in tables.items():
            path = directory / f"{name}-{part}.parquet"
            if replay:
                pd.testing.assert_frame_equal(table, pd.read_parquet(path), check_exact=True)
            else:
                table.to_parquet(path, index=False)
            output_hashes[str(path)] = sha256(path)
        path = directory / f"{name}-frontier.json"
        if replay:
            if frontier != read(path):
                raise ValueError("Changed policy search")
        else:
            write_json(path, frontier)
        output_hashes[str(path)] = sha256(path)
        arms[name] = arm
    refs = references(kind, fold)
    if any(s["eligible_episodes"] != arms["tag"]["scores"]["eligible_episodes"] for s in refs.values()):
        raise ValueError("Changed comparison cohort")
    result = {
        "kind": kind,
        "fold": fold,
        "signature": signature,
        "arms": arms,
        "references": refs,
        "model_sha256": sha256(weights),
        "outputs": output_hashes,
    }
    if replay:
        if result != read(target):
            raise ValueError("Changed tag evaluation")
    else:
        write_json(target, result)
    print("VERIFIED" if replay else "TAG SCORE", kind, fold, arms["tag"]["scores"], flush=True)
    return result


def metrics(row):
    return {
        "tag": row["arms"]["tag"]["scores"],
        "control": row["arms"]["control"]["scores"],
        **row["references"],
    }


def screen(rows):
    if not rows:
        raise ValueError("No tag comparison periods")
    keys = set(metrics(rows[0]))
    if any(set(metrics(r)) != keys for r in rows):
        raise ValueError("Comparison periods omit a reference")
    scores = {k: pooled([metrics(r)[k] for r in rows]) for k in sorted(keys)}
    candidate = scores["tag"]
    passed = all(
        primary_score(candidate) > 1.05 * primary_score(v) and candidate["f1"] >= v["f1"]
        for k, v in scores.items()
        if k != "tag"
    )
    return {"scores": scores, "passed_screen": passed}


def run(root=ROOT, replay=False):
    plan = prepare(root)
    inputs = load_inputs()
    report = {}
    completed = []
    for kind in KINDS:
        folds = tuple(FIVE) if kind == "flood" else ("screen_1", "screen_2")
        rows = [evaluate(kind, fold, inputs, root, replay) for fold in folds]
        selection = screen(rows)
        completed.extend((kind, fold) for fold in folds)
        row = {"selection": selection, "periods": rows, "serving_changed": False, "goal_achieved": False}
        if kind != "flood" and selection["passed_screen"]:
            extra = [evaluate(kind, fold, inputs, root, replay) for fold in ("confirmation", *STRESS)]
            completed.extend((kind, fold) for fold in ("confirmation", *STRESS))
            may = metrics(extra[0])
            stress = screen(extra[1:])
            passed_may = all(
                primary_score(may["tag"]) > primary_score(v) and may["tag"]["f1"] >= 0.95 * v["f1"]
                for k, v in may.items()
                if k != "tag"
            )
            passed_stress = stress["passed_screen"] and all(
                metrics(r)["tag"]["f1"] >= 0.9 * v["f1"]
                for r in extra[1:]
                for k, v in metrics(r).items()
                if k != "tag"
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
        print("TAG RESULT", kind, selection, row["status"], flush=True)
        gc.collect()
    if replay:
        if report != read(root / "report.json"):
            raise ValueError("Changed full-scope tag report")
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
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare(args.output) if args.prepare_only else run(args.output, args.replay)
