"""v21: refresh raw report-count windows on the proven 15-minute grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.experiments.cadence_capacity import subdivide_evaluation_slots
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts, select
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.quarter_count_features import carry_features, refresh_counts
from moscollector.experiments.waiting_time_research import base_directory, reference_result
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import calibrate, calibrated, model_input

KINDS = ("access", "fire", "fault")
COUNT_FOLDER = PROCESSED / "quarter-counts-v21"
PLAN = {
    "scope": "adaptive_retrospective_feature_freshness_followup_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "reason": "V20 access improved with15min warnings but scores stayed hourly. Test whether counts from new raw observations within the hour improve predictions, with cadence held fixed.",
    "weights": "Unchanged v9 count weights for access/fire; unchanged v19 channel-novelty count weights for fault, selected in the previous study. Both arms use identical tree weights. No tree training, additional feature columns or external data.",
    "arms": "Both15min: held_control carries all last-whole-hour features; fresh_candidate refreshes44 raw report counts for1/6/24/168h and, where present, six derived6h-vs168h burst ratios. All other inputs stay at the same whole-hour values, including measurements, recurrence and channel-novelty features. This tests partial raw-feature freshness, not complete15min feature recomputation.",
    "raw_definition": "Exactly original canonical per-channel timestamp aggregation, bool_or alarm, conflicting values unknown, original signal definitions. Count windows[t-window,t) at quarter boundaries. Raw buckets summed to hours and every valid original hourly count/burst feature must match exactly.",
    "causality": "No observations at/after forecast time. Both arms use v20 whole-hour availability for confirmed episodes,24h pending expiry,one-to-one matching,unchanged24h target. Both separately calibrate probability and mean count on original calibration dates with25h purge; policy selected only in the original policy interval.",
    "cohort": "Identical15min slots inside adjacent eligible hourly points, no gaps or final extension; exact original event identities. Offline eligibility is not an input/runtime signal. False-alert denominator is identical original hourly exposure.",
    "policy": "Frozen v20 mean_retarget policy grid and joint90/90 objective; >=10alerts/episodes for supported status, false alerts<=.25/object-day during policy selection; same explicit empty fallback.",
    "anchor": "Access: completed v20 quarter candidate (must reproduce as held control). Fire: original mean90 count policy, in addition to reproducing v20 quarter control on screen. Fault: completed v19 hourly count policy. These are historical count-family comparisons, not deployed fire/fault classifier comparisons.",
    "selection": "Nov/Feb pooled fresh candidate must improve primary min(P/.9,R/.9,1) by>5% against BOTH held_control and anchor, with no pooled F1 loss. Only passing kinds continue; no substitution.",
    "confirmation": "May improves primary withF1>=95% against both. Pooled Dec/Mar primary improves>5% withoutF1 loss and each month's F1>=90% each reference. Research gates do not establish90/90 or activate production.",
    "june": "No June labels, policy selection or evaluation. Previously used historical months are adaptively reused; not independent confirmation.",
}


def source_files(kind, fold):
    if kind == "fault":
        base = Path("artifacts/research-v19/fault") / fold
        return (
            base / "model.cbm",
            base / "fit.json",
            {period: base / f"{period}.parquet" for period in ("policy", "test")},
        )
    base = base_directory(kind, fold)
    return (
        base / f"{kind}.cbm",
        base / f"{kind}.json",
        {period: base / f"{kind}-{period}.parquet" for period in ("policy", "test")},
    )


def anchor(kind, fold):
    if kind == "access":
        return read(Path("artifacts/research-v20/access") / fold / "result.json")["scores"]
    if kind == "fault":
        return read(Path("artifacts/research-v19/fault") / fold / "result.json")["scores"]
    return reference_result(kind, fold)


def evaluate(root, kind, fold, frame, dense, episodes, counts):
    directory = root / kind / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    weights, metadata, archived = source_files(kind, fold)
    meta = read(metadata)
    model = CatBoostRegressor()
    model.load_model(str(weights))
    columns = meta["features"]
    directory.mkdir(parents=True, exist_ok=True)
    tables = {name: {} for name in ("held_control", "fresh_candidate")}
    exposure, changed = {}, {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, meta["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        hourly = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        slots = subdivide_evaluation_slots(hourly)
        held = carry_features(hourly, slots, columns)
        fresh = refresh_counts(held, counts)
        whole = fresh.as_of.eq(fresh.source_time)
        pd.testing.assert_frame_equal(held.loc[whole, columns], fresh.loc[whole, columns])
        raw_arrays = {}
        for name, inputs in (("held_control", held), ("fresh_candidate", fresh)):
            pred = inputs[["object_id", "as_of", "source_time"]].copy()
            pred["raw"] = model.predict(
                model_input(inputs, columns), prediction_type="RawFormulaVal", thread_count=2
            )
            if not np.isfinite(pred.raw).all():
                raise ValueError("Nonfinite forecast")
            assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
            tables[name][period] = pred
            raw_arrays[name] = pred.raw.to_numpy()
        difference = np.abs(raw_arrays["fresh_candidate"] - raw_arrays["held_control"])
        changed[period] = {
            "rows": len(held),
            "changed_rows": int((difference > 1e-10).sum()),
            "max_raw_difference": float(difference.max(initial=0)),
        }
        if period in archived:
            saved = pd.read_parquet(archived[period])
            shared = (
                tables["held_control"][period]
                .loc[whole]
                .merge(saved, on=["object_id", "as_of"], validate="one_to_one")
            )
            assert len(shared) == len(saved) == len(hourly)
            np.testing.assert_allclose(
                calibrated(shared.raw, meta["calibration"]), shared.probability, atol=1e-10, rtol=1e-10
            )
            np.testing.assert_allclose(
                np.exp(np.clip(shared.raw, -20, 20)) * meta["rate_scale"],
                shared.expected_count,
                atol=1e-10,
                rtol=1e-10,
            )
        exposure[period] = len(hourly) / 24
    arms = {}
    for name, predictions in tables.items():
        print("START fresh counts", kind, fold, name, changed["test"], flush=True)
        cal = predictions["calibration"]
        actual = episode_counts(cal, episodes)
        calibration = calibrate(cal.raw.to_numpy(), actual > 0)
        scale = float(actual.sum() / np.exp(np.clip(cal.raw.to_numpy(), -20, 20)).sum())
        for table in predictions.values():
            table["probability"] = calibrated(table.raw, calibration)
            table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
        policy, frontier = select(predictions["policy"], episodes, 0.25, exposure["policy"])
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, policy)
        metrics = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
        predictions["policy"].to_parquet(directory / f"{name}-policy.parquet", index=False)
        test.to_parquet(directory / f"{name}-test.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {
            "scores": metrics,
            "policy": policy,
            "calibration": calibration,
            "rate_scale": scale,
            "calibration_rows": len(cal),
            "exposure_days": exposure["test"],
        }
        print("DONE fresh counts", kind, fold, name, metrics, flush=True)
    # Frozen access/fire held-score results must replay the completed v20 arm.
    old_result = Path("artifacts/research-v20") / kind / fold / "result.json"
    if kind != "fault" and old_result.exists():
        old = read(old_result)
        control = arms["held_control"]
        for field in ("true_alerts", "alerts", "eligible_episodes"):
            assert control["scores"][field] == old["scores"][field]
        assert control["policy"] == old["arms"]["quarter_candidate"]["policy"]
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "scores": arms["fresh_candidate"]["scores"],
        "control": arms["held_control"]["scores"],
        "reference": anchor(kind, fold),
        "changed_raw_predictions": changed,
        "whole_hour_feature_parity": True,
        "whole_hour_archived_prediction_parity": True,
        "identical_episode_cohort": True,
        "source_weights_sha256": sha256(weights),
        "periods": meta["periods"],
        "features": columns,
    }
    for other in (result["control"], result["reference"]):
        assert result["scores"]["eligible_episodes"] == other["eligible_episodes"]
    write_json(target, result)
    return result


def lock_plan(root):
    sources = [
        PROCESSED / "features.parquet",
        PROCESSED / "features-dense-channel-novelty.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/research-v20/report.json"),
        Path("artifacts/research-v19/report.json"),
        COUNT_FOLDER / "build.json",
    ]
    for year in (2025, 2026):
        sources.extend([COUNT_FOLDER / f"counts-{year}.parquet", COUNT_FOLDER / f"counts-{year}.json"])
    for manifest in [
        COUNT_FOLDER / "build.json",
        *(COUNT_FOLDER / f"counts-{year}.json" for year in (2025, 2026)),
    ]:
        for category in ("inputs", "outputs"):
            for p, digest in read(manifest)[category].items():
                if sha256(Path(p)) != digest:
                    raise ValueError(f"Raw quarter count provenance changed: {p}")
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            weights, metadata, saved = source_files(kind, fold)
            sources.extend([weights, metadata, *saved.values()])
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
                carry_features,
                refresh_counts,
                select,
                cohort,
                evaluator_for,
                episode_counts,
                mask,
                calibrate,
                subdivide_evaluation_slots,
                align_opportunities,
                pooled,
                reference_result,
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
    frame = pd.read_parquet(
        PROCESSED / "features.parquet",
        columns=["object_id", "as_of", "eligible"],
        filters=[("as_of", "<", pd.Timestamp("2026-06-01"))],
    )
    dense = pd.read_parquet(PROCESSED / "features-dense-channel-novelty.parquet")
    assert dense.as_of.lt(pd.Timestamp("2026-06-01")).all()
    all_episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    counts = pd.concat(
        [pd.read_parquet(COUNT_FOLDER / f"counts-{year}.parquet") for year in (2025, 2026)], ignore_index=True
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            eps = all_episodes[all_episodes.kind.eq(kind)]
            rows = [
                evaluate(root, kind, fold, frame, dense, eps, counts) for fold in ("screen_1", "screen_2")
            ]
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
            print("SELECT fresh counts", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete screening all kinds first")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        eps = all_episodes[all_episodes.kind.eq(kind)]
        rows = [evaluate(root, kind, fold, frame, dense, eps, counts) for fold in ("confirmation", *STRESS)]
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
            "REPORT fresh counts", kind, {k: v for k, v in report[kind].items() if k != "periods"}, flush=True
        )
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v21"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
