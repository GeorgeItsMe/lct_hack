"""Evaluate 15-minute warning opportunities with frozen hourly count scores.

The old score is a lagged input, calibrated separately against the unchanged
24-hour target AT THE NEW TIME. This is adaptive retrospective research, not
new telemetry, new tree weights, or an independently tested release.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.goal90_research import STRESS, PendingSimulator, pooled, primary_score, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import calibrate, calibrated, model_input
from moscollector.waiting_time_research import base_directory, reference_result

KINDS = ("access", "fire")
PLAN = {
    "scope": "adaptive_retrospective_cadence_followup_not_blind_test",
    "motivation": "Exact future-informed capacity diagnostic found hourly recall below90% in December for access/fire. It does not establish predictive performance. Fault has no demonstrated hourly capacity deficit and is outside this isolated cadence study.",
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "weights": "Frozen original24h count weights, identical in both arms. No tree fitting or hyperparameter search.",
    "arms": "hourly_control and quarter_candidate. Each has a separately fitted sigmoid and mean-count scale on its own calibration grid, using the original calibration dates and25h purge. Old3h-calibrated hourly predictions remain a separate archived anchor.",
    "features": "Latest whole-hour raw model score carried forward at most45min, never interpolated from a future score. Treat this as a lagged feature, NOT an unchanged forecast window. Labels are original episodes in[current_time,current_time+24h) at every new time.",
    "eligibility": "Add15/30/45min slots only between adjacent valid hourly evaluation slots; no gaps or final-horizon extension. Assert exact original event identities. This offline future-coverage eligibility mask is NOT an input or runtime schedule.",
    "causality": "Episode information becomes available only at floor(start+70min to hour)+1h. This exactly reproduces old strict70min confirmation on whole hours, and does not disclose future hourly coverage at quarter times. One confirmation resolves at most one pending warning; pending expires after24h.",
    "policy": "Same mean_retarget grid: capacity multipliers .5,1,1.5,2; margins .25,.5,1,2,3,5; floors0,.5,.75,.9,.95. Policy period only. At most one warning per arm slot; >=10alerts and >=10episodes for supported status. FP budget .25/object-day, with identical original hourly exposure in both arms. Explicit zero-alert fallback if no policy fits budget.",
    "selection": "Nov/Feb pooled: quarter must improve min(P/.9,R/.9,1) by>5% against BOTH recalibrated hourly control and archived mean90 anchor, without pooled F1 loss. Only passing kinds proceed; no replacement after results.",
    "confirmation": "May must improve primary with F1>=95% against both references. Pooled Dec/Mar must improve primary>5%, no pooled F1 loss, each month's F1>=90% its reference. Gates are research eligibility, not achievement of90/90.",
    "unchanged": "Same24h horizon, same grouped original episodes, one-to-one matching, beforeJune only, no relabeling, exclusion, production change or automatic promotion.",
}


def hold_hourly_scores(hourly, slots):
    """Backward-only score lookup; unknown future scores cannot affect a row."""
    if hourly.duplicated(["object_id", "as_of"]).any():
        raise ValueError("Duplicate hourly scores")
    if not hourly.as_of.eq(hourly.as_of.dt.floor("h")).all():
        raise ValueError("Scores must be on whole hours")
    out = []
    for obj, rows in slots.groupby("object_id", sort=True):
        rows = rows.sort_values("as_of").copy()
        source = hourly[hourly.object_id.eq(obj)].sort_values("as_of")
        times = source.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        before = np.searchsorted(times, at, side="right") - 1
        if not len(times) or np.any(before < 0):
            raise ValueError("No past score for forecast slot")
        age = at - times[before]
        if np.any(age >= HOUR_NS):
            raise ValueError("Stale score across missing hourly coverage")
        rows["source_time"] = pd.to_datetime(times[before])
        rows["source_age_minutes"] = age / HOUR_NS * 60
        rows["raw"] = source.raw.to_numpy()[before]
        out.append(rows)
    if not out:
        return pd.DataFrame(columns=["object_id", "as_of", "source_time", "source_age_minutes", "raw"])
    result = pd.concat(out, ignore_index=True)
    if not np.isfinite(result.raw).all():
        raise ValueError("Nonfinite hourly score")
    return result


class FinePendingSimulator(PendingSimulator):
    """Conservative whole-hour information release, even on finer grids."""

    def alerts(self, capacity, probability, margin, floor):
        capacity, probability = np.asarray(capacity), np.asarray(probability)
        if any(x.shape != (self.n,) or not np.isfinite(x).all() for x in (capacity, probability)):
            raise ValueError("Invalid capacity/probability vector")
        if margin <= 0 or not 0 <= floor <= 1:
            raise ValueError("Invalid policy")
        result = np.zeros(self.n)
        delay = 70 * 60 * 1_000_000_000
        for ids, times, events in self.groups:
            available = ((events + delay) // HOUR_NS + 1) * HOUR_NS
            pending, next_event = [], 0
            for i, now in zip(ids, times, strict=True):
                while next_event < len(events) and available[next_event] <= now:
                    observed = events[next_event]
                    next_event += 1
                    for j, issued in enumerate(pending):
                        if issued <= observed < issued + 24 * HOUR_NS:
                            pending.pop(j)
                            break
                pending = [issued for issued in pending if now - issued < 24 * HOUR_NS]
                if probability[i] >= floor and capacity[i] >= len(pending) + margin:
                    result[i] = 1
                    pending.append(now)
        return result


def evaluator_for(pred, episodes, cadence, exposure_days):
    evaluator = EventEvaluator(pred, episodes, cadence)
    # Both arms cover the SAME original observable hourly exposure. More
    # forecast opportunities neither create exposure nor dilute false alarms.
    evaluator.opportunity_days = exposure_days
    return evaluator


def cohort(pred, episodes, cadence):
    return {
        obj: tuple(events)
        for obj, _, _, events in EventEvaluator(pred, episodes, cadence).groups
        if len(events)
    }


def policy_alerts(pred, episodes, policy):
    return FinePendingSimulator(pred, episodes).alerts(
        pred.expected_count.to_numpy() * float(policy["capacity"]),
        pred.probability.to_numpy(),
        policy["margin"],
        policy["floor"],
    )


def select(pred, episodes, cadence, exposure_days):
    simulator = FinePendingSimulator(pred, episodes)
    evaluator = evaluator_for(pred, episodes, cadence, exposure_days)
    options = []
    for multiplier in (0.5, 1, 1.5, 2):
        for margin in (0.25, 0.5, 1, 2, 3, 5):
            for floor in (0, 0.5, 0.75, 0.9, 0.95):
                values = simulator.alerts(
                    pred.expected_count.to_numpy() * multiplier, pred.probability, margin, floor
                )
                scores = evaluator.evaluate(values, 0.5, cadence)
                options.append({"capacity": multiplier, "margin": margin, "floor": floor, **scores})
    budget = [r for r in options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    fallback = {
        "capacity": 0,
        "margin": 1,
        "floor": 0,
        **evaluator.evaluate(np.zeros(len(pred)), 0.5, cadence),
    }
    chosen = dict(
        max(
            supported or budget or [fallback],
            key=lambda r: (primary_score(r), r["f1"], r["recall"], r["precision"]),
        )
    )
    chosen["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    return chosen, options


def evaluate(root, kind, fold, frame, dense, episodes):
    directory = root / kind / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    base = base_directory(kind, fold)
    meta = read(base / f"{kind}.json")
    model = CatBoostRegressor()
    model.load_model(str(base / f"{kind}.cbm"))
    directory.mkdir(parents=True, exist_ok=True)
    hourly = {}
    parity = {}
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, meta["periods"][period]))
        reference = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], reference)
        if not len(rows):
            raise ValueError(f"Missing {period} coverage")
        pred = rows[["object_id", "as_of"]].copy()
        pred["raw"] = model.predict(
            model_input(rows, meta["features"]), prediction_type="RawFormulaVal", thread_count=2
        )
        if period != "calibration":
            saved = pd.read_parquet(base / f"{kind}-{period}.parquet")
            shared = pred.merge(saved, on=["object_id", "as_of"], validate="one_to_one")
            assert len(shared) == len(pred) == len(saved)
            np.testing.assert_allclose(
                calibrated(shared.raw, meta["calibration"]), shared.probability, atol=1e-10, rtol=1e-10
            )
            np.testing.assert_allclose(
                np.exp(np.clip(shared.raw, -20, 20)) * meta["rate_scale"],
                shared.expected_count,
                atol=1e-10,
                rtol=1e-10,
            )
            parity[period] = True
        hourly[period] = pred
    arms = {}
    for name, cadence in (("hourly_control", 1.0), ("quarter_candidate", 0.25)):
        print("START cadence", kind, fold, name, flush=True)
        tables = {}
        for period, scores in hourly.items():
            slots = scores[["object_id", "as_of"]] if cadence == 1 else subdivide_evaluation_slots(scores)
            tables[period] = hold_hourly_scores(scores, slots)
            assert cohort(tables[period], episodes, cadence) == cohort(scores, episodes, 1)
        cal = tables["calibration"]
        counts = episode_counts(cal, episodes)
        calibration = calibrate(cal.raw.to_numpy(), counts > 0)
        scale = float(counts.sum() / np.exp(np.clip(cal.raw.to_numpy(), -20, 20)).sum())
        for table in tables.values():
            table["probability"] = calibrated(table.raw, calibration)
            table["expected_count"] = np.exp(np.clip(table.raw, -20, 20)) * scale
        policy_pred = tables["policy"]
        if cadence == 1:
            new_sim = FinePendingSimulator(policy_pred, episodes)
            old_sim = PendingSimulator(policy_pred, episodes)
            for capacity, margin, floor in ((0.5, 0.25, 0), (1, 1, 0.75), (2, 3, 0.95)):
                values = policy_pred.expected_count.to_numpy() * capacity
                np.testing.assert_array_equal(
                    new_sim.alerts(values, policy_pred.probability, margin, floor),
                    old_sim.alerts(values, policy_pred.probability, margin, floor),
                )
        chosen, frontier = select(policy_pred, episodes, cadence, len(hourly["policy"]) / 24)
        test = tables["test"]
        test["alert"] = policy_alerts(test, episodes, chosen)
        metrics = evaluator_for(test, episodes, cadence, len(hourly["test"]) / 24).evaluate(
            test.alert, 0.5, cadence
        )
        tables["policy"].to_parquet(directory / f"{name}-policy.parquet", index=False)
        test.to_parquet(directory / f"{name}-test.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {
            "scores": metrics,
            "policy": chosen,
            "calibration": calibration,
            "rate_scale": scale,
            "calibration_rows": len(cal),
            "exposure_days": len(hourly["test"]) / 24,
            "cadence_hours": cadence,
        }
        print("DONE cadence", kind, fold, name, metrics, flush=True)
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "scores": arms["quarter_candidate"]["scores"],
        "control": arms["hourly_control"]["scores"],
        "reference": reference_result(kind, fold),
        "identical_episode_cohort": True,
        "hourly_pending_confirmation_parity": True,
        "archived_prediction_parity": parity,
        "periods": meta["periods"],
        "source_model_sha256": sha256(base / f"{kind}.cbm"),
        "source_features": meta["features"],
    }
    for other in (result["control"], result["reference"]):
        assert result["scores"]["eligible_episodes"] == other["eligible_episodes"]
    write_json(target, result)
    return result


def lock_plan(root):
    sources = [
        PROCESSED / "features.parquet",
        PROCESSED / "features-dense-round4.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/hourly_capacity_audit.json"),
    ]
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            base = base_directory(kind, fold)
            sources.extend(
                base / f"{kind}{suffix}" for suffix in (".json", ".cbm", "-policy.parquet", "-test.parquet")
            )
            sources.append(
                Path("artifacts/research-v12") / fold / "uncorrected.json"
                if kind == "access"
                else Path("artifacts/research-v13-policy") / f"{kind}-{fold}.json"
            )
    code = {
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (
                subdivide_evaluation_slots,
                align_opportunities,
                episode_counts,
                PendingSimulator.__init__,
                mask,
                calibrate,
                base_directory,
                EventEvaluator.__init__,
            )
        ),
    }
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in sorted(code)},
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    # JSON normalizes tuples before equality validation.
    import json

    plan = json.loads(json.dumps(plan))
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
    dense = pd.read_parquet(PROCESSED / "features-dense-round4.parquet")
    assert dense.as_of.lt(pd.Timestamp("2026-06-01")).all()
    all_episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            episodes = all_episodes[all_episodes.kind.eq(kind)]
            rows = [evaluate(root, kind, fold, frame, dense, episodes) for fold in ("screen_1", "screen_2")]
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
            print("SELECT cadence", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Finish screening both kinds first")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        episodes = all_episodes[all_episodes.kind.eq(kind)]
        rows = [evaluate(root, kind, fold, frame, dense, episodes) for fold in ("confirmation", *STRESS)]
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
        print("REPORT cadence", kind, {k: v for k, v in report[kind].items() if k != "periods"}, flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v20"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
