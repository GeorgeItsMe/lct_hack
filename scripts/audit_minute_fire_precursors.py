"""v32: fixed causal fire-onset baselines and retrospective fault precursors."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.event_trigger_grid import trigger_alerts, trigger_grid
from moscollector.fine_cadence_research import cohort, evaluator_for
from moscollector.goal90_research import pooled, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

ROOT = Path("artifacts/research-v32")
FOLDS = ("screen_1", "screen_2", "confirmation", "stress_1", "stress_2")
RESOLUTIONS = (1, 60, 900, 3600)
COOLDOWNS = (0, 24)
PLAN = {
    "scope": "Adaptive descriptive trigger audit on all five already-used historical fault periods. Not model selection, a new blind test, or a solution for the full90/90 goal.",
    "hypothesis": "Previous grouped-channel case audit found fire states preceding faults by only a few minutes. Whole-hour aggregates can lose this timing. Quantify available onset signals, actual false-warning load, and the effect of release resolution before committing to a new event-stream predictor.",
    "resolutions_seconds": RESOLUTIONS,
    "cooldowns_hours": COOLDOWNS,
    "signals": ["fire"],
    "rule": "Use every raw channel fire onset, not only channels later known to fail. Release strictly at floor(onset/step)+1 step for1s/1min/15min/1h. At most one trigger per object and release time. Report every fixed arm: no cooldown, or24h since last warning. No learning, calibration, probability thresholds, pending-event confirmations, best-arm selection or production activation.",
    "coverage": "Retain all original valid whole-hour anchors; extra triggers only between adjacent valid hours. Last hour and gaps are never extended. This offline future-coverage mask is not a runtime feature. Require exact217-event total and exact per-period episode identities versus original hourly and archived quarter grids. Exposure is original hours/24, never trigger count or source message count.",
    "metric": "Original24h window[t,t+24h), same one-to-one event matching. Explicit causal cooldown before evaluation. Actual warnings and false-warning burden retained even if they exceed0.25/object-day. Reported metrics are these simple rules, not a trained ML model; no weak-score exclusion or fallback to zero warnings.",
    "case_diagnostic": "For each eligible fault use its original labeled member channels only to DESCRIBE whether any raw fire onset in its preceding24h could trigger strictly before the grouped fault start on each grid. Membership is hindsight, never an input to warning rules. This is not an upper bound on predictability, nor removal of false warnings. Source grouped labels and original members unchanged.",
    "latency_limits": "One-second source timestamps and immediate hypothetical processing; network and ingestion latency are unknown. One-second arm is a timing sensitivity check, not a verified operational latency. Onset after the fault has begun cannot count as a precursor. No June data/labels are evaluated; physical incidents and work schedules remain unverified.",
}


def freeze():
    sources = {
        PROCESSED / "episodes.parquet",
        Path("artifacts/research-v31/report.json"),
        Path("artifacts/fault_precursor_audit.json"),
    }
    for year in (2025, 2026):
        manifest_path = PROCESSED / "channel-novelty-v18" / f"onsets-{year}.json"
        sources.add(manifest_path)
        manifest = read(manifest_path)
        for category in ("inputs", "outputs"):
            for name, value in manifest[category].items():
                path = Path(name)
                if sha256(path) != value:
                    raise ValueError(f"Changed raw onset provenance: {path}")
                sources.add(path)
    for fold in FOLDS:
        directory = Path("artifacts/research-v31/fault") / fold
        result = read(directory / "result.json")
        sources.update(
            (
                directory / "result.json",
                directory / "pretrained-test.parquet",
                Path(result["source_control_directory"]) / "global_control-test.parquet",
            )
        )
    code = {
        Path(__file__),
        Path(trigger_grid.__code__.co_filename),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(evaluator_for.__code__.co_filename),
        Path("tests/test_event_trigger_grid.py"),
    }
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
        "code_hashes": {str(p): sha256(p) for p in sorted(code)},
    }
    # JSON round-trip normalizes tuple configuration fields before equality.
    import json

    plan = json.loads(json.dumps(plan))
    path = ROOT / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Trigger-audit inputs/code changed; preserve this audit and use a new directory")
    if not path.exists():
        write_json(path, plan)
    return plan


def run():
    plan = freeze()
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet",
        filters=[("kind", "==", "fault"), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
    )
    onsets = pd.concat(
        [
            pd.read_parquet(
                PROCESSED / "channel-novelty-v18" / f"onsets-{year}.parquet",
                filters=[("signal", "==", "fire"), ("ts", "<", pd.Timestamp("2026-06-01"))],
            )
            for year in (2025, 2026)
        ],
        ignore_index=True,
    )
    fire_times = {
        (int(obj), int(channel)): rows.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        for (obj, channel), rows in onsets.sort_values("ts").groupby(["object_id", "channel_id"])
    }
    members = episodes.set_index(["object_id", "start_ts"]).channel_ids.to_dict()
    periods, cases = [], []
    for fold in FOLDS:
        source = Path("artifacts/research-v31/fault") / fold
        (ROOT / fold).mkdir(parents=True, exist_ok=True)
        before = read(source / "result.json")
        archived = pd.read_parquet(source / "pretrained-test.parquet")
        hourly = archived.loc[
            archived.as_of.eq(archived.as_of.dt.floor("h")), ["object_id", "as_of"]
        ].reset_index(drop=True)
        exposure = len(hourly) / 24
        assert exposure == before["exposure_days"]
        original_cohort = cohort(hourly, episodes, 1)
        assert original_cohort == cohort(archived, episodes, 0.25)
        control = pd.read_parquet(Path(before["source_control_directory"]) / "global_control-test.parquet")
        assert cohort(control, episodes, 0.25) == original_cohort
        assert (
            evaluator_for(control, episodes, 0.25, exposure).evaluate(control.alert, 0.5, 0.25)
            == before["controls"]["global_control"]
        )
        row = {
            "fold": fold,
            "eligible_episodes": before["scores"]["eligible_episodes"],
            "exposure_days": exposure,
            "arms": {},
            "reference_catboost": before["controls"]["global_control"],
            "availability": {},
        }
        for seconds in RESOLUTIONS:
            slots = trigger_grid(hourly, onsets, seconds)
            assert cohort(slots, episodes, seconds / 3600) == original_cohort
            n_triggers = int(slots.trigger_count.gt(0).sum())
            row["availability"][str(seconds)] = {
                "forecast_rows": len(slots),
                "trigger_times": n_triggers,
                "onsets_released": int(slots.trigger_count.sum()),
            }
            released = {
                int(obj): g.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                for obj, g in slots.loc[slots.trigger_count.gt(0)].groupby("object_id")
            }
            for cooldown in COOLDOWNS:
                alerts = trigger_alerts(slots, cooldown)
                metrics = evaluator_for(slots, episodes, seconds / 3600, exposure).evaluate(
                    alerts, 0.5, max(seconds / 3600, cooldown)
                )
                assert (
                    metrics["alerts"] == int(alerts.sum())
                    and metrics["eligible_episodes"] == row["eligible_episodes"]
                )
                row["arms"][f"release_{seconds}s_cooldown_{cooldown}h"] = {
                    "scores": metrics,
                    "within_original_fp_budget": metrics["false_alerts_per_object_day"] <= 0.25,
                }
            step = seconds * 1_000_000_000
            for obj, events in original_cohort.items():
                for event in events:
                    support = []
                    for channel in members[(obj, pd.Timestamp(event))]:
                        ts = fire_times.get((obj, int(channel)), np.empty(0, dtype=np.int64))
                        past = ts[(ts > event - 24 * HOUR_NS) & (ts < event)]
                        candidate = (past // step + 1) * step
                        usable = candidate[
                            (candidate < event)
                            & np.isin(candidate, released.get(obj, np.empty(0, dtype=np.int64)))
                        ]
                        support.extend(usable.tolist())
                    cases.append(
                        {
                            "fold": fold,
                            "object_id": obj,
                            "start_ts": pd.Timestamp(event).isoformat(),
                            "resolution_seconds": seconds,
                            "member_fire_onset_visible_strictly_before_fault": bool(support),
                            "max_lead_minutes": (event - min(support)) / (HOUR_NS / 60) if support else None,
                        }
                    )
            slots.to_parquet(ROOT / fold / f"slots-{seconds}.parquet", index=False)
        periods.append(row)
        write_json(ROOT / fold / "result.json", row)
        print(
            "DONE trigger audit",
            fold,
            {k: (v["scores"]["true_alerts"], v["scores"]["alerts"]) for k, v in row["arms"].items()},
            flush=True,
        )
    assert sum(p["eligible_episodes"] for p in periods) == 217
    totals = {arm: pooled([p["arms"][arm]["scores"] for p in periods]) for arm in periods[0]["arms"]}
    availability = {
        str(seconds): {
            "eligible_episodes": sum(c["resolution_seconds"] == seconds for c in cases),
            "member_fire_onset_visible_strictly_before_fault": sum(
                c["resolution_seconds"] == seconds and c["member_fire_onset_visible_strictly_before_fault"]
                for c in cases
            ),
        }
        for seconds in RESOLUTIONS
    }
    report = {
        "scope": PLAN["scope"],
        "plan_sha256": sha256(ROOT / "plan.json"),
        "periods": periods,
        "pooled": totals,
        "precursor_availability": availability,
        "case_diagnostic": cases,
        "reference_catboost": pooled([p["reference_catboost"] for p in periods]),
        "same_episode_cohorts": True,
        "automatic_activation": False,
        "new_blind_test": False,
        "full_scope_goal_achieved": False,
        "source_hashes": plan["source_hashes"],
        "code_hashes": plan["code_hashes"],
    }
    write_json(ROOT / "report.json", report)
    print("POOLED", totals, flush=True)
    print("PRECURSOR AVAILABILITY", availability, flush=True)


if __name__ == "__main__":
    run()
