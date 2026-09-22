"""Describe fixed v42 warnings and policy-only cross-comparisons; no selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.fine_cadence_research import evaluator_for
from moscollector.goal90_research import read
from moscollector.prepare import sha256, write_json
from moscollector.temporal_count_research import CUTOFF, KEYS, PLAN, ROOT
from moscollector.temporal_count_verification import verified_evidence

COUNTS = (
    "true_alerts",
    "false_empty",
    "false_redundant",
    "eligible_episodes",
    "missed_episodes",
    "lead_under_1h",
    "lead_1_to_6h",
    "lead_at_least_6h",
)


def late_confirmation_audit(pred, episodes, cadence, exposure):
    """Compare owners on a FIXED warning schedule; never issue new warnings.

    A causal comparison ledger retains expired warnings for late confirmations.
    Its ownership is checked against the unchanged offline one-to-one scorer
    only for events actually revealed by each forecast tick. This is a state
    audit, not a forecast improvement or a counterfactual policy rollout.
    """
    evaluator = evaluator_for(pred, episodes, cadence, exposure)
    totals = dict.fromkeys(
        (
            "confirmed_events",
            "confirmed_events_with_original_owner",
            "expired_original_warning_resolutions",
            "wrong_warning_resolutions",
            "lost_confirmations",
            "spurious_resolutions",
            "warnings_issued_while_under_reserved",
            "under_reserved_true_warnings",
            "under_reserved_false_warnings",
        ),
        0,
    )
    differences = []
    lifetime, delay = 24 * HOUR_NS, 70 * 60 * 10**9
    for obj, ids, times, events in evaluator.groups:
        warnings = times[pred.alert.to_numpy()[ids] >= 0.5]
        warning_set = set(map(int, warnings))
        offline_owner, true_warnings, pointer = {}, set(), 0
        for issued in warnings:
            pointer = max(pointer, int(np.searchsorted(events, issued)))
            if pointer < len(events) and events[pointer] - issued < lifetime:
                offline_owner[pointer] = int(issued)
                true_warnings.add(int(issued))
                pointer += 1
        release = ((events + delay) // HOUR_NS + 1) * HOUR_NS
        old, retained, next_event = [], [], 0
        for now in times:
            while next_event < len(events) and release[next_event] <= now:
                event = events[next_event]
                owners = []
                for queue in (old, retained):
                    owner = next((i for i, w in enumerate(queue) if w <= event < w + lifetime), None)
                    owners.append(queue.pop(owner) if owner is not None else None)
                old_owner, retained_owner = owners
                if retained_owner != offline_owner.get(next_event):
                    raise ValueError("Retained causal ledger disagrees with original one-to-one ownership")
                totals["confirmed_events"] += 1
                if retained_owner is not None:
                    totals["confirmed_events_with_original_owner"] += 1
                    totals["expired_original_warning_resolutions"] += int(now - retained_owner >= lifetime)
                if old_owner != retained_owner:
                    reason = (
                        "lost_confirmations"
                        if old_owner is None
                        else "spurious_resolutions"
                        if retained_owner is None
                        else "wrong_warning_resolutions"
                    )
                    totals[reason] += 1
                    differences.append(
                        {
                            "object_id": obj,
                            "event_position": next_event,
                            "event_at": str(pd.Timestamp(event)),
                            "observed_at": str(pd.Timestamp(now)),
                            "reason": reason,
                            "old_warning_at": str(pd.Timestamp(old_owner)) if old_owner is not None else None,
                            "original_warning_at": str(pd.Timestamp(retained_owner))
                            if retained_owner is not None
                            else None,
                        }
                    )
                next_event += 1
            old = [w for w in old if now - w < lifetime]
            # After all newly visible events were processed, no removed warning
            # can own any event whose strict <24h horizon is still unconfirmed.
            retained = [w for w in retained if now - w < lifetime + delay + HOUR_NS]
            retained_active = sum(now - w < lifetime for w in retained)
            if int(now) in warning_set:
                if len(old) < retained_active:
                    totals["warnings_issued_while_under_reserved"] += 1
                    totals["under_reserved_true_warnings"] += int(int(now) in true_warnings)
                    totals["under_reserved_false_warnings"] += int(int(now) not in true_warnings)
                old.append(int(now))
                retained.append(int(now))
    return {
        "totals": totals,
        "ownership_differences": differences,
        "confirmed_prefix_ownership_verified": True,
    }


def warning_audit(pred, episodes, expected, cadence, exposure):
    if not pred.alert.isin([0, 1]).all():
        raise ValueError("Expected fixed binary warning flags")
    evaluator = evaluator_for(pred, episodes, cadence, exposure)
    if evaluator.evaluate(pred.alert, 0.5, cadence) != expected:
        raise ValueError("Warning audit differs from archived event scores")
    matches, cohort = {}, {}
    empty = redundant = 0
    alerts = pred.alert.to_numpy()
    for obj, ids, times, events in evaluator.groups:
        cohort[obj] = tuple(map(int, events))
        next_event = 0
        for at in times[alerts[ids] >= 0.5]:
            next_event = max(next_event, int(np.searchsorted(events, at)))
            if next_event < len(events) and events[next_event] - at < 24 * HOUR_NS:
                # Position in the exact common cohort preserves duplicate starts.
                matches[(obj, next_event)] = float((events[next_event] - at) / HOUR_NS)
                next_event += 1
            else:
                future = np.searchsorted(events, at + 24 * HOUR_NS) - np.searchsorted(events, at)
                empty += int(future == 0)
                redundant += int(future > 0)
    leads = np.asarray(list(matches.values()))
    stats = {
        "true_alerts": len(matches),
        "false_empty": empty,
        "false_redundant": redundant,
        "eligible_episodes": evaluator.events,
        "missed_episodes": evaluator.events - len(matches),
        "lead_under_1h": int((leads < 1).sum()),
        "lead_1_to_6h": int(((leads >= 1) & (leads < 6)).sum()),
        "lead_at_least_6h": int((leads >= 6).sum()),
    }
    if len(matches) != expected["true_alerts"] or empty + redundant != expected["false_alerts"]:
        raise ValueError("Warning error categories do not preserve all alerts")
    return {"stats": stats, "matches": matches, "cohort": cohort}


def paired(candidate, reference):
    if candidate["cohort"] != reference["cohort"]:
        raise ValueError("Paired warning audit changed episode cohort")
    a, b = candidate["matches"], reference["matches"]
    common = sorted(a.keys() & b.keys())
    return {
        "both_find": len(common),
        "candidate_only": len(a.keys() - b.keys()),
        "reference_only": len(b.keys() - a.keys()),
        "candidate_earlier": sum(a[k] > b[k] for k in common),
        "reference_earlier": sum(a[k] < b[k] for k in common),
        "same_warning_time": sum(a[k] == b[k] for k in common),
        "median_paired_lead_difference_hours": float(np.median([a[k] - b[k] for k in common]))
        if common
        else None,
    }


def policy_lookup(frontier, chosen):
    keys = ("capacity", "margin", "floor")
    found = [r for r in frontier if all(float(r[k]) == float(chosen[k]) for k in keys)]
    if len(found) == 1:
        return found[0]
    if not found and chosen["capacity"] == 0:
        return None  # Explicit zero fallback is not one of the grid rows.
    raise ValueError("Selected policy missing or duplicated in full preceding-period frontier")


def audit(root=ROOT, target=Path("artifacts/temporal_count_error_audit.json")):
    proof = verified_evidence(root)
    report = proof["report"]
    episode_path = Path("data/processed/episodes.parquet")
    episodes = pd.read_parquet(episode_path, filters=[("start_ts", "<", CUTOFF)])
    sources = {episode_path, root / "report.json", root / "plan.json", root / "weight-replay.json"}
    periods, comparisons, cross, late = [], [], [], []
    for kind, outcome in report.items():
        eps = episodes.loc[episodes.kind.eq(kind)]
        cadence = 1 if kind == "flood" else 0.25
        for result in outcome["periods"]:
            fold = result["fold"]
            directory = root / kind / fold
            sources.add(directory / "result.json")
            exposure = result["signature"]["parts"]["test"]["rows"] / 24
            analyses, original_keys = {}, None
            for arm in PLAN["arms"]:
                path = directory / f"{arm}-test.parquet"
                frontier_path = directory / f"{arm}-frontier.json"
                sources.update((path, frontier_path))
                pred, frontier = pd.read_parquet(path), read(frontier_path)
                if not pred.as_of.lt(CUTOFF).all():
                    raise ValueError("Post-May error audit inputs")
                if original_keys is None:
                    original_keys = pred[KEYS]
                else:
                    pd.testing.assert_frame_equal(pred[KEYS], original_keys, check_exact=True)
                analyses[arm] = warning_audit(pred, eps, result["arms"][arm]["scores"], cadence, exposure)
                late.append(
                    {
                        "kind": kind,
                        "fold": fold,
                        "arm": arm,
                        **late_confirmation_audit(pred, eps, cadence, exposure),
                    }
                )
                periods.append({"kind": kind, "fold": fold, "arm": arm, **analyses[arm]["stats"]})
                if len(frontier) != (120 if kind == "flood" else 456):
                    raise ValueError("Incomplete policy-only frontier")
                for donor in PLAN["arms"]:
                    chosen = result["arms"][donor]["policy"]
                    row = policy_lookup(frontier, chosen)
                    if donor == arm and row is not None and any(chosen[k] != v for k, v in row.items()):
                        raise ValueError("Chosen policy differs from own archived frontier")
                    cross.append(
                        {
                            "kind": kind,
                            "fold": fold,
                            "forecast_arm": arm,
                            "policy_from": donor,
                            "preceding_policy_scores": row,
                            "explicit_zero_fallback": row is None,
                            "within_fp_budget": row["false_alerts_per_object_day"] <= 0.25
                            if row is not None
                            else True,
                        }
                    )
            for arm in PLAN["arms"]:
                if arm != "profile":
                    comparisons.append(
                        {
                            "kind": kind,
                            "fold": fold,
                            "reference": arm,
                            **paired(analyses["profile"], analyses[arm]),
                        }
                    )
    totals = pd.DataFrame(periods).groupby(["kind", "arm"], as_index=False)[list(COUNTS)].sum()
    code = {
        Path(__file__),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(evaluator_for.__code__.co_filename),
        Path(verified_evidence.__code__.co_filename),
        Path("tests/test_temporal_count_diagnostics.py"),
    }
    code = {p.resolve().relative_to(Path.cwd()) for p in code}
    result = {
        "scope": "Descriptive audit of ALL completed v42 kinds, folds and four fixed arms. No new fits, policy selection, cohort removal, June evaluation or serving changes. No claim that hindsight error removal is an achievable predictor.",
        "policy_cross_scope": "Look up only archived preceding-policy frontiers at parameters already selected by each arm. These are NOT additional test policies; no test forecasts are changed or evaluated with swapped parameters.",
        "pairing_scope": "All original episode positions are retained, including identical starts. Paired leads condition on the exact events both arms found, while all gained and lost events remain explicit.",
        "error_meaning": "Empty means no eligible future episode in the original24h window. Redundant means future episodes exist but were already matched to earlier warnings. Both remain false alerts in full metrics.",
        "periods": periods,
        "totals": totals.to_dict("records"),
        "paired_leads": comparisons,
        "policy_only_cross_comparisons": cross,
        "late_confirmation_audit": late,
        "late_confirmation_scope": "Both ledgers consume only confirmations already released at the current original forecast tick. The comparison retains expired warnings until late confirmations are safe to discard and exactly matches offline ownership on the revealed prefix. Warning schedules remain fixed: counts of warnings issued while old reservations are missing are associations, not a claimed number of removable false alerts or a new policy's performance.",
        "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
        "code_hashes": {str(p): sha256(p) for p in sorted(code)},
        "selection_changed": False,
        "serving_changed": False,
        "goal_achieved": False,
    }
    write_json(target, result)
    print(totals.to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    audit()
