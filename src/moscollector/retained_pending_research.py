"""v43: isolate delayed-confirmation ownership on frozen v42 count forecasts."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from moscollector.fine_cadence_research import evaluator_for, policy_alerts
from moscollector.flood_research import FIVE, validate_hashes
from moscollector.goal90_research import pooled, primary_score, read
from moscollector.prepare import sha256, write_json
from moscollector.retained_pending import retained_alerts, select_retained
from moscollector.temporal_count_research import CUTOFF, KINDS
from moscollector.temporal_count_verification import verified_evidence as verify_parent

ROOT = Path("artifacts/research-v43")
PARENT = Path("artifacts/research-v42")
ARMS = ("retained_selected", "retained_fixed_control", "legacy_control")
PLAN = {
    "scope": "adaptive_retrospective_late_confirmation_ownership_study_not_blind_test",
    "motivation": "Fixed-warning v42 audit proves expired owners are deleted before delayed confirmations arrive, causing lost/wrong ownership. Correctness does not imply better event precision/recall: many warnings associated with under-reservation were true. Test actual causal rollout without hindsight removal.",
    "goal": "Both event precision and recall>=.90 on all four types; no single-head or row-metric substitute.",
    "kinds": KINDS,
    "folds": {k: list(FIVE) if k == "flood" else ["screen_1", "screen_2"] for k in KINDS},
    "forecasts": "Exactly the11 completed v42 count_control policy/test tables, frozen calibrated mean/probability, no refit or recalibration. All query keys, negative opportunities, episode identities,24h horizons, exposures and quarter cadence (hourly flood) remain identical.",
    "algorithm": "Keep unmatched warnings for24h+70min+1h, but reserve capacity only while age<24h. Process ALL newly released confirmations before pruning, even after gaps. Original strict whole-hour release after70min and FIFO event ownership are unchanged. Only possibly emitting ticks may be visited; independent unbounded-history/full-tick reference must agree.",
    "arms": ARMS,
    "selection": "Only retained_selected is a candidate: retune original456/120 grid on preceding policy period, same FP<=.25/original-object-day and supported/fallback rules. retained_fixed_control uses original chosen parameters with corrected ledger, diagnostic only; never substituted as candidate. legacy_control exactly replays original flags and event scores from the already weight-verified parent.",
    "references": "All original historical and stronger recent v33 access/v35 fire/v34 fault references from v42. Flood additionally compares v42 uniform_old9/28/40, whose prior result was seen; it is a comparator, not a new candidate. No choice of reference after seeing v43.",
    "gates": "Fixed eleven-period first-stage experiment: access/fire/fault pooled Nov/Feb, flood ALL five folds and40events. Candidate primary min(P/.9,R/.9,1)>1.05*EVERY other arm/reference and noF1 loss. Passing is only eligibility for a separately frozen confirmation protocol, not release or90/90 achievement. No additional months chosen during this study.",
    "invariants": "No test threshold selection, episode removal, physical-incident relabeling, June evaluation, serving edits or automatic activation. V42's failed profile is not changed. All existing artifacts/code stay frozen.",
}


def prepare(root=ROOT):
    spec = json.loads(json.dumps(PLAN))
    target = root / "plan.json"
    if target.exists():
        result = read(target)
        if any(result[k] != v for k, v in spec.items()):
            raise ValueError("Frozen retained-warning specification changed")
        validate_hashes(result)
        return result
    parent = verify_parent(PARENT)
    audit_path = Path("artifacts/temporal_count_error_audit.json")
    audit = read(audit_path)
    validate_hashes(audit)
    sources = {**parent["source_hashes"], **audit["source_hashes"], str(audit_path): sha256(audit_path)}
    sources[str(PARENT / "weight-replay.json")] = sha256(PARENT / "weight-replay.json")
    code = {
        Path(__file__),
        Path("tests/test_retained_pending.py"),
        Path("tests/test_retained_pending_research.py"),
    }
    code.update(
        Path(m.__file__)
        for n, m in sys.modules.items()
        if n.startswith("moscollector.") and getattr(m, "__file__", "").endswith(".py")
    )
    code = {p.resolve().relative_to(Path.cwd()) for p in code}
    result = {
        **spec,
        "source_hashes": sources,
        "code_hashes": {
            **parent["code_hashes"],
            **audit["code_hashes"],
            **{str(p): sha256(p) for p in sorted(code)},
        },
    }
    validate_hashes(result)
    write_json(target, result)
    return read(target)


def compare(rows):
    if not rows or any(set(r["arms"]) != set(ARMS) for r in rows):
        raise ValueError("Missing retained comparison arms")
    metrics = [{**{a: v["scores"] for a, v in r["arms"].items()}, **r["references"]} for r in rows]
    names = set(metrics[0])
    if any(set(m) != names for m in metrics):
        raise ValueError("Missing retained reference")
    if any(len({v["eligible_episodes"] for v in m.values()}) != 1 for m in metrics):
        raise ValueError("Changed retained episode cohort")
    scores = {name: pooled([m[name] for m in metrics]) for name in sorted(names)}
    candidate = scores["retained_selected"]
    passed = all(
        primary_score(candidate) > 1.05 * primary_score(s) and candidate["f1"] >= s["f1"]
        for name, s in scores.items()
        if name != "retained_selected"
    )
    return {"scores": scores, "passed_screen": passed}


def save_json(target, value, replay):
    if target.exists():
        if read(target) != value:
            raise ValueError(f"Changed retained result: {target}")
    elif replay:
        raise ValueError(f"Missing retained result: {target}")
    else:
        write_json(target, value)


def evaluate(kind, fold, episodes, root=ROOT, replay=False):
    parent = read(PARENT / kind / fold / "result.json")
    directory = root / kind / fold
    eps = episodes.loc[episodes.kind.eq(kind)]
    cadence = 1 if kind == "flood" else 0.25
    exposure = {p: parent["signature"]["parts"][p]["rows"] / 24 for p in ("policy", "test")}
    tables = {
        p: pd.read_parquet(PARENT / kind / fold / f"count_control-{p}.parquet") for p in ("policy", "test")
    }
    if any(not t.as_of.lt(CUTOFF).all() for t in tables.values()):
        raise ValueError("Post-May retained-warning inputs")
    old = parent["arms"]["count_control"]
    # Parent weight replay already proved every legacy grid row and selection.
    for table in tables.values():
        flags = policy_alerts(table, eps, old["policy"])
        if not (flags == table.alert.to_numpy()).all():
            raise ValueError("Changed legacy warning schedule")
    selected, frontier = select_retained(tables["policy"], eps, cadence, exposure["policy"])
    arms, outputs = {}, {}
    for arm in ARMS:
        policy = selected if arm == "retained_selected" else old["policy"]
        part_scores = {}
        for part, original in tables.items():
            table = original.copy()
            if arm != "legacy_control":
                table["alert"] = retained_alerts(table, eps, policy)
            pd.testing.assert_frame_equal(
                table.drop(columns="alert"), original.drop(columns="alert"), check_exact=True
            )
            evaluator = evaluator_for(table, eps, cadence, exposure[part])
            part_scores[part] = evaluator.evaluate(table.alert, 0.5, cadence)
            if arm == "legacy_control" and part == "test" and part_scores[part] != old["scores"]:
                raise ValueError("Changed legacy event scores")
            target = directory / f"{arm}-{part}.parquet"
            if target.exists():
                pd.testing.assert_frame_equal(table, pd.read_parquet(target), check_exact=True)
            elif replay:
                raise ValueError("Missing retained forecast table")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                table.to_parquet(target, index=False)
            outputs[str(target)] = sha256(target)
        arms[arm] = {"policy": policy, "scores": part_scores["test"], "policy_scores": part_scores["policy"]}
    if any(selected[k] != v for k, v in arms["retained_selected"]["policy_scores"].items()):
        raise ValueError("Selected corrected policy does not reproduce its frontier row")
    target = directory / "retained-frontier.json"
    save_json(target, frontier, replay)
    outputs[str(target)] = sha256(target)
    refs = dict(parent["references"])
    if kind == "flood":
        refs["deadline_uniform_old"] = parent["arms"]["uniform_old"]["scores"]
    result = {
        "kind": kind,
        "fold": fold,
        "plan_sha256": sha256(root / "plan.json"),
        "parent_sha256": sha256(PARENT / kind / fold / "result.json"),
        "exposure": exposure,
        "arms": arms,
        "references": refs,
        "outputs": outputs,
    }
    compare([result])  # Check complete same-cohort comparators in each month.
    save_json(directory / "result.json", result, replay)
    print(
        "VERIFIED" if replay else "RETAINED",
        kind,
        fold,
        {
            a: [s["scores"][k] for k in ("true_alerts", "alerts", "eligible_episodes")]
            for a, s in arms.items()
        },
        flush=True,
    )
    return result


def run(root=ROOT, replay=False):
    plan = prepare(root)
    episodes = pd.read_parquet("data/processed/episodes.parquet", filters=[("start_ts", "<", CUTOFF)])
    report = {}
    for kind in KINDS:
        rows = [evaluate(kind, fold, episodes, root, replay) for fold in PLAN["folds"][kind]]
        selection = compare(rows)
        if kind == "flood" and selection["scores"]["retained_selected"]["eligible_episodes"] != 40:
            raise ValueError("Changed flood40 cohort")
        report[kind] = {
            "periods": rows,
            "selection": selection,
            "status": "requires_separate_confirmation" if selection["passed_screen"] else "screen_rejected",
            "serving_changed": False,
            "goal_achieved": False,
        }
        print("RETAINED RESULT", kind, selection, flush=True)
    save_json(root / "report.json", report, replay)
    if replay:
        sources = {
            **plan["source_hashes"],
            **{str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "policy-replay.json"},
        }
        proof = {
            "status": "exact_policy_replay_passed",
            "source_hashes": sources,
            "code_hashes": plan["code_hashes"],
            "report": report,
            "new_model_fits": 0,
        }
        validate_hashes(proof)
        write_json(root / "policy-replay.json", proof)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    prepare(args.output) if args.prepare_only else run(args.output, args.replay)
