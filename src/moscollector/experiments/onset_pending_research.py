"""v36: fresh occurrence probabilities with frozen pending-count warning capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.experiments.fresh_counts_research import anchor
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.experiments.onset_binary_research import KINDS
from moscollector.experiments.onset_binary_research import evaluate as evaluate_components
from moscollector.experiments.onset_channel_features import CATS, FOLDER, KEYS
from moscollector.experiments.onset_policy import select_policy
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

PRIOR = Path("artifacts/research-v35")
ARMS = ("pending_candidate", "count_control")
REFERENCES = ("count_control", "direct_control", "reference")
PLAN = {
    "scope": "adaptive_retrospective_onset_probability_pending_policy_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "motivation": "V35 fire passes screening/May but fails stress. December selected a3h direct cooldown. Its row AP rises from.282 policy to.466 test, and353/404 events have some above-threshold prior opportunity, but only86 are matched. Shared opportunities and false/redundant warnings mean this is NOT attainable recall, and not all losses can be attributed to cooldown. Test occurrence probability combined with a recurrence-aware pending-count budget, without changing the weights or labels.",
    "candidate": "Use frozenv35 classifier probabilities on strictly-past fresh channel context, and exact old-weight minute expected_count fromv34/v35pending controls. Do not use v33 fresh count means. Replace the direct long cooldown with the existing causal pending-state mechanism and1min minimum spacing. Model scores and calibration remain frozen; only warning composition/policy changes.",
    "controls": "Count_control has the identical expected_count and pending simulator but its own original count probability; full456 policy reselection must exactly reproducev34/v35pending archives. Direct_control is the samev35 classifier with its frozen direct threshold/cooldown. Historical count anchor is a third reference. This is a composition comparison, not a loss-only or calibration-only claim.",
    "policy": "Same frozen456 floor/capacity/margin options and joint min(P/.9,R/.9,1),thenF1,R,P. Fit on original preceding policy period only, >=10alerts/events support and FP<=.25/original object-day. Outcome availability remains floor(start+70min to hour)+1hour, at most one pending warning cleared per confirmation,24h expiry. Floor0 disables the classifier gate and must be reported. Test outcomes never choose a rule.",
    "data": "Exactlyv35 minute opportunities, original24h labels/event identities,25h purge, original hourly exposure, and v35 independent calibration periods. No new features, resampling or target exclusions. All existing fire additional periods are already known and reused, not independent confirmation.",
    "missing_components": "Screening components must already exist inv35. Only passing kinds may evaluate additional months. Reuse originalv35 results when present. Missing periods are evaluated by byte-frozenv35code under separate parent-linked component/count/pending plans; their classifiers and matched count controls preserve original training. No completed prior study is extended or overwritten.",
    "screen": "Pooled Nov/Feb candidate primary improves>5% against count_control,direct_control,historical reference with noF1 loss. Only passing kinds continue, no variant substitutions.",
    "confirmation": "May primary improves andF1>=95% of every reference. Dec/Mar primary improves>5% with no pooledF1 loss; each month'sF1>=90% of every reference. Research eligibility is not90/90 achievement or automatic activation.",
    "limits": "No June or new labels. Historical adaptive reuse, proxy sensor episodes, static catalog and unknown delivery latency remain limitations. Flood unsupported; full goal includes all types. No automatic activation.",
}


def compose(binary, count):
    if binary.empty or binary.duplicated(KEYS).any() or count.duplicated(KEYS).any():
        raise ValueError("Empty or duplicate composition opportunities")
    pd.testing.assert_frame_equal(binary[KEYS], count[KEYS])
    out = count[[*KEYS, "raw", "expected_count"]].rename(columns={"raw": "raw_count"}).copy()
    out["raw_binary"] = binary.raw.to_numpy()
    out["probability"] = binary.probability.to_numpy()
    if (
        not np.isfinite(out[["raw_count", "raw_binary", "probability", "expected_count"]]).all().all()
        or not out.probability.between(0, 1).all()
        or out.expected_count.lt(0).any()
    ):
        raise ValueError("Invalid occurrence/count composition")
    return out


def metric(row, name):
    return row[name] if name in ("direct_control", "reference") else row["arms"][name]["scores"]


def screening(rows):
    scores = {
        name: pooled([metric(r, name) for r in rows]) for name in (*ARMS, "direct_control", "reference")
    }
    c = scores["pending_candidate"]
    return {
        **scores,
        "passed_screen": all(
            primary_score(c) > 1.05 * primary_score(scores[name]) and c["f1"] >= scores[name]["f1"]
            for name in REFERENCES
        ),
    }


def confirmation(rows):
    may, stress = (
        metric(rows[0], "pending_candidate"),
        pooled([metric(r, "pending_candidate") for r in rows[1:]]),
    )
    passed_may = all(
        primary_score(may) > primary_score(metric(rows[0], name))
        and may["f1"] >= 0.95 * metric(rows[0], name)["f1"]
        for name in REFERENCES
    )
    passed_stress = all(
        primary_score(stress) > 1.05 * primary_score(pooled([metric(r, name) for r in rows[1:]]))
        and stress["f1"] >= pooled([metric(r, name) for r in rows[1:]])["f1"]
        and all(metric(r, "pending_candidate")["f1"] >= 0.9 * metric(r, name)["f1"] for r in rows[1:])
        for name in REFERENCES
    )
    return {
        "passed_may": passed_may,
        "passed_stress": passed_stress,
        "research_eligible": bool(passed_may and passed_stress),
    }


def component_source(root, kind, fold, episodes):
    source = PRIOR / kind / fold
    if (source / "result.json").exists():
        return source
    if fold.startswith("screen_"):
        raise ValueError("Missing declared onset screening component")
    if not read(root / "selection.json")[kind]["passed_screen"]:
        raise ValueError("Additional components require screening success")
    child = root / "component_controls"
    evaluate_components(
        child,
        kind,
        fold,
        pd.read_parquet(PROCESSED / "features-channel-novelty.parquet"),
        pd.read_parquet(PROCESSED / "features-dense-channel-novelty.parquet"),
        pd.read_parquet(FOLDER / "context.parquet", read_dictionary=CATS),
        pd.read_parquet(FOLDER / "triggers.parquet"),
        episodes,
    )
    return child / kind / fold


def evaluate(root, kind, fold, episodes):
    directory = root / kind / fold
    source = component_source(root, kind, fold, episodes)
    component = read(source / "result.json")
    pending = Path(component["pending_control"])
    old = read(pending / "result.json")
    provenance = {
        "plan_sha256": sha256(root / "plan.json"),
        "component": str(source),
        "component_result_sha256": sha256(source / "result.json"),
        "pending": str(pending),
        "pending_result_sha256": sha256(pending / "result.json"),
    }
    target = directory / "result.json"
    if target.exists():
        result = read(target)
        if any(result[k] != v for k, v in provenance.items()):
            raise ValueError("Changed pending composition provenance")
        return result
    if component["exposure"] != old["exposure"]:
        raise ValueError("Component exposures differ")
    tables = {name: {} for name in ARMS}
    for part in ("calibration", "policy", "test"):
        binary = pd.read_parquet(source / f"binary_candidate-{part}.parquet")
        count = pd.read_parquet(pending / f"minute_candidate-{part}.parquet")
        tables["pending_candidate"][part] = compose(binary, count)
        tables["count_control"][part] = count.drop(columns="alert", errors="ignore")
        if cohort(binary, episodes, 1 / 60) != cohort(count, episodes, 1 / 60):
            raise ValueError("Pending composition changes event identities")
    directory.mkdir(parents=True, exist_ok=True)
    arms = {}
    for name, predictions in tables.items():
        print("START onset pending policy", kind, fold, name, flush=True)
        policy, frontier = select_policy(predictions["policy"], episodes, component["exposure"]["policy"])
        for part in ("policy", "test"):
            predictions[part]["alert"] = policy_alerts(predictions[part], episodes, policy)
        scores = evaluator_for(predictions["test"], episodes, 1 / 60, component["exposure"]["test"]).evaluate(
            predictions["test"].alert, 0.5, 1 / 60
        )
        replay = evaluator_for(
            predictions["policy"], episodes, 1 / 60, component["exposure"]["policy"]
        ).evaluate(predictions["policy"].alert, 0.5, 1 / 60)
        if any(policy[k] != v for k, v in replay.items()):
            raise ValueError("Pending policy replay changed")
        if name == "count_control":
            archived = old["arms"]["minute_candidate"]
            if policy != archived["policy"] or scores != archived["scores"]:
                raise ValueError("Matched pending policy did not reproduce archive")
            if frontier != read(pending / "minute_candidate-frontier.json"):
                raise ValueError("Matched pending frontier changed")
            for part, pred in predictions.items():
                pd.testing.assert_frame_equal(
                    pred, pd.read_parquet(pending / f"minute_candidate-{part}.parquet")
                )
        for part, pred in predictions.items():
            pred.to_parquet(directory / f"{name}-{part}.parquet", index=False)
        write_json(directory / f"{name}-frontier.json", frontier)
        arms[name] = {"policy": policy, "scores": scores}
        print("DONE onset pending", kind, fold, name, scores, flush=True)
    historical = anchor(kind, fold)
    direct = component["arms"]["binary_candidate"]["scores"]
    if (
        any(a["scores"]["eligible_episodes"] != historical["eligible_episodes"] for a in arms.values())
        or direct["eligible_episodes"] != historical["eligible_episodes"]
    ):
        raise ValueError("Reference denominator changed")
    result = {
        "kind": kind,
        "fold": fold,
        **provenance,
        "arms": arms,
        "direct_control": direct,
        "reference": historical,
        "exposure": component["exposure"],
        "count_control_exact_parity": True,
        "same_episode_cohort": True,
    }
    write_json(target, result)
    return result


def lock_plan(root):
    prior = read(PRIOR / "plan.json")
    audit_path = root / "motivation-audit.json"
    audit = read(audit_path)
    for manifest in (prior, audit):
        for category in ("source_hashes", "code_hashes"):
            for path, digest in manifest[category].items():
                if sha256(Path(path)) != digest:
                    raise ValueError(f"Changed onset pending input: {path}")
    sources = {p for p in PRIOR.rglob("*") if p.suffix in (".json", ".cbm", ".parquet")}
    sources.add(audit_path)
    code = (Path(__file__), Path("tests/test_onset_pending_research.py"))
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "source_hashes": {**prior["source_hashes"], **{str(p): sha256(p) for p in sorted(sources)}},
                "code_hashes": {
                    **prior["code_hashes"],
                    **audit["code_hashes"],
                    **{str(p): sha256(p) for p in code},
                },
            }
        )
    )
    path = root / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Changed onset pending study; use a new directory")
    if not path.exists():
        write_json(path, plan)
    child = root / "component_controls"
    for destination, parent, old in (
        (child / "plan.json", path, PRIOR / "plan.json"),
        (child / "count_controls/plan.json", child / "plan.json", Path("artifacts/research-v33/plan.json")),
        (child / "pending_controls/plan.json", child / "plan.json", Path("artifacts/research-v34/plan.json")),
    ):
        child_plan = {
            "parent_plan_sha256": sha256(parent),
            "frozen_source_plan_sha256": sha256(old),
            "purpose": "Only missing additional-month matched components for a screening-passing v36 kind; prior studies remain frozen.",
        }
        if destination.exists() and read(destination) != child_plan:
            raise ValueError("Changed onset pending component plan")
        if not destination.exists():
            write_json(destination, child_plan)


def run(root, stage):
    lock_plan(root)
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selected = {}
        for kind in KINDS:
            rows = [
                evaluate(root, kind, fold, episodes.loc[episodes.kind.eq(kind)])
                for fold in ("screen_1", "screen_2")
            ]
            selected[kind] = screening(rows)
            write_json(root / "selection.json", selected)
            print("SELECT onset pending", kind, selected[kind], flush=True)
        return
    selected, report = read(root / "selection.json"), {}
    if set(selected) != set(KINDS):
        raise ValueError("Complete all pending screening first")
    for kind in KINDS:
        if not selected[kind]["passed_screen"]:
            report[kind] = {
                "selection": selected[kind],
                "research_eligible": False,
                "status": "screen_failed",
            }
            continue
        rows = [
            evaluate(root, kind, fold, episodes.loc[episodes.kind.eq(kind)])
            for fold in ("confirmation", *STRESS)
        ]
        gates = confirmation(rows)
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected[kind],
            "periods": rows,
            "five_period_pooled": {
                name: pooled([metric(r, name) for r in rows])
                for name in (*ARMS, "direct_control", "reference")
            },
            **gates,
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT onset pending", kind, gates, report[kind]["five_period_pooled"], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v36"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
