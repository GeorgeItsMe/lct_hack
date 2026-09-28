"""v45: fixed probability lower bound for two frozen count forecasters."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.experiments.count_probability_bound import project
from moscollector.experiments.fine_cadence_research import evaluator_for
from moscollector.experiments.flood_research import validate_hashes
from moscollector.experiments.goal90_research import pooled, primary_score, read
from moscollector.experiments.retained_pending import retained_alerts, select_retained
from moscollector.experiments.retained_pending_research import save_json
from moscollector.experiments.temporal_count_research import CUTOFF, KEYS, KINDS
from moscollector.experiments.tweedie_count_research import PLAN as PARENT_PLAN
from moscollector.experiments.tweedie_count_verification import verified_evidence as verify_parent
from moscollector.prepare import sha256, write_json

ROOT = Path("artifacts/research-v45")
PARENT = Path("artifacts/research-v44")
CANDIDATES = ("projected_poisson", "projected_tweedie")
CONTROLS = ("poisson_control", "tweedie_control", "legacy_control")
PLAN = {
    "scope": "adaptive_retrospective_necessary_probability_count_bound_not_blind_test",
    "goal": "Both event precision and recall>=.90 across all four original incident types.",
    "motivation": "V44 fault emitted no warnings: all preceding-policy expected counts<.091, below smallest decision requirement .25/2=.125. Its separate sigmoid occurrence probability can exceed its calibrated mean count. For nonnegative integerN, indicator(N>0)<=N, hence P(N>0)<=E[N]. V43 preceding-policy audit also found664violations on fault and454380on flood; access/fire had none. Known V44 test extrema helped identify the issue, so this is explicitly adaptive research.",
    "candidates": CANDIDATES,
    "kinds": KINDS,
    "folds": PARENT_PLAN["folds"],
    "projection": "For ALL original rows, replace calibrated expected_count by max(expected_count,probability). Keep probability/raw score/query keys unchanged. This enforces only a necessary inequality, not a full joint distribution or valid joint calibration. It may destroy marginal mean calibration and increase false warnings. No labels enter projection, no per-kind exceptions, free coefficients, power choice, new weights or recalibration. Existing decision multipliers remain heuristics.",
    "sources": "Both frozen forecasters included for every kind/fold: original Poisson from v42/v43 and fixed Tweedie from fully weight-replayed v44. No choice of the more promising source after new outcomes. Unchanged rows and all negative opportunities remain. Calibration projection is diagnostic only, not a new fitting stage.",
    "policy": "Each candidate independently selects original456/120 rules on its preceding policy interval only. Same corrected retained-ledger semantics, strict70min whole-hour confirmation,24h horizon and same event identities/exposure. FP<=.25/original object-day, original support/zero fallback. Full unchanged-source policy grids and flags must reproduce v43/v44.",
    "controls": "Unprojected corrected Poisson, unprojected Tweedie, original legacy Poisson, all historical/stronger v33/v35/v34 references. Flood also retains v42uniform_old9/28/40. Each candidate must beat BOTH unprojected forecasters and every older comparator; competing projected candidates are selected by the same predeclared rule.",
    "selection": "Fixed11period first stage: pooled Nov/Feb for access/fire/fault; allfive flood periods/all40events. Each candidate needs primary min(P/.9,R/.9,1)>1.05*EVERY unprojected comparator and noF1 loss. Among passing candidates choose primary,F1,R,P,projected_poisson on exact tie. Passing only permits a separately frozen confirmation, not release or90/90achievement.",
    "invariants": "No June evaluation, event omission/relabeling, hindsight warning removal, test threshold fitting, serving edits or automatic activation. All parent studies stay unchanged. Sensor proxy episodes are not verified physical incidents; reused historical months are not blind evidence.",
}


def source_path(kind, fold, base, part):
    if base == "tweedie":
        return PARENT / kind / fold / f"tweedie-{part}.parquet"
    if base != "poisson":
        raise ValueError("Unknown fixed count source")
    if part == "calibration":
        return Path("artifacts/research-v42") / kind / fold / "count_control-calibration.parquet"
    return Path("artifacts/research-v43") / kind / fold / f"retained_selected-{part}.parquet"


def prepare(root=ROOT):
    spec = json.loads(json.dumps(PLAN))
    path = root / "plan.json"
    if path.exists():
        plan = read(path)
        if any(plan[k] != v for k, v in spec.items()):
            raise ValueError("Frozen count-bound specification changed")
        validate_hashes(plan)
        return plan
    parent = verify_parent(PARENT)
    sources = {
        **parent["source_hashes"],
        str(PARENT / "weight-replay.json"): sha256(PARENT / "weight-replay.json"),
    }
    code = {
        Path(__file__),
        Path("tests/test_count_probability_bound.py"),
        Path("tests/test_count_bound_research.py"),
    }
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
    write_json(path, plan)
    return read(path)


def selection(rows):
    if not rows:
        raise ValueError("Missing bound comparison periods")
    metrics = []
    for row in rows:
        refs = (
            {"object_weekday", "deadline_uniform_old"} if row["kind"] == "flood" else {"historical", "recent"}
        )
        if set(row["arms"]) != set(CANDIDATES + CONTROLS) or set(row["references"]) != refs:
            raise ValueError("Missing bound arm or reference")
        scores = {**{a: x["scores"] for a, x in row["arms"].items()}, **row["references"]}
        if len({s["eligible_episodes"] for s in scores.values()}) != 1:
            raise ValueError("Changed bound episode cohort")
        metrics.append(scores)
    names = set(metrics[0])
    if any(set(m) != names for m in metrics):
        raise ValueError("Changed bound comparators across periods")
    scores = {name: pooled([m[name] for m in metrics]) for name in sorted(names)}
    passed = {
        name: all(
            primary_score(scores[name]) > 1.05 * primary_score(scores[c])
            and scores[name]["f1"] >= scores[c]["f1"]
            for c in names - set(CANDIDATES)
        )
        for name in CANDIDATES
    }
    eligible = [name for name in CANDIDATES if passed[name]]
    chosen = (
        max(
            eligible,
            key=lambda name: (
                primary_score(scores[name]),
                scores[name]["f1"],
                scores[name]["recall"],
                scores[name]["precision"],
                -CANDIDATES.index(name),
            ),
        )
        if eligible
        else None
    )
    return {"scores": scores, "candidate_passed": passed, "selected": chosen}


def evaluate(kind, fold, episodes, root=ROOT, replay=False):
    parent = read(PARENT / kind / fold / "result.json")
    directory = root / kind / fold
    eps = episodes.loc[episodes.kind.eq(kind)]
    cadence = 1 if kind == "flood" else 0.25
    exposure = {p: parent["signature"]["parts"][p]["rows"] / 24 for p in ("calibration", "policy", "test")}
    arms = {
        alias: {"scores": parent["arms"][old]["scores"], "policy": parent["arms"][old]["policy"]}
        for alias, old in (
            ("poisson_control", "poisson_corrected"),
            ("tweedie_control", "tweedie"),
            ("legacy_control", "poisson_legacy"),
        )
    }
    outputs, original_keys = {}, {}
    for name in CANDIDATES:
        base = name.removeprefix("projected_")
        tables, diagnostics = {}, {}
        for part in ("calibration", "policy", "test"):
            original = pd.read_parquet(source_path(kind, fold, base, part))
            if not original.as_of.lt(CUTOFF).all():
                raise ValueError("Post-May count-bound inputs")
            if part in original_keys:
                pd.testing.assert_frame_equal(original[KEYS], original_keys[part], check_exact=True)
            else:
                original_keys[part] = original[KEYS]
            tables[part], diagnostics[part] = project(original)
            if part in ("policy", "test"):
                expected_policy = arms[f"{base}_control"]["policy"]
                np.testing.assert_array_equal(retained_alerts(original, eps, expected_policy), original.alert)
                if part == "policy":
                    old_policy, old_frontier = select_retained(original, eps, cadence, exposure[part])
                    frontier_path = (
                        PARENT / kind / fold / "tweedie-frontier.json"
                        if base == "tweedie"
                        else Path("artifacts/research-v43") / kind / fold / "retained-frontier.json"
                    )
                    if old_policy != expected_policy or old_frontier != read(frontier_path):
                        raise ValueError("Changed unprojected control grid/policy")
                else:
                    if (
                        evaluator_for(original, eps, cadence, exposure[part]).evaluate(
                            original.alert, 0.5, cadence
                        )
                        != arms[f"{base}_control"]["scores"]
                    ):
                        raise ValueError("Changed original source event scores")
        policy, frontier = select_retained(tables["policy"], eps, cadence, exposure["policy"])
        scores = {}
        for part in ("policy", "test"):
            tables[part]["alert"] = retained_alerts(tables[part], eps, policy)
            scores[part] = evaluator_for(tables[part], eps, cadence, exposure[part]).evaluate(
                tables[part].alert, 0.5, cadence
            )
        if any(policy[k] != v for k, v in scores["policy"].items()):
            raise ValueError("Chosen projected policy differs from frontier")
        for part, table in tables.items():
            target = directory / f"{name}-{part}.parquet"
            if target.exists():
                pd.testing.assert_frame_equal(table, pd.read_parquet(target), check_exact=True)
            elif replay:
                raise ValueError("Missing count-bound table")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                table.to_parquet(target, index=False)
            outputs[str(target)] = sha256(target)
        target = directory / f"{name}-frontier.json"
        save_json(target, frontier, replay)
        outputs[str(target)] = sha256(target)
        arms[name] = {
            "scores": scores["test"],
            "policy_scores": scores["policy"],
            "policy": policy,
            "projection": diagnostics,
        }
    result = {
        "kind": kind,
        "fold": fold,
        "plan_sha256": sha256(root / "plan.json"),
        "parent_sha256": sha256(PARENT / kind / fold / "result.json"),
        "arms": arms,
        "references": parent["references"],
        "outputs": outputs,
    }
    selection([result])
    save_json(directory / "result.json", result, replay)
    print(
        "VERIFIED" if replay else "BOUND",
        kind,
        fold,
        {
            name: [arms[name]["scores"][k] for k in ("true_alerts", "alerts", "eligible_episodes")]
            for name in CANDIDATES
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
        chosen = selection(rows)
        if kind == "flood" and chosen["scores"][CANDIDATES[0]]["eligible_episodes"] != 40:
            raise ValueError("Changed bound flood40 cohort")
        report[kind] = {
            "periods": rows,
            "selection": chosen,
            "status": "requires_separate_confirmation" if chosen["selected"] else "screen_rejected",
            "serving_changed": False,
            "goal_achieved": False,
        }
        if not replay:
            write_json(root / "report.json", report)
        print("BOUND RESULT", kind, chosen, flush=True)
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
