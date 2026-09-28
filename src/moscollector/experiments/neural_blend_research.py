"""v29: fixed convex pools of frozen CatBoost, MLP and GRU count forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.experiments.binary_gate_research import KINDS, select_gate_policy
from moscollector.experiments.fine_cadence_research import cohort, evaluator_for, policy_alerts
from moscollector.experiments.goal90_research import STRESS, pooled, primary_score, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

COMPONENTS = ("global_control", "mlp", "gru")
BLENDS = {
    "tree_mlp": {"global_control": 0.5, "mlp": 0.5},
    "tree_gru": {"global_control": 0.5, "gru": 0.5},
    "three_way": {name: 1 / 3 for name in COMPONENTS},
}
PLAN = {
    "scope": "adaptive_retrospective_fixed_neural_pool_study_not_blind_validation",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "folds": FOLDS,
    "stress_folds": STRESS,
    "blends": BLENDS,
    "hypothesis": "V28 architectures miss different events. Access GRU finds237 screen episodes missed by CatBoost and misses128 found by CatBoost. Test complementary forecast information through fixed convex pools, not future-informed union/removal of warnings.",
    "inputs": "Exact frozen v28 calibrated probabilities and mean counts, same whole-hour snapshots held on15min grid. Pool probability and count separately with identical fixed weights. No learned mixing weights, new calibration, warning union, changed episode definitions or changed warning horizon.",
    "control": "Replay all three component policy/test metrics and alerts exactly. Each blend must beat CatBoost and historical anchor and every participating neural component. Same event identities and original hourly exposure; independent456-policy choice on preceding policy period for every blend.",
    "screen": "Nov/Feb pooled primary min(P/.9,R/.9,1)>1.05*every relevant reference, with noF1 loss. Select highest primary,F1,R,P among passing fixed blends only, declared order breaks ties. No per-kind weights selected outside these three fixed variants.",
    "confirmation": "Only passing kinds: selected blend stays fixed. May improves primary withF1>=95%every relevant reference. Pooled Dec/Mar primary>1.05*every relevant reference,noF1 loss; each monthF1>=90%every relevant reference. Additional MLP/GRU models use exactly frozen v28 training/codec/earlystop/device procedures in a separate v29 controls directory. Both networks fitted for complete component comparisons; no v28 outcome is rewritten.",
    "limits": "Fixed mixtures can change warning timing, so descriptive union overlap is not an upper bound. Earlier months are adaptively reused, not independent future evidence. Flood remains unsupported and full-scope90/90 unmet unless actually proved. No automatic activation. No June labels or source mutation.",
}


def blend_predictions(tables, weights):
    """Pool forecasts only after strict identity/order and finite-domain checks."""
    values = np.asarray(list(weights.values()), dtype=float)
    if (
        not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1)
    ):
        raise ValueError("Blend weights must be a convex combination")
    first = tables[next(iter(weights))]
    keys = first[["object_id", "as_of"]].reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError("Duplicate blend opportunities")
    output = keys.copy()
    output["probability"], output["expected_count"] = 0.0, 0.0
    for name, weight in weights.items():
        frame = tables[name].reset_index(drop=True)
        if not frame[["object_id", "as_of"]].equals(keys):
            raise ValueError("Blend opportunities differ or are reordered")
        p, count = frame.probability.to_numpy(), frame.expected_count.to_numpy()
        if (
            not np.isfinite(p).all()
            or not np.isfinite(count).all()
            or np.any((p < 0) | (p > 1))
            or np.any(count < 0)
        ):
            raise ValueError("Invalid component forecasts")
        output["probability"] += weight * p
        output["expected_count"] += weight * count
    return output


def references(variant):
    return ("global_control", "reference", *(name for name in BLENDS[variant] if name != "global_control"))


def control_directory(root, kind, fold, device):
    original = Path("artifacts/research-v28") / kind / fold
    if fold in ("screen_1", "screen_2"):
        return original
    controls = root / "controls"
    if not (controls / kind / fold / "result.json").exists():
        import torch

        from moscollector.experiments.neural_count_research import TRAINING, evaluate

        if str(torch.__version__) != read(root / "plan.json")["torch_version"]:
            raise ValueError("Frozen PyTorch version changed")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Frozen MPS device unavailable; no silent fallback")
        torch.set_num_threads(TRAINING["cpu_threads"])
        frame = pd.read_parquet(PROCESSED / "features-channel-novelty.parquet")
        dense = pd.read_parquet(PROCESSED / "features-dense-channel-novelty.parquet")
        if not all(f.as_of.lt(pd.Timestamp("2026-06-01")).all() for f in (frame, dense)):
            raise ValueError("Post-May neural input")
        episodes = pd.read_parquet(
            PROCESSED / "episodes.parquet",
            filters=[("kind", "==", kind), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
        )
        evaluate(controls, kind, fold, frame, dense, episodes, device)
    return controls / kind / fold


def evaluate(root, kind, fold, episodes, device):
    directory = root / kind / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    source = control_directory(root, kind, fold, device)
    old = read(source / "result.json")
    exposure = old["exposure_days"]
    tables = {
        period: {name: pd.read_parquet(source / f"{name}-{period}.parquet") for name in COMPONENTS}
        for period in ("policy", "test")
    }
    reference_policy = tables["policy"]["global_control"]
    original_hours = int(reference_policy.as_of.eq(reference_policy.as_of.dt.floor("h")).sum())
    reference_test = tables["test"]["global_control"]
    assert int(reference_test.as_of.eq(reference_test.as_of.dt.floor("h")).sum()) / 24 == exposure
    for name in COMPONENTS:
        policy = old["arms"][name]["policy"]
        policy_rows = tables["policy"][name]
        policy_scores = evaluator_for(policy_rows, episodes, 0.25, original_hours / 24).evaluate(
            policy_alerts(policy_rows, episodes, policy), 0.5, 0.25
        )
        assert all(policy[key] == value for key, value in policy_scores.items())
        before = tables["test"][name]
        alerts = policy_alerts(before, episodes, policy)
        np.testing.assert_array_equal(alerts, before.alert)
        scores = evaluator_for(before, episodes, 0.25, exposure).evaluate(alerts, 0.5, 0.25)
        assert scores == old["arms"][name]["scores"]
    directory.mkdir(parents=True, exist_ok=True)
    arms = {}
    for variant, weights in BLENDS.items():
        predictions = {period: blend_predictions(table, weights) for period, table in tables.items()}
        for period, pred in predictions.items():
            assert cohort(pred, episodes, 0.25) == cohort(tables[period]["global_control"], episodes, 0.25)
        # Original hours are explicitly present in the held quarter grid. The
        # last hour of each valid segment is retained without adding quarters.
        print("START blend policy", kind, fold, variant, flush=True)
        policy, frontier = select_gate_policy(predictions["policy"], episodes, original_hours / 24)
        test = predictions["test"]
        test["alert"] = policy_alerts(test, episodes, policy)
        scores = evaluator_for(test, episodes, 0.25, exposure).evaluate(test.alert, 0.5, 0.25)
        assert scores["eligible_episodes"] == old["reference"]["eligible_episodes"]
        for period, pred in predictions.items():
            pred.to_parquet(directory / f"{variant}-{period}.parquet", index=False)
        write_json(directory / f"{variant}-frontier.json", frontier)
        arms[variant] = {"scores": scores, "policy": policy, "weights": weights}
        print("DONE blend policy", kind, fold, variant, scores, flush=True)
    result = {
        "kind": kind,
        "fold": fold,
        "arms": arms,
        "controls": {name: old["arms"][name]["scores"] for name in COMPONENTS},
        "reference": old["reference"],
        "same_episode_cohort": True,
        "component_alert_and_score_parity": True,
        "source_directory": str(source),
        "exposure_days": exposure,
    }
    write_json(target, result)
    return result


def lock_plan(root):
    previous = Path("artifacts/research-v28/plan.json")
    old_plan = read(previous)
    sources = {Path(p) for p in old_plan["source_hashes"]}
    sources.update(
        (
            previous,
            Path("artifacts/research-v28/report.json"),
            Path("artifacts/neural_complementarity_audit.json"),
        )
    )
    for category in ("source_hashes", "code_hashes"):
        for source, digest in old_plan[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed v28 prerequisite: {source}")
    for kind in KINDS:
        for fold in ("screen_1", "screen_2"):
            directory = Path("artifacts/research-v28") / kind / fold
            sources.update((directory / "result.json", directory / "codec.json"))
            for variant in COMPONENTS:
                sources.update(directory / f"{variant}-{p}.parquet" for p in ("policy", "test"))
            for variant in ("mlp", "gru"):
                sources.update(directory / "models" / variant / p for p in ("model.pt", "fit.json"))
    code = {Path(p) for p in old_plan["code_hashes"]}
    code.update((Path(__file__), Path("scripts/research/audit_neural_complementarity.py")))
    plan = json.loads(
        json.dumps(
            {
                **PLAN,
                "device": old_plan["device"],
                "torch_version": old_plan["torch_version"],
                "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
                "code_hashes": {str(p): sha256(p) for p in sorted(code)},
            }
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / "plan.json"
    if target.exists() and read(target) != plan:
        raise ValueError("Blend study inputs/code changed; use a new output directory")
    if not target.exists():
        write_json(target, plan)
    control_plan = {
        "scope": "Additional matched v28 neural training needed only for a v29 screen-selected blend; does not change the v28 study.",
        "parent_plan_sha256": sha256(target),
        "frozen_training_source_plan_sha256": sha256(previous),
    }
    control_path = root / "controls/plan.json"
    if control_path.exists() and read(control_path) != control_plan:
        raise ValueError("Neural control plan changed")
    write_json(control_path, control_plan)
    return plan


def scores_for(row, key):
    if key == "reference":
        return row["reference"]
    return row["controls"][key] if key in COMPONENTS else row["arms"][key]["scores"]


def run(root, stage):
    plan = lock_plan(root)
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            rows = [
                evaluate(root, kind, fold, episodes.loc[episodes.kind.eq(kind)], plan["device"])
                for fold in ("screen_1", "screen_2")
            ]
            controls = {
                name: pooled([scores_for(row, name) for row in rows]) for name in (*COMPONENTS, "reference")
            }
            variants = {}
            for variant in BLENDS:
                scores = pooled([scores_for(row, variant) for row in rows])
                variants[variant] = {
                    "scores": scores,
                    "passed_screen": all(
                        primary_score(scores) > 1.05 * primary_score(controls[k])
                        and scores["f1"] >= controls[k]["f1"]
                        for k in references(variant)
                    ),
                }
            eligible = [name for name in BLENDS if variants[name]["passed_screen"]]
            winner = (
                max(
                    eligible,
                    key=lambda name: tuple(
                        variants[name]["scores"][k] for k in ("primary_score", "f1", "recall", "precision")
                    ),
                )
                if eligible
                else None
            )
            selection[kind] = {
                "variants": variants,
                "controls": controls,
                "selected_variant": winner,
                "passed_screen": winner is not None,
            }
            write_json(root / "selection.json", selection)
            print("SELECT blend", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Complete all blend screening before confirmation")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        variant = selected["selected_variant"]
        rows = [
            evaluate(root, kind, fold, episodes.loc[episodes.kind.eq(kind)], plan["device"])
            for fold in ("confirmation", *STRESS)
        ]
        may, stress = scores_for(rows[0], variant), pooled([scores_for(row, variant) for row in rows[1:]])
        passed_may = all(
            primary_score(may) > primary_score(scores_for(rows[0], k))
            and may["f1"] >= 0.95 * scores_for(rows[0], k)["f1"]
            for k in references(variant)
        )
        passed_stress = all(
            primary_score(stress) > 1.05 * primary_score(pooled([scores_for(row, k) for row in rows[1:]]))
            and stress["f1"] >= pooled([scores_for(row, k) for row in rows[1:]])["f1"]
            and all(scores_for(row, variant)["f1"] >= 0.9 * scores_for(row, k)["f1"] for row in rows[1:])
            for k in references(variant)
        )
        rows = [read(root / kind / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": {
                key: pooled([scores_for(row, key) for row in rows])
                for key in (*BLENDS, *COMPONENTS, "reference")
            },
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT blend", kind, variant, report[kind]["five_period_pooled"][variant], flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v29"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
