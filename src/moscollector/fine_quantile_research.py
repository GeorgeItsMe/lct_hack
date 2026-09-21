"""v22: frozen count quantiles on v20's causal quarter-hour warning grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.cadence_research import align_opportunities
from moscollector.count_research import episode_counts
from moscollector.fine_cadence_research import FinePendingSimulator, cohort, evaluator_for
from moscollector.goal90_research import ALPHAS, STRESS, noncrossing, pooled, primary_score, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.quarter_count_features import carry_features
from moscollector.research import FOLDS, mask
from moscollector.train import model_input

PLAN = {
    "scope": "adaptive_retrospective_quantile_cadence_followup_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kind": "access",
    "folds": FOLDS,
    "stress_folds": STRESS,
    "reason": "V20 improved access with15min warnings; v21 raw-count freshness gains did not pass screening. Independently test the previously trained v11 quantile model on the same v20 grid, without adopting rejected v21 fresh inputs.",
    "weights": "Frozen v11 MultiQuantile weights alpha.1,.25,.5,.75,.9; same original66features held at whole-hour values. No training, new inputs, loss search or changed labels.",
    "calibration": "Carry last whole-hour raw quantile outputs forward at most45min as lagged inputs. Refit each scalar residual quantile correction on original calibration dates with25h purge against episode counts in[NEW_time,NEW_time+24h). Nonnegative clipping and per-row sorting enforce noncrossing outputs. Calibration-period marginal coverage is diagnostic, not guaranteed conditional coverage.",
    "probability_gate": "Exact frozen v20 quarter-candidate probabilities/calibration, common to both mean and quantile policies. No new probability model.",
    "cadence": "Exact v20 quarter opportunities and event identities. Same whole-hour confirmation availability,24h pending expiry,one warning per15min,one-to-one matching,identical original hourly exposure for FP budget. Offline eligibility is not a runtime signal.",
    "policy": "Original quantile grid: one alpha,.25/.5/1/2 margin, floor0/.5/.75/.9/.95. Policy period only; >=10alerts/episodes support, false alerts<=.25/object-day. Max primary min(P/.9,R/.9,1), thenF1,R,P. Explicit empty fallback.",
    "reference": "Completed v20 quarter mean-count policy; original hourly v11 quantile is descriptive secondary comparison. Tree-model and warning-capacity changes are jointly tested against the stronger v20 family, not attributed solely to cadence.",
    "screen": "Nov/Feb pooled primary must improve>5% against v20 with noF1 loss; only passing candidate continues.",
    "confirmation": "May improves primary withF1>=95%reference; pooled Dec/Mar primary improves>5%, no pooledF1 loss, each month'sF1>=90%reference. Research gates do not imply90/90 or automatic activation.",
    "june": "No June evaluation, selection or labels. Adaptively reused historical months do not provide new independent validation. Access success would not establish success for all incident types.",
}


def alerts(pred, episodes, policy):
    if policy["quantile"] is None:
        return np.zeros(len(pred))
    return FinePendingSimulator(pred, episodes).alerts(
        pred[f"q_{policy['quantile']}"], pred.probability, policy["margin"], policy["floor"]
    )


def select(pred, episodes, exposure):
    simulator = FinePendingSimulator(pred, episodes)
    evaluator = evaluator_for(pred, episodes, 0.25, exposure)
    options = []
    for q in ALPHAS:
        for margin in (0.25, 0.5, 1, 2):
            for floor in (0, 0.5, 0.75, 0.9, 0.95):
                warning = simulator.alerts(pred[f"q_{q}"], pred.probability, margin, floor)
                options.append(
                    {
                        "quantile": q,
                        "margin": margin,
                        "floor": floor,
                        **evaluator.evaluate(warning, 0.5, 0.25),
                    }
                )
    budget = [r for r in options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    fallback = {
        "quantile": None,
        "margin": 1,
        "floor": 0,
        **evaluator.evaluate(np.zeros(len(pred)), 0.5, 0.25),
    }
    chosen = dict(
        max(
            supported or budget or [fallback],
            key=lambda r: (primary_score(r), r["f1"], r["recall"], r["precision"]),
        )
    )
    chosen["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    return chosen, options


def evaluate(root, fold, frame, dense, episodes):
    directory = root / fold
    target = directory / "result.json"
    if target.exists():
        return read(target)
    base = Path("artifacts/research-v11") / fold / "quantile"
    mean_root = Path("artifacts/research-v20/access") / fold
    meta, reference = read(base / "fit.json"), read(mean_root / "result.json")
    assert meta["periods"] == reference["periods"]
    if sha256(base / "access.cbm") != meta["model_sha256"]:
        raise ValueError("Frozen quantile weights changed")
    model = CatBoostRegressor()
    model.load_model(str(base / "access.cbm"))
    raw_names = [f"raw_{q}" for q in ALPHAS]
    tables, exposure = {}, {}
    directory.mkdir(parents=True, exist_ok=True)
    for period in ("calibration", "policy", "test"):
        dates = tuple(map(pd.Timestamp, meta["periods"][period]))
        old = frame.loc[mask(frame, *dates), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *dates)], old)
        hourly = rows[["object_id", "as_of"]].copy()
        raw = model.predict(model_input(rows, meta["features"]), thread_count=2)
        if raw.shape != (len(hourly), len(ALPHAS)) or not np.isfinite(raw).all():
            raise ValueError("Invalid raw quantile outputs")
        hourly[raw_names] = raw
        if period != "calibration":
            saved = pd.read_parquet(base / f"access-{period}.parquet")
            shared = hourly.merge(saved, on=["object_id", "as_of"], validate="one_to_one")
            assert len(shared) == len(saved) == len(hourly)
            np.testing.assert_allclose(
                noncrossing(shared[raw_names].to_numpy(), meta["corrections"]),
                shared[[f"q_{q}" for q in ALPHAS]].to_numpy(),
                atol=1e-10,
                rtol=1e-10,
            )
        pred = carry_features(hourly, subdivide_evaluation_slots(hourly), raw_names)
        assert cohort(pred, episodes, 0.25) == cohort(hourly, episodes, 1)
        if period != "calibration":
            mean = pd.read_parquet(mean_root / f"quarter_candidate-{period}.parquet")
            pred = pred.merge(
                mean[["object_id", "as_of", "probability"]], on=["object_id", "as_of"], validate="one_to_one"
            )
            assert len(pred) == len(mean)
        exposure[period] = len(hourly) / 24
        tables[period] = pred
    cal = tables["calibration"]
    counts = episode_counts(cal, episodes)
    raw = cal[raw_names].to_numpy()
    correction = np.array([np.quantile(counts - raw[:, i], q) for i, q in enumerate(ALPHAS)])
    for pred in tables.values():
        pred[[f"q_{q}" for q in ALPHAS]] = noncrossing(pred[raw_names].to_numpy(), correction)
    coverage = {str(q): float(np.mean(counts <= cal[f"q_{q}"])) for q in ALPHAS}
    print("START quarter quantile", fold, "calibration_rows", len(cal), flush=True)
    policy, frontier = select(tables["policy"], episodes, exposure["policy"])
    write_json(directory / "policy.json", {"selected": policy, "options": frontier})
    test = tables["test"]
    test["alert"] = alerts(test, episodes, policy)
    scores = evaluator_for(test, episodes, 0.25, exposure["test"]).evaluate(test.alert, 0.5, 0.25)
    assert scores["eligible_episodes"] == reference["scores"]["eligible_episodes"]
    for period in ("policy", "test"):
        tables[period].to_parquet(directory / f"{period}.parquet", index=False)
    result = {
        "kind": "access",
        "fold": fold,
        "scores": scores,
        "reference": reference["scores"],
        "old_hourly_quantile": read(base / "result.json")["scores"],
        "policy": policy,
        "corrections": correction.tolist(),
        "calibration_rows": len(cal),
        "calibration_coverage_not_test": coverage,
        "identical_episode_cohort": True,
        "hourly_raw_quantile_parity": True,
        "shared_probability_is_frozen_v20": True,
        "source_model_sha256": sha256(base / "access.cbm"),
        "periods": meta["periods"],
    }
    write_json(target, result)
    print("DONE quarter quantile", fold, scores, flush=True)
    return result


def lock_plan(root):
    sources = [
        PROCESSED / "features.parquet",
        PROCESSED / "features-dense-round4.parquet",
        PROCESSED / "episodes.parquet",
        Path("artifacts/research-v20/report.json"),
        Path("artifacts/research-v21/report.json"),
    ]
    for fold in (*FOLDS, *STRESS):
        base = Path("artifacts/research-v11") / fold / "quantile"
        sources.extend(
            base / name
            for name in (
                "access.cbm",
                "fit.json",
                "result.json",
                "access-policy.parquet",
                "access-test.parquet",
            )
        )
        mean = Path("artifacts/research-v20/access") / fold
        sources.extend(
            mean / name
            for name in ("result.json", "quarter_candidate-policy.parquet", "quarter_candidate-test.parquet")
        )
    code = {
        Path(__file__),
        *(
            Path(f.__code__.co_filename)
            for f in (
                noncrossing,
                episode_counts,
                carry_features,
                evaluator_for,
                FinePendingSimulator.__init__,
                cohort,
                model_input,
                subdivide_evaluation_slots,
                align_opportunities,
                mask,
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
    dense = pd.read_parquet(PROCESSED / "features-dense-round4.parquet")
    assert dense.as_of.lt(pd.Timestamp("2026-06-01")).all()
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet",
        filters=[("kind", "==", "access"), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
    )
    if stage == "screen":
        rows = [evaluate(root, fold, frame, dense, episodes) for fold in ("screen_1", "screen_2")]
        c, r = (pooled([item[key] for item in rows]) for key in ("scores", "reference"))
        selection = {
            "candidate": c,
            "reference": r,
            "passed_screen": primary_score(c) > 1.05 * primary_score(r) and c["f1"] >= r["f1"],
        }
        write_json(root / "selection.json", selection)
        print("SELECT quarter quantile", selection, flush=True)
        return
    selection = read(root / "selection.json")
    if not selection["passed_screen"]:
        write_json(
            root / "report.json",
            {"selection": selection, "research_eligible": False, "status": "screen_failed"},
        )
        return
    rows = [evaluate(root, fold, frame, dense, episodes) for fold in ("confirmation", *STRESS)]
    may = rows[0]
    c, r = (pooled([item[key] for item in rows[1:]]) for key in ("scores", "reference"))
    passed_may = (
        primary_score(may["scores"]) > primary_score(may["reference"])
        and may["scores"]["f1"] >= 0.95 * may["reference"]["f1"]
    )
    passed_stress = (
        primary_score(c) > 1.05 * primary_score(r)
        and c["f1"] >= r["f1"]
        and all(item["scores"]["f1"] >= 0.9 * item["reference"]["f1"] for item in rows[1:])
    )
    rows = [read(root / fold / "result.json") for fold in ("screen_1", "screen_2")] + rows
    report = {
        "selection": selection,
        "periods": rows,
        "five_period_pooled": pooled([item["scores"] for item in rows]),
        "five_period_reference": pooled([item["reference"] for item in rows]),
        "passed_may": passed_may,
        "passed_stress": passed_stress,
        "research_eligible": bool(passed_may and passed_stress),
        "automatic_activation": False,
    }
    write_json(root / "report.json", report)
    print("REPORT quarter quantile", {k: v for k, v in report.items() if k != "periods"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v22"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
