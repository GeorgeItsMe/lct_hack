"""Predeclared research toward 90% event precision AND recall, without activation.

Two changes are separated: retarget the existing mean-count warning policy, and
replace its warning capacity with calibrated conditional count quantiles. The
original labels, hourly opportunities, temporal purges and one-to-one matching
are preserved. All results remain adaptive retrospective research.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from moscollector.alert_diagnostics import HOUR_NS, EventEvaluator
from moscollector.cadence_research import align_opportunities, assert_same_episode_cohort
from moscollector.count_extension_research import validate_opportunities
from moscollector.count_research import episode_counts, pending_alerts
from moscollector.paths import PROCESSED
from moscollector.precision_research import columns_for, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, model_input

ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)
STRESS = {"stress_1": "2025-12-01", "stress_2": "2026-03-01"}
FAMILIES = ("mean_retarget", "quantile")
PLAN = {
    "scope": "adaptive_retrospective_research_not_blind_test",
    "primary_goals": {"precision": 0.9, "recall": 0.9},
    "interpretation": "Conservative interpretation while clarification is pending: BOTH event precision and recall >=.90. Neither accuracy nor precision alone substitutes. Access is the first research head, not proof for all incident types.",
    "folds": FOLDS,
    "stress_folds": STRESS,
    "families": {
        "mean_retarget": "Unchanged v9 mean-count weights and probability. Policy margin .25,.5,1,2,3,5; capacity multiplier .5,1,1.5,2; probability floor0,.5,.75,.9,.95.",
        "quantile": "CatBoost MultiQuantile alpha .1,.25,.5,.75,.9; original recent_reference features, depth6,l2=8,lr=.06,max600,seed42,CPU4,early stopping100 on MultiQuantile. A separate scalar residual quantile correction per output is fitted only on calibration rows. Nonnegative clipping and per-row sorting enforce noncrossing capacity. V9 probability supplies the same optional probability gate. Policy selects one quantile, margin .25,.5,1,2 and floor0,.5,.75,.9,.95.",
    },
    "unchanged": "Same v9 train/validation/calibration/policy dates with25h purge; same grouped episode starts within[t,t+24h), hourly cohort and one-to-one matching;70min confirmation delay. No June reads, relabeling, event exclusion or automatic activation.",
    "policy": "Select only on each preceding policy period, >=10 alerts and >=10 episodes for supported status, <=.25 false alerts/object-day. Maximize min(P/.9,R/.9,1), then F1, recall, precision. If support unavailable, report low_support; no claim of meeting target.",
    "selection": "Choose the family with largest pooled primary score, then F1 on Nov/Feb. Require >5% relative primary score improvement against archived v9 and no F1 loss. Evaluate only that screen-selected family on May and Dec/Mar; no replacement after those results.",
    "confirmation": "For eligibility require improvement in May primary score with F1>=95% reference, and >5% primary score gain on pooled Dec/Mar with no pooled F1 loss and each month F1>=90% reference. These are research gates, not achievement of90%.",
    "success": "Report90/90 per month and pooled, >=10 alerts and episodes in every evaluated month. Retrospective success cannot establish future or independently tested final-weight performance. All incident types remain in scope of the overall solution.",
    "diagnostics": "Policy-period frontiers and perfect-future schedules are labeled diagnostic, never presented as predictive performance. Secondary P90/R50 and F190 are descriptive only, not alternative success criteria.",
    "sources": ["https://catboost.ai/docs/en/concepts/loss-functions-regression"],
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def primary_score(scores):
    return min(scores["precision"] / 0.9, scores["recall"] / 0.9, 1.0)


def pooled(rows):
    tp, alerts, events = (sum(r[k] for r in rows) for k in ("true_alerts", "alerts", "eligible_episodes"))
    result = {
        "true_alerts": tp,
        "alerts": alerts,
        "eligible_episodes": events,
        "precision": tp / alerts if alerts else 0,
        "recall": tp / events if events else 0,
        "f1": 2 * tp / (alerts + events) if alerts + events else 0,
    }
    result["primary_score"] = primary_score(result)
    result["both_90"] = result["precision"] >= 0.9 and result["recall"] >= 0.9
    return result


class PendingSimulator:
    """Array-based equivalent of v9 pending_alerts; all resolutions are causal."""

    def __init__(self, predictions, episodes):
        self.n = len(predictions)
        if predictions.duplicated(["object_id", "as_of"]).any():
            raise ValueError("Duplicate forecast opportunities")
        self.groups = []
        for obj, rows in predictions.reset_index(drop=True).groupby("object_id"):
            rows = rows.sort_values("as_of")
            times = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            events = episodes.loc[episodes.object_id.eq(obj), "start_ts"].sort_values()
            events = events.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            self.groups.append((rows.index.to_numpy(), times, events))

    def alerts(self, capacity, probability, margin, floor):
        capacity, probability = np.asarray(capacity), np.asarray(probability)
        if any(x.shape != (self.n,) or not np.isfinite(x).all() for x in (capacity, probability)):
            raise ValueError("Invalid capacity/probability vector")
        if margin <= 0 or not 0 <= floor <= 1:
            raise ValueError("Invalid policy")
        result = np.zeros(self.n)
        delay = 70 * 60 * 1_000_000_000
        for ids, times, events in self.groups:
            pending = []
            next_event = 0
            for i, now in zip(ids, times, strict=True):
                while next_event < len(events) and events[next_event] + delay < now:
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


def select_policy(pred, episodes, family):
    simulator, evaluator = PendingSimulator(pred, episodes), EventEvaluator(pred, episodes, 1)
    floors = (0, 0.5, 0.75, 0.9, 0.95)
    probability = pred.probability.to_numpy()
    if family == "mean_retarget":
        capacities = [(str(m), pred.expected_count.to_numpy() * m) for m in (0.5, 1, 1.5, 2)]
        margins = (0.25, 0.5, 1, 2, 3, 5)
    else:
        capacities = [(str(q), pred[f"q_{q}"].to_numpy()) for q in ALPHAS]
        margins = (0.25, 0.5, 1, 2)
    options = []
    for key, capacity in capacities:
        for margin in margins:
            for floor in floors:
                alerts = simulator.alerts(capacity, probability, margin, floor)
                scores = evaluator.evaluate(alerts, 0.5, 1)
                options.append({"capacity": key, "margin": margin, "floor": floor, **scores})
    budget = [r for r in options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    selected = dict(
        max(supported or budget, key=lambda r: (primary_score(r), r["f1"], r["recall"], r["precision"]))
    )
    selected["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    diagnostic = {
        "period": "preceding_policy_only",
        "recall_at_precision_90": max((r["recall"] for r in supported if r["precision"] >= 0.9), default=0),
        "precision_at_recall_90": max((r["precision"] for r in supported if r["recall"] >= 0.9), default=0),
        "max_f1": max((r["f1"] for r in supported), default=0),
        "any_both_90": any(r["precision"] >= 0.9 and r["recall"] >= 0.9 for r in supported),
        "perfect_future_schedule_not_model": evaluator.clairvoyant_schedule(1),
    }
    return selected, options, diagnostic


def apply_policy(pred, episodes, family, policy):
    if family == "mean_retarget":
        capacity = pred.expected_count.to_numpy() * float(policy["capacity"])
    else:
        capacity = pred[f"q_{policy['capacity']}"].to_numpy()
    return PendingSimulator(pred, episodes).alerts(
        capacity, pred.probability, policy["margin"], policy["floor"]
    )


def base_directory(fold):
    return Path("artifacts/research-access-stress" if fold in STRESS else "artifacts/research-v9") / fold


def noncrossing(values, correction):
    result = np.asarray(values) + np.asarray(correction)
    if result.ndim != 2 or result.shape[1] != len(ALPHAS) or not np.isfinite(result).all():
        raise ValueError("Invalid quantile predictions")
    return np.sort(np.maximum(result, 0), axis=1)


def fit_quantiles(frame, dense, episodes, directory, test_begin, base):
    path = directory / "fit.json"
    if path.exists():
        return read(path)
    periods = periods_for(test_begin, "access")
    validate_opportunities(frame, dense, periods)
    used = np.logical_or.reduce([mask(frame, *periods[k]) for k in ("train", "validation", "calibration")])
    training = frame.loc[used].reset_index(drop=True)
    masks = {k: mask(training, *periods[k]) for k in ("train", "validation", "calibration")}
    columns = columns_for(training, "recent_reference", "access")
    counts = episode_counts(training, episodes)
    if not np.array_equal(counts > 0, training.target_access.to_numpy().astype(bool)):
        raise ValueError("Count target differs from original episode definition")
    x = model_input(training, columns)
    loss = "MultiQuantile:alpha=" + ",".join(map(str, ALPHAS))
    model = CatBoostRegressor(
        iterations=600,
        depth=6,
        l2_leaf_reg=8,
        learning_rate=0.06,
        loss_function=loss,
        eval_metric=loss,
        random_seed=42,
        thread_count=4,
        cat_features=CATEGORICAL,
        allow_writing_files=False,
        early_stopping_rounds=100,
        verbose=100,
    )
    print("START quantile", directory.parent.name, int(masks["train"].sum()), flush=True)
    model.fit(
        x[masks["train"]],
        counts[masks["train"]],
        eval_set=(x[masks["validation"]], counts[masks["validation"]]),
    )
    raw = model.predict(x[masks["calibration"]])
    actual = counts[masks["calibration"]]
    corrections = np.array([np.quantile(actual - raw[:, i], q) for i, q in enumerate(ALPHAS)])
    calibrated = noncrossing(raw, corrections)
    coverage = {str(q): float(np.mean(actual <= calibrated[:, i])) for i, q in enumerate(ALPHAS)}
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(directory / "access.cbm"))
    for period in ("policy", "test"):
        reference = frame.loc[mask(frame, *periods[period]), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *periods[period])], reference)
        pred = rows[["object_id", "as_of"]].copy()
        quantiles = noncrossing(model.predict(model_input(rows, columns)), corrections)
        for i, q in enumerate(ALPHAS):
            pred[f"q_{q}"] = quantiles[:, i]
        base_pred = pd.read_parquet(base / f"access-{period}.parquet")
        pred = pred.merge(
            base_pred[["object_id", "as_of", "probability"]], on=["object_id", "as_of"], validate="one_to_one"
        )
        if len(pred) != len(rows) or len(pred) != len(base_pred):
            raise ValueError("Quantile and mean predictions have different opportunities")
        assert_same_episode_cohort(pred, reference, episodes)
        pred.to_parquet(directory / f"access-{period}.parquet", index=False)
    result = {
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "features": columns,
        "alphas": ALPHAS,
        "corrections": corrections.tolist(),
        "calibration_coverage_not_test": coverage,
        "best_iteration": model.best_iteration_,
        "binary_target_parity": True,
        "training_rows": int(masks["train"].sum()),
        "model_sha256": sha256(directory / "access.cbm"),
    }
    write_json(path, result)
    return result


def evaluate(output, fold, family, episodes, frame=None, dense=None):
    directory, base = output / fold / family, base_directory(fold)
    path = directory / "result.json"
    if path.exists():
        return read(path)
    directory.mkdir(parents=True, exist_ok=True)
    if family == "quantile":
        fit_quantiles(frame, dense, episodes, directory, {**FOLDS, **STRESS}[fold], base)
    pred_root = base if family == "mean_retarget" else directory
    policy_pred = pd.read_parquet(pred_root / "access-policy.parquet")
    if family == "mean_retarget":
        old = read(base / "access.json")["policies"]["pending"]
        fast = PendingSimulator(policy_pred, episodes).alerts(
            policy_pred.expected_count, policy_pred.probability, old["margin"], old["probability_floor"]
        )
        np.testing.assert_array_equal(
            fast, pending_alerts(policy_pred, episodes, old["margin"], old["probability_floor"])
        )
    policy, options, diagnostic = select_policy(policy_pred, episodes, family)
    # Freeze policy before opening/evaluating this family's test forecasts.
    write_json(directory / "policy.json", {"selected": policy, "options": options, "diagnostic": diagnostic})
    test = pd.read_parquet(pred_root / "access-test.parquet")
    test["candidate_alert"] = apply_policy(test, episodes, family, policy)
    scores = EventEvaluator(test, episodes, 1).evaluate(test.candidate_alert, 0.5, 1)
    reference = read(base / "access.json")["scores"]["pending"]
    if scores["eligible_episodes"] != reference["eligible_episodes"]:
        raise ValueError("Changed episode denominator")
    test.to_parquet(directory / "evaluated.parquet", index=False)
    result = {
        "kind": "access",
        "fold": fold,
        "family": family,
        "policy": policy,
        "diagnostic": diagnostic,
        "scores": scores,
        "reference": reference,
    }
    write_json(path, result)
    print("DONE", fold, family, json.dumps(scores), flush=True)
    return result


def lock_plan(output):
    sources = [
        PROCESSED / n for n in ("features.parquet", "features-dense-round4.parquet", "episodes.parquet")
    ]
    for fold in (*FOLDS, *STRESS):
        sources.extend(
            base_directory(fold) / n
            for n in ("access.json", "access.cbm", "access-policy.parquet", "access-test.parquet")
        )
    code = [
        Path(__file__),
        Path(episode_counts.__code__.co_filename),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(periods_for.__code__.co_filename),
        Path(align_opportunities.__code__.co_filename),
        Path(model_input.__code__.co_filename),
        Path(mask.__code__.co_filename),
    ]
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in code},
    }
    plan = json.loads(json.dumps(plan))
    path = output / "plan.json"
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != plan:
            raise ValueError("Study inputs or implementation changed; use a new study directory")
    else:
        write_json(path, plan)


def run(output, stage):
    lock_plan(output)
    episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet",
        filters=[("start_ts", "<", pd.Timestamp("2026-06-01")), ("kind", "==", "access")],
    )
    if stage == "policy":
        for fold in ("screen_1", "screen_2"):
            evaluate(output, fold, "mean_retarget", episodes)
        return
    if stage == "train":
        frame = pd.read_parquet(
            PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
        )
        dense = pd.read_parquet(PROCESSED / "features-dense-round4.parquet", columns=frame.columns.tolist())
        for fold in ("screen_1", "screen_2"):
            evaluate(output, fold, "quantile", episodes, frame, dense)
        return
    screen = {
        family: [read(output / fold / family / "result.json") for fold in ("screen_1", "screen_2")]
        for family in FAMILIES
    }
    candidates = {k: pooled([r["scores"] for r in v]) for k, v in screen.items()}
    reference = pooled([r["reference"] for r in screen["mean_retarget"]])
    selected = max(candidates, key=lambda k: (primary_score(candidates[k]), candidates[k]["f1"]))
    chosen = candidates[selected]
    eligible = primary_score(chosen) > 1.05 * primary_score(reference) and chosen["f1"] >= reference["f1"]
    selection = {
        "selected": selected,
        "candidates": candidates,
        "reference": reference,
        "passed_screen": eligible,
    }
    write_json(output / "selection.json", selection)
    print("SELECTION", json.dumps(selection), flush=True)
    if not eligible:
        return
    frame = dense = None
    if selected == "quantile":
        frame = pd.read_parquet(
            PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
        )
        dense = pd.read_parquet(PROCESSED / "features-dense-round4.parquet", columns=frame.columns.tolist())
    results = [evaluate(output, fold, selected, episodes, frame, dense) for fold in ("confirmation", *STRESS)]
    may, stress = results[0], results[1:]
    candidate_extra = pooled([r["scores"] for r in stress])
    reference_extra = pooled([r["reference"] for r in stress])
    passed_may = (
        primary_score(may["scores"]) > primary_score(may["reference"])
        and may["scores"]["f1"] >= 0.95 * may["reference"]["f1"]
    )
    passed_extra = (
        primary_score(candidate_extra) > 1.05 * primary_score(reference_extra)
        and candidate_extra["f1"] >= reference_extra["f1"]
        and all(r["scores"]["f1"] >= 0.9 * r["reference"]["f1"] for r in stress)
    )
    all_rows = screen[selected] + results
    report = {
        "selection": selection,
        "periods": all_rows,
        "five_period_pooled": pooled([r["scores"] for r in all_rows]),
        "five_period_reference": pooled([r["reference"] for r in all_rows]),
        "passed_may": passed_may,
        "passed_stress": passed_extra,
        "research_eligible": bool(passed_may and passed_extra),
        "automatic_activation": False,
        "new_blind_test": False,
    }
    write_json(output / "report.json", report)
    print("REPORT", json.dumps({k: v for k, v in report.items() if k != "periods"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v11"))
    parser.add_argument("--stage", choices=("policy", "train", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
