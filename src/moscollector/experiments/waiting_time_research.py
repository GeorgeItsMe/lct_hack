"""v15: censored residual waiting time for the same recurring sensor episodes.

AFT is an auxiliary training objective, not a change of evaluation horizon or
incident definition. Probabilities are calibrated independently for the original
24h binary target. Never activates models or reads June evaluation rows.
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
from moscollector.experiments.count_extension_research import validate_opportunities
from moscollector.experiments.goal90_research import (
    STRESS,
    apply_policy,
    pooled,
    primary_score,
    read,
    select_policy,
)
from moscollector.paths import PROCESSED
from moscollector.precision_research import columns_for, periods_for
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS, mask
from moscollector.train import CATEGORICAL, calibrate, calibrated, model_input

KINDS = ("access", "fire", "fault")
FAMILIES = ("aft_threshold", "aft_count_gate")
PLAN = {
    "scope": "adaptive_retrospective_waiting_time_study_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": KINDS,
    "hypothesis": "Training on residual time to the next episode may rank imminent events better than a binary24h target or mean24h count. Evaluate both pure threshold/cooldown and a probability gate on existing mean-count warning capacity.",
    "target": "Next episode start per object in[t,t+24h). Exact waiting time in hours for observed starts, clipped below at1/60h solely to keep log-time finite. No event in that window => right-censoring [24,-1]. At24h is censored. Assert observed-event mask equals original binary labels; changes beyond24h cannot alter targets.",
    "fit": "CatBoost SurvivalAft:dist=Normal;scale=1, depth6,l2=8,lr=.04,max800,seed42,CPU4,earlystop100 on same censored loss. Same v9 recent_reference features (access66,others94), same train/validation/calibration/policy dates and25h purges. This models conditional residual waiting time at each snapshot, not independent equipment lifetimes or a full recurrent point process.",
    "calibration": "Same independent sigmoid calibration used previously, applied to minus raw predicted log-time against the original24h binary target. No claim that exp(raw) is a calibrated lead-time estimate.",
    "families": {
        "aft_threshold": "On preceding policy data only, thresholds .01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.75,.8,.85,.9,.95,.98,.99,1.01 plus probability quantiles .8,.9,.95,.975,.99,.995; cooldown1,2,3,6,12,24h. Same90/90 score thenF1,recall,precision;>=10alerts,>=10episodes for supported status;<=.25FP/object-day.",
        "aft_count_gate": "Frozen v9 expected counts and unchanged v11 mean_retarget grid; replace only the independently calibrated probability gate with AFT probability. Floor0 means AFT has no effect and is reported explicitly. Same causal pending budget and70min resolution delay.",
    },
    "reference": "V9 count family with90/90 policy from v12 uncorrected(access) and v13(fire/fault). Same exact hourly opportunities and episodes. Not a comparison to the deployed fire/fault classifiers.",
    "selection": "Per kind choose highest pooled90/90 primary score thenF1 over Nov/Feb. Require>5% primary-score gain and noF1 loss versus reference. Only selected family for passing kinds proceeds to May and Dec/Mar; no replacing family afterward.",
    "confirmation": "May improves primary score and preserves>=95%F1. Dec/Mar pooled improves primary score>5% with noF1 loss; each extra monthF1>=90%reference. Intermediate research gates only; report actual90/90 achievement and monthly support separately. Never auto-promote.",
    "unchanged": "Original24h one-to-one episode evaluation, labels, opportunities and70min confirmation delay. No June evaluation, external training data or activation. Earlier dates are adaptively reused, not a new blind test.",
    "sources": [
        "https://catboost.ai/docs/en/concepts/loss-functions-regression#SurvivalAft",
        "https://github.com/catboost/tutorials/blob/master/regression/survival.ipynb",
    ],
}


def waiting_time_labels(frame, episodes):
    """Return AFT bounds and the exact original24h observed-event indicator."""
    labels = np.column_stack((np.full(len(frame), 24.0), np.full(len(frame), -1.0)))
    observed = np.zeros(len(frame), dtype=bool)
    for obj, rows in frame.reset_index(drop=True).groupby("object_id"):
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        events = episodes.loc[episodes.object_id.eq(obj), "start_ts"].sort_values()
        events = events.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        if not len(events):
            continue
        following = np.searchsorted(events, at, side="left")
        valid = following < len(events)
        wait = (events[np.minimum(following, len(events) - 1)] - at) / HOUR_NS
        valid &= (wait >= 0) & (wait < 24)
        ids = rows.index.to_numpy()[valid]
        exact = np.maximum(wait[valid], 1 / 60)
        labels[ids, 0] = labels[ids, 1] = exact
        observed[ids] = True
    return labels, observed


def base_directory(kind, fold):
    if kind == "access" and fold in STRESS:
        return Path("artifacts/research-access-stress") / fold
    base = Path("artifacts/research-v9") / fold
    return base if (base / f"{kind}.json").exists() else Path("artifacts/research-v10b") / fold / "reference"


def reference_result(kind, fold):
    if kind == "access":
        return read(Path("artifacts/research-v12") / fold / "uncorrected.json")["scores"]
    return read(Path("artifacts/research-v13-policy") / f"{kind}-{fold}.json")["scores"]


def threshold_policy(pred, episodes):
    evaluator = EventEvaluator(pred, episodes, 1)
    thresholds = np.unique(
        np.r_[
            [
                0.01,
                0.025,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.98,
                0.99,
                1.01,
            ],
            np.quantile(pred.probability, [0.8, 0.9, 0.95, 0.975, 0.99, 0.995]),
        ]
    )
    options = [
        {"threshold": float(t), **evaluator.evaluate(pred.probability, t, c)}
        for c in (1, 2, 3, 6, 12, 24)
        for t in thresholds
    ]
    budget = [r for r in options if (r["false_alerts_per_object_day"] or 0) <= 0.25]
    supported = [r for r in budget if r["alerts"] >= 10]
    selected = dict(
        max(supported or budget, key=lambda r: (primary_score(r), r["f1"], r["recall"], r["precision"]))
    )
    selected["status"] = "supported" if supported and evaluator.events >= 10 else "low_support"
    return selected, options


def threshold_alerts(pred, policy):
    alerts = np.zeros(len(pred))
    for _, rows in pred.reset_index(drop=True).groupby("object_id"):
        previous = None
        for row in rows.sort_values("as_of").itertuples():
            now = pd.Timestamp(row.as_of).value
            if row.probability < policy["threshold"]:
                continue
            if previous is not None and now - previous < policy["cooldown_hours"] * HOUR_NS:
                continue
            alerts[row.Index], previous = 1, now
    return alerts


def fit(frame, dense, episodes, directory, kind, fold):
    path = directory / "fit.json"
    if path.exists():
        meta = read(path)
        if sha256(directory / "model.cbm") != meta["model_sha256"]:
            raise ValueError("Cached AFT weights changed")
        return meta
    periods = periods_for({**FOLDS, **STRESS}[fold], kind)
    validate_opportunities(frame, dense, periods)
    used = np.logical_or.reduce([mask(frame, *periods[k]) for k in ("train", "validation", "calibration")])
    training = frame.loc[used].reset_index(drop=True)
    masks = {k: mask(training, *periods[k]) for k in ("train", "validation", "calibration")}
    columns = columns_for(training, "recent_reference", kind)
    labels, observed = waiting_time_labels(training, episodes)
    if not np.array_equal(observed, training[f"target_{kind}"].to_numpy().astype(bool)):
        raise ValueError("Waiting-time labels differ from the original24h episode target")
    x = model_input(training, columns)
    loss = "SurvivalAft:dist=Normal;scale=1"
    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        l2_leaf_reg=8,
        learning_rate=0.04,
        loss_function=loss,
        eval_metric=loss,
        random_seed=42,
        thread_count=4,
        cat_features=CATEGORICAL,
        allow_writing_files=False,
        early_stopping_rounds=100,
        verbose=100,
    )
    print("START aft", kind, fold, int(masks["train"].sum()), flush=True)
    model.fit(
        x[masks["train"]],
        labels[masks["train"]],
        eval_set=(x[masks["validation"]], labels[masks["validation"]]),
    )
    raw = model.predict(x[masks["calibration"]], prediction_type="RawFormulaVal")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite AFT calibration predictions")
    calibration = calibrate(-raw, observed[masks["calibration"]])
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(directory / "model.cbm"))
    for period in ("policy", "test"):
        reference = frame.loc[mask(frame, *periods[period]), ["object_id", "as_of"]]
        rows = align_opportunities(dense.loc[mask(dense, *periods[period])], reference)
        pred = rows[["object_id", "as_of"]].copy()
        raw = model.predict(model_input(rows, columns), prediction_type="RawFormulaVal")
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite AFT forecast")
        pred["raw_log_wait"] = raw
        pred["probability"] = calibrated(-raw, calibration)
        base = pd.read_parquet(base_directory(kind, fold) / f"{kind}-{period}.parquet")
        pred = pred.merge(
            base[["object_id", "as_of", "expected_count"]], on=["object_id", "as_of"], validate="one_to_one"
        )
        if len(pred) != len(rows) or len(pred) != len(base):
            raise ValueError("Different AFT and count forecast opportunities")
        assert_same_episode_cohort(pred, reference, episodes)
        pred.to_parquet(directory / f"{period}.parquet", index=False)
    meta = {
        "kind": kind,
        "fold": fold,
        "features": columns,
        "periods": {k: list(map(str, v)) for k, v in periods.items()},
        "calibration": calibration,
        "training_rows": int(masks["train"].sum()),
        "best_iteration": model.best_iteration_,
        "observed_target_parity": True,
        "model_sha256": sha256(directory / "model.cbm"),
    }
    write_json(path, meta)
    return meta


def evaluate(root, kind, fold, families, frame, dense, episodes):
    directory = root / kind / fold
    fit(frame, dense, episodes, directory, kind, fold)
    results = {}
    for family in families:
        path = directory / f"{family}.json"
        if path.exists():
            results[family] = read(path)
            continue
        pred = pd.read_parquet(directory / "policy.parquet")
        if family == "aft_threshold":
            policy, options = threshold_policy(pred, episodes)
        else:
            policy, options, _ = select_policy(pred, episodes, "mean_retarget")
        write_json(directory / f"{family}-policy.json", {"selected": policy, "options": options})
        test = pd.read_parquet(directory / "test.parquet")
        alerts = (
            threshold_alerts(test, policy)
            if family == "aft_threshold"
            else apply_policy(test, episodes, "mean_retarget", policy)
        )
        test["candidate_alert"] = alerts
        scores = EventEvaluator(test, episodes, 1).evaluate(alerts, 0.5, 1)
        if family == "aft_threshold":
            original = EventEvaluator(test, episodes, 1).evaluate(
                test.probability, policy["threshold"], policy["cooldown_hours"]
            )
            if scores["true_alerts"] != original["true_alerts"] or scores["alerts"] != original["alerts"]:
                raise ValueError("Threshold planner differs from evaluator")
        reference = reference_result(kind, fold)
        if scores["eligible_episodes"] != reference["eligible_episodes"]:
            raise ValueError("Different episode denominator")
        test.to_parquet(directory / f"{family}-evaluated.parquet", index=False)
        result = {
            "kind": kind,
            "fold": fold,
            "family": family,
            "policy": policy,
            "scores": scores,
            "reference": reference,
            "uses_aft_at_decision": family != "aft_count_gate" or policy["floor"] > 0,
        }
        write_json(path, result)
        results[family] = result
        print("DONE", kind, fold, family, scores, flush=True)
    return results


def lock_plan(root):
    sources = [
        PROCESSED / n for n in ("features.parquet", "features-dense-round4.parquet", "episodes.parquet")
    ]
    for kind in KINDS:
        for fold in (*FOLDS, *STRESS):
            base = base_directory(kind, fold)
            sources.extend(
                base / f"{kind}{suffix}" for suffix in (".json", "-policy.parquet", "-test.parquet")
            )
            sources.append(
                Path("artifacts/research-v12") / fold / "uncorrected.json"
                if kind == "access"
                else Path("artifacts/research-v13-policy") / f"{kind}-{fold}.json"
            )
    code = [
        Path(__file__),
        Path(EventEvaluator.__init__.__code__.co_filename),
        Path(align_opportunities.__code__.co_filename),
        Path(columns_for.__code__.co_filename),
        Path(select_policy.__code__.co_filename),
        Path(calibrate.__code__.co_filename),
        Path(mask.__code__.co_filename),
        Path(validate_opportunities.__code__.co_filename),
    ]
    plan = {
        **PLAN,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in code},
    }
    plan = json.loads(json.dumps(plan))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("Study inputs or implementation changed; use another directory")
    else:
        write_json(path, plan)


def run(root, stage):
    lock_plan(root)
    frame = pd.read_parquet(
        PROCESSED / "features.parquet", filters=[("as_of", "<", pd.Timestamp("2026-06-01"))]
    )
    dense = pd.read_parquet(PROCESSED / "features-dense-round4.parquet", columns=frame.columns.tolist())
    all_episodes = pd.read_parquet(
        PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    if stage == "screen":
        selection = {}
        for kind in KINDS:
            episodes = all_episodes[all_episodes.kind.eq(kind)]
            rows = [
                evaluate(root, kind, fold, FAMILIES, frame, dense, episodes)
                for fold in ("screen_1", "screen_2")
            ]
            candidates = {family: pooled([r[family]["scores"] for r in rows]) for family in FAMILIES}
            reference = pooled([r[FAMILIES[0]]["reference"] for r in rows])
            family = max(candidates, key=lambda f: (primary_score(candidates[f]), candidates[f]["f1"]))
            candidate = candidates[family]
            selection[kind] = {
                "family": family,
                "candidates": candidates,
                "reference": reference,
                "passed_screen": primary_score(candidate) > 1.05 * primary_score(reference)
                and candidate["f1"] >= reference["f1"],
            }
            write_json(root / "selection.json", selection)
            print("SELECT", kind, selection[kind], flush=True)
        return
    selection = read(root / "selection.json")
    if set(selection) != set(KINDS):
        raise ValueError("Finish screening all kinds first")
    report = {}
    for kind, selected in selection.items():
        if not selected["passed_screen"]:
            report[kind] = {"selection": selected, "research_eligible": False, "status": "screen_failed"}
            continue
        family = selected["family"]
        episodes = all_episodes[all_episodes.kind.eq(kind)]
        rows = [
            evaluate(root, kind, fold, [family], frame, dense, episodes)[family]
            for fold in ("confirmation", *STRESS)
        ]
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
        rows = [read(root / kind / fold / f"{family}.json") for fold in ("screen_1", "screen_2")] + rows
        report[kind] = {
            "selection": selected,
            "periods": rows,
            "five_period_pooled": pooled([item["scores"] for item in rows]),
            "five_period_reference": pooled([item["reference"] for item in rows]),
            "passed_may": passed_may,
            "passed_stress": passed_stress,
            "research_eligible": bool(passed_may and passed_stress),
            "automatic_activation": False,
        }
        write_json(root / "report.json", report)
        print("REPORT", kind, {k: v for k, v in report[kind].items() if k != "periods"}, flush=True)
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-v15"))
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
