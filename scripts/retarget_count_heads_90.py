"""Apply a fixed past-only 90/90 policy-selection rule to fault/fire count heads.

This extends the diagnostic beyond access. Reference is the archived v9 count
family, not the active classifier/ensemble; no production comparison or promotion
is inferred from these numbers.
"""

from pathlib import Path

import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.goal90_research import STRESS, apply_policy, pooled, read, select_policy
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

root = Path("artifacts/research-v13-policy")
root.mkdir(exist_ok=True)
paths = {}
sources = [PROCESSED / "episodes.parquet"]
for kind in ("fault", "fire"):
    for fold in (*FOLDS, *STRESS):
        old = Path("artifacts/research-v9") / fold
        base = (
            old if (old / f"{kind}.json").exists() else Path("artifacts/research-v10b") / fold / "reference"
        )
        paths[kind, fold] = base
        sources.extend(base / f"{kind}{suffix}" for suffix in (".json", "-policy.parquet", "-test.parquet"))
plan = {
    "scope": "adaptive_retrospective_policy_diagnostic_not_blind_test",
    "goals": {"precision": 0.9, "recall": 0.9},
    "kinds": ["fault", "fire"],
    "folds": {**FOLDS, **STRESS},
    "rule": "Fixed v11 mean_retarget grid and90/90 criterion. Select solely on preceding policy period for every kind/month; evaluate all five months regardless of result. >=10 alerts and episodes for supported policy. No new model or family selection.",
    "reference": "Archived v9 count family; NOT the deployed legacy fault classifier or fire ensemble. Results cannot establish gain against production.",
    "unchanged": "24h labels, one-to-one event matching, hourly opportunities and70min confirmation delay; no June reads or automatic activation. Flood remains unsupported by too few episodes.",
    "source_hashes": {str(p): sha256(p) for p in sources},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(select_policy.__code__.co_filename),
            Path(EventEvaluator.__init__.__code__.co_filename),
        )
    },
}
plan_path = root / "plan.json"
if plan_path.exists():
    if read(plan_path) != plan:
        raise ValueError("Inputs or implementation changed; use another study directory")
else:
    write_json(plan_path, plan)
episodes = pd.read_parquet(sources[0], filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
report = {"plan": plan, "heads": {}, "automatic_activation": False}
for kind in ("fault", "fire"):
    eps = episodes[episodes.kind.eq(kind)]
    rows = []
    for fold in (*FOLDS, *STRESS):
        path = root / f"{kind}-{fold}.json"
        if path.exists():
            rows.append(read(path))
            continue
        base = paths[kind, fold]
        policy_pred = pd.read_parquet(base / f"{kind}-policy.parquet")
        policy, options, diagnostic = select_policy(policy_pred, eps, "mean_retarget")
        write_json(
            root / f"{kind}-{fold}-policy.json",
            {"selected": policy, "options": options, "diagnostic": diagnostic},
        )
        pred = pd.read_parquet(base / f"{kind}-test.parquet")
        alerts = apply_policy(pred, eps, "mean_retarget", policy)
        scores = EventEvaluator(pred, eps, 1).evaluate(alerts, 0.5, 1)
        reference = read(base / f"{kind}.json")["scores"]["pending"]
        assert scores["eligible_episodes"] == reference["eligible_episodes"]
        item = {
            "fold": fold,
            "policy": policy,
            "scores": scores,
            "reference": reference,
            "diagnostic": diagnostic,
        }
        write_json(path, item)
        rows.append(item)
        print("DONE", kind, fold, scores, flush=True)
    report["heads"][kind] = {
        "periods": rows,
        "pooled": pooled([r["scores"] for r in rows]),
        "reference_pooled": pooled([r["reference"] for r in rows]),
    }
    write_json(root / "report.json", report)
assert {str(p): sha256(p) for p in sources} == plan["source_hashes"]
write_json(Path("artifacts/goal90_other_heads.json"), report)
print("POOLED", {k: v["pooled"] for k, v in report["heads"].items()}, flush=True)
