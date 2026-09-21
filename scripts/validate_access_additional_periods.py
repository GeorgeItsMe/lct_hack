"""Evaluate the already-selected access family on fixed additional past months."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moscollector.count_extension_research import validate_opportunities
from moscollector.count_research import fit_one, pooled
from moscollector.model_registry import active_version
from moscollector.paths import PROCESSED
from moscollector.precision_research import periods_for
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-access-stress")
root.mkdir(exist_ok=True)
folds = {"stress_1": "2025-12-01", "stress_2": "2026-03-01"}
sources = [PROCESSED / n for n in ("features.parquet", "features-dense-round4.parquet", "episodes.parquet")]
plan = {
    "scope": "additional_retrospective_evaluation_of_selected_access_family_not_blind_test",
    "folds": folds,
    "model": "Unchanged v9 Poisson recent_reference and pending-warning policy; refit at each historical origin. No alternative configuration or test-label threshold selection.",
    "source_hashes": {str(p): sha256(p) for p in sources},
    "implementation_sha256": sha256(Path(__file__)),
    "count_implementation_sha256": sha256(Path(fit_one.__code__.co_filename)),
    "selection": "No new selection or automatic model activation. Report each extra period and all five months regardless of performance.",
    "limits": "These dates previously appeared in training/calibration. Final deployed weights are not independently tested by historical refits.",
}
path = root / "plan.json"
if path.exists():
    assert json.loads(path.read_text()) == plan
else:
    write_json(path, plan)
frame = pd.read_parquet(sources[0], filters=[("as_of", "<", pd.Timestamp("2026-06-01"))])
dense = pd.read_parquet(sources[1], columns=frame.columns.tolist())
episodes = pd.read_parquet(sources[2], filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
results = {}
for fold, date in folds.items():
    validate_opportunities(frame, dense, periods_for(date, "access"))
    results[fold] = fit_one(frame, dense, episodes, root / fold, "access", date)
original = {
    fold: json.loads((Path("artifacts/research-v9") / fold / "access.json").read_text())
    for fold in ("screen_1", "screen_2", "confirmation")
}
all_results = {**original, **results}
report = {
    "created_at": datetime.now(UTC).isoformat(),
    "plan": plan,
    "model_version": active_version(),
    "periods": {
        fold: {
            "test_period": r["periods"]["test"],
            "scores": r["scores"]["pending"],
            "policy": r["policies"]["pending"],
        }
        for fold, r in all_results.items()
    },
    "additional_pooled": pooled(list(results.values()), "pending"),
    "five_period_pooled": pooled(list(all_results.values()), "pending"),
    "new_model_selection": False,
    "final_refit_has_new_blind_test": False,
}
assert {str(p): sha256(p) for p in sources} == plan["source_hashes"]
write_json(Path("artifacts/access_additional_validation.json"), report)
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
