"""Compact evidence for the operational model, separate from the June report."""

import json
from pathlib import Path

from moscollector.count_research import pooled
from moscollector.model_registry import active_version
from moscollector.prepare import write_json

root = Path("artifacts/research-v9")
confirmation = json.loads((root / "confirmation.json").read_text())
if not confirmation["access"]["passes"]:
    raise ValueError("The count model did not pass confirmation")
rows, results = [], []
for fold, label in (("screen_1", "Ноябрь 2025"), ("screen_2", "Февраль 2026"), ("confirmation", "Май 2026")):
    result = json.loads((root / fold / "access.json").read_text())
    results.append(result)
    rows.append({"period": label, **result["scores"]["pending"]})
rows.append({"period": "Суммарно", **pooled(results, "pending")})
uncertainty = json.loads(Path("artifacts/research_round3_uncertainty.json").read_text())
write_json(
    Path("artifacts/operational_quality.json"),
    {
        "model_version": active_version(),
        "kind": "access",
        "scope": "adaptive_retrospective_not_new_blind_test",
        "rows": rows,
        "f1_gain_interval": uncertainty["models"]["access"]["by_parent"]["percentile_95"]["f1_difference"],
        "target_precision": 0.75,
        "target_recall": 0.5,
        "june_report_unchanged": True,
    },
)
