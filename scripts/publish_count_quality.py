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
extra_path = Path("artifacts/access_additional_validation.json")
extra = json.loads(extra_path.read_text()) if extra_path.exists() else None
if extra:
    if extra["model_version"] != active_version():
        raise ValueError("Additional evaluation refers to a different active bundle")
    for fold, label in (("stress_1", "Декабрь 2025"), ("stress_2", "Март 2026")):
        rows.append({"period": label, **extra["periods"][fold]["scores"]})
    order = ("Ноябрь 2025", "Декабрь 2025", "Февраль 2026", "Март 2026", "Май 2026")
    rows.sort(key=lambda row: order.index(row["period"]))
    total = extra["five_period_pooled"]
else:
    total = pooled(results, "pending")
period_count = len(rows)
passed = sum(row["precision"] >= 0.75 and row["recall"] >= 0.5 for row in rows)
target_met = total["precision"] >= 0.75 and total["recall"] >= 0.5
summary = (
    f"Цель 75% / 50% {'выполнена' if target_met else 'не выполнена'} суммарно по {period_count} периодам. "
    f"Обе цели выполнены отдельно в {passed} из {period_count} месяцев."
)
below = []
for row in rows:
    missed = [
        name
        for name, value, target in (("Precision", row["precision"], 0.75), ("Recall", row["recall"], 0.5))
        if value < target
    ]
    if missed:
        below.append(f"{row['period']}: ниже цели {', '.join(missed)}")
rows.append({"period": f"Суммарно: {period_count} периодов", **total})
uncertainty = json.loads(Path("artifacts/research_round3_uncertainty.json").read_text())
write_json(
    Path("artifacts/operational_quality.json"),
    {
        "model_version": active_version(),
        "kind": "access",
        "scope": "adaptive_retrospective_not_new_blind_test",
        "rows": rows,
        "target_summary": summary,
        "below_target": "; ".join(below),
        "additional_periods": bool(extra),
        "f1_gain_interval": uncertainty["models"]["access"]["by_parent"]["percentile_95"]["f1_difference"],
        "target_precision": 0.75,
        "target_recall": 0.5,
        "june_report_unchanged": True,
    },
)
