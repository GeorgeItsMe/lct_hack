"""Freeze model selection BEFORE running the final June evaluation."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
a = root / "artifacts"
if (a / "evaluation_report.json").exists():
    raise SystemExit("Final test has already been opened; refusing automatic model reselection.")
v3 = a / "experiments/v3"
v3.mkdir(parents=True, exist_ok=True)
if not (v3 / "validation_report.json").exists():
    shutil.copytree(a / "models", v3 / "models", dirs_exist_ok=True)
    shutil.copy2(a / "validation_report.json", v3 / "validation_report.json")
reports = {
    v: json.loads((a / f"experiments/{v}/validation_report.json").read_text()) for v in ("v1", "v2", "v3")
}
selection = {
    "selected_at": datetime.now(UTC).isoformat(),
    "test_opened": False,
    "rule": "Maximum policy-period event F1, then precision; minimum 10 eligible policy events. Flood uses v3 with warnings disabled.",
    "models": {},
}
report = dict(reports["v3"])
report["models"] = {}
for kind in ("fault", "fire", "flood", "access"):
    candidates = [(v, d["models"][kind]["policy_period"]) for v, d in reports.items()]
    chosen = "v3" if kind == "flood" else max(candidates, key=lambda x: (x[1]["f1"], x[1]["precision"]))[0]
    for ext in ("cbm", "json"):
        shutil.copy2(a / f"experiments/{chosen}/models/{kind}.{ext}", a / f"models/{kind}.{ext}")
    meta_path = a / f"models/{kind}.json"
    meta = json.loads(meta_path.read_text())
    meta["training_range"] = reports[chosen]["splits"]["train"]
    meta["selected_experiment"] = chosen
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    report["models"][kind] = {
        **reports[chosen]["models"][kind],
        "training_range": meta["training_range"],
        "selected_experiment": chosen,
    }
    selection["models"][kind] = {
        "experiment": chosen,
        "candidates": dict(candidates),
        "sha256": hashlib.sha256((a / f"models/{kind}.cbm").read_bytes()).hexdigest(),
    }
report["splits"]["train"] = ["2022-01-01", "2026-03-01"]
report["model_selection"] = selection
(a / "model_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2))
(a / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print({k: v["experiment"] for k, v in selection["models"].items()})
