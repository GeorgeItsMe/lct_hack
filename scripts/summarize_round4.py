"""Publish complete v10 evidence without changing production or previous reports."""

import json
from datetime import UTC, datetime
from pathlib import Path

from moscollector.model_registry import active_version, load_bundle
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v10b")


def read(name):
    return json.loads((root / name).read_text())


stages = {name: read(name + ".json") for name in ("selection", "confirmation", "stress")}
for name, result in stages.items():
    assert set(result) == {"fault", "fire"}, f"Incomplete {name}"
plan = read("plan.json")
for path, digest in plan["source_hashes"].items():
    assert sha256(Path(path)) == digest
for path, digest in plan["code_hashes"].items():
    assert sha256(Path(path)) == digest
if (root / "continuation.json").exists():
    for path, digest in read("continuation.json")["reused_files_sha256"].items():
        assert sha256(root / path) == digest
for kind, meta in json.loads(Path("artifacts/model_selection.json").read_text())["models"].items():
    assert sha256(Path("artifacts/models") / f"{kind}.cbm") == meta["sha256"]
version = active_version()
load_bundle(version)
report = {
    "created_at": datetime.now(UTC).isoformat(),
    "scope": "adaptive_retrospective_research_not_new_blind_test",
    "plan": plan,
    "hourly_feature_parity": json.loads(Path("artifacts/round4_feature_parity.json").read_text()),
    "continuation_provenance": read("continuation.json") if (root / "continuation.json").exists() else None,
    **stages,
    "models": {str(p.relative_to(root)): json.loads(p.read_text()) for p in sorted(root.glob("*/*/*.json"))},
    "eligible_for_further_release_checks": [
        kind for kind in ("fault", "fire") if all(stages[s][kind]["passes"] for s in stages)
    ],
    "active_model_version_at_report": version,
    "automatic_promotion": False,
    "original_model_hashes_verified": True,
    "source_hashes_verified": True,
    "external_training_data": False,
}
write_json(Path("artifacts/research_round4_report.json"), report)
print(
    json.dumps({k: v for k, v in report.items() if k not in ("models", "plan")}, ensure_ascii=False, indent=2)
)
