"""Consolidate completed research; do not alter any active model or frozen test."""

import json
from datetime import UTC, datetime
from pathlib import Path

from moscollector.model_registry import active_version
from moscollector.prepare import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def read(path):
    return json.loads(path.read_text())


def run():
    frozen = read(ARTIFACTS / "model_selection.json")
    for kind, meta in frozen["models"].items():
        if sha256(ARTIFACTS / "models" / f"{kind}.cbm") != meta["sha256"]:
            raise ValueError(f"Frozen model changed: {kind}")
    repair = read(ARTIFACTS / "censoring_repair.json")
    if sha256(ROOT / "data/processed/features.parquet") != repair["unchanged_original_features_sha256"]:
        raise ValueError("Original features changed")
    studies = {}
    for version in ("v6b", "v7b", "v8", "v9"):
        directory = ARTIFACTS / f"research-{version}"
        selection, confirmation = (read(directory / f"{name}.json") for name in ("selection", "confirmation"))
        expected = {"fault", "fire", "access"} | ({"flood"} if version in ("v6b", "v7b") else set())
        if set(selection) != expected or set(confirmation) != expected:
            raise ValueError(f"Study {version} is incomplete")
        studies[version] = {
            "plan": read(directory / "plan.json"),
            "selection": selection,
            "confirmation": confirmation,
            "individual_models": {
                str(p.relative_to(directory)): read(p)
                for p in sorted(
                    directory.glob("*/*.json") if version == "v9" else directory.glob("*/*/*.json")
                )
            },
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "adaptive_retrospective_research_not_new_blind_test",
        "targets": {"precision": 0.75, "recall": 0.5, "brief_precision": 0.7},
        "evaluation_unit": "one_to_one_grouped_sensor_proxy_episode",
        "row_metrics_are_separate": True,
        "new_external_training_data": False,
        "original_models_and_features_verified_unchanged": True,
        "active_model_version_at_report": active_version(),
        "automatic_promotion": False,
        "invalidated_studies": ["v6", "v7"],
        "censoring_repair": repair,
        "cadence_parity": read(ARTIFACTS / "dense_rich_feature_parity.json"),
        "policy_only_comparison": read(ARTIFACTS / "policy_comparison.json"),
        "studies": studies,
        "limitations": [
            "Repeatedly used historical periods; May is a confirmation period, not a new blind test.",
            "Cadence/threshold improvements are distinct from gains caused by model weights.",
            "Proxy sensor episodes are not confirmed physical incidents.",
            "Sparse fault/flood events and temporal drift limit conclusions.",
            "Some loglinear fits reached the iteration limit without convergence.",
            "Passing a relative-improvement gate does not imply reaching the P/R targets.",
        ],
    }
    write_json(ARTIFACTS / "research_round3_report.json", report)
    for version, study in studies.items():
        for kind, result in study["confirmation"].items():
            print(version, kind, result)


if __name__ == "__main__":
    run()
