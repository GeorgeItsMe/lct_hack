"""Publish completed 90/90 research evidence, without altering serving quality."""

from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from platform import python_version

from moscollector.goal90_research import read
from moscollector.model_registry import active_version, load_bundle
from moscollector.prepare import sha256, write_json

root = Path("artifacts/research-v11")
selection = read(root / "selection.json")
v11 = read(root / "report.json") if selection["passed_screen"] else {"selection": selection}
v12_root = Path("artifacts/research-v12")
v12_selection = read(v12_root / "selection.json")
v12 = read(v12_root / "report.json") if v12_selection["passed_screen"] else {"selection": v12_selection}
v13 = read(Path("artifacts/goal90_other_heads.json"))
v14_root = Path("artifacts/research-v14")
v14_selection = read(v14_root / "selection.json")
v14 = read(v14_root / "report.json") if v14_selection["passed_screen"] else {"selection": v14_selection}
studies = {
    "v11": root,
    "v12": v12_root,
    "v13": Path("artifacts/research-v13-policy"),
    "v14": v14_root,
    "v15": Path("artifacts/research-v15"),
    "v16": Path("artifacts/research-v16"),
    "v17": Path("artifacts/research-v17"),
}
plans = {}
original_hashes = read(Path("artifacts/hourly_feature_parity.json"))["source_files_unchanged"]
for source, digest in original_hashes.items():
    if sha256(Path(source)) != digest:
        raise ValueError(f"Changed original feature/episode/June evidence: {source}")
for name, folder in studies.items():
    plan = read(folder / "plan.json")
    for category in ("source_hashes", "code_hashes"):
        for source, digest in plan[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed {category}: {source}")
    plans[name] = plan
ordered_folder = Path("data/processed/ordered-states-v17")
ordered_build = read(ordered_folder / "build.json")
ordered_manifests = [ordered_build] + [read(p) for p in sorted(ordered_folder.glob("transitions-*.json"))]
for manifest in ordered_manifests:
    for category in ("inputs", "outputs"):
        for source, digest in manifest[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed ordered-feature provenance: {source}")
new_weights = {}
for name in ("v15", "v16", "v17"):
    for meta_path in studies[name].rglob("fit.json"):
        meta = read(meta_path)
        weights = meta_path.parent / "model.cbm"
        if sha256(weights) != meta["model_sha256"]:
            raise ValueError(f"Changed research weights: {weights}")
        new_weights[str(weights)] = meta["model_sha256"]
version = active_version()
load_bundle(version)
if version != "op-d87946a7b7fd":
    raise ValueError("Serving model changed during non-promoting research")
report = {
    "created_at": datetime.now(UTC).isoformat(),
    "environment": {
        "python": python_version(),
        **{
            p: package_version(p)
            for p in ("catboost", "numpy", "pandas", "pyarrow", "scikit-learn", "duckdb")
        },
    },
    "goal_interpretation": "Both event precision and recall>=90%; user clarification pending. Accuracy and precision-only are not success substitutes.",
    "scope": "Completed adaptive retrospective research, not an independent final-weight test. Overall solution includes all incident types.",
    "active_version": version,
    "serving_weights_changed": False,
    "plans": plans,
    "access_v11": v11,
    "access_v12": v12,
    "access_v14": v14,
    "other_count_heads_v13": v13["heads"],
    "waiting_time_v15": read(studies["v15"] / "report.json"),
    "near_term_capacity_v16": read(studies["v16"] / "report.json"),
    "ordered_states_v17": read(studies["v17"] / "report.json"),
    "ordered_feature_build": ordered_build,
    "ordered_feature_raw_provenance_verified": True,
    "additional_research_weights_verified": new_weights,
    "recurrence_regimes": read(Path("artifacts/recurrence_regime_audit.json")),
    "fault_precursor_cases": read(Path("artifacts/fault_precursor_audit.json")),
    "ordered_uncertainty": read(Path("artifacts/ordered_state_uncertainty.json")),
    "uncertainty": read(Path("artifacts/goal90_uncertainty.json")),
    "flood": {"status": "unsupported_too_few_episodes", "target_achievement_claimed": False},
    "baseline_access_five_months": read(Path("artifacts/access_additional_validation.json"))[
        "five_period_pooled"
    ],
    "errors": read(Path("artifacts/goal90_error_diagnostics.json"))["totals"],
    "source_and_code_hashes_verified": True,
    "original_features_episodes_and_june_hashes_verified": original_hashes,
    "new_blind_test": False,
    "goal_achieved": False,
    "promotion_decision": "Retain current bundle. V11 passes its intermediate research gates but falls well short of90/90 and lowers precision; December false-warning workload rises. Conditional F1 gain intervals include zero. V12,V14,V15,V16 fail their predeclared gates; V17 decisions are recorded separately per kind. No experimental gate automatically activates weights.",
    "limitations": "No individual-head result can establish 90/90 for the entire solution. F1 gains involving lower precision are recorded explicitly. No candidate is promoted automatically.",
}
target = Path("artifacts/goal90_research_report.json")
write_json(target, report)
print("Saved", target, "active", version, "goal achieved", report["goal_achieved"])
