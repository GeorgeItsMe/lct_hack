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
    "v18": Path("artifacts/research-v18"),
    "v19": Path("artifacts/research-v19"),
    "v20": Path("artifacts/research-v20"),
    "v21": Path("artifacts/research-v21"),
    "v22": Path("artifacts/research-v22"),
    "v23": Path("artifacts/research-v23"),
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
novelty_folder = Path("data/processed/channel-novelty-v18")
novelty_build = read(novelty_folder / "build.json")
for manifest in [novelty_build] + [read(p) for p in sorted(novelty_folder.glob("onsets-*.json"))]:
    for category in ("inputs", "outputs"):
        for source, digest in manifest[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed channel-novelty provenance: {source}")
new_weights = {}
quarter_folder = Path("data/processed/quarter-counts-v21")
quarter_build = read(quarter_folder / "build.json")
quarter_manifests = [read(p) for p in sorted(quarter_folder.glob("counts-*.json"))]
for manifest in [quarter_build, *quarter_manifests]:
    for category in ("inputs", "outputs"):
        for source, digest in manifest[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed raw quarter-count provenance: {source}")
phase_folder = Path("data/processed/phase-augmentation-v23")
phase_build = read(phase_folder / "build.json")
phase_manifests = [read(p) for p in sorted(phase_folder.glob("counts-*.json"))]
for manifest in [phase_build, *phase_manifests]:
    for category in ("inputs", "outputs"):
        for source, digest in manifest[category].items():
            if sha256(Path(source)) != digest:
                raise ValueError(f"Changed phase augmentation provenance: {source}")
for name in ("v15", "v16", "v17", "v18", "v19", "v23"):
    for meta_path in studies[name].rglob("fit.json"):
        meta = read(meta_path)
        weights = meta_path.parent / "model.cbm"
        if sha256(weights) != meta["model_sha256"]:
            raise ValueError(f"Changed research weights: {weights}")
        new_weights[str(weights)] = meta["model_sha256"]
capacity = read(Path("artifacts/hourly_capacity_audit.json"))
for category in ("inputs", "code_hashes"):
    for source, digest in capacity[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed cadence-capacity evidence: {source}")
fine_uncertainty = read(Path("artifacts/fine_cadence_uncertainty.json"))
for category in ("source_hashes", "code_hashes"):
    for source, digest in fine_uncertainty[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed fine-cadence evidence: {source}")
fine_errors = read(Path("artifacts/fine_cadence_error_audit.json"))
for category in ("source_hashes", "code_hashes"):
    for source, digest in fine_errors[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed fine-cadence error audit: {source}")
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
            for p in ("catboost", "numpy", "pandas", "pyarrow", "scikit-learn", "duckdb", "scipy")
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
    "channel_novelty_classifier_v18": read(studies["v18"] / "report.json"),
    "channel_novelty_count_v19": read(studies["v19"] / "report.json"),
    "channel_novelty_uncertainty": read(Path("artifacts/channel_novelty_uncertainty.json")),
    "channel_novelty_recurrence": read(Path("artifacts/channel_novelty_recurrence_audit.json")),
    "channel_novelty_feature_build": novelty_build,
    "channel_novelty_raw_provenance_verified": True,
    "hourly_capacity_diagnostic": capacity,
    "fine_cadence_v20": read(studies["v20"] / "report.json"),
    "fine_cadence_uncertainty": fine_uncertainty,
    "fresh_counts_v21": read(studies["v21"] / "report.json"),
    "fine_quantiles_v22": read(studies["v22"] / "report.json"),
    "raw_quarter_feature_build": quarter_build,
    "raw_quarter_aggregation_manifests": quarter_manifests,
    "raw_quarter_count_provenance_verified": True,
    "phase_training_v23": read(studies["v23"] / "report.json"),
    "phase_training_screen_periods": {
        str(p.relative_to(studies["v23"])): read(p)
        for p in sorted(studies["v23"].glob("*/screen_*/result.json"))
    },
    "phase_augmentation_build": phase_build,
    "phase_augmentation_raw_manifests": phase_manifests,
    "phase_augmentation_provenance_verified": True,
    "fine_cadence_error_audit": fine_errors,
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
    "promotion_decision": "Retain current bundle. V11 passes intermediate gates but falls well short of90/90 and lowers precision. V12,V14,V15,V16,V17,V18 fail final gates. V19 fault passes research gates against its count anchor but precision remains.269 and recall.272, with2/92 quiet-history episodes found. Its positive conditional F1 gain interval does not establish superiority to the deployed classifier or the full goal. V19 access/fire fail screening. V20 access passes all research gates with P.714/R.714/F1.714 on five months, improving both metrics over its recalibrated hourly control; compared with the archived operational family it improves recall but loses precision and adds false alerts. Conditional paired gains do not establish independent future performance. V20 fire fails screening. V21 fresh counts and V22 quarter quantiles slightly improve access screening F1 but do not clear their >5% primary improvement gates; no extra-month evaluation or promotion. V21 fire/fault also fail screening. V23 phase-augmented training gives access P.716/R.735/F1.725 on screen but <5% primary gain over matched old weights; fire loses precision and fault finds only3/46 events. All v23 kinds fail screening. No experimental gate automatically activates weights.",
    "limitations": "No individual-head result can establish 90/90 for the entire solution. F1 gains involving lower precision are recorded explicitly. No candidate is promoted automatically.",
}
target = Path("artifacts/goal90_research_report.json")
write_json(target, report)
print("Saved", target, "active", version, "goal achieved", report["goal_achieved"])
