"""Publish completed 90/90 research evidence, without altering serving quality."""

from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from platform import python_version

import pandas as pd

from moscollector.event_trigger_grid import trigger_alerts, trigger_grid
from moscollector.fine_cadence_research import cohort, evaluator_for
from moscollector.goal90_research import pooled, read
from moscollector.minute_cadence_verification import verify as verify_minute_cadence
from moscollector.model_registry import active_version, load_bundle
from moscollector.onset_binary_verification import verify as verify_onset_binary
from moscollector.onset_pending_verification import verify as verify_onset_pending
from moscollector.onset_verification import verify as verify_onset_study
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
    "v24": Path("artifacts/research-v24"),
    "v25": Path("artifacts/research-v25"),
    "v26": Path("artifacts/research-v26"),
    "v27": Path("artifacts/research-v27"),
    "v28": Path("artifacts/research-v28"),
    "v29": Path("artifacts/research-v29"),
    "v30": Path("artifacts/research-v30"),
    "v31": Path("artifacts/research-v31"),
    "v32": Path("artifacts/research-v32"),
    "v33": Path("artifacts/research-v33"),
    "v34": Path("artifacts/research-v34"),
    "v35": Path("artifacts/research-v35"),
    "v36": Path("artifacts/research-v36"),
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
for name in ("v15", "v16", "v17", "v18", "v19", "v23", "v30", "v33", "v35", "v36"):
    for meta_path in studies[name].rglob("fit.json"):
        meta = read(meta_path)
        weights = meta_path.parent / "model.cbm"
        if sha256(weights) != meta["model_sha256"]:
            raise ValueError(f"Changed research weights: {weights}")
        new_weights[str(weights)] = meta["model_sha256"]
probability_weights = {}
for folder in (studies["v24"], studies["v25"] / "controls"):
    for meta_path in folder.glob("*/*/binary/fit.json"):
        meta = read(meta_path)
        weights = Path(meta["weights_file"])
        if sha256(weights) != meta["model_sha256"]:
            raise ValueError(f"Changed probability weights: {weights}")
        probability_weights[str(weights)] = meta["model_sha256"]
        if meta["newly_fitted"]:
            new_weights[str(weights)] = meta["model_sha256"]
conditional_models = {}
for meta_path in studies["v25"].glob("*/*/conditional/fit.json"):
    meta = read(meta_path)
    if meta["mode"] == "catboost":
        weights = meta_path.parent / "model.cbm"
        if sha256(weights) != meta["model_sha256"]:
            raise ValueError(f"Changed conditional count weights: {weights}")
        new_weights[str(weights)] = meta["model_sha256"]
    elif meta["mode"] != "constant" or meta["constant_extra"] < 0:
        raise ValueError(f"Invalid conditional model: {meta_path}")
    conditional_models[str(meta_path)] = meta
object_kind_models = {}
object_kind_metadata = list(studies["v26"].glob("*/*/groups/*/fit.json"))
object_kind_metadata.extend((studies["v27"] / "controls").glob("*/*/groups/*/fit.json"))
for meta_path in object_kind_metadata:
    meta = read(meta_path)
    weights = Path(meta["weights_file"])
    if sha256(weights) != meta["model_sha256"]:
        raise ValueError(f"Changed object-kind weights: {weights}")
    if meta["mode"] == "specialist":
        new_weights[str(weights)] = meta["model_sha256"]
    elif meta["mode"] != "global_fallback":
        raise ValueError(f"Unknown object-kind model mode: {meta_path}")
    object_kind_models[str(meta_path)] = meta
neural_models = {}
neural_plan_hash = sha256(studies["v28"] / "plan.json")
neural_report = read(studies["v28"] / "report.json")
if set(neural_report) != {"access", "fire", "fault"}:
    raise ValueError("Neural final report is incomplete")
expected_neural_fits = set()
for kind, outcome in neural_report.items():
    folds = ["screen_1", "screen_2"]
    if outcome["selection"]["passed_screen"]:
        if len(outcome["periods"]) != 5:
            raise ValueError(f"Neural confirmation is incomplete: {kind}")
        folds.extend(["confirmation", "stress_1", "stress_2"])
    for fold in folds:
        for variant in ("mlp", "gru"):
            expected_neural_fits.add(str(studies["v28"] / kind / fold / "models" / variant / "fit.json"))
for meta_path in sorted(studies["v28"].glob("*/*/models/*/fit.json")):
    meta = read(meta_path)
    signature = meta["signature"]
    weights = meta_path.parent / "model.pt"
    codec = meta_path.parents[2] / "codec.json"
    result = read(meta_path.parents[2] / "result.json")
    if sha256(weights) != meta["model_sha256"]:
        raise ValueError(f"Changed neural weights: {weights}")
    if signature["plan_sha256"] != neural_plan_hash:
        raise ValueError(f"Neural model belongs to a different study: {meta_path}")
    if sha256(codec) != signature["codec_sha256"] or signature["codec_sha256"] != result["codec_sha256"]:
        raise ValueError(f"Changed neural preprocessing: {codec}")
    if signature["device"] != plans["v28"]["device"] or signature["torch"] != plans["v28"]["torch_version"]:
        raise ValueError(f"Changed neural backend: {meta_path}")
    if result["fits"][meta["variant"]] != meta:
        raise ValueError(f"Neural result/fit mismatch: {meta_path}")
    neural_models[str(meta_path)] = meta
if set(neural_models) != expected_neural_fits:
    raise ValueError("Neural weights do not match the predeclared completed stages")
neural_overlap = read(Path("artifacts/neural_complementarity_audit.json"))
for category in ("source_hashes", "code_hashes"):
    for source, digest in neural_overlap[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed neural complementarity audit: {source}")
blend_report = read(studies["v29"] / "report.json")
if set(blend_report) != {"access", "fire", "fault"}:
    raise ValueError("Neural blend report is incomplete")
blend_control_plan = studies["v29"] / "controls/plan.json"
if (
    read(blend_control_plan)["parent_plan_sha256"] != sha256(studies["v29"] / "plan.json")
    or read(blend_control_plan)["frozen_training_source_plan_sha256"] != neural_plan_hash
):
    raise ValueError("Neural blend control plan differs from parent")
blend_neural_models, expected_blend_fits = {}, set()
for kind, outcome in blend_report.items():
    if outcome["selection"]["passed_screen"]:
        if len(outcome["periods"]) != 5:
            raise ValueError(f"Neural blend confirmation incomplete: {kind}")
        for fold in ("confirmation", "stress_1", "stress_2"):
            for variant in ("mlp", "gru"):
                expected_blend_fits.add(
                    str(studies["v29"] / "controls" / kind / fold / "models" / variant / "fit.json")
                )
for meta_path in sorted((studies["v29"] / "controls").glob("*/*/models/*/fit.json")):
    meta = read(meta_path)
    signature = meta["signature"]
    weights = meta_path.parent / "model.pt"
    codec = meta_path.parents[2] / "codec.json"
    result = read(meta_path.parents[2] / "result.json")
    if sha256(weights) != meta["model_sha256"] or signature["plan_sha256"] != sha256(blend_control_plan):
        raise ValueError(f"Changed additional neural control: {meta_path}")
    if sha256(codec) != signature["codec_sha256"] or result["codec_sha256"] != signature["codec_sha256"]:
        raise ValueError(f"Changed neural blend codec: {codec}")
    if signature["device"] != plans["v29"]["device"] or signature["torch"] != plans["v29"]["torch_version"]:
        raise ValueError(f"Changed neural blend backend: {meta_path}")
    if result["fits"][meta["variant"]] != meta:
        raise ValueError(f"Neural blend control/result mismatch: {meta_path}")
    blend_neural_models[str(meta_path)] = meta
if set(blend_neural_models) != expected_blend_fits:
    raise ValueError("Neural blend controls differ from declared completed stages")
peer_build = read(Path("data/processed/peer-context-v30/build.json"))
for category in ("inputs", "outputs", "code_hashes"):
    for source, digest in peer_build[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed cross-object feature provenance: {source}")
peer_report = read(studies["v30"] / "report.json")
if set(peer_report) != {"access", "fire", "fault"}:
    raise ValueError("Peer count study is incomplete")
peer_models, expected_peer_models = {}, set()
for kind, outcome in peer_report.items():
    folds = ["screen_1", "screen_2"]
    if outcome["selection"]["passed_screen"]:
        if len(outcome["periods"]) != 5:
            raise ValueError(f"Peer confirmation incomplete: {kind}")
        folds.extend(["confirmation", "stress_1", "stress_2"])
    expected_peer_models.update(str(studies["v30"] / kind / fold / "peer/fit.json") for fold in folds)
for meta_path in sorted(studies["v30"].glob("*/*/peer/fit.json")):
    meta = read(meta_path)
    if meta["signature"]["plan_sha256"] != sha256(studies["v30"] / "plan.json"):
        raise ValueError(f"Peer weights belong to another study: {meta_path}")
    if meta["added_columns"] != peer_build["columns"] or len(meta["added_columns"]) != 64:
        raise ValueError(f"Unexpected peer feature set: {meta_path}")
    if read(meta_path.parents[1] / "result.json")["fit"] != meta:
        raise ValueError(f"Peer fit/result mismatch: {meta_path}")
    peer_models[str(meta_path)] = meta
if set(peer_models) != expected_peer_models:
    raise ValueError("Peer weights differ from the declared completed stages")
selfsup_root = studies["v31"]
selfsup_report = read(selfsup_root / "report.json")
if set(selfsup_report) != {"access", "fire", "fault"}:
    raise ValueError("Self-supervised report is incomplete")
selfsup_plan_hash = sha256(selfsup_root / "plan.json")
selfsup_control_plan = selfsup_root / "controls/plan.json"
if (
    read(selfsup_control_plan)["parent_plan_sha256"] != selfsup_plan_hash
    or read(selfsup_control_plan)["frozen_training_source_plan_sha256"] != neural_plan_hash
):
    raise ValueError("Self-supervised control plan differs from its sources")
selfsup_models, selfsup_controls = {}, {}
expected_selfsup_fits, expected_selfsup_controls = set(), set()
for kind, outcome in selfsup_report.items():
    folds = ["screen_1", "screen_2"]
    if outcome["selection"]["passed_screen"]:
        if len(outcome["periods"]) != 5:
            raise ValueError(f"Self-supervised confirmation incomplete: {kind}")
        folds.extend(["confirmation", "stress_1", "stress_2"])
    for fold in folds:
        directory = selfsup_root / kind / fold
        result = read(directory / "result.json")
        aggregate = read(directory / "fit.json")
        source = Path(result["source_control_directory"])
        control = read(source / "result.json")
        if result["fit"] != aggregate or sha256(directory / "codec.json") != sha256(source / "codec.json"):
            raise ValueError(f"Changed self-supervised aggregate or codec: {directory}")
        if not all(
            aggregate[key]
            for key in (
                "same_count_architecture",
                "same_rows_and_codec_as_control",
                "only_history_weights_transferred",
            )
        ):
            raise ValueError(f"Self-supervised comparability check failed: {directory}")
        for stage in ("pretext", "count"):
            meta_path = directory / "models" / stage / "fit.json"
            expected_selfsup_fits.add(str(meta_path))
            meta = read(meta_path)
            signature = meta["signature"]
            if (
                signature["plan_sha256"] != selfsup_plan_hash
                or signature["kind"] != kind
                or signature["fold"] != fold
                or signature["stage"] != stage
                or meta["stage"] != stage
                or signature["codec_sha256"] != aggregate["codec_sha256"]
                or signature["codec_sha256"] != sha256(directory / "codec.json")
                or signature["device"] != plans["v31"]["device"]
                or signature["torch"] != plans["v31"]["torch_version"]
                or signature["sizes"] != control["fits"]["gru"]["sizes"]
                or signature["count_config"] != control["fits"]["gru"]["config"]
                or sha256(meta_path.parent / "model.pt") != meta["model_sha256"]
                or aggregate[stage] != meta
            ):
                raise ValueError(f"Changed self-supervised stage: {meta_path}")
            selfsup_models[str(meta_path)] = meta
        count, pretext = aggregate["count"], aggregate["pretext"]
        if (
            count["signature"]["pretext_model_sha256"] != pretext["model_sha256"]
            or count["signature"]["pretext_fit_sha256"] != sha256(directory / "models/pretext/fit.json")
            or count["parameter_count"] != control["fits"]["gru"]["parameter_count"]
        ):
            raise ValueError(f"Broken pretext-to-count transfer provenance: {directory}")
        if source.is_relative_to(selfsup_root / "controls"):
            for variant in ("mlp", "gru"):
                expected_selfsup_controls.add(str(source / "models" / variant / "fit.json"))
actual_selfsup_fits = {str(p) for p in selfsup_root.glob("*/*/models/*/fit.json")}
if actual_selfsup_fits != expected_selfsup_fits:
    raise ValueError("Self-supervised stages differ from declared completed periods")
for meta_path in sorted((selfsup_root / "controls").glob("*/*/models/*/fit.json")):
    meta = read(meta_path)
    signature = meta["signature"]
    result = read(meta_path.parents[2] / "result.json")
    if (
        signature["plan_sha256"] != sha256(selfsup_control_plan)
        or signature["codec_sha256"] != sha256(meta_path.parents[2] / "codec.json")
        or signature["codec_sha256"] != result["codec_sha256"]
        or signature["device"] != plans["v31"]["device"]
        or signature["torch"] != plans["v31"]["torch_version"]
        or sha256(meta_path.parent / "model.pt") != meta["model_sha256"]
        or result["fits"][meta["variant"]] != meta
    ):
        raise ValueError(f"Changed self-supervised scratch control: {meta_path}")
    selfsup_controls[str(meta_path)] = meta
if set(selfsup_controls) != expected_selfsup_controls:
    raise ValueError("Self-supervised scratch controls differ from declared completed periods")
selfsup_error_audit = read(Path("artifacts/self_supervised_fault_error_audit.json"))
for category in ("source_hashes", "code_hashes"):
    for source, digest in selfsup_error_audit[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed self-supervised fault error audit: {source}")
for variant in ("pretrained", "global_control", "scratch_gru"):
    if (
        selfsup_error_audit["totals"][variant]
        != selfsup_report["fault"]["five_period_pooled"][variant]["true_alerts"]
    ):
        raise ValueError("Self-supervised audit differs from the final event totals")
trigger_root = studies["v32"]
trigger_report = read(trigger_root / "report.json")
if (
    trigger_report["plan_sha256"] != sha256(trigger_root / "plan.json")
    or trigger_report["source_hashes"] != plans["v32"]["source_hashes"]
    or trigger_report["code_hashes"] != plans["v32"]["code_hashes"]
    or {p["fold"] for p in trigger_report["periods"]}
    != {"screen_1", "screen_2", "confirmation", "stress_1", "stress_2"}
    or len(trigger_report["periods"]) != 5
):
    raise ValueError("Minute-trigger audit is incomplete or belongs to another plan")
trigger_onsets = pd.concat(
    [
        pd.read_parquet(
            f"data/processed/channel-novelty-v18/onsets-{year}.parquet",
            filters=[("signal", "==", "fire"), ("ts", "<", pd.Timestamp("2026-06-01"))],
        )
        for year in (2025, 2026)
    ],
    ignore_index=True,
)
trigger_episodes = pd.read_parquet(
    "data/processed/episodes.parquet",
    filters=[("kind", "==", "fault"), ("start_ts", "<", pd.Timestamp("2026-06-01"))],
)
trigger_slot_hashes, trigger_case_keys = {}, set()
for period in trigger_report["periods"]:
    fold = period["fold"]
    if read(trigger_root / fold / "result.json") != period:
        raise ValueError("Minute-trigger period/report mismatch")
    old = pd.read_parquet(selfsup_root / "fault" / fold / "pretrained-test.parquet")
    hourly = old.loc[old.as_of.eq(old.as_of.dt.floor("h")), ["object_id", "as_of"]].reset_index(drop=True)
    original_cohort = cohort(hourly, trigger_episodes, 1)
    if len(hourly) / 24 != period["exposure_days"]:
        raise ValueError("Trigger grid changed exposure denominator")
    for seconds in plans["v32"]["resolutions_seconds"]:
        target = trigger_root / fold / f"slots-{seconds}.parquet"
        regenerated = trigger_grid(hourly, trigger_onsets, seconds)
        stored = pd.read_parquet(target)
        pd.testing.assert_frame_equal(regenerated, stored)
        if cohort(stored, trigger_episodes, seconds / 3600) != original_cohort:
            raise ValueError("Trigger grid changed eligible episode identities")
        trigger_slot_hashes[str(target)] = sha256(target)
        for obj, events in original_cohort.items():
            trigger_case_keys.update(
                (fold, obj, pd.Timestamp(event).isoformat(), seconds) for event in events
            )
        for cooldown in plans["v32"]["cooldowns_hours"]:
            alerts = trigger_alerts(stored, cooldown)
            actual = evaluator_for(
                stored, trigger_episodes, seconds / 3600, period["exposure_days"]
            ).evaluate(alerts, 0.5, max(seconds / 3600, cooldown))
            saved = period["arms"][f"release_{seconds}s_cooldown_{cooldown}h"]
            if actual != saved["scores"] or saved["within_original_fp_budget"] != (
                actual["false_alerts_per_object_day"] <= 0.25
            ):
                raise ValueError("Trigger-warning rule/metric replay failed")
for arm, actual in trigger_report["pooled"].items():
    if actual != pooled([p["arms"][arm]["scores"] for p in trigger_report["periods"]]):
        raise ValueError("Trigger pooled metrics differ from period counts")
actual_keys = {
    (c["fold"], c["object_id"], c["start_ts"], c["resolution_seconds"])
    for c in trigger_report["case_diagnostic"]
}
if actual_keys != trigger_case_keys or len(actual_keys) != len(trigger_report["case_diagnostic"]):
    raise ValueError("Trigger case audit lost or duplicated original events")
for seconds in plans["v32"]["resolutions_seconds"]:
    cases = [c for c in trigger_report["case_diagnostic"] if c["resolution_seconds"] == seconds]
    actual = {
        "eligible_episodes": len(cases),
        "member_fire_onset_visible_strictly_before_fault": sum(
            c["member_fire_onset_visible_strictly_before_fault"] for c in cases
        ),
    }
    if actual != trigger_report["precursor_availability"][str(seconds)]:
        raise ValueError("Trigger precursor totals differ from cases")
    if any(
        (not (0 < c["max_lead_minutes"] < 1440))
        if c["member_fire_onset_visible_strictly_before_fault"]
        else c["max_lead_minutes"] is not None
        for c in cases
    ):
        raise ValueError("Invalid precursor timing")
object_kind_audit = read(Path("artifacts/object_kind_error_audit.json"))
for category in ("source_hashes", "code_hashes"):
    for source, digest in object_kind_audit[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed object-kind audit: {source}")
kind_policy_path = studies["v26"] / "access/screen_2/specialist_candidate-test.parquet"
kind_result_path = studies["v26"] / "access/screen_2/result.json"
kind_predictions = pd.read_parquet(kind_policy_path)
kind_house = kind_predictions.loc[kind_predictions.object_kind.eq("controlHouse")]
kind_policy = read(kind_result_path)["arms"]["specialist_candidate"]["policy"]
kind_policy_diagnostic = {
    "scope": "Descriptive used-test prediction audit, not a newly selected threshold.",
    "rows": len(kind_house),
    "warnings": int(kind_house.alert.sum()),
    "max_expected_count": float(kind_house.expected_count.max()),
    "max_effective_capacity": float(kind_house.expected_count.max() * kind_policy["capacity"]),
    "minimum_capacity_before_any_pending_warning": kind_policy["margin"],
    "source_hashes": {str(p): sha256(p) for p in (kind_policy_path, kind_result_path)},
}
assert kind_policy_diagnostic["warnings"] == 0
assert kind_policy_diagnostic["max_effective_capacity"] < kind_policy["margin"]
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
onset_verified = verify_onset_study(studies["v33"])
minute_cadence_verified = verify_minute_cadence(studies["v34"])
onset_binary_verified = verify_onset_binary(studies["v35"])
onset_pending_verified = verify_onset_pending(studies["v36"], onset_binary_verified, minute_cadence_verified)
onset_binary_errors = read(studies["v35"] / "error-audit.json")
for category in ("source_hashes", "code_hashes"):
    for source, digest in onset_binary_errors[category].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Changed binary onset error audit: {source}")
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
    "binary_gate_v24": read(studies["v24"] / "report.json"),
    "binary_gate_screen_periods": {
        str(p.relative_to(studies["v24"])): read(p)
        for p in sorted(studies["v24"].glob("*/screen_*/result.json"))
    },
    "probability_weights_verified": probability_weights,
    "two_part_count_v25": read(studies["v25"] / "report.json"),
    "two_part_count_screen_periods": {
        str(p.relative_to(studies["v25"])): read(p)
        for p in sorted(studies["v25"].glob("*/screen_*/result.json"))
    },
    "conditional_count_models_verified": conditional_models,
    "object_kind_error_audit": object_kind_audit,
    "object_kind_experts_v26": read(studies["v26"] / "report.json"),
    "object_kind_screen_periods": {
        str(p.relative_to(studies["v26"])): read(p)
        for p in sorted(studies["v26"].glob("*/screen_*/result.json"))
    },
    "object_kind_models_verified": object_kind_models,
    "group_policy_v27": read(studies["v27"] / "report.json"),
    "group_policy_screen_periods": {
        str(p.relative_to(studies["v27"])): read(p)
        for p in sorted(studies["v27"].glob("*/screen_*/result.json"))
    },
    "neural_count_v28": neural_report,
    "onset_count_v33": onset_verified,
    "minute_cadence_v34": minute_cadence_verified,
    "onset_binary_v35": onset_binary_verified,
    "onset_pending_v36": onset_pending_verified,
    "onset_pending_decision": "V36 combines frozen v35 occurrence probabilities with the exact old minute-grid count capacity and delayed pending-state policy, without new trees. All six matched count controls reproduce archived forecasts, policies, complete456-option frontiers and metrics. Screening access1922/2771/2667 P.694/R.721 adds27 true and34 false alerts versus pending control; precision falls. Fire140/278/288 P.504/R.486 improves pending control but versus direct classifier adds7 true and36 false alerts, loses F1 despite a5.26% primary gain and fails the no-F1-loss guard. Fault10/27/46 P.370/R.217 removes15 false alerts at cost of1 true; recall/primary decline. All fail, so no additional periods or model fits. Motivation audit describes higher December row AP and353/404 fire episodes with some above-threshold prior opportunity versus86 direct true alerts; shared opportunities mean this is not attainable recall or evidence that cooldown alone causes misses. No activation, June reuse or90/90 claim.",
    "onset_binary_error_audit": onset_binary_errors,
    "minute_fire_precursor_v32": trigger_report,
    "minute_trigger_slot_hashes_verified": trigger_slot_hashes,
    "minute_trigger_decision": "V32 is a fixed-rule timing audit, not a newly trained model. All8 arms are reported across the same217 faults. One-minute release makes past fire onsets in hindsight-known member channels available before42 faults versus7 at hourly release;35 additional cases, none lost. This membership audit is not a predictor or a predictability bound. Causal minute rule using ALL fire onset channels yields67/5125/217 P.013/R.309 without suppression, or33/264/217 P.125/R.152 with24h cooldown. Corresponding hourly rules yield38/737/217 and19/266/217. Rapid raw signals are worth evaluating with a learned gate; naive firing is rejected because precision remains low and unrestricted minute alerts exceed the original FP budget in4/5 periods. Unknown transport latency limits one-second sensitivity results. No model or warning policy activated, no June reuse, full90/90 not reached.",
    "self_supervised_v31": selfsup_report,
    "self_supervised_stages_verified": selfsup_models,
    "self_supervised_scratch_controls_verified": selfsup_controls,
    "self_supervised_fault_error_audit": selfsup_error_audit,
    "self_supervised_decision": "V31 trains9 telemetry predictors and9 identically sized GRU count networks, with only history weights transferred and fresh count optimizers. Access fails screen1983/2851/2667 P.696/R.744/F1.719; fire improves on scratch GRU145/341/288 P.425/R.503/F1.461 but fails CatBoost/anchor gates. Fault passes screen13/28/46 P.464/R.283, then fails May0/15/29 and stress December5/31/106, March0/0/36. Five months18/74/217 P.243/R.083/F1.124 versus CatBoost58/223/217 P.260/R.267/F1.264;109 fewer false warnings cost40 true warnings and both metrics decline. Reused3 v29 scratch GRU controls; no additional controls fitted. Fault error audit finds4/92 quiet-history episodes versus1/92 control, but only13/92 within-day recurrence versus49/92. All original events remain included. No activation, June reuse, substitution, or90/90 claim; telemetry reconstruction improvement is not forecast quality, and extra pretraining compute is not matched to scratch training.",
    "self_supervised_screen_periods": {
        str(p.relative_to(selfsup_root)): read(p) for p in sorted(selfsup_root.glob("*/screen_*/result.json"))
    },
    "peer_context_v30": peer_report,
    "peer_context_feature_build": peer_build,
    "peer_context_models_verified": peer_models,
    "peer_context_decision": "V30 adds64 same-time strictly-past cross-object report features while retaining original66/94/144 columns and fitting6 Poisson models under the original protocol. New peer inputs exactly match on151632 shared original/dense rows and preserve all old columns. Access1942/2803/2667 P.693/R.728/F1.710 adds49 true and51 false warnings versus control. Fire133/277/288 P.480/R.462/F1.471 adds2 true and removes17 false, but primary improvement2.94% is below predeclared>5%. Fault12/99/46 P.121/R.261/F1.166 adds1 true and24 false. All fail screen; no additional-month evaluation, feature substitution or activation. Historical static catalog and proxy-label limitations remain; full-scope90/90 is not reached.",
    "peer_context_screen_periods": {
        str(p.relative_to(studies["v30"])): read(p)
        for p in sorted(studies["v30"].glob("*/screen_*/result.json"))
    },
    "neural_complementarity_audit": neural_overlap,
    "neural_blend_v29": blend_report,
    "neural_blend_screen_periods": {
        str(p.relative_to(studies["v29"])): read(p)
        for p in sorted(studies["v29"].glob("*/screen_*/result.json"))
    },
    "neural_blend_control_models_verified": blend_neural_models,
    "neural_blend_decision": "V29 fixes three convex probability/count pools before evaluation; no fitted mixing weights. Access/fire fail screening. Fault CatBoost/MLP passes screen8/34/46 and May7/34/29, but fails stress: December17/35/106 versus44/131/106 control, March0/2/36 versus0/0/36. Five months32/105/217 P.305/R.147/F1.199 versus58/223/217 P.260/R.267/F1.264;92 fewer false warnings cost26 true warnings. No activation, variant substitution or90/90 claim. Six additional neural models were trained only for declared extra-month component comparisons, preserving the v28 report.",
    "neural_count_screen_periods": {
        str(p.relative_to(studies["v28"])): read(p)
        for p in sorted(studies["v28"].glob("*/screen_*/result.json"))
    },
    "neural_models_verified": neural_models,
    "neural_compute_preflight": read(Path("artifacts/neural_compute_preflight.json")),
    "neural_count_decision": "V28 trains12 MLP/GRU count models in an optional PyTorch environment on local MPS. All six frozen CatBoost controls replay v26. Access GRU2002/2886/2667 P.694/R.751/F1.721 adds109 true and74 false warnings versus matched control; precision and primary criterion decline. Fire GRU143/376/288 P.380/R.497/F1.431 loses precision/F1. Fault GRU11/226/46 P.049/R.239/F1.081 adds152 false warnings with no true-warning gain; MLP finds5/46. Both variants fail screening for every kind, so no extra-month evaluation or activation. Current-only MLP is included, but architectures are not parameter-matched. Best fully checked access remains v20; full-scope90/90 is not reached.",
    "neural_verification": {
        "base_suite": "161 passed, 2 skipped (optional PyTorch modules)",
        "neural_suite": "175 passed, including occurrence/count alignment and delayed pending rearming, explicitly weighted classifier PRAUC, count-independent direct warnings and all-reference/per-month guards, strictly past channel context, preserved snapshot mass/event cohorts, delayed confirmation and interrupted pretext/count checkpoint resume on CPU and MPS",
        "scope": "Code and provenance checks, not evidence of forecast quality. Base environment unchanged; optional PyTorch dependency stays outside serving requirements.",
    },
    "object_kind_policy_diagnostic": kind_policy_diagnostic,
    "object_kind_decision": "V26 trains12 separate count models for the two original catalog object kinds, with a matched group-calibration control. All global controls exactly reproduce v24. On screen access P.685/R.726/F1.705 adds only1 true alert versus calibration alone, with precision below global control. Fire P.405/R.410 loses both metrics. Fault P.212/R.152 finds7/46 rather than11, losing recall/F1. All fail screening; no extra-month evaluation or activation. Subgroup diagnostic gains are not full-scope success.",
    "tested_group_policy_hypothesis": "V26 access screen_2 specialist maximum expected count in controlHouse is.76576489988272, below its shared capacity1/margin1 policy. V27 tested separate catalog-group policies on both frozen global and specialist forecasts, using only preceding policy periods and full event denominators. Access/fire fail screening. Fault global weights pass screening but fail May and stress confirmation: five months53/165/217 P.321/R.244/F1.277 versus matched shared58/223/217 P.260/R.267/F1.264.53 fewer false warnings cost5 true warnings; no activation or90/90 claim. Six additional expert models were fitted only to complete declared matched controls.",
    "latest_probability_studies_decision": "V24 separate binary gate fails all screening gates: access loses precision, fire primary gain4.94% is below predeclared>5%, fault fails its historical count anchor. V25 adds six conditional extra-count models: access P.696/R.736/F1.715 but only1.01% primary gain over binary control and precision below count control; fire loses precision; fault P.084/R.174. All fail screening, so no additional-month evaluation or activation. Expanded-grid control improvements are not attributed to the binary model. Best fully checked access candidate remains v20, not90/90.",
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
