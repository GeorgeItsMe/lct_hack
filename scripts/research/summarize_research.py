"""Publish reproducible research comparisons without modifying frozen predictions."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from moscollector.model_registry import active_version
from moscollector.prepare import write_json
from moscollector.train import alert_metrics
from moscollector.uncertainty import COUNT_COLUMNS, bootstrap

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
v4 = ARTIFACTS / "research-v4"
v5 = ARTIFACTS / "research-v5"


def read(path):
    return json.loads(path.read_text())


def sensitivity(kind, variant, expected):
    episodes = pd.read_parquet(
        ROOT / "data/processed/episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
    )
    episodes = episodes[episodes.kind.eq(kind)]
    counts = []
    for fold in ("screen_1", "screen_2", "confirmation"):
        suffix = "-blend" if variant == "blend" else ""
        candidate = pd.read_parquet(v5 / fold / f"{kind}{suffix}-predictions.parquet")
        reference = pd.read_parquet(v4 / fold / "reference" / f"{kind}-predictions.parquet")
        if (
            not candidate[["object_id", "as_of"]]
            .reset_index(drop=True)
            .equals(reference[["object_id", "as_of"]].reset_index(drop=True))
        ):
            raise ValueError("Paired comparison does not share forecast opportunities")
        candidate_threshold = read(v5 / fold / f"{kind}{suffix}.json")["threshold"]
        reference_threshold = read(v4 / fold / "reference" / f"{kind}.json")["threshold"]
        for obj, group in candidate.groupby("object_id"):
            events = episodes[episodes.object_id.eq(obj)]
            c = alert_metrics(group, events, candidate_threshold)
            r = alert_metrics(reference[reference.object_id.eq(obj)], events, reference_threshold)
            if c["eligible_episodes"] != r["eligible_episodes"]:
                raise ValueError("Eligible episode counts differ")
            counts.append(
                {
                    "object_id": obj,
                    "model_tp": c["true_alerts"],
                    "model_alerts": c["alerts"],
                    "episodes": c["eligible_episodes"],
                    "baseline_tp": r["true_alerts"],
                    "baseline_alerts": r["alerts"],
                }
            )
    counts = pd.DataFrame(counts).groupby("object_id")[COUNT_COLUMNS].sum().reset_index()
    c, r = expected["pooled_candidate"], expected["pooled_reference"]
    if counts[COUNT_COLUMNS].sum().tolist() != [c["tp"], c["alerts"], c["events"], r["tp"], r["alerts"]]:
        raise ValueError("Cluster totals do not match research report")
    objects = pd.read_parquet(ROOT / "data/processed/objects.parquet")
    parents = (
        counts.merge(objects[["object_id", "parent_id"]], on="object_id", validate="one_to_one")
        .groupby("parent_id")[COUNT_COLUMNS]
        .sum()
    )
    return {
        "scope": "conditional_post_selection_sensitivity_not_new_validation",
        "by_parent": bootstrap(parents.to_numpy()),
        "notes": "Paired histories across all three periods; fixed models and thresholds. Does not account for model selection or uncertainty on future months.",
    }


selection = read(ARTIFACTS / "model_selection.json")
for kind, meta in selection["models"].items():
    digest = hashlib.sha256((ARTIFACTS / "models" / f"{kind}.cbm").read_bytes()).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"Frozen model changed: {kind}")
comparison = read(v5 / "comparison.json")
if set(comparison) != {"fault", "fire", "access"}:
    raise ValueError("Research has not finished")
models = {}
for kind, result in comparison.items():
    rows = []
    for fold, label in (
        ("screen_1", "Ноябрь 2025"),
        ("screen_2", "Февраль 2026"),
        ("confirmation", "Май 2026"),
    ):
        reference = read(v4 / fold / "reference" / f"{kind}.json")["test"]
        context_path = v4 / fold / "context" / f"{kind}.json"
        rows.append(
            {
                "period": label,
                "reference": reference["f1"],
                "context": read(context_path)["test"]["f1"] if context_path.exists() else None,
                "regularized": result["variants"]["regularized"]["periods"][fold]["candidate"]["f1"],
                "blend": result["variants"]["blend"]["periods"][fold]["candidate"]["f1"],
                "eligible_episodes": reference["eligible_episodes"],
            }
        )
    models[kind] = {"selected": result["selected"], "rows": rows, "variants": result["variants"]}
    if result["selected"] != "reference":
        models[kind]["uncertainty"] = sensitivity(
            kind, result["selected"], result["variants"][result["selected"]]
        )
report = {
    "scope": "adaptive_retrospective_not_new_blind_test",
    "deployed_changed": active_version() != "legacy",
    "operational_model_version": active_version(),
    "external_training_data_added": False,
    "original_report_sha256": hashlib.sha256((ARTIFACTS / "evaluation_report.json").read_bytes()).hexdigest(),
    "plans": {"context": read(v4 / "plan.json"), "stability": read(v5 / "plan.json")},
    "screen_selection": read(v4 / "selection.json"),
    "context_confirmation": read(v4 / "confirmation.json"),
    "models": models,
}
write_json(ARTIFACTS / "research_report.json", report)
print({kind: value["selected"] for kind, value in models.items()})
