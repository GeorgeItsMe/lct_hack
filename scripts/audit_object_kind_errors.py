"""Describe catalog object-kind errors without removing any hard events."""

from pathlib import Path

import pandas as pd

from moscollector.alert_diagnostics import EventEvaluator
from moscollector.goal90_research import STRESS, pooled, read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.research import FOLDS

objects_path, channels_path = PROCESSED / "objects.parquet", PROCESSED / "channels.parquet"
episodes_path = PROCESSED / "episodes.parquet"
objects = pd.read_parquet(objects_path)
channels = pd.read_parquet(channels_path)
episodes = pd.read_parquet(episodes_path, filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
sources = {objects_path, channels_path, episodes_path}
rows = []
specs = [("v20_access", "access", fold) for fold in (*FOLDS, *STRESS)] + [
    ("v24_count_control", kind, fold)
    for kind in ("access", "fire", "fault")
    for fold in ("screen_1", "screen_2")
]
for family, kind, fold in specs:
    folder = (
        Path("artifacts/research-v20/access") / fold
        if family == "v20_access"
        else Path("artifacts/research-v24") / kind / fold
    )
    prediction = folder / (
        "quarter_candidate-test.parquet"
        if family == "v20_access"
        else "count_probability_control-test.parquet"
    )
    result = folder / "result.json"
    sources.update((prediction, result))
    pred = pd.read_parquet(prediction).merge(
        objects[["object_id", "object_kind"]], on="object_id", validate="many_to_one", how="left"
    )
    assert pred.object_kind.notna().all()
    expected = read(result)["scores" if family == "v20_access" else "control"]
    parts = []
    for group, view in pred.groupby("object_kind"):
        view = view.reset_index(drop=True)
        eps = episodes[episodes.kind.eq(kind) & episodes.object_id.isin(view.object_id)]
        m = EventEvaluator(view, eps, 0.25).evaluate(view.alert, 0.5, 0.25)
        parts.append(m)
        rows.append(
            {
                "family": family,
                "kind": kind,
                "fold": fold,
                "object_kind": group,
                "true_alerts": m["true_alerts"],
                "alerts": m["alerts"],
                "eligible_episodes": m["eligible_episodes"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "objects": view.object_id.nunique(),
            }
        )
    totals = pooled(parts)
    assert all(totals[k] == expected[k] for k in ("true_alerts", "alerts", "eligible_episodes"))
summary = []
for (family, kind, group), selected in pd.DataFrame(rows).groupby(["family", "kind", "object_kind"]):
    summary.append(
        {"family": family, "kind": kind, "object_kind": group, **pooled(selected.to_dict("records"))}
    )
catalog = channels.merge(
    objects[["object_id", "object_kind"]], on="object_id", validate="many_to_one", how="left"
)
report = {
    "scope": "Descriptive audit of already-used historical evaluations; not independent validation or post-hoc subgroup success. Full solution must still cover every object and incident kind.",
    "catalog_limits": "Object-kind labels already existed as original model inputs. Sensor counts describe the supplied static catalog, not historical install dates or causal event attribution to sensor types.",
    "summary": summary,
    "periods": rows,
    "catalog_sensor_counts": catalog.groupby(["object_kind", "sensor_type"])
    .size()
    .rename("channels")
    .reset_index()
    .to_dict("records"),
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(EventEvaluator.__init__.__code__.co_filename),
            Path(pooled.__code__.co_filename),
        )
    },
}
write_json(Path("artifacts/object_kind_error_audit.json"), report)
print(pd.DataFrame(summary).to_string(index=False))
