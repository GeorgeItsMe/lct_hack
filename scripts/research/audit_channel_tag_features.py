"""Direct set-based audit of tag co-activity snapshots without episode labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.experiments.channel_tag_features import COLUMNS, FOLDER, KEYS, SIGNALS, catalog_groups
from moscollector.prepare import sha256, write_json

ROOT = Path("artifacts/research-v41-feature-check")
TIMES = tuple(map(pd.Timestamp, ("2023-01-01", "2024-07-01", "2025-11-01", "2026-05-01")))


def direct(q, history, groups):
    expected = dict.fromkeys(COLUMNS, 0.0)
    for name, g in history.groupby("tag_group", sort=False):
        size = int(groups.loc[groups.tag_group.eq(name), "channel_id"].nunique())
        last6 = g.loc[g.ts.ge(q - pd.Timedelta(hours=6))]
        active = {
            s: {h: set(rows.loc[rows.signal.eq(s), "channel_id"]) for h, rows in ((6, last6), (24, g))}
            for s in SIGNALS
        }
        all6, types6 = set(last6.channel_id), last6.signal.nunique()
        for signal in SIGNALS:
            for hours in (6, 24):
                count = len(active[signal][hours])
                expected[f"tag_{signal}_groups_{hours}h"] += count > 0
                for stat, value in (
                    (f"max_channels_{hours}h", count),
                    (f"max_fraction_{hours}h", count / size),
                ):
                    key = f"tag_{signal}_{stat}"
                    expected[key] = max(expected[key], value)
            if active[signal][6]:
                for stat, value in (
                    ("max_other_channels_6h", len(all6 - active[signal][6])),
                    ("max_signal_types_6h", types6),
                ):
                    key = f"tag_{signal}_{stat}"
                    expected[key] = max(expected[key], value)
    return expected


def run():
    onsets = [FOLDER / f"onsets-{y}.parquet" for y in range(2022, 2027)]
    catalog_path, context_path = FOLDER.parent / "channels.parquet", FOLDER / "context.parquet"
    sources = [*onsets, catalog_path, context_path, FOLDER / "build.json"]
    code = [Path(__file__), Path(catalog_groups.__code__.co_filename)]
    plan = {
        "scope": "Implementation audit of64features, no labels or performance claims.",
        "times": list(map(str, TIMES)),
        "rtol": 1e-7,
        "atol": 1e-7,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(p): sha256(p) for p in code},
    }
    path = ROOT / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Tag snapshot audit input/code changed")
    if not path.exists():
        write_json(path, plan)
    stored = pd.read_parquet(context_path, filters=[("as_of", "in", list(TIMES))])
    groups = catalog_groups(pd.read_parquet(catalog_path))
    rows, residuals = [], []
    for at in TIMES:
        history = pd.read_parquet(
            onsets, filters=[("ts", ">=", at - pd.Timedelta(hours=24)), ("ts", "<", at)]
        )
        history = history.merge(
            groups[["channel_id", "tag_group"]], on="channel_id", how="left", validate="many_to_one"
        )
        for q in stored.loc[stored.as_of.eq(at)].itertuples(index=False):
            values = direct(
                at,
                history.loc[history.object_id.eq(q.object_id)],
                groups.loc[groups.object_id.eq(q.object_id)],
            )
            actual = np.array([getattr(q, c) for c in COLUMNS])
            reference = np.array([values[c] for c in COLUMNS])
            np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-7)
            residuals.append(float(np.max(np.abs(actual - reference), initial=0)))
            rows.append({"object_id": q.object_id, "as_of": at, **values})
        print("DIRECT TAG VERIFIED", at, len(history), flush=True)
    table = pd.DataFrame(rows, columns=[*KEYS, *COLUMNS])
    target = ROOT / "direct-values.parquet"
    table.to_parquet(target, index=False)
    build = json.loads((FOLDER / "build.json").read_text())
    report = {
        **plan,
        "plan_sha256": sha256(path),
        "status": "direct_onset_snapshots_passed",
        "queries": len(table),
        "objects": int(table.object_id.nunique()),
        "feature_values_compared": len(table) * len(COLUMNS),
        "max_absolute_difference": max(residuals),
        "direct_values_sha256": sha256(target),
        "build_rows": build["rows"],
        "catalog_support": build["catalog_support"],
        "features": COLUMNS,
        "limitations": "Direct sets use complete canonical onsets and literal catalog groups, not the feature transform. Original six families match prior full extraction; new water/pump entries have predicate tests. This does not verify physical adjacency or every feature row, and no incident labels are read.",
        "serving_changed": False,
        "goal_achieved": False,
    }
    write_json(Path("artifacts/channel_tag_feature_audit.json"), report)
    return report


if __name__ == "__main__":
    print(run()["status"])
