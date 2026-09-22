"""Independently compare fixed numeric snapshots directly with raw past records."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from moscollector.numeric_channel_features import COLUMNS, FAMILIES, FOLDER, KEYS, STATS
from moscollector.prepare import sha256, write_json

TIMES = tuple(map(pd.Timestamp, ("2023-01-01", "2024-07-01", "2025-11-01", "2026-05-01")))
ROOT = Path("artifacts/research-v40-feature-check")


def direct(history, at):
    """Direct raw-record loops, no cached hourly values or rolling transforms."""
    result = {}
    for family, kind in FAMILIES.items():
        seen, logged = 0, 0
        values, ages, shifts = [], [], []
        ranges = {6: [], 24: []}
        for _, h in history.loc[history.sensor_type.eq(kind)].groupby("channel_id"):
            h = h.sort_values("ts")
            last = h.iloc[-1]
            age = (at - last.ts).total_seconds() / 3600
            valid = age <= 168 and pd.notna(last.x)
            logged += age <= 168
            old = h.loc[h.ts.lt(at - pd.Timedelta(hours=24))]
            if valid:
                values.append(float(last.x))
                ages.append(age)
                if len(old):
                    past = old.iloc[-1]
                    old_age = (at - pd.Timedelta(hours=24) - past.ts).total_seconds() / 3600
                    if old_age <= 168 and pd.notna(past.x):
                        shifts.append((last.x - past.x) / (1 + abs(past.x)))
            for hours in (6, 24):
                window = h.loc[h.ts.ge(at - pd.Timedelta(hours=hours)), "x"].dropna()
                if len(window):
                    ranges[hours].append((window.max() - window.min()) / (1 + abs(window.mean())))
                    if hours == 24:
                        seen += 1
        stats = dict.fromkeys(STATS, np.nan)
        stats.update(
            seen_channels_24h=seen,
            logged_channels_168h=logged,
            numeric_channels_168h=len(values),
            shift_supported_channels=len(shifts),
        )
        if values:
            stats.update(
                last_mean=np.mean(values),
                last_min=min(values),
                last_max=max(values),
                last_std=np.std(values),
                last_age_mean_h=np.mean(ages),
            )
        if shifts:
            stats.update(
                relative_shift_24h_max=max(shifts),
                relative_shift_24h_min=min(shifts),
                relative_shift_24h_abs_mean=np.abs(shifts).mean(),
            )
        for hours, observations in ranges.items():
            if observations:
                stats[f"relative_range_{hours}h_max"] = max(observations)
                stats[f"relative_range_{hours}h_mean"] = np.mean(observations)
        result.update({f"num_{family}_{k}": v for k, v in stats.items()})
    return result


def run():
    sources = [FOLDER.parent / f"events-{y}.parquet" for y in range(2022, 2027)]
    sources += [FOLDER.parent / "channels.parquet", FOLDER / "context.parquet", FOLDER / "build.json"]
    plan = {
        "scope": "Feature-only fixed snapshots; no incident labels or predictions.",
        "times": list(map(str, TIMES)),
        "rtol": 1e-6,
        "atol": 1e-6,
        "source_hashes": {str(p): sha256(p) for p in sources},
        "code_hashes": {str(Path(__file__)): sha256(Path(__file__))},
    }
    path = ROOT / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Raw numeric check plan changed")
    if not path.exists():
        write_json(path, plan)
    stored = pd.read_parquet(FOLDER / "context.parquet", filters=[("as_of", "in", list(TIMES))])
    c = duckdb.connect()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='1GB'")
    c.execute("SET temp_directory='tmp/numeric-raw-check'")
    c.read_parquet([str(p) for p in sources[:5]]).create_view("events")
    c.read_parquet(str(sources[5])).create_view("channels")
    checked, residuals = [], []
    for at in TIMES:
        raw = c.execute(
            """WITH canonical AS (
            SELECT e.channel_id,e.ts,ch.object_id,ch.sensor_type,
              CASE WHEN count(DISTINCT e.value)=1 THEN min(e.numeric_value) END AS value
            FROM events e JOIN channels ch USING(channel_id)
            WHERE e.ts>=? AND e.ts<? AND e.alarm IS NOT NULL
              AND ch.sensor_type IN ('Датчик температуры','Газовый датчик')
            GROUP BY 1,2,3,4)
            SELECT *,CASE WHEN (sensor_type='Датчик температуры' AND value BETWEEN -50 AND 100)
              OR (sensor_type='Газовый датчик' AND value BETWEEN 0 AND 100) THEN value END AS x
            FROM canonical ORDER BY object_id,channel_id,ts""",
            [at - pd.Timedelta(hours=192), at],
        ).fetchdf()
        for row in stored.loc[stored.as_of.eq(at)].itertuples(index=False):
            expected = direct(raw.loc[raw.object_id.eq(row.object_id)], at)
            actual = np.array([getattr(row, name) for name in COLUMNS])
            reference = np.array([expected[name] for name in COLUMNS])
            np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6, equal_nan=True)
            difference = np.abs(actual - reference)
            residuals.append(float(np.max(difference[np.isfinite(difference)], initial=0)))
            checked.append({"object_id": row.object_id, "as_of": at, **expected})
        print("RAW NUMERIC VERIFIED", at, len(raw), flush=True)
    c.close()
    frame = pd.DataFrame(checked, columns=[*KEYS, *COLUMNS])
    expected_path = ROOT / "direct-values.parquet"
    frame.to_parquet(expected_path, index=False)
    build = json.loads((FOLDER / "build.json").read_text())
    report = {
        **plan,
        "plan_sha256": sha256(path),
        "status": "direct_raw_snapshots_passed",
        "queries": len(frame),
        "objects": int(frame.object_id.nunique()),
        "feature_values_compared": len(frame) * len(COLUMNS),
        "max_absolute_difference": max(residuals),
        "direct_values_sha256": sha256(expected_path),
        "build_rows": build["rows"],
        "features": COLUMNS,
        "coverage": build["coverage"],
        "limitations": "Fixed snapshots verify implementation, not all feature rows or predictive value. No episode labels read.",
        "serving_changed": False,
        "goal_achieved": False,
    }
    write_json(Path("artifacts/numeric_feature_audit.json"), report)
    return report


if __name__ == "__main__":
    print(run()["status"])
