"""Raw quarter-hour report counts with exact original whole-hour semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.features import COUNT_COLUMNS
from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import connection, sha256, write_json

BEFORE = "2026-06-01"
STARTS = {2025: "2025-09-24", 2026: "2026-01-01"}
BURST_SIGNALS = (
    "fault_reports",
    "power_reports",
    "unknown_reports",
    "smoke_reports",
    "flood_reports",
    "access_reports",
)
UPDATED_COLUMNS = [f"{signal}_{hours}h" for hours in (1, 6, 24, 168) for signal in COUNT_COLUMNS]


def aggregate_counts(con):
    """Same canonicalization and signal definitions as features.aggregate_year.

    Callers register already time-filtered `source` and the channel catalog.
    No episode table, transition end, duration or future label is read.
    """
    con.execute("""CREATE TABLE canonical AS
        SELECT channel_id,ts,
            CASE WHEN count(DISTINCT value)=1 THEN min(value) ELSE NULL END AS value,
            CASE WHEN count(DISTINCT value)=1 THEN min(numeric_value) ELSE NULL END AS numeric_value,
            bool_or(alarm) AS alarm, count(DISTINCT value)>1 AS ambiguous
        FROM source WHERE ts IS NOT NULL AND channel_id IS NOT NULL AND alarm IS NOT NULL
        GROUP BY channel_id,ts""")
    con.execute("""CREATE TABLE typed AS SELECT e.*,c.object_id,c.sensor_type,
        CASE WHEN ambiguous THEN -1
             WHEN value='Неисправен' THEN 1
             WHEN value='Обнаружен дым' AND sensor_type='Датчик дыма' THEN 2
             WHEN value='Температура выше 40ºC' AND sensor_type='Тепловой датчик' THEN 2
             WHEN value='Затоплен' AND sensor_type='Состояние насоса' THEN 3
             WHEN value='Не замкнут' AND sensor_type='Датчик затопления' AND alarm THEN 3
             WHEN alarm AND sensor_type IN ('КД Дверь','КД Люк','КД АВ','9-секционный люк',
                  'Датчик движения','Стекло') AND value IN ('Не замкнут','Обнаружено движение','Стекло') THEN 4
             ELSE 0 END AS state_code
        FROM canonical e JOIN channels c USING(channel_id)""")
    return con.sql("""SELECT object_id,time_bucket(INTERVAL '15 minutes',ts) AS bucket,
        count(*) AS events, count_if(alarm) AS alarms,
        count_if(value='Неисправен') AS fault_reports,
        count_if(value IN ('Обесточен','Питание от батарей','Батарея разряжена','Отключено устройство')) AS power_reports,
        count_if(value IN ('Неопределен','Не определено')) AS unknown_reports,
        count_if(state_code=2) AS smoke_reports, count_if(state_code=3) AS flood_reports,
        count_if(state_code=4) AS access_reports,
        count_if(sensor_type='Состояние насоса' AND value IN ('Включен','Выключен','Работают все насосы в АНС')) AS pump_switches,
        count_if(ambiguous) AS ambiguous,
        count_if(numeric_value IN (-100,255,-3276,-127) OR starts_with(value,'01.01.1970')) AS technical_codes
        FROM typed GROUP BY 1,2 ORDER BY 1,2""")


def extract(year, folder):
    target = folder / f"counts-{year}.parquet"
    metadata = folder / f"counts-{year}.json"
    raw = PROCESSED / f"events-{year}.parquet"
    old_hourly = PROCESSED / f"hourly-{year}.parquet"
    catalog = PROCESSED / "channels.parquet"
    inputs = {
        str(p): sha256(p)
        for p in (raw, old_hourly, catalog, Path(__file__), Path("src/moscollector/features.py"))
    }
    if metadata.exists():
        meta = read(metadata)
        if meta["inputs"] != inputs or meta["outputs"] != {str(target): sha256(target)}:
            raise ValueError("Quarter-count cache changed; use a new directory")
        return target
    folder.mkdir(parents=True, exist_ok=True)
    con = connection()
    con.read_parquet(str(raw)).filter(
        f"ts >= TIMESTAMP '{STARTS[year]}' AND ts < TIMESTAMP '{BEFORE}'"
    ).create_view("source")
    con.read_parquet(str(catalog)).create_view("channels")
    aggregate_counts(con).write_parquet(str(target), compression="zstd")
    con.close()
    frame = pd.read_parquet(target)
    expected = pd.read_parquet(
        old_hourly,
        columns=["object_id", "hour", *COUNT_COLUMNS],
        filters=[("hour", ">=", pd.Timestamp(STARTS[year])), ("hour", "<", pd.Timestamp(BEFORE))],
    )
    actual = (
        frame.assign(hour=frame.bucket.dt.floor("h"))
        .groupby(["object_id", "hour"], as_index=False)[COUNT_COLUMNS]
        .sum(min_count=1)
    )
    keys = ["object_id", "hour"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[keys], expected[keys])
    np.testing.assert_array_equal(actual[COUNT_COLUMNS].to_numpy(), expected[COUNT_COLUMNS].to_numpy())
    write_json(
        metadata,
        {
            "inputs": inputs,
            "outputs": {str(target): sha256(target)},
            "start": STARTS[year],
            "before": BEFORE,
            "rows": len(frame),
            "raw_canonical_reports": int(frame.events.sum()),
            "original_hourly_rows_exact": len(expected),
            "signal_definitions": "Unchanged original report-count definitions; no episode labels used.",
        },
    )
    print("Saved raw quarter counts", year, len(frame), "hourly parity", len(expected), flush=True)
    return target


def carry_features(hourly, slots, columns):
    """Carry the latest whole-hour inputs, identified by exact source time."""
    if not hourly.as_of.eq(hourly.as_of.dt.floor("h")).all():
        raise ValueError("Expected whole-hour source features")
    if slots.duplicated(["object_id", "as_of"]).any():
        raise ValueError("Duplicate target slots")
    required = list(dict.fromkeys(["object_id", "as_of", *columns]))
    lookup = hourly[required].rename(columns={"as_of": "source_time"})
    result = slots[["object_id", "as_of"]].copy()
    result["source_time"] = result.as_of.dt.floor("h")
    result = result.merge(
        lookup, on=["object_id", "source_time"], how="left", validate="many_to_one", indicator=True
    )
    if not result._merge.eq("both").all():
        raise ValueError("No whole-hour source within the current hour")
    return result.drop(columns="_merge")


def refresh_counts(frame, counts):
    """Exact counts in [t-window,t) at quarter boundaries; all other inputs held.

    Whole-hour values must match the frozen original features. The six burst
    ratios, where used by a model, are refreshed from those same count windows.
    """
    out = frame.reset_index(drop=True).copy()
    if not out.as_of.eq(out.as_of.dt.floor("15min")).all():
        raise ValueError("Only quarter-hour forecast boundaries are supported")
    if counts.duplicated(["object_id", "bucket"]).any():
        raise ValueError("Duplicate raw count buckets")
    if not counts.bucket.eq(counts.bucket.dt.floor("15min")).all():
        raise ValueError("Counts must have quarter-hour buckets")
    selected = [name for name in UPDATED_COLUMNS if name in out]
    matrices = {name: np.zeros(len(out), dtype=np.float32) for name in selected}
    for obj, rows in out.groupby("object_id", sort=False):
        source = counts[counts.object_id.eq(obj)].sort_values("bucket")
        values = source[COUNT_COLUMNS].fillna(0).to_numpy(dtype=np.int64)
        if np.any(values < 0):
            raise ValueError("Negative report count")
        cumul = np.vstack([np.zeros((1, len(COUNT_COLUMNS)), dtype=np.int64), np.cumsum(values, axis=0)])
        times = source.bucket.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        right = np.searchsorted(times, at, side="left")
        for hours in (1, 6, 24, 168):
            left = np.searchsorted(times, at - hours * HOUR_NS, side="left")
            sums = cumul[right] - cumul[left]
            for j, signal in enumerate(COUNT_COLUMNS):
                name = f"{signal}_{hours}h"
                if name in matrices:
                    matrices[name][rows.index] = sums[:, j].astype(np.float32)
    for name, values in matrices.items():
        out[name] = values
    for signal in BURST_SIGNALS:
        name = f"{signal}_burst"
        if name in out:
            out[name] = (out[f"{signal}_6h"] / (1 + out[f"{signal}_168h"] / 28)).astype(np.float32)
    return out


def build(folder):
    files = [extract(year, folder) for year in STARTS]
    counts = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    source = PROCESSED / "features-dense-channel-novelty.parquet"
    columns = [*UPDATED_COLUMNS, *(f"{s}_burst" for s in BURST_SIGNALS)]
    reference = pd.read_parquet(source, columns=["object_id", "as_of", "eligible", *columns])
    # The frozen builder deliberately invalidates the first week of each year.
    # Check every valid hourly row; no filled-in year-boundary history is used.
    reference = reference[reference.eligible].reset_index(drop=True)
    actual = refresh_counts(reference, counts)
    pd.testing.assert_frame_equal(reference, actual)
    report = {
        "inputs": {
            str(p): sha256(p)
            for p in [source, Path(__file__), *files, *(folder / f"counts-{year}.json" for year in STARTS)]
        },
        "outputs": {},
        "valid_hourly_feature_rows_exact": len(reference),
        "refreshed_columns": columns,
        "raw_periods": STARTS,
        "before": BEFORE,
        "whole_hour_feature_parity": True,
        "new_training_weights": False,
    }
    write_json(folder / "build.json", report)
    print("Whole-hour count feature parity", len(reference), len(columns), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=PROCESSED / "quarter-counts-v21")
    build(parser.parse_args().folder)
