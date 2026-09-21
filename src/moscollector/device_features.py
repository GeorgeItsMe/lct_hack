"""Causal device-state summaries, including operating modes lost by old counters.

State is only last reported value, not proof of physical health. Histories reset
at each year, consistently with the original preprocessing. Source June rows are
excluded for the research build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.paths import PROCESSED
from moscollector.prepare import connection, sha256, write_json

# Fixed from catalog/domain meanings, not selected using validation outcomes.
PREDICATES = {
    "fault": "value='Неисправен'",
    "unknown": "value IN ('Неопределен','Не определено')",
    "disabled": "value IN ('Выключен','Отключено устройство')",
    "no_power": "value='Обесточен'",
    "ups_battery": "sensor_type='ИБП' AND value='Питание от батарей'",
    "ups_bad": "sensor_type='ИБП' AND value IN ('Батарея неисправна','Батарея разряжена','Неисправен')",
    "armed": "sensor_type='Состояние охраны' AND value='На охране'",
    "disarmed": "sensor_type='Состояние охраны' AND value='Снято с охраны'",
    "many_faults": "sensor_type='Состояние охраны' AND value='Много неисправных устройств'",
    "pump_on": "sensor_type='Состояние насоса' AND value IN ('Включен','Работают все насосы в АНС')",
    "fan_on": "sensor_type='Состояние вентилятора' AND value='Включен'",
    "smoke": "sensor_type='Датчик дыма' AND value='Обнаружен дым'",
    "motion": "sensor_type='Датчик движения' AND value='Обнаружено движение'",
    "door_open": "sensor_type IN ('КД Дверь','КД Люк','КД АВ','9-секционный люк') AND value='Не замкнут'",
    "gas_detected": "sensor_type='Газовый датчик' AND value='Обнаружен газ'",
}


def aggregate(year: int, before: str):
    target = PROCESSED / f"device-hourly-{year}-{before}.parquet"
    if target.exists():
        return target
    c = connection()
    c.read_parquet(str(PROCESSED / f"events-{year}.parquet")).filter(
        f"ts < TIMESTAMP '{pd.Timestamp(before).isoformat()}'"
    ).create_view("source")
    c.read_parquet(str(PROCESSED / "channels.parquet")).create_view("channels")
    # Same conflict handling as the original canonical log; no arbitrary last row.
    c.execute("""CREATE TABLE typed AS WITH canonical AS (
        SELECT channel_id, ts, CASE WHEN count(DISTINCT value)=1 THEN min(value) ELSE NULL END AS value,
               count(DISTINCT value)>1 AS ambiguous, bool_or(alarm) alarm
        FROM source WHERE ts IS NOT NULL AND channel_id IS NOT NULL AND alarm IS NOT NULL
        GROUP BY channel_id, ts)
        SELECT e.*, c.object_id, c.sensor_type FROM canonical e
        JOIN channels c USING(channel_id)""")
    flags = ", ".join(
        f"CAST(coalesce(({predicate}),false) AS INT) AS s_{key}" for key, predicate in PREDICATES.items()
    )
    c.execute(f"CREATE TABLE flags AS SELECT *, {flags} FROM typed")
    deltas = ", ".join(f"s_{k} - coalesce(lag(s_{k}) OVER w,0) AS d_{k}" for k in PREDICATES)
    c.execute(f"""CREATE TABLE deltas AS SELECT *, {deltas},
                 lag(value) OVER w IS DISTINCT FROM value AS changed
                 FROM flags WINDOW w AS (PARTITION BY channel_id ORDER BY ts)""")
    aggregations = []
    for key in PREDICATES:
        aggregations += [f"sum(d_{key}) AS delta_{key}", f"sum(s_{key}) AS reports_{key}"]
    aggregations += [
        "count_if(changed) AS changes",
        "count_if(ambiguous) AS conflicts",
        "count(DISTINCT channel_id) FILTER(WHERE s_fault=1 OR s_unknown=1 OR s_no_power=1) AS bad_channels",
    ]
    c.execute(f"""COPY (SELECT object_id,date_trunc('hour',ts) AS hour,
                 {", ".join(aggregations)} FROM deltas GROUP BY 1,2 ORDER BY 1,2)
                 TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    c.close()
    print(f"aggregated device states {year}", flush=True)
    return target


def transform(times, hourly):
    times = pd.DatetimeIndex(times)
    chunks = []
    for year in sorted(times.year.unique()):
        at = times[times.year == year]
        hist = hourly[hourly.hour.dt.year.eq(year)]
        # Do not assume a zero initial state before the first observed state.
        index = pd.date_range(f"{year}-01-01", at.max(), freq="h")
        hist = hist.set_index("hour").reindex(index).fillna(0)
        ids = index.get_indexer(at)
        extra = {}
        for key in PREDICATES:
            state = hist[f"delta_{key}"].cumsum().shift(1)
            extra[f"dev_last_reported_{key}"] = state.to_numpy()[ids]
            for hours in (6, 24, 168):
                extra[f"dev_{key}_reports_{hours}h"] = (
                    hist[f"reports_{key}"].shift(1).rolling(hours).sum().to_numpy()[ids]
                )
            extra[f"dev_{key}_state_hours_24h"] = state.rolling(24).sum().to_numpy()[ids]
        for col in ("changes", "conflicts", "bad_channels"):
            for hours in (6, 24, 168):
                extra[f"dev_{col}_{hours}h"] = hist[col].shift(1).rolling(hours).sum().to_numpy()[ids]
        chunks.append(pd.DataFrame(extra, index=at).astype(np.float32))
    return pd.concat(chunks).reindex(times)


def build(output: Path, before="2026-06-01"):
    original = PROCESSED / "features-sequence.parquet"
    features = pd.read_parquet(original, filters=[("as_of", "<", pd.Timestamp(before))])
    years = sorted(features.as_of.dt.year.unique())
    files = [aggregate(int(year), before) for year in years]
    hourly = pd.concat([pd.read_parquet(f) for f in files])
    extras = []
    for obj, rows in features.groupby("object_id", sort=True):
        extra = transform(rows.as_of, hourly[hourly.object_id.eq(obj)])
        extra.index = rows.index
        extras.append(extra)
    result = pd.concat([features, pd.concat(extras).reindex(features.index)], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    write_json(
        output.with_suffix(".json"),
        {
            "rows": len(result),
            "columns": result.columns.tolist(),
            "before": before,
            "source_sha256": sha256(original),
            "sha256": sha256(output),
            "device_summaries": {str(f): sha256(f) for f in files},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/features-device.parquet"))
    parser.add_argument("--before", default="2026-06-01")
    args = parser.parse_args()
    build(args.output, args.before)
