"""Past-only numeric histories, preserving each channel before object pooling."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import connection, sha256, write_json

FOLDER = PROCESSED / "numeric-channels-v40"
CUTOFF = pd.Timestamp("2026-06-01")
KEYS = ["object_id", "as_of"]
FAMILIES = {"temperature": "Датчик температуры", "gas": "Газовый датчик"}
STATS = (
    "seen_channels_24h",
    "logged_channels_168h",
    "numeric_channels_168h",
    "last_mean",
    "last_min",
    "last_max",
    "last_std",
    "last_age_mean_h",
    "relative_range_6h_max",
    "relative_range_6h_mean",
    "relative_range_24h_max",
    "relative_range_24h_mean",
    "relative_shift_24h_max",
    "relative_shift_24h_min",
    "relative_shift_24h_abs_mean",
    "shift_supported_channels",
)
COLUMNS = [f"num_{family}_{stat}" for family in FAMILIES for stat in STATS]


def create_hourly(c, before=CUTOFF):
    """Registered events/channels views, exact old duplicate/range rules.

    Keep the last report even if invalid: a later fault or conflicting record
    must invalidate a previous numeric value. A bucket ending at t uses ts<t.
    """
    c.execute(
        """CREATE TABLE numeric_canonical AS
        SELECT e.channel_id,e.ts,c.object_id,c.sensor_type,
          CASE WHEN count(DISTINCT e.value)=1 THEN min(e.numeric_value) END AS x
        FROM events e JOIN channels c USING(channel_id)
        WHERE e.ts<? AND e.alarm IS NOT NULL
          AND c.sensor_type IN ('Датчик температуры','Газовый датчик')
        GROUP BY e.channel_id,e.ts,c.object_id,c.sensor_type""",
        [before],
    )
    c.execute("""CREATE TABLE numeric_hourly AS WITH clean AS (
        SELECT *,CASE WHEN (sensor_type='Датчик температуры' AND x BETWEEN -50 AND 100)
          OR (sensor_type='Газовый датчик' AND x BETWEEN 0 AND 100) THEN x END AS value
        FROM numeric_canonical)
        SELECT object_id,channel_id,sensor_type,
          date_trunc('hour',ts)+INTERVAL '1 hour' AS as_of,
          count(value) AS n,sum(value) AS total,min(value) AS lo,max(value) AS hi,
          max(ts) AS last_ts,last(value ORDER BY ts) AS last_value
        FROM clean GROUP BY 1,2,3,4 ORDER BY 1,2,4""")


def extract(year, folder=FOLDER):
    target, meta_path = folder / f"hourly-{year}.parquet", folder / f"hourly-{year}.json"
    inputs = [PROCESSED / f"events-{year}.parquet", PROCESSED / "channels.parquet"]
    original = PROCESSED / f"hourly-{year}.parquet"
    hashes = {str(p): sha256(p) for p in (*inputs, original, Path(__file__))}
    if meta_path.exists():
        meta = read(meta_path)
        if meta["inputs"] != hashes or sha256(target) != meta["output_sha256"]:
            raise ValueError("Numeric extraction cache changed")
        return target
    folder.mkdir(parents=True, exist_ok=True)
    c = connection()
    c.read_parquet(str(inputs[0])).create_view("events")
    c.read_parquet(str(inputs[1])).create_view("channels")
    create_hourly(c)
    c.read_parquet(str(original)).create_view("original")
    parity = c.execute("""WITH pooled AS (
        SELECT object_id,sensor_type,as_of-INTERVAL '1 hour' AS hour,
          sum(total)/sum(n) AS mean,max(hi) AS hi
        FROM numeric_hourly GROUP BY 1,2,3 HAVING sum(n)>0)
        SELECT p.sensor_type,count(*) AS rows,
          count_if(CASE WHEN p.sensor_type='Датчик температуры'
            THEN o.temperature IS NULL OR o.temperature_max IS NULL ELSE o.gas IS NULL END) AS missing,
          max(abs(p.mean-CASE WHEN p.sensor_type='Датчик температуры'
            THEN o.temperature ELSE o.gas END)) AS mean_difference,
          max(CASE WHEN p.sensor_type='Датчик температуры'
            THEN abs(p.hi-o.temperature_max) ELSE 0 END) AS maximum_difference
        FROM pooled p LEFT JOIN original o USING(object_id,hour)
        GROUP BY 1 ORDER BY 1""").fetchdf()
    if (
        len(parity) != 2
        or parity.missing.sum()
        or (parity.mean_difference > 1e-8).any()
        or (parity.maximum_difference > 1e-8).any()
    ):
        raise ValueError("Original hourly numeric aggregates differ")
    c.sql("SELECT * FROM numeric_hourly").write_parquet(str(target), compression="zstd")
    count = c.execute("SELECT count(*) FROM numeric_hourly").fetchone()[0]
    c.close()
    write_json(
        meta_path,
        {
            "inputs": hashes,
            "rows": count,
            "parity": parity.to_dict("records"),
            "output_sha256": sha256(target),
        },
    )
    print("NUMERIC EXTRACT", year, count, flush=True)
    return target


def _divide(total, count):
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def transform(queries, hourly):
    queries = queries[KEYS].reset_index(drop=True)
    if queries.isna().any().any() or queries.duplicated(KEYS).any():
        raise ValueError("Invalid numeric feature query")
    if not queries.as_of.eq(queries.as_of.dt.floor("h")).all():
        raise ValueError("Numeric summaries require completed whole hours")
    if not hourly.empty:
        if hourly.duplicated(["channel_id", "as_of"]).any():
            raise ValueError("Duplicate numeric channel hour")
        if not (
            hourly.as_of.eq(hourly.as_of.dt.floor("h"))
            & hourly.last_ts.lt(hourly.as_of)
            & hourly.last_ts.ge(hourly.as_of - pd.Timedelta(hours=1))
        ).all():
            raise ValueError("Numeric bucket contains unavailable observations")
    result = pd.DataFrame(np.nan, index=queries.index, columns=COLUMNS, dtype=float)
    for obj, rows in queries.groupby("object_id", sort=False):
        index = pd.date_range(rows.as_of.min() - pd.Timedelta(hours=192), rows.as_of.max(), freq="h")
        q = index.get_indexer(rows.as_of)
        local = hourly.loc[hourly.object_id.eq(obj)]
        for family, sensor_type in FAMILIES.items():
            n = len(rows)
            count, logged, seen, age = (np.zeros(n) for _ in range(4))
            total, square = np.zeros(n), np.zeros(n)
            minimum, maximum = np.full(n, np.inf), np.full(n, -np.inf)
            ranges = {w: [np.zeros(n), np.full(n, -np.inf), np.zeros(n)] for w in (6, 24)}
            shift_count, shift_abs = np.zeros(n), np.zeros(n)
            shift_min, shift_max = np.full(n, np.inf), np.full(n, -np.inf)
            for _, group in local.loc[local.sensor_type.eq(sensor_type)].groupby("channel_id", sort=False):
                h = group.set_index("as_of").reindex(index)
                observed = h.last_ts.notna().to_numpy()
                previous = np.maximum.accumulate(np.where(observed, np.arange(len(index)), -1))
                safe = np.maximum(previous, 0)
                last_ts = h.last_ts.to_numpy(dtype="datetime64[ns]")[safe]
                last_age = (index.to_numpy() - last_ts) / np.timedelta64(1, "h")
                recent = (previous >= 0) & (last_age <= 168)
                values = h.last_value.to_numpy()[safe].copy()
                valid = recent & np.isfinite(values)
                values[~valid] = np.nan
                v, ok = values[q], valid[q]
                count += ok
                logged += recent[q]
                total += np.where(ok, v, 0)
                square += np.where(ok, v * v, 0)
                minimum = np.minimum(minimum, np.where(ok, v, np.inf))
                maximum = np.maximum(maximum, np.where(ok, v, -np.inf))
                age += np.where(ok, last_age[q], 0)
                for w in (6, 24):
                    counts = h.n.fillna(0).rolling(w, min_periods=1).sum()
                    mean = h.total.fillna(0).rolling(w, min_periods=1).sum() / counts.replace(0, np.nan)
                    span = h.hi.rolling(w, min_periods=1).max() - h.lo.rolling(w, min_periods=1).min()
                    relative = (span / (1 + mean.abs())).to_numpy()[q]
                    supported = np.isfinite(relative)
                    ranges[w][0] += np.where(supported, relative, 0)
                    ranges[w][1] = np.maximum(ranges[w][1], np.where(supported, relative, -np.inf))
                    ranges[w][2] += supported
                    if w == 24:
                        seen += counts.to_numpy()[q] > 0
                old_values = pd.Series(values).shift(24).to_numpy()[q]
                shift = (v - old_values) / (1 + np.abs(old_values))
                supported = np.isfinite(shift)
                shift_count += supported
                shift_abs += np.where(supported, np.abs(shift), 0)
                shift_min = np.minimum(shift_min, np.where(supported, shift, np.inf))
                shift_max = np.maximum(shift_max, np.where(supported, shift, -np.inf))
            mean = _divide(total, count)
            stats = {
                "seen_channels_24h": seen,
                "logged_channels_168h": logged,
                "numeric_channels_168h": count,
                "last_mean": mean,
                "last_min": np.where(count > 0, minimum, np.nan),
                "last_max": np.where(count > 0, maximum, np.nan),
                "last_std": np.sqrt(np.maximum(_divide(square, count) - mean**2, 0)),
                "last_age_mean_h": _divide(age, count),
                "relative_shift_24h_max": np.where(shift_count > 0, shift_max, np.nan),
                "relative_shift_24h_min": np.where(shift_count > 0, shift_min, np.nan),
                "relative_shift_24h_abs_mean": _divide(shift_abs, shift_count),
                "shift_supported_channels": shift_count,
            }
            for w, (sums, maxima, counts) in ranges.items():
                stats[f"relative_range_{w}h_max"] = np.where(counts > 0, maxima, np.nan)
                stats[f"relative_range_{w}h_mean"] = _divide(sums, counts)
            for stat, values in stats.items():
                result.loc[rows.index, f"num_{family}_{stat}"] = values
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite numeric summary")
    return result.astype(np.float32)


def build(folder=FOLDER):
    paths = [extract(y, folder) for y in range(2022, 2027)]
    sources = [
        PROCESSED / n
        for n in (
            "features.parquet",
            "features-hourly-research.parquet",
            "features-channel-novelty.parquet",
            "features-dense-channel-novelty.parquet",
        )
    ]
    hashes = {
        "inputs": {str(p): sha256(p) for p in (*sources, *paths)},
        "code_hashes": {str(Path(__file__)): sha256(Path(__file__))},
    }
    target, manifest = folder / "context.parquet", folder / "build.json"
    if manifest.exists():
        old = read(manifest)
        if any(old[k] != v for k, v in hashes.items()) or sha256(target) != old["output_sha256"]:
            raise ValueError("Numeric feature cache changed")
        return old
    queries = (
        pd.concat(
            [pd.read_parquet(p, columns=KEYS, filters=[("as_of", "<", CUTOFF)]) for p in sources],
            ignore_index=True,
        )
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
    )
    schema = pa.schema(
        [("object_id", pa.int64()), ("as_of", pa.timestamp("ns")), *[(c, pa.float32()) for c in COLUMNS]]
    )
    rows, coverage = 0, []
    with pq.ParquetWriter(target, schema, compression="zstd") as writer:
        for obj, group in queries.groupby("object_id", sort=True):
            group = group.reset_index(drop=True)
            hourly = pd.read_parquet(paths, filters=[("object_id", "==", int(obj))])
            output = pd.concat([group, transform(group, hourly)], axis=1)
            writer.write_table(pa.Table.from_pandas(output, schema=schema, preserve_index=False))
            rows += len(output)
            coverage.append(
                {
                    "object_id": int(obj),
                    "queries": len(group),
                    **{
                        family: int(output[f"num_{family}_numeric_channels_168h"].gt(0).sum())
                        for family in FAMILIES
                    },
                }
            )
            print("NUMERIC CONTEXT", obj, len(group), flush=True)
    report = {
        **hashes,
        "rows": rows,
        "features": COLUMNS,
        "coverage": coverage,
        "before": str(CUTOFF),
        "output_sha256": sha256(target),
        "extraction_manifests": {
            str(folder / f"hourly-{y}.json"): sha256(folder / f"hourly-{y}.json") for y in range(2022, 2027)
        },
    }
    write_json(manifest, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    print(build(parser.parse_args().folder)["rows"])
