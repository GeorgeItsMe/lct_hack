"""Reproducible, bounded-memory CSV ingestion and source audit.

The original files remain unchanged. Technical codes are preserved as evidence;
they are not silently converted into physical measurements or failure labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import duckdb
import pandas as pd

from moscollector.paths import ARTIFACTS, PROCESSED, RAW

LOG = logging.getLogger(__name__)
CHANNEL_COLUMNS = {
    "ид_канала_данных": "channel_id",
    "тип_инж_системы": "system_type",
    "тип_датчика": "sensor_type",
    "тег_инженерной_системы": "system_tag",
    "название_датчика": "sensor_name",
    "ид_объект": "object_id",
}
OBJECT_COLUMNS = {
    "ид_объект": "object_id",
    "иерархия_уровень": "level",
    "родитель": "parent_id",
    "вид_объекта": "object_kind",
    "диспетчерское_название_объекта": "object_name",
}


def connection() -> duckdb.DuckDBPyConnection:
    (ARTIFACTS / "cache").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='4GB'")
    con.execute("SET temp_directory=?", [str(ARTIFACTS / "cache")])
    return con


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(con, query: str):
    result = con.execute(query)
    columns = [x[0] for x in result.description]
    return [dict(zip(columns, r, strict=True)) for r in result.fetchall()]


def write_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def prepare_catalogs(raw: Path = RAW, out: Path = PROCESSED):
    out.mkdir(parents=True, exist_ok=True)
    channels = pd.read_csv(raw / "справочник_каналов_датчиков.csv").rename(columns=CHANNEL_COLUMNS)
    objects = pd.read_csv(raw / "справочник_объектов_диспетчер.csv").rename(columns=OBJECT_COLUMNS)
    for frame, key in [(channels, "channel_id"), (objects, "object_id")]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Catalog key {key} is missing or duplicated")
    if not channels.object_id.isin(objects.object_id).all():
        raise ValueError("Channel catalog contains unknown object_id")
    channels.to_parquet(out / "channels.parquet", index=False)
    objects.to_parquet(out / "objects.parquet", index=False)
    info = {
        "channel_count": len(channels),
        "object_count": len(objects),
        "sensor_types": channels.sensor_type.value_counts().to_dict(),
        "system_types": channels.system_type.value_counts().to_dict(),
        "catalog_source": "provided_dataset",
        "coordinates_available": False,
    }
    write_json(ARTIFACTS / "catalog_audit.json", info)
    LOG.info("Catalogs: %s channels, %s objects", len(channels), len(objects))
    return info


def extract_year(year: int, raw: Path = RAW) -> Path:
    target = raw / f"ext-journal-{year}.csv"
    archive = raw / f"ext-journal-{year}.7z"
    if target.exists():
        return target
    import py7zr

    with py7zr.SevenZipFile(archive) as z:
        if z.getnames() != [target.name]:
            raise ValueError(f"Unexpected archive contents: {archive.name}")
    LOG.info("Extracting %s", archive.name)
    # bsdtar uses the system libarchive and is much faster on macOS.
    try:
        p = subprocess.run(["tar", "-xf", str(archive), "-C", str(raw)], capture_output=True)
    except FileNotFoundError:
        p = None
    if p is None or p.returncode:
        with py7zr.SevenZipFile(archive) as z:
            z.extract(path=raw, targets=[target.name])
    return target


def prepare_year(year: int, raw: Path = RAW, out: Path = PROCESSED, force: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"events-{year}.parquet"
    audit_path = ARTIFACTS / f"audit-{year}.json"
    if target.exists() and audit_path.exists() and not force:
        LOG.info("Already prepared %s", year)
        return json.loads(audit_path.read_text(encoding="utf-8"))
    source = extract_year(year, raw)
    start = time.monotonic()
    con = connection()
    # all_varchar avoids mixed sensor values being lost to CSV type inference.
    con.read_csv(str(source), header=True, all_varchar=True, strict_mode=True).create_view("raw_events")
    normalized = """
        SELECT try_cast(ид_события AS BIGINT) AS event_id,
               try_cast(ид_канала_данных AS BIGINT) AS channel_id,
               try_cast(дата || ' ' || время AS TIMESTAMP) AS ts,
               CASE lower(trim(тревожное)) WHEN 't' THEN true WHEN 'true' THEN true
                    WHEN 'f' THEN false WHEN 'false' THEN false ELSE NULL END AS alarm,
               trim(значение_датчика) AS value,
               try_cast(replace(trim(значение_датчика), ',', '.') AS DOUBLE) AS numeric_value
        FROM raw_events
    """
    # Copy via relation avoids interpolating filesystem paths into SQL.
    con.sql(normalized).write_parquet(str(target), compression="zstd")
    con.read_parquet(str(target)).create_view("events")
    con.read_parquet(str(out / "channels.parquet")).create_view("channels")
    summary = records(
        con,
        """
        SELECT count(*) AS rows, min(ts) AS start, max(ts) AS end,
               count(DISTINCT channel_id) AS channels, count_if(alarm) AS alarms,
               count_if(ts IS NULL OR channel_id IS NULL OR alarm IS NULL) AS invalid_rows,
               count_if(numeric_value IS NOT NULL) AS numeric_rows,
               count_if(numeric_value IN (-100,255,-3276,-127)) AS technical_code_rows,
               count_if(channel_id NOT IN (SELECT channel_id FROM channels)) AS unmapped_rows
        FROM events
    """,
    )[0]
    daily = records(
        con,
        """
        SELECT cast(ts AS DATE) AS day, count(*) AS rows, count_if(alarm) AS alarms,
               count(DISTINCT channel_id) AS channels
        FROM events GROUP BY 1 ORDER BY 1
    """,
    )
    states = records(
        con,
        """
        SELECT value, count(*) AS rows, count_if(alarm) AS alarms,
               count(DISTINCT channel_id) AS channels
        FROM events WHERE numeric_value IS NULL GROUP BY value ORDER BY rows DESC LIMIT 80
    """,
    )
    by_type = records(
        con,
        """
        SELECT coalesce(c.sensor_type,'Без справочника') AS sensor_type,
               count(*) AS rows, count_if(e.alarm) AS alarms,
               count_if(e.value='Неисправен') AS fault_rows,
               count(DISTINCT e.channel_id) AS channels
        FROM events e LEFT JOIN channels c USING(channel_id)
        GROUP BY 1 ORDER BY rows DESC
    """,
    )
    info = {
        "year": year,
        "source": source.name,
        "sha256": sha256(source),
        "timezone": "Europe/Moscow (assumption; source has no offset)",
        "quarantined": year == 2021,
        "quarantine_reason": "Monitoring migration reported in team brief" if year == 2021 else None,
        "summary": summary,
        "daily": daily,
        "states": states,
        "by_sensor_type": by_type,
        "preparation_seconds": round(time.monotonic() - start, 2),
    }
    write_json(audit_path, info)
    con.close()
    LOG.info("Prepared %s: %s", year, summary)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2026])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    prepare_catalogs()
    for year in args.years:
        prepare_year(year, force=args.force)


if __name__ == "__main__":
    main()
