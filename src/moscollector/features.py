"""Causal object-level features and explicitly defined proxy incident episodes.

Every feature row at t contains observations strictly before t. Targets describe
new grouped episodes in [t, t+24h), not whether an alarm is already active.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.prepare import connection, write_json

LOG = logging.getLogger(__name__)
KINDS = {1: "fault", 2: "fire", 3: "flood", 4: "access"}
KIND_NAMES = {
    "fault": "Неисправность оборудования",
    "fire": "Пожарный сигнал",
    "flood": "Сигнал подтопления",
    "access": "Охранный сигнал",
}
COUNT_COLUMNS = [
    "events",
    "alarms",
    "fault_reports",
    "power_reports",
    "unknown_reports",
    "smoke_reports",
    "flood_reports",
    "access_reports",
    "pump_switches",
    "ambiguous",
    "technical_codes",
]


def aggregate_year(year: int, force: bool = False):
    hourly_file = PROCESSED / f"hourly-{year}.parquet"
    episode_file = PROCESSED / f"episodes-{year}.parquet"
    if hourly_file.exists() and episode_file.exists() and not force:
        return
    start = time.monotonic()
    c = connection()
    c.read_parquet(str(PROCESSED / f"events-{year}.parquet")).create_view("events")
    c.read_parquet(str(PROCESSED / "channels.parquet")).create_view("channels")
    # One observation per channel/second. Conflicting states remain unknown.
    c.execute("""CREATE TABLE canonical AS
        SELECT e.channel_id, e.ts, min(e.event_id) AS event_id,
               CASE WHEN count(DISTINCT e.value)=1 THEN min(e.value) ELSE NULL END AS value,
               CASE WHEN count(DISTINCT e.value)=1 THEN min(e.numeric_value) ELSE NULL END AS numeric_value,
               bool_or(e.alarm) AS alarm, count(DISTINCT e.value)>1 AS ambiguous,
               count(*) AS source_rows
        FROM events e WHERE e.ts IS NOT NULL AND e.channel_id IS NOT NULL AND e.alarm IS NOT NULL
        GROUP BY e.channel_id,e.ts
    """)
    c.execute("""CREATE TABLE typed AS SELECT e.*, c.object_id,c.sensor_type,c.sensor_name,
        CASE WHEN ambiguous THEN -1
             WHEN value='Неисправен' THEN 1
             WHEN value='Обнаружен дым' AND sensor_type='Датчик дыма' THEN 2
             WHEN value='Температура выше 40ºC' AND sensor_type='Тепловой датчик' THEN 2
             WHEN value='Затоплен' AND sensor_type='Состояние насоса' THEN 3
             WHEN value='Не замкнут' AND sensor_type='Датчик затопления' AND alarm THEN 3
             WHEN alarm AND sensor_type IN ('КД Дверь','КД Люк','КД АВ','9-секционный люк',
                  'Датчик движения','Стекло') AND value IN ('Не замкнут','Обнаружено движение','Стекло') THEN 4
             ELSE 0 END AS state_code
        FROM canonical e JOIN channels c USING(channel_id)
    """)
    c.sql("""SELECT object_id,date_trunc('hour',ts) AS hour,
        count(*) AS events, count_if(alarm) AS alarms,
        count_if(value='Неисправен') AS fault_reports,
        count_if(value IN ('Обесточен','Питание от батарей','Батарея разряжена','Отключено устройство')) AS power_reports,
        count_if(value IN ('Неопределен','Не определено')) AS unknown_reports,
        count_if(state_code=2) AS smoke_reports, count_if(state_code=3) AS flood_reports,
        count_if(state_code=4) AS access_reports,
        count_if(sensor_type='Состояние насоса' AND value IN ('Включен','Выключен','Работают все насосы в АНС')) AS pump_switches,
        count_if(ambiguous) AS ambiguous,
        count_if(numeric_value IN (-100,255,-3276,-127) OR starts_with(value,'01.01.1970')) AS technical_codes,
        avg(numeric_value) FILTER(WHERE sensor_type='Датчик температуры' AND numeric_value BETWEEN -50 AND 100) AS temperature,
        max(numeric_value) FILTER(WHERE sensor_type='Датчик температуры' AND numeric_value BETWEEN -50 AND 100) AS temperature_max,
        avg(numeric_value) FILTER(WHERE sensor_type='Газовый датчик' AND numeric_value BETWEEN 0 AND 100) AS gas,
        count(DISTINCT channel_id) AS reporting_channels
        FROM typed GROUP BY 1,2 ORDER BY 1,2
    """).write_parquet(str(hourly_file), compression="zstd")
    # First observed state is left-censored, so it cannot prove a NEW episode.
    c.execute("""CREATE TABLE transitions AS WITH prev AS (
        SELECT *,lag(state_code) OVER(PARTITION BY channel_id ORDER BY ts) AS previous_code
        FROM typed)
        SELECT * FROM prev WHERE previous_code IS NULL OR previous_code<>state_code
    """)
    end = c.execute("SELECT max(ts) FROM canonical").fetchone()[0]
    episodes = c.execute(
        """WITH bounded AS (
        SELECT *,lead(ts) OVER(PARTITION BY channel_id ORDER BY ts) AS end_ts
        FROM transitions)
        SELECT channel_id,object_id,sensor_type,sensor_name,ts AS start_ts,end_ts,state_code,
               previous_code,event_id,value,
               coalesce(end_ts,?) - ts AS duration,
               end_ts IS NULL AS right_censored
        FROM bounded WHERE state_code>0 AND previous_code IS NOT NULL AND previous_code>=0
          AND (state_code<>1 OR coalesce(end_ts,?)-ts>=INTERVAL 1 HOUR)
        ORDER BY object_id,state_code,ts,channel_id
    """,
        [end, end],
    ).df()
    episodes["kind"] = episodes.state_code.map(KINDS)
    episodes["duration_seconds"] = episodes.duration.dt.total_seconds()
    episodes.drop(columns=["duration"]).to_parquet(episode_file, index=False)
    audit = {
        "year": year,
        "canonical_rows": c.execute("SELECT count(*) FROM canonical").fetchone()[0],
        "collapsed_rows": c.execute("SELECT sum(source_rows-1) FROM canonical").fetchone()[0],
        "ambiguous_channel_seconds": c.execute("SELECT count(*) FROM canonical WHERE ambiguous").fetchone()[
            0
        ],
        "channel_episodes": episodes.kind.value_counts().to_dict(),
        "seconds": round(time.monotonic() - start, 2),
    }
    write_json(ARTIFACTS / f"episode-audit-{year}.json", audit)
    c.close()
    LOG.info("Episodes %s: %s", year, audit)


def group_episodes(channel_episodes: pd.DataFrame) -> pd.DataFrame:
    """Fixed 10-minute window anchored to first onset, so bursts cannot chain for days."""
    if channel_episodes.empty:
        return pd.DataFrame(
            {
                "object_id": pd.Series(dtype="int64"),
                "kind": pd.Series(dtype="str"),
                "start_ts": pd.Series(dtype="datetime64[ns]"),
                "last_onset": pd.Series(dtype="datetime64[ns]"),
                "end_ts": pd.Series(dtype="datetime64[ns]"),
                "channel_count": pd.Series(dtype="int64"),
                "channel_ids": pd.Series(dtype="object"),
                "sensor_names": pd.Series(dtype="object"),
                "duration_seconds": pd.Series(dtype="float64"),
                "right_censored": pd.Series(dtype="bool"),
                "source_event_id": pd.Series(dtype="int64"),
                "episode_id": pd.Series(dtype="str"),
            }
        )
    rows = []
    for (obj, kind), group in channel_episodes.groupby(["object_id", "kind"], sort=True):
        group = group.sort_values(["start_ts", "channel_id"])
        buffer = []
        first = None

        def emit(buffer, obj, kind):
            b = pd.DataFrame(buffer)
            rows.append(
                {
                    "object_id": int(obj),
                    "kind": kind,
                    "start_ts": b.start_ts.min(),
                    "last_onset": b.start_ts.max(),
                    "end_ts": b.end_ts.max(),
                    "channel_count": int(b.channel_id.nunique()),
                    "channel_ids": sorted(map(int, b.channel_id.unique())),
                    "sensor_names": list(dict.fromkeys(b.sensor_name.astype(str)))[:8],
                    "duration_seconds": float(b.duration_seconds.max()),
                    "right_censored": bool(b.right_censored.any()),
                    "source_event_id": int(b.event_id.iloc[0]),
                }
            )

        for r in group.to_dict("records"):
            if (
                first is not None
                and (pd.Timestamp(r["start_ts"]) - pd.Timestamp(first)).total_seconds() > 600
            ):
                emit(buffer, obj, kind)
                buffer = []
                first = None
            if first is None:
                first = r["start_ts"]
            buffer.append(r)
        if buffer:
            emit(buffer, obj, kind)
    result = pd.DataFrame(rows).sort_values(["start_ts", "object_id", "kind"]).reset_index(drop=True)
    result["episode_id"] = [f"EP-{i + 1:07d}" for i in range(len(result))]
    return result


def future_counts(counts: np.ndarray, horizon: int) -> np.ndarray:
    """Inclusive current hour, exclusive t+horizon. The counts represent future onsets."""
    totals = np.r_[0, np.cumsum(counts)]
    end = np.minimum(np.arange(len(counts)) + horizon, len(counts))
    return totals[end] - totals[:-1]


def build_dataset(
    years: list[int],
    step_hours: int = 3,
    *,
    output_path: Path | None = None,
    episode_output_path: Path | None = None,
    audit_output_path: Path | None = None,
    before: str | None = None,
):
    alternate_outputs = (output_path, episode_output_path, audit_output_path)
    if (before or any(alternate_outputs)) and not all(alternate_outputs):
        raise ValueError("Research/cutoff builds require all three explicit output paths")
    years = sorted(y for y in years if y != 2021)
    channel_episodes = pd.concat(
        [
            pd.read_parquet(
                PROCESSED / f"episodes-{y}.parquet",
                filters=[("start_ts", "<", pd.Timestamp(before))] if before else None,
            )
            for y in years
        ],
        ignore_index=True,
    )
    episodes = group_episodes(channel_episodes)
    channels = pd.read_parquet(PROCESSED / "channels.parquet")
    objects = pd.read_parquet(PROCESSED / "objects.parquet")
    object_lookup = objects.set_index("object_id")
    all_frames = []
    retained = []
    for year in years:
        hourly = pd.read_parquet(
            PROCESSED / f"hourly-{year}.parquet",
            filters=[("hour", "<", pd.Timestamp(before))] if before else None,
        )
        audit = json.loads((ARTIFACTS / f"audit-{year}.json").read_text())
        start = pd.Timestamp(audit["summary"]["start"]).floor("h")
        end = pd.Timestamp(audit["summary"]["end"]).floor("h")
        if before:
            end = min(end, pd.Timestamp(before).ceil("h") - pd.Timedelta(hours=1))
        index = pd.date_range(start, end, freq="h")
        # A complete absence of fleet telemetry is a source gap, not a healthy label.
        fleet = hourly.groupby("hour").events.sum().reindex(index, fill_value=0)
        known = (fleet > 0).astype(np.int8)
        # Keep a full week of history and 25h of observable target + confirmation window.
        past_valid = known.shift(1, fill_value=0).rolling(168, min_periods=168).sum().eq(168)
        future_valid = future_counts(known.to_numpy(), 25) == 25
        valid = past_valid.to_numpy() & future_valid
        year_episodes = episodes[episodes.start_ts.dt.year.eq(year)].copy()
        ep_hours = index.get_indexer(year_episodes.start_ts.dt.floor("h"))
        # Drop episode labels whose first confirmation hour crosses a source outage.
        confirm_ok = np.zeros(len(year_episodes), dtype=bool)
        inside = ep_hours >= 0
        k = known.to_numpy()
        pos = ep_hours[inside]
        confirm_ok[inside] = (k[pos] > 0) & (k[np.minimum(pos + 1, len(k) - 1)] > 0)
        year_episodes = year_episodes[confirm_ok]
        retained.append(year_episodes)
        for obj, history in hourly.groupby("object_id"):
            history = history.set_index("hour").reindex(index)
            # Counts say how many logged reports arrived, never that an unobserved device is healthy.
            observed = history[COUNT_COLUMNS].fillna(0).astype(np.float32)
            frame = pd.DataFrame(index=index)
            for hours in (1, 6, 24, 168):
                sums = observed.shift(1).rolling(hours, min_periods=hours).sum()
                for col in COUNT_COLUMNS:
                    frame[f"{col}_{hours}h"] = sums[col].astype(np.float32)
            for col in (
                "fault_reports",
                "power_reports",
                "unknown_reports",
                "smoke_reports",
                "flood_reports",
                "access_reports",
            ):
                seen = observed[col].shift(1).fillna(0).gt(0)
                last = pd.Series(np.where(seen, np.arange(len(index)), np.nan), index=index).ffill()
                frame[f"{col}_recency_h"] = (np.arange(len(index)) - last).fillna(1e5).astype(np.float32)
                frame[f"{col}_burst"] = frame[f"{col}_6h"] / (1 + frame[f"{col}_168h"] / 28)
            for col in ("temperature", "temperature_max", "gas"):
                # No back-fill: last seen value and its age are separate features.
                lag = history[col].shift(1)
                frame[col] = lag.ffill().astype(np.float32)
                observed_at = pd.Series(
                    np.where(lag.notna(), np.arange(len(index)), np.nan), index=index
                ).ffill()
                frame[f"{col}_age_h"] = (np.arange(len(index)) - observed_at).fillna(1e5).astype(np.float32)
                frame[f"{col}_change_24h"] = (frame[col] - frame[col].shift(24)).astype(np.float32)
                # Do not carry measurements across global data gaps or >7 days of staleness.
                frame.loc[(frame[f"{col}_age_h"] > 168) | ~past_valid, col] = np.nan
            frame["reporting_channels_24h"] = (
                history.reporting_channels.shift(1).rolling(24).max().astype(np.float32)
            )
            object_channels = channels[channels.object_id.eq(obj)]
            frame["channel_count"] = len(object_channels)
            for name, sensor in [
                ("temperature_channels", "Датчик температуры"),
                ("smoke_channels", "Датчик дыма"),
                ("pump_channels", "Состояние насоса"),
                ("water_channels", "Датчик затопления"),
            ]:
                frame[name] = int(object_channels.sensor_type.eq(sensor).sum())
            frame["hour"] = index.hour
            frame["day_of_week"] = index.dayofweek
            frame["month"] = index.month
            frame["weekend"] = (index.dayofweek >= 5).astype(int)
            frame["object_id"] = int(obj)
            frame["parent_id"] = str(object_lookup.loc[obj, "parent_id"])
            frame["object_kind"] = str(object_lookup.loc[obj, "object_kind"])
            frame["as_of"] = index
            for kind in KINDS.values():
                target_eps = year_episodes[year_episodes.object_id.eq(obj) & year_episodes.kind.eq(kind)]
                # Episode existence/duration becomes observable only AFTER confirmation.
                # Fixed cascade grouping also waits until its 10-minute window closes.
                available = (
                    (target_eps.start_ts + pd.Timedelta(minutes=70))
                    .sort_values()
                    .to_numpy(dtype="datetime64[ns]")
                )
                at = index.to_numpy(dtype="datetime64[ns]")
                right = np.searchsorted(available, at, side="left")
                for window in (24, 168, 720):
                    left = np.searchsorted(available, at - np.timedelta64(window, "h"), side="left")
                    frame[f"past_{kind}_episodes_{window}h"] = (right - left).astype(np.float32)
                recency = np.full(len(index), 1e5, dtype=np.float32)
                has_prior = right > 0
                recency[has_prior] = (at[has_prior] - available[right[has_prior] - 1]) / np.timedelta64(
                    1, "h"
                )
                frame[f"past_{kind}_recency_h"] = recency
                target_counts = (
                    target_eps.groupby(target_eps.start_ts.dt.floor("h"))
                    .size()
                    .reindex(index, fill_value=0)
                    .to_numpy()
                )
                frame[f"target_{kind}"] = (future_counts(target_counts, 24) > 0).astype(np.int8)
            frame["eligible"] = valid
            frame = frame.iloc[::step_hours].copy()
            all_frames.append(frame.reset_index(drop=True))
        LOG.info("Features completed for %s", year)
    dataset = pd.concat(all_frames, ignore_index=True).sort_values(["as_of", "object_id"])
    dataset.to_parquet(output_path or PROCESSED / "features.parquet", index=False, compression="zstd")
    final_eps = pd.concat(retained, ignore_index=True).sort_values("start_ts")
    final_eps.to_parquet(episode_output_path or PROCESSED / "episodes.parquet", index=False)
    summary = {
        "years": years,
        "rows": len(dataset),
        "eligible_rows": int(dataset.eligible.sum()),
        "cadence_hours": step_hours,
        "horizon_hours": 24,
        "failure_min_duration_seconds": 3600,
        "cascade_window_minutes": 10,
        "label_type": "proxy_sensor_episodes",
        "episodes": final_eps.kind.value_counts().to_dict(),
        "positive_rows": {k: int(dataset.loc[dataset.eligible, f"target_{k}"].sum()) for k in KINDS.values()},
        "feature_count": len([x for x in dataset if not x.startswith("target_")]) - 2,
        "limitations": [
            "Incident labels are sensor proxies, not confirmed physical incidents.",
            "Catalog is a current snapshot; historical equipment inventory is unavailable.",
            "Last known states assume changes are logged; no per-device heartbeat is available.",
            "Each year starts a new state history; first observations are left-censored.",
        ],
    }
    write_json(audit_output_path or ARTIFACTS / "feature_audit.json", summary)
    LOG.info("Dataset: %s", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    p.add_argument("--force", action="store_true")
    p.add_argument("--step-hours", type=int, default=3)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for y in args.years:
        if y != 2021:
            aggregate_year(y, args.force)
    build_dataset(args.years, args.step_hours)


if __name__ == "__main__":
    main()
