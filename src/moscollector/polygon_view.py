"""Polygon results for the service: report, June feed of the demo world and its dataset.

Everything here is read from files ``scripts/polygon_run.py`` wrote; the service neither
generates worlds nor trains polygon models. The full polygon journal of a day is assembled
on request from the real archive slice and the demo world's injected records.
"""

from __future__ import annotations

import io
import json
import zipfile
from functools import lru_cache

import duckdb
import pandas as pd

from moscollector.paths import ARTIFACTS, BUNDLE, PROCESSED

JOURNAL_COLUMNS = ["ид_события", "ид_канала_данных", "дата", "время", "значение_датчика", "тревожное"]
README = """# Полигон: синтетический датасет мира {world}

Реальный журнал СМВУ 2026 года остаётся фоном. Поверх него добавлены записи инцидентов с
известной правдой и реестры, которых нет в выгрузке организаторов. Частоты и формы сигналов —
допущения команды, описанные в docs/POLYGON.md.

| Файл | Содержимое |
|---|---|
| permits.csv | Наряды-допуски: объект, начало, окончание, вид работ, бригада |
| passports.csv | Паспорта насосов и вентиляторов: дата ввода, последнее ТО, ресурс (допущение) |
| repairs.csv | Журнал аварийных ремонтов после отказов |
| truth.csv | Правда: каждый вставленный инцидент, его время, объект и каналы |
| injected.csv | Вставленные записи в формате выгрузки СМВУ, с номером инцидента |
| suppressed.csv | Окна реальных записей канала, заменённые сценарием износа или роста газа |

Полный журнал полигона за сутки (реальные записи плюс вставленные) отдаёт
`GET /api/polygon/journal.csv?day=2026-06-15`; его можно загрузить как журнал аналитика.
"""


def folder():
    own = ARTIFACTS / "polygon"
    return own if (own / "report.json").exists() else BUNDLE / "artifacts" / "polygon"


def available():
    return (folder() / "report.json").exists()


@lru_cache(maxsize=1)
def report():
    return json.loads((folder() / "report.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def feed():
    return json.loads((folder() / "feed.json").read_text(encoding="utf-8"))


def as_journal(events: pd.DataFrame) -> pd.DataFrame:
    """Organizers' export format: date and time apart, alarm flag as t/f."""
    ts = pd.to_datetime(events.ts)
    return pd.DataFrame(
        {
            "ид_события": events.event_id.astype("int64"),
            "ид_канала_данных": events.channel_id.astype("int64"),
            "дата": ts.dt.strftime("%Y-%m-%d"),
            "время": ts.dt.strftime("%H:%M:%S"),
            "значение_датчика": events.value.astype(str),
            "тревожное": events.alarm.map({True: "t", False: "f"}),
        }
    )


def dataset_zip() -> bytes:
    base = folder()
    injected = pd.read_parquet(base / "injected.parquet")
    journal = as_journal(injected).assign(инцидент=injected.incident_id.to_numpy())
    suppressed = pd.read_parquet(base / "suppressed.parquet")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("permits.csv", "passports.csv", "repairs.csv", "truth.csv"):
            z.write(base / name, name)
        z.writestr("injected.csv", journal.to_csv(index=False))
        z.writestr("suppressed.csv", suppressed.to_csv(index=False))
        z.writestr("README.md", README.format(world=report()["design"]["feed_world"]))
    return buffer.getvalue()


def journal_day(day: str) -> str:
    """Real records of the day without the replaced windows, plus the injected ones."""
    start = pd.Timestamp(day).normalize()
    test = [pd.Timestamp(x) for x in report()["design"]["test"]]
    if not test[0] <= start < test[1]:
        raise ValueError("Журнал полигона доступен за дни июня 2026 года")
    end = start + pd.Timedelta(days=1)
    base = folder()
    con = duckdb.connect()
    real = con.execute(
        "SELECT event_id, channel_id, ts, value, alarm FROM read_parquet(?) WHERE ts >= ? AND ts < ?",
        [str(PROCESSED / "events-2026.parquet"), start.to_pydatetime(), end.to_pydatetime()],
    ).df()
    con.close()
    for s in pd.read_parquet(base / "suppressed.parquet").itertuples():
        real = real[~(real.channel_id.eq(s.channel_id) & real.ts.ge(s.start) & real.ts.lt(s.end))]
    injected = pd.read_parquet(base / "injected.parquet")
    injected = injected[injected.ts.ge(start) & injected.ts.lt(end)]
    events = pd.concat([real, injected[real.columns]], ignore_index=True).sort_values(["ts", "event_id"])
    return as_journal(events).to_csv(index=False)


# --- Registries of the demo world: dispatcher queue labels and the card's situation check ---

ACCESS_SCENARIOS = {"intrusion", "intrusion_probe"}
FIRE_SCENARIOS = {"real_fire", "hot_work_smoke"}
HOUR = pd.Timedelta(hours=1)


@lru_cache(maxsize=1)
def _registry(name: str) -> pd.DataFrame:
    frame = pd.read_csv(folder() / name)
    for column in ("start", "end", "installed", "last_maintenance"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def permits() -> pd.DataFrame:
    return _registry("permits.csv")


def _moment(value) -> pd.Timestamp:
    moment = pd.Timestamp(value)
    if moment.tzinfo is not None:
        moment = moment.tz_convert("Europe/Moscow").tz_localize(None)
    return moment


def _permit(row) -> dict:
    return {
        "permit_id": row.permit_id,
        "work_type": row.work_type,
        "start": row.start.isoformat(),
        "end": row.end.isoformat(),
        "brigade": row.brigade,
    }


def active_permits(object_id: int, at) -> pd.DataFrame:
    at = _moment(at)
    p = permits()
    return p[p.object_id.eq(int(object_id)) & p.start.le(at) & p.end.ge(at)].sort_values(
        ["start", "permit_id"], kind="stable"
    )


def tags(object_id: int, at, scenario: str) -> list[dict]:
    """Registry labels of one alarm or warning, as the dispatcher sees them in the queue."""
    active = active_permits(object_id, at)
    hot = active[active.work_type.eq("Огневые работы")]
    result = []
    if len(hot):
        result.append({"kind": "hot_work", "label": f"Огневые работы по наряду {hot.permit_id.iloc[0]}"})
    other = active[~active.work_type.eq("Огневые работы")]
    if len(other):
        row = other.iloc[0]
        result.append(
            {"kind": "permit", "label": f"Работа по наряду {row.permit_id} · {row.work_type.lower()}"}
        )
    if scenario in ACCESS_SCENARIOS and active.empty:
        result.append({"kind": "no_permit", "label": "Доступ без наряда"})
    if scenario in FIRE_SCENARIOS and hot.empty:
        result.append({"kind": "no_hot_work", "label": "Огневых работ по наряду нет"})
    return result


def queue(as_of, hours: int = 24) -> dict:
    """Feed rows of the demo world within ``hours`` before ``as_of``, with registry labels."""
    end = _moment(as_of)
    begin = end - pd.Timedelta(hours=hours)
    rows = []
    for row in feed():
        at = pd.Timestamp(row["time"])
        if begin < at <= end:
            rows.append({**row, "tags": tags(row["object_id"], at, row["scenario"])})
    rows.sort(key=lambda r: r["time"], reverse=True)
    design = report()["design"]
    return {
        "world": design["feed_world"],
        "period": design["test"],
        "begin": begin.isoformat(),
        "end": end.isoformat(),
        "rows": rows,
        "feed_total": len(feed()),
        "note": "Инциденты и реестры нарядов синтетические, фоном служит реальный журнал СМВУ.",
    }


def situation(object_id: int, at) -> dict:
    """Situation check from the demo world's registries around one moment on one object."""
    at = _moment(at)
    obj = int(object_id)
    p = permits()
    near = p[p.object_id.eq(obj) & p.end.ge(at - 12 * HOUR) & p.start.le(at + 12 * HOUR)].sort_values(
        ["start", "permit_id"], kind="stable"
    )
    repairs = _registry("repairs.csv")
    repairs = repairs[
        repairs.object_id.eq(obj) & repairs.end.ge(at - 48 * HOUR) & repairs.start.le(at + 48 * HOUR)
    ]
    units = _registry("passports.csv")
    units = units[units.object_id.eq(obj)]
    return {
        "world": report()["design"]["feed_world"],
        "at": at.isoformat(),
        "permits": [{**_permit(r), "active": bool(r.start <= at <= r.end)} for r in near.itertuples()],
        "hot_works": [_permit(r) for r in near[near.work_type.eq("Огневые работы")].itertuples()],
        "repairs": [
            {
                "incident_id": r.incident_id,
                "work": r.work,
                "start": r.start.isoformat(),
                "end": r.end.isoformat(),
            }
            for r in repairs.itertuples()
        ],
        "units": {
            "pumps": int(units.group.eq("pump").sum()),
            "fans": int(units.group.eq("fan").sum()),
        },
        "tags": tags(obj, at, ""),
    }


def units(as_of) -> dict:
    """Pumps and fans of the demo world: passports and the June failure forecasts."""
    at = _moment(as_of)
    passports = _registry("passports.csv")
    # The polygon warns per unit; the feed has no unit name, so one row per object and moment.
    warnings, seen = [], {}
    for r in feed():
        if r["scenario"] not in ("pump_failure", "fan_failure"):
            continue
        key = (r["time"], r["object_id"], r["scenario"], r["outcome"])
        if key in seen:
            seen[key]["count"] += 1
            continue
        seen[key] = {**r, "count": 1}
        warnings.append(seen[key])
    rows = []
    for w in warnings:
        issued = pd.Timestamp(w["time"])
        unit = passports[passports.object_id.eq(int(w["object_id"]))]
        unit = unit[unit.group.eq("pump" if w["scenario"] == "pump_failure" else "fan")]
        oldest = unit.sort_values(["installed", "channel_id"], kind="stable").head(1)
        rows.append(
            {
                "time": w["time"],
                "object_id": w["object_id"],
                "object_name": w["object_name"],
                "unit": "Насос" if w["scenario"] == "pump_failure" else "Вентилятор",
                "units_on_object": len(unit),
                "units_warned": w["count"],
                "oldest_installed": oldest.installed.iloc[0].date().isoformat() if len(oldest) else None,
                "last_maintenance": unit.last_maintenance.max().date().isoformat() if len(unit) else None,
                "outcome": w["outcome"],
                "truth": w["truth"],
                "lead_hours": w["lead_hours"],
                "active": bool(issued <= at < issued + 24 * HOUR),
            }
        )
    rows.sort(key=lambda r: r["time"], reverse=True)
    return {
        "world": report()["design"]["feed_world"],
        "period": report()["design"]["test"],
        "pumps": int(passports.group.eq("pump").sum()),
        "fans": int(passports.group.eq("fan").sum()),
        "warnings": rows,
        "active": sum(r["active"] for r in rows),
    }
