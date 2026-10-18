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
