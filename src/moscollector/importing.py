"""Bounded file ingestion. Each batch is isolated from the frozen research data.

Accepted layouts, matched by column name regardless of case and spacing:
- the monitoring journal as delivered to the teams: ид_события, ид_канала_данных,
  дата, время, значение_датчика, тревожное (f/t or false/true);
- the conditional schema of Appendix 1 of the task: ИД канала данных, Текущее значение,
  Дата записи (no alarm flag);
- the service's own layout: channel_id, ts, value, alarm.
CSV (comma, semicolon or tab; UTF-8 or Windows-1251), XLSX, JSON, XML, or a ZIP with one
such file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pandas as pd
from defusedxml import ElementTree

from moscollector.paths import ARTIFACTS, PROCESSED, RUNTIME

MAX_BYTES = 300 * 1024 * 1024
MAX_ROWS = 2_000_000
# Share of malformed rows that is skipped with a report; above it the file is rejected.
MAX_INVALID_SHARE = 0.05
FORMATS = ("csv", "xlsx", "json", "xml", "zip")
ALIASES = {
    "channel_id": "channel_id",
    "channel": "channel_id",
    "ид_канала_данных": "channel_id",
    "ид_канала": "channel_id",
    "канал": "channel_id",
    "ts": "ts",
    "timestamp": "ts",
    "datetime": "ts",
    "date_time": "ts",
    "дата_время": "ts",
    "дата_и_время": "ts",
    "дата_записи": "ts",
    "дата": "date",
    "date": "date",
    "время": "time",
    "time": "time",
    "value": "value",
    "значение": "value",
    "значение_датчика": "value",
    "текущее_значение": "value",
    "alarm": "alarm",
    "тревога": "alarm",
    "тревожное": "alarm",
    "тревожный": "alarm",
    "признак_тревоги": "alarm",
    "id": "event_id",
    "event_id": "event_id",
    "ид_события": "event_id",
    "ид_записи_журнала": "event_id",
}
ALARM_WORDS = {
    "true": True,
    "false": False,
    "t": True,
    "f": False,
    "1": True,
    "0": False,
    "да": True,
    "нет": False,
    "yes": True,
    "no": False,
}
# Values that are alarms whatever the channel; used only when the file has no alarm flag
# and the archive has no statistics for the channel.
ALARM_VALUES = {
    "Неисправен",
    "Обесточен",
    "Отключено устройство",
    "Питание от батарей",
    "Батарея разряжена",
    "Обнаружен дым",
    "Обнаружен газ",
    "Затоплен",
    "Рычаг сдернут",
}


def canonical(name) -> str:
    text = str(name).replace("﻿", "").strip().strip('"').strip().lower().replace("ё", "е")
    return re.sub(r"[\s\-.]+", "_", text)


def _read_csv(content: bytes) -> pd.DataFrame:
    # Decode whole lines only: a cut inside a multi-byte letter must not look like cp1251.
    sample = content[:65536]
    newline = b"\n"
    if len(content) > len(sample) and newline in sample:
        sample = sample[: sample.rindex(newline)]
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            head = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Кодировка файла не распознана: сохраните CSV в UTF-8")
    first = head.splitlines()[0] if head else ""
    try:
        delimiter = csv.Sniffer().sniff(first, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return pd.read_csv(
        io.BytesIO(content),
        sep=delimiter,
        nrows=MAX_ROWS + 1,
        dtype=str,
        encoding=encoding,
        keep_default_na=False,
        skipinitialspace=True,
    )


def _unzip(content: bytes):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [
            m
            for m in archive.infolist()
            if not m.is_dir()
            and not m.filename.startswith("__MACOSX")
            and m.filename.rsplit(".", 1)[-1].lower() in ("csv", "xlsx", "json", "xml")
        ]
        if len(members) != 1:
            raise ValueError("В ZIP должен быть ровно один файл CSV, XLSX, JSON или XML")
        if members[0].file_size > MAX_BYTES:
            raise ValueError(f"Распакованный файл превышает {MAX_BYTES // 1024 // 1024} МБ")
        return archive.read(members[0]), members[0].filename.rsplit(".", 1)[-1].lower()


def read_events(content: bytes, extension: str) -> pd.DataFrame:
    if not content or len(content) > MAX_BYTES:
        raise ValueError(f"Размер файла должен быть от 1 байта до {MAX_BYTES // 1024 // 1024} МБ")
    extension = (extension or "").lower()
    if extension == "zip":
        content, extension = _unzip(content)
    try:
        if extension == "json":
            rows = json.loads(content)
            frame = pd.DataFrame(rows.get("events", []) if isinstance(rows, dict) else rows, dtype=str)
        elif extension == "xml":
            root = ElementTree.fromstring(content)
            frame = pd.DataFrame(
                [{child.tag: child.text for child in node} for node in root.iter("event")], dtype=str
            )
        elif extension == "csv":
            frame = _read_csv(content)
        elif extension == "xlsx":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(x.file_size for x in archive.infolist()) > MAX_BYTES:
                    raise ValueError("Распакованный XLSX слишком большой")
            frame = pd.read_excel(io.BytesIO(content), nrows=MAX_ROWS + 1, dtype=str, engine="openpyxl")
        else:
            raise ValueError("Поддерживаются CSV, XLSX, JSON, XML и ZIP")
    except ValueError as error:
        if "Поддерживаются" in str(error) or "слишком большой" in str(error) or "Кодировка" in str(error):
            raise
        raise ValueError(f"Не удалось прочитать {extension.upper()}: проверьте формат файла") from error
    except Exception as error:
        raise ValueError(f"Не удалось прочитать {extension.upper()}: проверьте формат файла") from error
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"В одном импорте должно быть от 1 до {MAX_ROWS:,} записей".replace(",", " "))
    renamed = {c: ALIASES.get(canonical(c), canonical(c)) for c in frame.columns}
    frame = frame.rename(columns=renamed)
    if frame.columns.duplicated().any():
        duplicated = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError(f"Неоднозначные столбцы: {', '.join(duplicated)}")
    if "ts" not in frame.columns and "date" in frame.columns:
        date = frame["date"].fillna("").astype(str).str.strip()
        time = frame["time"].fillna("").astype(str).str.strip() if "time" in frame.columns else ""
        frame["ts"] = (date + " " + time).str.strip() if "time" in frame.columns else date
    elif "ts" not in frame.columns and "time" in frame.columns:
        frame["ts"] = frame["time"]
    missing = [
        label
        for field, label in (
            ("channel_id", "идентификатор канала (ид_канала_данных / channel_id)"),
            ("ts", "время записи (дата и время / Дата записи / ts)"),
            ("value", "значение (значение_датчика / Текущее значение / value)"),
        )
        if field not in frame.columns
    ]
    if missing:
        raise ValueError("Не найдены столбцы: " + "; ".join(missing))
    return frame


def parse_moments(raw: pd.Series) -> pd.Series:
    """Naive times are Moscow time; values with an offset are converted to it."""
    text = raw.fillna("").astype(str).str.strip()
    result = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    zoned = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if zoned.any():
        parsed = pd.to_datetime(text[zoned], utc=True, errors="coerce", format="ISO8601")
        result[zoned] = parsed.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    naive = ~zoned & text.ne("")
    if naive.any():
        iso = pd.to_datetime(text[naive], errors="coerce", format="ISO8601")
        result[naive] = iso
        rest = naive & result.isna()
        if rest.any():
            result[rest] = pd.to_datetime(text[rest], errors="coerce", dayfirst=True, format="mixed")
    return result


def normalize_events(
    frame: pd.DataFrame,
    known_channels: set[int],
    as_of: str | None = None,
    alarm_lookup=None,
    strict_time: bool = False,
):
    """Validate, clean and date one batch.

    Malformed rows, rows of channels missing from the catalog, rows at or after the forecast
    moment and rows of another year are skipped and counted, so a large journal is not
    rejected for a few bad lines. Without ``as_of`` the moment follows the last record.
    """
    frame = frame.copy()
    total = len(frame)
    channel = pd.to_numeric(frame.channel_id, errors="coerce")
    ts = parse_moments(frame.ts)
    value = frame.value.fillna("").astype(str).str.strip()
    alarm_inferred = "alarm" not in frame.columns
    if alarm_inferred:
        alarms = pd.Series(pd.NA, index=frame.index, dtype="object")
    else:
        alarms = frame.alarm.fillna("").astype(str).str.strip().str.lower().map(ALARM_WORDS)
    invalid = (
        channel.isna()
        | (channel % 1 != 0)
        | ts.isna()
        | (alarms.isna() if not alarm_inferred else False)
        | value.eq("")
        | value.str.len().gt(500)
    )
    invalid_rows = int(invalid.sum())
    if invalid_rows and invalid_rows > MAX_INVALID_SHARE * total:
        reasons = []
        if channel.isna().any() or (channel % 1 != 0).any():
            reasons.append("идентификатор канала не число")
        if ts.isna().any():
            reasons.append("время не распознано")
        if not alarm_inferred and alarms.isna().any():
            reasons.append("флаг тревоги не t/f, true/false или 1/0")
        if value.eq("").any():
            reasons.append("пустое значение")
        raise ValueError(
            f"Невалидных строк: {invalid_rows} из {total} ({'; '.join(reasons)}). "
            f"Первые номера строк: {(frame.index[invalid][:5] + 2).tolist()}"
        )
    keep = ~invalid
    frame, channel, ts, value, alarms = frame[keep], channel[keep], ts[keep], value[keep], alarms[keep]
    unknown = ~channel.isin(known_channels)
    if unknown.all():
        raise ValueError("Ни один канал пакета не найден в справочнике. Обновите справочник перед импортом")
    # Channels missing from the catalog (e.g. dismantled equipment) cannot be placed on an object.
    # They are skipped and reported instead of rejecting the whole batch.
    skipped_channels = sorted(int(c) for c in channel[unknown].unique())
    skipped_rows = int(unknown.sum())
    keep = ~unknown
    channel, ts, value, alarms = channel[keep], ts[keep], value[keep], alarms[keep]
    if as_of:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("Europe/Moscow").tz_localize(None)
        if pd.isna(cutoff) or cutoff != cutoff.floor("5min"):
            raise ValueError("Момент прогноза должен соответствовать пятиминутной сетке по МСК")
        auto = False
    else:
        # Right after the last record, on the five-minute grid the models use.
        cutoff = (ts.max() + pd.Timedelta(seconds=1)).ceil("5min")
        auto = True
    late = ts.ge(cutoff)
    other_year = ~late & ts.dt.year.ne(cutoff.year)
    if strict_time and (late.any() or other_year.any()):
        # The stream keeps a monotonic watermark: a batch with future rows is refused whole.
        raise ValueError("Все записи должны предшествовать моменту прогноза и принадлежать тому же году")
    usable = ~late & ~other_year
    if not usable.any():
        raise ValueError(
            "Все записи позже момента прогноза или относятся к другому году. "
            "Оставьте момент пустым, чтобы взять его по последней записи"
        )
    channel, ts, value, alarms = channel[usable], ts[usable], value[usable], alarms[usable]
    inferred_alarms = 0
    if alarm_inferred:
        guessed = (
            alarm_lookup(channel.astype("int64"), value)
            if alarm_lookup is not None
            else value.isin(ALARM_VALUES)
        )
        alarms = guessed
        inferred_alarms = int(guessed.sum())
    result = pd.DataFrame(
        {"channel_id": channel.astype("int64"), "ts": ts, "value": value, "alarm": alarms.astype(bool)}
    )
    result["numeric_value"] = pd.to_numeric(value.str.replace(",", ".", regex=False), errors="coerce")
    duplicates = int(result.duplicated(["channel_id", "ts", "value", "alarm"]).sum())
    result = result.drop_duplicates(["channel_id", "ts", "value", "alarm"]).reset_index(drop=True)
    # Stable local IDs; the input's identifier is not allowed to overwrite archive rows.
    result["event_id"] = range(9_000_000_000, 9_000_000_000 + len(result))
    return (
        result,
        cutoff,
        {
            "input_rows": total,
            "accepted_rows": len(result),
            "exact_duplicates": duplicates,
            "invalid_rows": invalid_rows,
            "unknown_channel_rows": skipped_rows,
            "unknown_channels": skipped_channels[:20],
            "unknown_channel_count": len(skipped_channels),
            "late_rows": int(late.sum()),
            "other_year_rows": int(other_year.sum()),
            "alarm_inferred": alarm_inferred,
            "inferred_alarm_rows": inferred_alarms,
            "as_of_auto": auto,
            "first_record": ts.min().isoformat(),
            "last_record": ts.max().isoformat(),
        },
    )


class AlarmVocabulary:
    """Alarm flag for files that have none, in order of evidence: what this channel reported for
    this value in the archive; what the value usually meant in the journal (the models learned
    from these flags); the organizers' state catalog for states the journal never showed.

    The catalog is not used first: it marks "Обнаружено движение" or "Не замкнут" as alarms,
    while the journal flags them only when the object is armed (0.2% and 12% of records)."""

    def __init__(self):
        self.by_channel = None
        self.by_value = None
        self.lock = threading.Lock()

    def _load(self):
        import duckdb

        self.by_value = {v: True for v in ALARM_VALUES}
        states = PROCESSED / "sensor_states.csv"
        if states.exists():
            catalog = pd.read_csv(states, dtype=str, encoding="utf-8-sig")
            flags = catalog["тревожное"].str.strip().str.lower().map(ALARM_WORDS)
            for name, flag in zip(catalog["название_состояния"].str.strip(), flags, strict=True):
                if pd.notna(flag):
                    self.by_value[name] = self.by_value.get(name, False) or bool(flag)
        journal = {}
        for path in sorted(ARTIFACTS.glob("audit-*.json")):
            for state in json.loads(path.read_text(encoding="utf-8")).get("states", []):
                rows, alarms = journal.get(state["value"], (0, 0))
                journal[state["value"]] = (rows + state["rows"], alarms + state["alarms"])
        self.by_value.update({v: a * 2 > r for v, (r, a) in journal.items() if r})
        archives = sorted(PROCESSED.glob("events-*.parquet"))
        if archives:
            con = duckdb.connect()
            frame = con.execute(
                "SELECT channel_id, value, avg(alarm::INT) > 0.5 AS alarm FROM read_parquet(?) GROUP BY 1, 2",
                [str(archives[-1])],
            ).df()
            con.close()
            self.by_channel = {(int(c), v): bool(a) for c, v, a in frame.itertuples(index=False)}
        else:
            self.by_channel = {}

    def __call__(self, channel: pd.Series, value: pd.Series) -> pd.Series:
        with self.lock:
            if self.by_channel is None:
                self._load()
        keys = list(zip(channel.tolist(), value.tolist(), strict=True))
        return pd.Series(
            [self.by_channel.get(k, self.by_value.get(k[1], False)) for k in keys], index=channel.index
        )


class ImportManager:
    def __init__(self):
        self.root = RUNTIME / "imports"
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="contour-import")
        self.slots = threading.BoundedSemaphore(4)
        self.lock = threading.Lock()
        self.channels = set(pd.read_parquet(PROCESSED / "channels.parquet").channel_id)
        self.alarms = AlarmVocabulary()
        for path in self.root.glob("*/status.json"):
            status = json.loads(path.read_text(encoding="utf-8"))
            if status["status"] in ("running", "queued"):
                status.update(status="failed", error="Расчёт прерван перезапуском. Загрузите пакет снова.")
                self._save(path.parent, status)

    @staticmethod
    def _save(directory, state):
        temporary = directory / "status.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "status.json")

    def submit(self, content, extension, as_of, user_id):
        frame, cutoff, counts = normalize_events(
            read_events(content, extension), self.channels, as_of, self.alarms
        )
        identity = hashlib.sha256(content + cutoff.isoformat().encode()).hexdigest()
        return self.submit_frame(frame, cutoff, counts, identity, extension, user_id)

    def submit_frame(self, frame, cutoff, counts, identity, extension, user_id, metadata=None):
        """Queue an immutable snapshot, also used by the accumulated stream."""
        from moscollector.model_registry import active_version

        with self.lock:
            version = active_version()
            previous = self.find(identity, model_version=version)
            if previous and previous["status"] != "failed":
                return previous
            if not self.slots.acquire(blocking=False):
                raise ValueError("Очередь из четырёх пакетов заполнена. Дождитесь завершения расчёта")
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            frame.to_parquet(directory / "input.parquet", index=False)
            state = {
                "id": job_id,
                "status": "queued",
                "sha256": identity,
                "as_of": cutoff.isoformat(),
                "created_at": datetime.now(UTC).isoformat(),
                "user_id": user_id,
                "format": extension,
                **counts,
                **(metadata or {}),
                "model_version": version,
            }
            self._save(directory, state)
            self.pool.submit(self._run, directory, state)
            return state.copy()

    def find(self, identity, model_version=None):
        candidates = [
            json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*/status.json")
        ]
        return next(
            (
                s
                for s in sorted(candidates, key=lambda s: s["created_at"], reverse=True)
                if s["sha256"] == identity
                and (model_version is None or s.get("model_version", "legacy") == model_version)
            ),
            None,
        )

    def _run(self, directory, state):
        try:
            state["status"] = "running"
            self._save(directory, state)
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, "-m", "moscollector.batch", str(directory), state["as_of"]],
                    stdout=log,
                    stderr=log,
                    timeout=300,
                    env={**os.environ, "PYTHONWARNINGS": "ignore::DeprecationWarning"},
                )
            if result.returncode:
                error_path = directory / "error.json"
                raise ValueError(
                    json.loads(error_path.read_text(encoding="utf-8"))["error"]
                    if error_path.exists()
                    else "Ошибка расчёта; проверьте журнал worker.log"
                )
            state["status"] = "complete"
        except Exception as error:
            state.update(status="failed", error=str(error))
        finally:
            state["finished_at"] = datetime.now(UTC).isoformat()
            self._save(directory, state)
            self.slots.release()

    def get(self, job_id, include_result=True):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise KeyError("Пакет не найден")
        directory = self.root / job_id
        if not (directory / "status.json").exists():
            raise KeyError("Пакет не найден")
        state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if include_result and state["status"] == "complete":
            state["result"] = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return state

    def list(self, mode=None):
        return sorted(
            [
                s
                for p in self.root.glob("*/status.json")
                if (s := json.loads(p.read_text(encoding="utf-8")))
                and (mode is None or s.get("mode") == mode)
            ],
            key=lambda x: x["created_at"],
            reverse=True,
        )[:100]

    def detail(self, job_id, object_id, kind, service):
        from moscollector.domain import data_recommendations
        from moscollector.service import clean

        state = self.get(job_id)
        if state["status"] != "complete":
            raise ValueError("Расчёт ещё не завершён")
        forecast = next(
            (r for r in state["result"]["forecasts"] if r["object_id"] == object_id and r["kind"] == kind),
            None,
        )
        if forecast is None:
            raise KeyError("Прогноз в пакете не найден")
        directory = self.root / job_id
        frame = pd.read_parquet(
            directory / "feature_snapshot.parquet", filters=[("object_id", "=", object_id)]
        )
        import duckdb

        cutoff = pd.Timestamp(state["as_of"])
        con = duckdb.connect()
        con.execute("SET threads=2")
        try:
            con.read_parquet(str(directory / "input.parquet")).create_view("incoming")
            con.read_parquet(str(PROCESSED / "channels.parquet")).create_view("channels")
            archive = PROCESSED / f"events-{cutoff.year}.parquet"
            if archive.exists():
                con.read_parquet(str(archive)).create_view("archive")
                con.execute(
                    "CREATE VIEW combined AS SELECT channel_id,ts,value,alarm FROM archive UNION ALL SELECT channel_id,ts,value,alarm FROM incoming"
                )
            else:
                con.execute("CREATE VIEW combined AS SELECT channel_id,ts,value,alarm FROM incoming")
            source = con.execute(
                "SELECT DISTINCT e.channel_id,e.ts,e.value,e.alarm,c.sensor_type FROM combined e JOIN channels c USING(channel_id) "
                "WHERE c.object_id=? AND e.ts<? AND e.ts>=? ORDER BY e.alarm DESC,e.ts DESC LIMIT 40",
                [object_id, cutoff.to_pydatetime(), (cutoff - pd.Timedelta(hours=24)).to_pydatetime()],
            ).df()
        finally:
            con.close()
        records = source.to_dict("records")
        details = data_recommendations(kind, records)
        return clean(
            {
                **forecast,
                "batch_id": job_id,
                "id": f"batch:{job_id}:{object_id}:{kind}",
                "as_of": state["as_of"],
                "model_version": state["result"].get("model_version", "legacy"),
                "explanation": service.explain_features(
                    frame, kind, state["result"].get("model_version", "legacy")
                ),
                "explanation_unit": "log_odds",
                "source_events": source.to_dict("records"),
                "recommendations": [r["text"] for r in details],
                "recommendation_details": details,
                "source": "stream" if state.get("mode") == "accumulated_stream" else "import",
            }
        )
