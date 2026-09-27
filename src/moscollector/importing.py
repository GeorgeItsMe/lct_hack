"""Bounded file ingestion. Each batch is isolated from the frozen research data."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pandas as pd
from defusedxml import ElementTree

from moscollector.paths import PROCESSED, RUNTIME

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
ALIASES = {
    "ид_канала_данных": "channel_id",
    "дата_время": "ts",
    "значение": "value",
    "тревога": "alarm",
    "timestamp": "ts",
    "id": "event_id",
    "time": "ts",
}


def read_events(content: bytes, extension: str) -> pd.DataFrame:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Размер файла должен быть от 1 байта до 20 МБ")
    try:
        if extension == "json":
            rows = json.loads(content)
            frame = pd.DataFrame(rows.get("events", []) if isinstance(rows, dict) else rows)
        elif extension == "xml":
            root = ElementTree.fromstring(content)
            frame = pd.DataFrame(
                [{child.tag: child.text for child in node} for node in root.findall("event")]
            )
        elif extension == "csv":
            frame = pd.read_csv(
                io.BytesIO(content),
                sep=None,
                engine="python",
                nrows=MAX_ROWS + 1,
                dtype=str,
                encoding="utf-8-sig",
                keep_default_na=False,
            )
        elif extension == "xlsx":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(x.file_size for x in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError("Распакованный XLSX превышает 100 МБ")
            frame = pd.read_excel(io.BytesIO(content), nrows=MAX_ROWS + 1, dtype=str, engine="openpyxl")
        else:
            raise ValueError("Поддерживаются CSV, XLSX, JSON и XML")
    except Exception as error:
        raise ValueError(
            f"Не удалось прочитать {extension.upper()}: проверьте формат и кодировку UTF-8"
        ) from error
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError("В одном импорте должно быть от 1 до 100 000 записей")
    frame = frame.rename(columns=ALIASES)
    if frame.columns.duplicated().any():
        raise ValueError("Неоднозначные имена столбцов")
    required = {"channel_id", "ts", "value", "alarm"}
    if not required.issubset(frame.columns):
        raise ValueError("Обязательные поля: channel_id, ts, value, alarm")
    return frame


def normalize_events(frame: pd.DataFrame, known_channels: set[int], as_of: str):
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("Europe/Moscow").tz_localize(None)
    if pd.isna(cutoff) or cutoff != cutoff.floor("5min"):
        raise ValueError("Момент прогноза должен соответствовать пятиминутной сетке по МСК")
    frame = frame.copy()
    channel = pd.to_numeric(frame.channel_id, errors="coerce")
    timestamps = []
    for value in frame.ts:
        try:
            t = pd.Timestamp(value)
            if t.tzinfo is not None:
                t = t.tz_convert("Europe/Moscow").tz_localize(None)
            timestamps.append(t)
        except (ValueError, TypeError):
            timestamps.append(pd.NaT)
    ts = pd.Series(timestamps, index=frame.index, dtype="datetime64[ns]")
    alarms = (
        frame.alarm.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "t": True, "f": False, "1": True, "0": False})
    )
    value = frame.value.fillna("").astype(str).str.strip()
    invalid = (
        channel.isna()
        | (channel % 1 != 0)
        | ts.isna()
        | alarms.isna()
        | value.eq("")
        | value.str.len().gt(500)
    )
    if invalid.any():
        raise ValueError(
            f"Невалидных строк: {int(invalid.sum())}. Первые номера: {(frame.index[invalid][:5] + 2).tolist()}"
        )
    unknown = ~channel.isin(known_channels)
    if unknown.any():
        raise ValueError(
            f"Неизвестных каналов: {int(channel[unknown].nunique())}. Обновите справочник перед импортом"
        )
    if ts.ge(cutoff).any() or not ts.dt.year.eq(cutoff.year).all():
        raise ValueError("Все записи должны предшествовать моменту прогноза и принадлежать тому же году")
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
        {"input_rows": len(frame), "accepted_rows": len(result), "exact_duplicates": duplicates},
    )


class ImportManager:
    def __init__(self):
        self.root = RUNTIME / "imports"
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="contour-import")
        self.slots = threading.BoundedSemaphore(4)
        self.lock = threading.Lock()
        self.channels = set(pd.read_parquet(PROCESSED / "channels.parquet").channel_id)
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
        frame, cutoff, counts = normalize_events(read_events(content, extension), self.channels, as_of)
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
        candidates = [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*/status.json")]
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
                if (s := json.loads(p.read_text(encoding="utf-8"))) and (mode is None or s.get("mode") == mode)
            ],
            key=lambda x: x["created_at"],
            reverse=True,
        )[:100]

    def detail(self, job_id, object_id, kind, service):
        from moscollector.domain import RECOMMENDATIONS
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
                "recommendations": RECOMMENDATIONS[kind],
            }
        )
