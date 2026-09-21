"""Durable, idempotent incoming telemetry and immutable prediction snapshots.

The gateway pushes already anonymized JSON/XML/CSV/XLSX batches. Each transaction
stores a receipt and new events together. A retry never doubles the history. A
failed or full prediction queue does not roll back accepted source events.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from moscollector.database import AuditLog, StreamBatch, StreamEvent
from moscollector.importing import normalize_events, read_events


def event_hash(row):
    # Canonical fields, not source event IDs: duplicates across encodings collapse,
    # while conflicting states in the same second survive for the feature encoder.
    payload = [int(row.channel_id), row.ts.isoformat(), row.value, bool(row.alarm)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def receipt(row):
    return {
        "id": row.id,
        "sha256": row.sha256,
        "as_of": row.as_of.isoformat(),
        "input_rows": row.input_rows,
        "inserted_rows": row.inserted_rows,
        "duplicate_rows": row.duplicate_rows,
        "received_at": row.received_at.isoformat() + "Z",
    }


class StreamManager:
    def __init__(self, factory, importer):
        self.factory = factory
        self.importer = importer
        self.lock = threading.RLock()

    def ingest(self, content, extension, as_of, user_id):
        frame, cutoff, counts = normalize_events(
            read_events(content, extension), self.importer.channels, as_of
        )
        records = []
        for row in frame.itertuples():
            records.append(
                {
                    "event_hash": event_hash(row),
                    "channel_id": int(row.channel_id),
                    "ts": row.ts.to_pydatetime(),
                    "value": row.value,
                    "numeric_value": None if pd.isna(row.numeric_value) else float(row.numeric_value),
                    "alarm": bool(row.alarm),
                }
            )
        identity = hashlib.sha256(
            (cutoff.isoformat() + "|" + "|".join(sorted(r["event_hash"] for r in records))).encode()
        ).hexdigest()
        with self.lock, self.factory() as db:
            previous = db.scalar(select(StreamBatch).where(StreamBatch.sha256 == identity))
            if previous:
                accepted = {**receipt(previous), "replayed": True}
                original_identity = hashlib.sha256(
                    f"stream:{previous.id}:{cutoff.isoformat()}".encode()
                ).hexdigest()
                original_job = self.importer.find(original_identity)
                if original_job and original_job.get("status") != "failed":
                    return {"receipt": accepted, "job": original_job}
            else:
                watermark = db.scalar(select(func.max(StreamBatch.as_of)))
                if watermark and cutoff < pd.Timestamp(watermark):
                    raise ValueError(
                        "Момент потока нельзя уменьшать. Опоздавшие события подавайте с текущим моментом"
                    )
                insert = pg_insert if db.bind.dialect.name == "postgresql" else sqlite_insert
                inserted = 0
                # Bound SQL bind parameters on both supported databases.
                for offset in range(0, len(records), 500):
                    statement = insert(StreamEvent).values(records[offset : offset + 500])
                    keys = db.scalars(
                        statement.on_conflict_do_nothing(index_elements=["event_hash"]).returning(
                            StreamEvent.event_hash
                        )
                    ).all()
                    inserted += len(keys)
                batch = StreamBatch(
                    sha256=identity,
                    as_of=cutoff.to_pydatetime(),
                    input_rows=counts["input_rows"],
                    inserted_rows=inserted,
                    duplicate_rows=counts["input_rows"] - inserted,
                    user_id=user_id,
                )
                db.add(batch)
                db.flush()
                accepted = {**receipt(batch), "replayed": False}
                db.add(
                    AuditLog(
                        user_id=user_id,
                        action="stream_batch_accepted",
                        detail=json.dumps(accepted, ensure_ascii=False),
                    )
                )
                db.commit()
            # Receipt is durable before CPU work. Retrying also recovers a queue
            # failure, using the latest immutable revision for the same cutoff.
            try:
                job = self.forecast(cutoff.isoformat(), user_id)
                return {"receipt": accepted, "job": job}
            except ValueError as error:
                return {"receipt": accepted, "job": None, "forecast_error": str(error)}

    def forecast(self, as_of, user_id):
        with self.lock, self.factory() as db:
            cutoff = pd.Timestamp(as_of)
            if cutoff.tzinfo is not None:
                cutoff = cutoff.tz_convert("Europe/Moscow").tz_localize(None)
            if pd.isna(cutoff) or cutoff != cutoff.floor("5min"):
                raise ValueError("Момент прогноза должен соответствовать пятиминутной сетке по МСК")
            revision = db.scalar(select(func.max(StreamBatch.id)))
            if revision is None:
                raise ValueError("Поток ещё не содержит событий")
            watermark = db.scalar(select(func.max(StreamBatch.as_of)))
            if cutoff != pd.Timestamp(watermark):
                raise ValueError("Расчёт потока доступен только на последнем принятом моменте")
            rows = (
                db.execute(
                    select(
                        StreamEvent.event_hash,
                        StreamEvent.channel_id,
                        StreamEvent.ts,
                        StreamEvent.value,
                        StreamEvent.numeric_value,
                        StreamEvent.alarm,
                    )
                    .where(
                        StreamEvent.ts >= datetime(cutoff.year, 1, 1),
                        StreamEvent.ts < cutoff,
                    )
                    .order_by(StreamEvent.ts, StreamEvent.event_hash)
                )
                .mappings()
                .all()
            )
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise ValueError("Нет событий текущего года перед моментом прогноза")
            frame["ts"] = pd.to_datetime(frame.ts)
            frame["event_id"] = frame.event_hash.map(lambda x: int(x[:15], 16))
            frame = frame.drop(columns="event_hash")
            identity = hashlib.sha256(f"stream:{revision}:{cutoff.isoformat()}".encode()).hexdigest()
            return self.importer.submit_frame(
                frame,
                cutoff,
                {"input_rows": len(frame), "accepted_rows": len(frame), "exact_duplicates": 0},
                identity,
                "stream",
                user_id,
                {
                    "mode": "accumulated_stream",
                    "stream_revision": revision,
                    "snapshot_at": datetime.now(UTC).isoformat(),
                },
            )

    def status(self):
        with self.factory() as db:
            rows, begin, end = db.execute(
                select(func.count(), func.min(StreamEvent.ts), func.max(StreamEvent.ts)).select_from(
                    StreamEvent
                )
            ).one()
            batches = db.scalars(select(StreamBatch).order_by(StreamBatch.id.desc()).limit(10)).all()
            jobs = [j for j in self.importer.list() if j.get("mode") == "accumulated_stream"]
            return {
                "source": "smvu_gateway",
                "source_connection": "push_api_ready",
                "external_system_connected": False,
                "events": rows,
                "first_event": begin.isoformat() if begin else None,
                "last_event": end.isoformat() if end else None,
                "as_of": max((b.as_of for b in batches), default=None),
                "receipts": [receipt(b) for b in batches],
                "latest_job": jobs[0] if jobs else None,
                "notice": "Накопленные копии пакетов в собственной БД. Подключение реальной СМВУ ещё не выполнено.",
            }
