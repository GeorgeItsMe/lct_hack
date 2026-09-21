import json

import pandas as pd
import pytest
from sqlalchemy import func, select

from moscollector.database import AuditLog, StreamBatch, StreamEvent, make_database
from moscollector.streaming import StreamManager


class SnapshotQueue:
    channels = {1, 2}

    def __init__(self):
        self.snapshots = {}
        self.jobs = {}
        self.full = False

    def submit_frame(self, frame, cutoff, counts, identity, extension, user_id, metadata=None):
        if self.full:
            raise ValueError("Очередь заполнена")
        self.snapshots.setdefault(identity, frame.copy(deep=True))
        self.jobs[identity] = {"id": identity[:32], "as_of": cutoff.isoformat(), **counts, **(metadata or {})}
        return self.jobs[identity]

    def find(self, identity):
        return self.jobs.get(identity)

    def list(self):
        return []


@pytest.fixture
def stream(tmp_path):
    engine, factory = make_database(f"sqlite:///{tmp_path / 'stream.db'}")
    manager = StreamManager(factory, SnapshotQueue())
    yield manager, factory
    engine.dispose()


def batch(*rows):
    return json.dumps(rows).encode()


def event(**change):
    return {"channel_id": 1, "ts": "2026-06-15T11:59:00", "value": "Норма", "alarm": False, **change}


def send(manager, *rows, at="2026-06-15T12:00"):
    return manager.ingest(batch(*rows), "json", at, 1)


def test_duplicate_delivery_and_cross_format_retry_are_idempotent(stream):
    manager, factory = stream
    first = send(manager, event(), event())
    duplicate = manager.ingest(
        pd.DataFrame([event()]).to_csv(index=False).encode(), "csv", "2026-06-15T12:00", 1
    )
    assert duplicate["receipt"]["replayed"]
    assert duplicate["receipt"]["id"] == first["receipt"]["id"]
    assert duplicate["job"]["id"] == first["job"]["id"]
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(StreamEvent)) == 1
        assert db.scalar(select(func.count()).select_from(StreamBatch)) == 1
        assert db.scalar(select(func.count()).select_from(AuditLog)) == 1


def test_next_batch_keeps_history_but_does_not_change_old_snapshot(stream):
    manager, _ = stream
    first = send(manager, event())
    second = send(
        manager, event(), event(ts="2026-06-15T12:02", value="Неисправен", alarm=True), at="2026-06-15T12:05"
    )
    snapshots = list(manager.importer.snapshots.values())
    assert [len(f) for f in snapshots] == [1, 2]
    assert second["receipt"]["inserted_rows"] == 1
    assert second["receipt"]["duplicate_rows"] == 1
    assert first["job"]["id"] != second["job"]["id"]
    assert snapshots[0].ts.max() < pd.Timestamp(first["job"]["as_of"])
    old_retry = send(manager, event())
    assert old_retry["job"]["id"] == first["job"]["id"]


def test_conflicting_same_second_states_remain_for_canonicalization(stream):
    manager, _ = stream
    send(manager, event(), event(value="Неисправен", alarm=True))
    frame = next(iter(manager.importer.snapshots.values()))
    assert len(frame) == 2
    assert frame.ts.nunique() == 1
    assert set(frame.value) == {"Норма", "Неисправен"}


def test_queue_failure_keeps_receipt_and_restart_recovers(stream):
    manager, factory = stream
    manager.importer.full = True
    result = send(manager, event())
    assert result["job"] is None and "forecast_error" in result
    restarted = StreamManager(factory, SnapshotQueue())
    retry = send(restarted, event())
    assert retry["receipt"]["replayed"] and retry["job"]
    assert restarted.status()["events"] == 1


def test_future_rows_and_regressive_watermark_are_rejected_atomically(stream):
    manager, factory = stream
    send(manager, event())
    with pytest.raises(ValueError):
        send(manager, event(ts="2026-06-15T12:05"), at="2026-06-15T12:05")
    with pytest.raises(ValueError):
        send(manager, event(ts="2026-06-15T11:50"), at="2026-06-15T11:55")
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(StreamEvent)) == 1
        assert db.scalar(select(func.count()).select_from(StreamBatch)) == 1


def test_late_events_revise_current_snapshot_without_rewriting_old(stream):
    manager, _ = stream
    first = send(manager, event())
    late = send(manager, event(ts="2026-06-15T11:45", channel_id=2))
    assert first["job"]["as_of"] == late["job"]["as_of"]
    assert first["job"]["stream_revision"] < late["job"]["stream_revision"]
    assert [len(f) for f in manager.importer.snapshots.values()] == [1, 2]


def test_inference_cannot_silently_jump_past_watermark(stream):
    manager, _ = stream
    send(manager, event())
    with pytest.raises(ValueError):
        manager.forecast("2026-06-15T12:10", 1)
