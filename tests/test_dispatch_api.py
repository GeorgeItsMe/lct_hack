from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from moscollector.database import AuditLog, hash_password, make_database, verify_password
from moscollector.service import prediction_id


@pytest.fixture
def client(tmp_path, monkeypatch):
    from moscollector import api

    engine, factory = make_database(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setattr(api, "SessionFactory", factory)
    monkeypatch.setattr(api, "DEMO_MODE", True)
    t = pd.Timestamp("2026-06-15 12:00:00")
    identifier = prediction_id(1, "fault", t)
    mock = SimpleNamespace(
        resolve_time=lambda _: t,
        forecast_rows=lambda _: [{"id": identifier}],
        object_map={1: {"object_name": "Объект Альфа"}},
        thresholds=lambda overrides=None: {"fault": 0.2, **(overrides or {})},
    )
    monkeypatch.setattr(api, "analytics", lambda: mock)
    with TestClient(api.app) as c:
        yield c, factory, identifier


def login(c, username):
    result = c.post("/api/auth/login", json={"username": username, "password": "contour-demo"})
    assert result.status_code == 200


def test_password_hash_is_salted_and_verified():
    first = hash_password("correct horse")
    assert first != hash_password("correct horse")
    assert verify_password("correct horse", first)
    assert not verify_password("wrong", first)


def test_anonymous_cannot_read_decisions(client):
    c, _, _ = client
    assert c.get("/api/decisions").status_code == 401


def test_analyst_cannot_mutate_operational_decisions(client):
    c, _, identifier = client
    login(c, "analyst")
    body = {
        "prediction_id": identifier,
        "object_id": 1,
        "kind": "fault",
        "as_of": "2026-06-15T12:00:00",
        "action": "dispatch",
        "reason": "sensor_pattern",
        "comment": "Проверка",
    }
    assert c.post("/api/decisions", json=body).status_code == 403
    assert c.put("/api/settings", json={"thresholds": {"fault": 0.1}}).status_code == 403


def test_dispatcher_decision_is_idempotent_audited_and_survives_read(client):
    c, factory, identifier = client
    login(c, "dispatcher")
    body = {
        "prediction_id": identifier,
        "object_id": 1,
        "kind": "fault",
        "as_of": "2026-06-15T12:00:00",
        "action": "monitor",
        "reason": "sensor_pattern",
        "comment": "Проверены соседние каналы",
    }
    assert c.post("/api/decisions", json=body).status_code == 200
    body["action"] = "maintenance"
    assert c.post("/api/decisions", json=body).status_code == 200
    rows = c.get("/api/decisions").json()
    assert len(rows) == 1
    assert rows[0]["action"] == "maintenance"
    from sqlalchemy import select

    with factory() as db:
        logs = db.scalars(select(AuditLog).where(AuditLog.action == "decision_saved")).all()
        assert len(logs) == 2
        assert '"previous": {"action": "monitor"' in logs[1].detail
    assert c.post("/api/decisions", json={**body, "prediction_id": "forged"}).status_code == 422


def test_admin_thresholds_validate_and_persist(client):
    c, _, _ = client
    login(c, "admin")
    assert c.put("/api/settings", json={"thresholds": {"fault": -1}}).status_code == 422
    assert c.put("/api/settings", json={"thresholds": {"invented": 0.5}}).status_code == 422
    assert c.put("/api/settings", json={"thresholds": {"fault": 0.15}}).status_code == 200
    assert c.get("/api/settings").json()["thresholds"]["fault"] == 0.15
    assert c.post("/api/auth/logout").status_code == 200
    assert c.get("/api/settings").status_code == 401
