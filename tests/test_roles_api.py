"""Role matrix and the workflows it enables: dispatcher → technician → analyst → unit head → admin."""

import csv
import io
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from moscollector.database import AuditLog, User, hash_password, make_database, seed_users
from moscollector.roles import DEMO_ACCOUNTS, PERMISSIONS, ROLES
from moscollector.service import prediction_id

T = pd.Timestamp("2026-06-15 12:00:00")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from moscollector import api, workorders

    engine, factory = make_database(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setattr(api, "SessionFactory", factory)
    monkeypatch.setattr(api, "DEMO_MODE", True)
    monkeypatch.setattr(api, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(workorders, "EMULATOR_STEP_SECONDS", 0.0)
    identifier = prediction_id(1, "fault", T)

    def detail(obj, kind, as_of=None, overrides=None):
        return {
            "id": prediction_id(obj, kind, T),
            "object_id": obj,
            "object_name": "Объект Альфа",
            "kind": kind,
            "as_of": T.isoformat(),
            "probability": 0.4,
            "threshold": 0.2,
            "risk": "high",
            "recommendations": ["Проверить питание и связь группы каналов на объекте."],
            "explanation": [{"label": "Сообщения о неисправности за 24 ч"}],
        }

    mock = SimpleNamespace(
        resolve_time=lambda *_: T,
        forecast_rows=lambda *_: [{"id": identifier}],
        object_map={1: {"object_name": "Объект Альфа"}},
        thresholds=lambda overrides=None: {"fault": 0.2, "fire": 0.3, **(overrides or {})},
        detail=detail,
        retrospective=lambda obj, kind, as_of: {"occurred": True},
        channel_states=lambda _: pd.DataFrame({"object_id": [1], "current_fault": [False]}),
        summary=lambda *_: {"as_of": T.isoformat(), "totals": {"warnings": 3}},
        threshold_preview=lambda kind, value, *_: {
            "current": {"threshold": 0.2, "precision": 0.3, "recall": 0.4},
            "proposed": {"threshold": value, "precision": 0.5, "recall": 0.3},
            "period_label": "policy",
        },
    )
    monkeypatch.setattr(api, "analytics", lambda: mock)
    with TestClient(api.app) as c:
        yield c, factory, identifier


def login(c, username):
    result = c.post("/api/auth/login", json={"username": username, "password": "contour-demo"})
    assert result.status_code == 200
    return result.json()["user"]


def decision_body(identifier, action="dispatch"):
    return {
        "prediction_id": identifier,
        "object_id": 1,
        "kind": "fault",
        "as_of": T.isoformat(),
        "action": action,
        "reason": "sensor_pattern",
        "comment": "Повторные сообщения о неисправности",
    }


def test_every_role_logs_in_with_its_own_permission_set(client):
    c, _, _ = client
    assert [r for _, _, r in DEMO_ACCOUNTS] == list(ROLES)
    for username, _, role in DEMO_ACCOUNTS:
        user = login(c, username)
        assert user["role"] == role and user["role_label"] == ROLES[role]
        assert user["permissions"] == sorted(p for p, roles in PERMISSIONS.items() if role in roles)
    login(c, "admin")
    assert "decisions.write" not in c.get("/api/auth/me").json()["permissions"]


@pytest.mark.parametrize(
    ("method", "path", "body", "allowed_role"),
    [
        ("post", "/api/decisions", "decision", "dispatcher"),
        ("post", "/api/work-orders", {"object_id": 1, "kind": "fault"}, "technician"),
        ("put", "/api/settings", {"thresholds": {"fault": 0.25}}, "manager"),
        ("get", "/api/labels", None, "analyst"),
        (
            "post",
            "/api/thresholds/proposals",
            {"kind": "fault", "threshold": 0.3, "rationale": "Снизить нагрузку"},
            "analyst",
        ),
        ("get", "/api/users", None, "admin"),
        ("get", "/api/audit", None, "admin"),
    ],
)
def test_write_actions_are_reserved_to_one_role(client, method, path, body, allowed_role):
    c, _, identifier = client
    payload = decision_body(identifier) if body == "decision" else body
    for username, _, role in DEMO_ACCOUNTS:
        login(c, username)
        response = getattr(c, method)(path, **({"json": payload} if payload is not None else {}))
        if role == allowed_role:
            assert response.status_code in (200, 201), (role, path, response.text)
        else:
            assert response.status_code == 403, (role, path, response.status_code)


def test_read_permissions_follow_the_matrix(client):
    c, _, _ = client
    expected = {
        "/api/tasks": {"dispatcher", "technician", "analyst", "manager"},
        "/api/work-orders": {"dispatcher", "technician", "analyst", "manager"},
        "/api/summary": {"analyst", "manager", "admin"},
        "/api/thresholds/proposals": {"analyst", "manager", "admin"},
        "/api/decisions": set(ROLES),
    }
    for username, _, role in DEMO_ACCOUNTS:
        login(c, username)
        for path, roles in expected.items():
            status = c.get(path).status_code
            assert (status == 200) == (role in roles), (role, path, status)


def test_decision_to_work_order_to_archive_outcome(client):
    c, factory, identifier = client
    login(c, "dispatcher")
    assert c.post("/api/decisions", json=decision_body(identifier, "inspect")).status_code == 200
    login(c, "technician")
    task = c.get("/api/tasks").json()
    assert len(task) == 1 and task[0]["action_label"] == "Направить бригаду на проверку"
    order = c.post("/api/work-orders", json={"decision_id": task[0]["id"]}).json()
    assert order["status"] == "draft" and order["number"].startswith("З-")
    assert "Направить бригаду на проверку" in order["description"]
    assert order["priority"] == "high"
    assert c.post("/api/work-orders", json={"decision_id": task[0]["id"]}).status_code == 409
    edited = c.patch(f"/api/work-orders/{order['id']}", json={"priority": "urgent"}).json()
    assert edited["priority_label"] == "Срочно"
    submitted = c.post(f"/api/work-orders/{order['id']}/submit").json()
    assert submitted["status"] == "submitted" and submitted["external_id"] == f"HD-{order['id']:06d}"
    assert c.patch(f"/api/work-orders/{order['id']}", json={"priority": "normal"}).status_code == 409
    # Statuses are read back from the (emulated) external system, never typed in by staff.
    synced = c.post("/api/work-orders/sync").json()
    assert synced["changed"][0]["status"] == "done" and synced["changed"][0]["outcome"] == "confirmed"
    login(c, "dispatcher")
    row = c.get("/api/decisions").json()[0]
    assert row["work_order"]["outcome_label"] == "Событие подтвердилось"
    with factory() as db:
        actions = set(db.scalars(select(AuditLog.action)))
    assert {"work_order_drafted", "work_order_submitted", "work_orders_synced"} <= actions


def test_analyst_verifies_labels_and_freezes_them_for_retraining(client, tmp_path):
    c, _, identifier = client
    login(c, "dispatcher")
    c.post("/api/decisions", json=decision_body(identifier, "false_alarm"))
    login(c, "analyst")
    assert c.post("/api/retraining").status_code == 422
    rows = c.get("/api/labels").json()["rows"]
    assert rows[0]["suggested_label"] == 0 and "ложное" in rows[0]["label_basis"]
    assert rows[0]["archive_occurred"] is True
    decision = rows[0]["id"]
    assert c.put(f"/api/labels/{decision}", json={"verdict": "accepted"}).status_code == 422
    assert c.put(f"/api/labels/{decision}", json={"verdict": "rejected"}).status_code == 422
    assert (
        c.put(
            f"/api/labels/{decision}", json={"verdict": "accepted", "label": 1, "note": "Архив"}
        ).status_code
        == 200
    )
    exported = list(csv.reader(io.StringIO(c.get("/api/labels/export.csv").text.lstrip("﻿"))))
    assert exported[1][4] == "1" and exported[1][5] == "false_alarm"
    request = c.post("/api/retraining").json()
    assert request["labels"] == 1 and request["positives"] == 1
    assert (tmp_path / "runtime" / "retraining" / request["file_name"]).exists()
    assert c.get("/api/labels").json()["retraining"][0]["status"] == "prepared"


def test_threshold_change_needs_analyst_proposal_and_manager_approval(client):
    c, _, _ = client
    login(c, "analyst")
    body = {"kind": "fault", "threshold": 0.35, "rationale": "Снизить ложные предупреждения"}
    proposal = c.post("/api/thresholds/proposals", json=body).json()
    assert proposal["status"] == "pending" and proposal["preview"]["proposed"]["threshold"] == 0.35
    assert c.post("/api/thresholds/proposals", json=body).status_code == 409
    assert c.post(f"/api/thresholds/proposals/{proposal['id']}/approve", json={}).status_code == 403
    login(c, "manager")
    assert c.post(f"/api/thresholds/proposals/{proposal['id']}/reject", json={}).status_code == 422
    approved = c.post(
        f"/api/thresholds/proposals/{proposal['id']}/approve", json={"note": "Согласовано"}
    ).json()
    assert approved["status"] == "approved" and approved["decided_by"] == "Руководитель подразделения"
    assert c.get("/api/settings").json()["thresholds"]["fault"] == 0.35
    assert c.post(f"/api/thresholds/proposals/{proposal['id']}/approve", json={}).status_code == 409


def test_admin_manages_users_but_cannot_lock_itself_out(client):
    c, factory, _ = client
    me = login(c, "admin")
    created = c.post(
        "/api/users",
        json={
            "username": "ivanova",
            "name": "Иванова А.",
            "role": "technician",
            "password": "long-enough-pass",
        },
    )
    assert created.status_code == 201
    assert (
        c.post(
            "/api/users", json={"username": "short", "name": "Кто-то", "role": "analyst", "password": "x"}
        ).status_code
        == 422
    )
    target = created.json()["id"]
    assert (
        c.patch(f"/api/users/{target}", json={"role": "manager", "scope": "node:3218"}).json()["scope"]
        == "node:3218"
    )
    assert c.patch(f"/api/users/{me['id']}", json={"active": False}).status_code == 422
    assert c.patch(f"/api/users/{me['id']}", json={"role": "analyst"}).status_code == 422
    assert c.patch(f"/api/users/{target}", json={"active": False}).json()["active"] is False
    login_attempt = c.post("/api/auth/login", json={"username": "ivanova", "password": "long-enough-pass"})
    assert login_attempt.status_code == 401
    with factory() as db:
        assert db.scalar(select(User).where(User.username == "ivanova")).role == "manager"


def test_demo_database_from_previous_release_gains_new_roles(tmp_path):
    engine, factory = make_database(f"sqlite:///{tmp_path / 'old.db'}")
    with factory() as db:
        for username, role in (("dispatcher", "dispatcher"), ("analyst", "analyst"), ("admin", "admin")):
            db.add(
                User(
                    username=username,
                    display_name=username,
                    role=role,
                    password_hash=hash_password("contour-demo"),
                )
            )
        db.commit()
    seed_users(factory, True)
    with factory() as db:
        assert set(db.scalars(select(User.username))) == {u for u, _, _ in DEMO_ACCOUNTS}
    engine.dispose()


def test_model_diagnostics_belong_to_the_analyst_only(client):
    c, _, _ = client
    for username in ("dispatcher", "technician", "manager", "admin"):
        login(c, username)
        for path in ("/api/evaluation", "/api/quality", "/api/evaluation/matches/fire"):
            assert c.get(path).status_code == 403, (username, path)


def test_stream_forecasts_join_the_queue_with_their_source(monkeypatch):
    from moscollector import api

    job = "b" * 32
    rows = [
        {
            "object_id": 1,
            "object_name": "Объект Альфа",
            "kind": "access",
            "kind_label": "Охранный сигнал",
            "probability": 0.8,
            "threshold": 0.3,
            "above_threshold": True,
            "recommendation": "Проверить режим охраны",
        },
        {
            "object_id": 1,
            "object_name": "Объект Альфа",
            "kind": "fire",
            "kind_label": "Пожарный сигнал",
            "probability": 0.5,
            "threshold": 0.3,
            "above_threshold": False,
            "recommendation": "Проверить датчики",
        },
    ]
    manager = SimpleNamespace(
        list=lambda mode=None: [{"id": job, "status": "complete", "mode": mode}],
        get=lambda _: {"result": {"as_of": "2026-06-15T12:05:00", "forecasts": rows}},
    )
    monkeypatch.setattr(api, "imports", lambda: manager)
    monkeypatch.setattr(api, "analytics", lambda: SimpleNamespace(object_map={1: {"parent_id": 7}}))
    monkeypatch.setattr(api, "SERVERLESS_MODE", False)
    snapshot = api.stream_snapshot()
    first, second = snapshot["forecasts"]
    assert snapshot["as_of"] == "2026-06-15T12:05:00"
    assert first["id"] == f"batch:{job}:1:access" and first["batch_id"] == job
    assert first["source"] == "stream" and first["parent_id"] == 7 and first["risk"] == "critical"
    assert first["valid_until"] == "2026-06-16T12:05:00"
    # Above the probability threshold but not confirmed by the warning policy: not a warning level.
    assert second["risk"] == "watch" and not second["above_threshold"]
