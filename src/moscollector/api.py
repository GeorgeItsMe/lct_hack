"""REST API for the dispatch workflow, authentication, replay, and evidence."""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal
from xml.etree import ElementTree

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from moscollector.authentication import authenticate
from moscollector.database import (
    AuditLog,
    Decision,
    LabelReview,
    LoginSession,
    RetrainRequest,
    Setting,
    ThresholdProposal,
    User,
    WorkOrder,
    hash_password,
    make_database,
    seed_users,
)
from moscollector.domain import ACTIONS, KIND_LABELS, REASONS, RISK_LEVELS, WORK_ACTIONS, risk_level
from moscollector.paths import ARTIFACTS, ROOT, RUNTIME
from moscollector.roles import DENIED, ROLES, allowed, permissions_for
from moscollector.service import AnalyticsService, prediction_id
from moscollector.workorders import utc_iso

DEMO_MODE = os.getenv("CONTOUR_DEMO", "true").lower() == "true"
SERVERLESS_MODE = os.getenv("CONTOUR_SERVERLESS", "false").lower() == "true"
engine, SessionFactory = make_database()
analytics_lock = threading.Lock()
_analytics = None
_imports = None
_stream = None
login_attempts = defaultdict(deque)


@asynccontextmanager
async def lifespan(app):
    seed_users(SessionFactory, DEMO_MODE)
    yield
    engine.dispose()


app = FastAPI(
    title="Контур — Москоллектор",
    version="0.1.0",
    lifespan=lifespan,
    description="Поддержка решений диспетчера. Внешние источники подключаются только на чтение.",
)


def analytics():
    global _analytics
    if _analytics is None:
        with analytics_lock:
            if _analytics is None:
                try:
                    _analytics = AnalyticsService()
                except FileNotFoundError as error:
                    raise HTTPException(
                        503, "Модель ещё готовится. Проверьте выполнение подготовки и обучения."
                    ) from error
    return _analytics


def imports():
    global _imports
    with analytics_lock:
        if _imports is None:
            from moscollector.importing import ImportManager

            _imports = ImportManager()
    return _imports


def stream():
    global _stream
    importer = imports()
    with analytics_lock:
        if _stream is None:
            from moscollector.streaming import StreamManager

            _stream = StreamManager(SessionFactory, importer)
    return _stream


def require_import_runtime():
    if SERVERLESS_MODE:
        raise HTTPException(
            503,
            "Импорт телеметрии требует постоянного worker-контейнера. На Vercel доступен архивный демостенд.",
        )


def current_user(request: Request):
    token = request.cookies.get("contour_session")
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    if not token:
        raise HTTPException(401, "Войдите в систему")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with SessionFactory() as db:
        login = db.get(LoginSession, token_hash)
        now = datetime.now(UTC).replace(tzinfo=None)
        if not login or login.expires_at < now:
            raise HTTPException(401, "Сеанс завершён. Войдите снова")
        user = db.get(User, login.user_id)
        if not user or not user.active:
            raise HTTPException(401, "Учётная запись недоступна")
        request.state.user_id = user.id
        return user


def require(permission):
    """Dependency enforcing one permission of the role matrix in roles.py."""

    def check(user=Depends(current_user)):
        if not allowed(user.role, permission):
            raise HTTPException(403, DENIED.get(permission, "Недостаточно прав"))
        return user

    check.__name__ = f"require_{permission.replace('.', '_')}"
    return check


def scope_parent(user):
    """Operational node the user is limited to, or None for the whole district."""
    scope = getattr(user, "scope", None) or "district"
    return int(scope.split(":", 1)[1]) if scope.startswith("node:") else None


def user_payload(user):
    return {
        "id": user.id,
        "name": user.display_name,
        "role": user.role,
        "role_label": ROLES.get(user.role, user.role),
        "scope": user.scope or "district",
        "permissions": permissions_for(user.role),
        "demo_mode": DEMO_MODE,
    }


def thresholds():
    with SessionFactory() as db:
        row = db.get(Setting, "thresholds")
        return json.loads(row.value) if row else {}


def audit(user_id, action, detail=None):
    with SessionFactory() as db:
        db.add(
            AuditLog(
                user_id=user_id,
                action=action,
                detail=json.dumps(detail or {}, ensure_ascii=False, default=str),
            )
        )
        db.commit()


@app.middleware("http")
async def record_request(request, call_next):
    started = time.monotonic()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    if hasattr(request.state, "user_id") and request.url.path.startswith("/api/"):
        audit(
            request.state.user_id,
            f"{request.method} {request.url.path}",
            {"status": response.status_code, "elapsed_ms": round((time.monotonic() - started) * 1000, 1)},
        )
    return response


@app.exception_handler(ValueError)
async def bad_value(request, error):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(KeyError)
async def not_found(request, error):
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": (ARTIFACTS / "evaluation_report.json").exists(),
        "demo_mode": DEMO_MODE,
        "database": "postgresql" if engine.dialect.name == "postgresql" else "sqlite",
        "persistent_database": engine.dialect.name == "postgresql",
        "serverless_mode": SERVERLESS_MODE,
    }


@app.get("/api/ready")
def ready():
    with engine.connect() as connection:
        from sqlalchemy import text

        connection.execute(text("SELECT 1"))
    service = analytics()
    from moscollector.model_registry import active_version, load_bundle

    operational_version = active_version()
    if operational_version != "legacy":
        load_bundle(operational_version)
    return {
        "status": "ready",
        "model_version": service.version,
        "operational_model_version": operational_version,
        "objects": len(service.objects),
    }


class LoginInput(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


@app.post("/api/auth/login")
def login(body: LoginInput, request: Request, response: Response):
    key = (request.client.host if request.client else "unknown", body.username)
    now = time.monotonic()
    attempts = login_attempts[key]
    while attempts and now - attempts[0] > 300:
        attempts.popleft()
    if len(attempts) >= 10:
        raise HTTPException(429, "Слишком много попыток. Повторите через пять минут")
    with SessionFactory() as db:
        user = db.scalar(select(User).where(User.username == body.username, User.active.is_(True)))
        if not user or not authenticate(user, body.password):
            attempts.append(now)
            audit(None, "login_failed", {"username": body.username})
            raise HTTPException(401, "Неверный логин или пароль")
        attempts.clear()
        token = secrets.token_urlsafe(32)
        db.add(
            LoginSession(
                token_hash=hashlib.sha256(token.encode()).hexdigest(),
                user_id=user.id,
                expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=8),
            )
        )
        db.commit()
        response.set_cookie(
            "contour_session", token, max_age=28800, httponly=True, samesite="strict", secure=not DEMO_MODE
        )
        audit(user.id, "login_success")
        return {"user": user_payload(user), "demo_mode": DEMO_MODE}


@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return user_payload(user)


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user=Depends(current_user)):
    token = request.cookies.get("contour_session", "")
    if request.headers.get("authorization", "").startswith("Bearer "):
        token = request.headers["authorization"][7:]
    with SessionFactory() as db:
        row = db.get(LoginSession, hashlib.sha256(token.encode()).hexdigest())
        if row:
            db.delete(row)
            db.commit()
    response.delete_cookie("contour_session")
    return {"ok": True}


@app.get("/api/overview")
def overview(as_of: str | None = None, user=Depends(require("forecasts.view"))):
    result = analytics().overview(as_of, thresholds())
    for row in result["forecasts"]:
        row["source"] = "archive"
    result["stream"] = stream_snapshot()
    parent = scope_parent(user)
    if parent is not None:
        result["forecasts"] = [r for r in result["forecasts"] if r.get("parent_id") == parent]
        if result["stream"]:
            result["stream"]["forecasts"] = [
                r for r in result["stream"]["forecasts"] if r.get("parent_id") == parent
            ]
    with SessionFactory() as db:
        decisions = db.scalars(select(Decision)).all()
        orders = latest_orders(db)
        by_id = {
            d.prediction_id: {
                "action": d.action,
                "reason": d.reason,
                "comment": d.comment,
                "id": d.id,
                "work_status": orders[d.id].status if d.id in orders else None,
                "work_outcome": orders[d.id].outcome if d.id in orders else None,
            }
            for d in decisions
        }
        for row in result["forecasts"] + (result["stream"] or {}).get("forecasts", []):
            row["decision"] = by_id.get(row["id"])
    return result


def stream_snapshot():
    """Latest completed forecast over the monitoring stream, shaped like archive forecasts.

    Stream data arrive from the monitoring system over the API without a person in the
    loop, so their warnings belong in the dispatcher's queue next to archive forecasts.
    """
    if SERVERLESS_MODE:
        return None
    try:
        manager = imports()
        job = next(
            (j for j in manager.list(mode="accumulated_stream") if j.get("status") == "complete"), None
        )
        if job is None:
            return None
        result = manager.get(job["id"])["result"]
    except (KeyError, OSError, ValueError):
        return None
    import pandas as pd

    service = analytics()
    moment = pd.Timestamp(result["as_of"])
    rows = []
    for r in result["forecasts"]:
        level = risk_level(r["probability"], r["threshold"])
        if not r["above_threshold"] and level in ("high", "critical"):
            level = "watch"  # the count-based warning policy did not confirm this one
        parent = service.object_map.get(r["object_id"], {}).get("parent_id")
        rows.append(
            {
                "id": f"batch:{job['id']}:{r['object_id']}:{r['kind']}",
                "object_id": r["object_id"],
                "object_name": r["object_name"],
                "parent_id": int(parent) if parent is not None and parent == parent else None,
                "kind": r["kind"],
                "kind_label": r["kind_label"],
                "probability": r["probability"],
                "threshold": r["threshold"],
                "risk": level,
                "risk_label": RISK_LEVELS[level],
                "above_threshold": r["above_threshold"],
                "as_of": result["as_of"],
                "horizon_hours": 24,
                "valid_until": (moment + pd.Timedelta(hours=24)).isoformat(),
                "recommendation": r["recommendation"],
                "model_version": result.get("model_version", "legacy"),
                "split": "stream",
                "source": "stream",
                "batch_id": job["id"],
            }
        )
    return {"job_id": job["id"], "as_of": result["as_of"], "forecasts": rows}


def latest_orders(db):
    """Most recent work order per decision."""
    rows = db.scalars(
        select(WorkOrder).where(WorkOrder.decision_id.is_not(None)).order_by(WorkOrder.id)
    ).all()
    return {o.decision_id: o for o in rows}


@app.get("/api/topology")
def topology(as_of: str | None = None, user=Depends(require("forecasts.view"))):
    return analytics().topology(as_of, thresholds())


@app.get("/api/forecast/{object_id}/{kind}")
def forecast(
    object_id: int,
    kind: Literal["fault", "fire", "flood", "access"],
    as_of: str | None = None,
    user=Depends(require("forecasts.view")),
):
    return analytics().detail(object_id, kind, as_of, thresholds())


@app.get("/api/retrospective/{object_id}/{kind}")
def retrospective(
    object_id: int, kind: Literal["fault", "fire", "flood", "access"], as_of: str, user=Depends(current_user)
):
    return analytics().retrospective(object_id, kind, as_of)


@app.get("/api/replay")
def replay(user=Depends(current_user)):
    service = analytics()
    return {
        "times": [t.isoformat() for t in service.times if t >= __import__("pandas").Timestamp("2026-05-01")],
        "default": service.resolve_time().isoformat(),
        "timezone": "Europe/Moscow",
        "step_hours": 3,
        "mode": "historical_replay",
    }


@app.get("/api/evaluation")
def evaluation(user=Depends(require("model.view"))):
    service = analytics()
    research_path = ARTIFACTS / "research_report.json"
    research = json.loads(research_path.read_text(encoding="utf-8")) if research_path.exists() else None
    quality_path = ARTIFACTS / "operational_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    return {
        **service.report,
        "uncertainty": getattr(service, "uncertainty", None),
        "research": research,
        "operational_quality": quality,
    }


@app.get("/api/evaluation/matches/{kind}")
def evaluation_matches(
    kind: Literal["fault", "fire", "flood", "access"], user=Depends(require("model.view"))
):
    return json.loads((ARTIFACTS / "predictions" / f"test-matches-{kind}.json").read_text(encoding="utf-8"))


@app.get("/api/quality")
def quality(user=Depends(require("model.view"))):
    return analytics().quality()


@app.get("/api/reasons")
def reasons(user=Depends(current_user)):
    return {"reasons": REASONS, "actions": ACTIONS}


class DecisionInput(BaseModel):
    prediction_id: str = Field(max_length=100)
    object_id: int
    kind: Literal["fault", "fire", "flood", "access"]
    as_of: str = Field(max_length=40)
    action: Literal["dispatch", "inspect", "monitor", "false_alarm", "maintenance"]
    reason: str = Field(max_length=80)
    comment: str = Field(default="", max_length=2000)
    batch_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


@app.post("/api/decisions")
def decide(body: DecisionInput, user=Depends(require("decisions.write"))):
    if body.reason not in {r["id"] for r in REASONS}:
        raise HTTPException(422, "Выберите причину из справочника")
    if body.reason == "other" and not body.comment.strip():
        raise HTTPException(422, "Укажите причину в комментарии")
    service = analytics()
    if body.batch_id:
        state = imports().get(body.batch_id)
        if state["status"] != "complete":
            raise HTTPException(422, "Расчёт ещё не завершён")
        if body.as_of != state["as_of"] or not any(
            r["object_id"] == body.object_id and r["kind"] == body.kind for r in state["result"]["forecasts"]
        ):
            raise HTTPException(422, "Прогноз не соответствует снимку пакета")
        import pandas as pd

        t = pd.Timestamp(state["as_of"])
        expected = f"batch:{body.batch_id}:{body.object_id}:{body.kind}"
    else:
        t = service.resolve_time(body.as_of)
        expected = prediction_id(body.object_id, body.kind, t)
        if not any(r["id"] == expected for r in service.forecast_rows(t)):
            raise HTTPException(404, "Прогноз не найден")
    if expected != body.prediction_id:
        raise HTTPException(422, "Идентификатор прогноза не соответствует данным")
    with SessionFactory() as db:
        row = db.scalar(select(Decision).where(Decision.prediction_id == expected))
        previous = {"action": row.action, "reason": row.reason, "comment": row.comment} if row else None
        if row is None:
            row = Decision(
                prediction_id=expected,
                object_id=body.object_id,
                kind=body.kind,
                forecast_at=t.isoformat(),
                user_id=user.id,
                action=body.action,
                reason=body.reason,
                comment=body.comment,
            )
            db.add(row)
        else:
            row.action = body.action
            row.reason = body.reason
            row.comment = body.comment
            row.user_id = user.id
            row.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.add(
            AuditLog(
                user_id=user.id,
                action="decision_saved",
                detail=json.dumps(
                    {
                        "prediction_id": expected,
                        "previous": previous,
                        "action": body.action,
                        "reason": body.reason,
                        "comment": body.comment,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        db.commit()
        return {"id": row.id, "prediction_id": row.prediction_id, "action": row.action, "saved": True}


@app.get("/api/decisions")
def decisions(user=Depends(require("decisions.view"))):
    return decision_rows()


def object_name(object_id):
    return analytics().object_map.get(object_id, {}).get("object_name", str(object_id))


def decision_rows(limit=1000):
    """Decisions with the work order they caused and the analyst's verdict (results of handling)."""
    from moscollector.workorders import serialize

    reasons = {r["id"]: r["label"] for r in REASONS}
    with SessionFactory() as db:
        rows = db.scalars(select(Decision).order_by(Decision.updated_at.desc()).limit(limit)).all()
        orders = latest_orders(db)
        reviews = {r.decision_id: r for r in db.scalars(select(LabelReview))}
        users = {u.id: u.display_name for u in db.scalars(select(User))}
        return [
            {
                "id": d.id,
                "prediction_id": d.prediction_id,
                "object_id": d.object_id,
                "object_name": object_name(d.object_id),
                "kind": d.kind,
                "forecast_at": d.forecast_at,
                "action": d.action,
                "action_label": ACTIONS.get(d.action, d.action),
                "reason": d.reason,
                "reason_label": reasons.get(d.reason, d.reason),
                "comment": d.comment,
                "created_at": utc_iso(d.created_at),
                "updated_at": utc_iso(d.updated_at),
                "user_id": d.user_id,
                "user_name": users.get(d.user_id),
                "requires_work": d.action in WORK_ACTIONS,
                "work_order": serialize(orders[d.id], object_name(d.object_id)) if d.id in orders else None,
                "label_review": {"verdict": reviews[d.id].verdict, "label": reviews[d.id].label}
                if d.id in reviews
                else None,
            }
            for d in rows
        ]


class SettingsInput(BaseModel):
    thresholds: dict[str, float]


@app.get("/api/settings")
def settings(user=Depends(require("forecasts.view"))):
    return {
        "thresholds": analytics().thresholds(thresholds()),
        "defaults": analytics().thresholds(),
        "horizon_hours": 24,
        "cooldown_hours": 24,
        "mode": "historical_replay",
    }


def validate_thresholds(values):
    if any(k not in KIND_LABELS or not 0.001 <= v <= 1.01 for k, v in values.items()):
        raise HTTPException(422, "Порог должен быть от 0,001 до 1,01; 1,01 отключает предупреждения")


def store_thresholds(db, user_id, values, source):
    previous = thresholds()
    merged = {**previous, **values}
    row = db.get(Setting, "thresholds")
    if row:
        row.value = json.dumps(merged)
    else:
        db.add(Setting(key="thresholds", value=json.dumps(merged)))
    db.add(
        AuditLog(
            user_id=user_id,
            action="thresholds_updated",
            detail=json.dumps({"previous": previous, "new": merged, "source": source}),
        )
    )
    return merged


@app.put("/api/settings")
def update_settings(body: SettingsInput, user=Depends(require("thresholds.approve"))):
    validate_thresholds(body.thresholds)
    with SessionFactory() as db:
        merged = store_thresholds(db, user.id, body.thresholds, "manager_direct")
        db.commit()
    return {"saved": True, "thresholds": analytics().thresholds(merged)}


@app.get("/api/audit")
def audit_records(user=Depends(require("audit.view"))):
    with SessionFactory() as db:
        users = {u.id: u for u in db.scalars(select(User))}
        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "user_name": users[r.user_id].display_name if r.user_id in users else None,
                "role": users[r.user_id].role if r.user_id in users else None,
                "action": r.action,
                "detail": json.loads(r.detail),
                "created_at": utc_iso(r.created_at),
            }
            for r in db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(300))
        ]


@app.get("/api/export/forecasts.csv")
def export_forecasts(as_of: str | None = None, user=Depends(current_user)):
    import csv

    rows = analytics().forecast_rows(as_of, thresholds())
    stream = io.StringIO()
    fields = [
        "id",
        "object_id",
        "object_name",
        "kind_label",
        "as_of",
        "probability",
        "threshold",
        "horizon_hours",
        "recommendation",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        "\ufeff" + stream.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=contour-forecasts.csv"},
    )


@app.get("/api/export/forecasts.xml")
def export_xml(as_of: str | None = None, user=Depends(current_user)):
    root = ElementTree.Element(
        "forecasts", horizon_hours="24", timezone="Europe/Moscow", label_type="proxy_sensor_episode"
    )
    for forecast in analytics().forecast_rows(as_of, thresholds()):
        node = ElementTree.SubElement(root, "forecast")
        for key in (
            "id",
            "object_id",
            "object_name",
            "kind",
            "probability",
            "threshold",
            "as_of",
            "recommendation",
        ):
            ElementTree.SubElement(node, key).text = str(forecast[key])
    return Response(
        ElementTree.tostring(root, encoding="utf-8", xml_declaration=True), media_type="application/xml"
    )


@app.get("/api/topology.geojson")
def geojson(as_of: str | None = None, user=Depends(current_user)):
    topology = analytics().topology(as_of, thresholds())
    # RFC 7946 allows null geometry for features without known geographic position.
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": n["id"], "geometry": None, "properties": n} for n in topology["nodes"]
        ],
        "coordinate_status": "not_provided",
        "description": topology["description"],
    }


@app.get("/api/topology.wkt")
def topology_wkt(user=Depends(current_user)):
    return Response(
        "GEOMETRYCOLLECTION EMPTY\n", media_type="text/plain", headers={"X-Coordinate-Status": "not-provided"}
    )


@app.post("/api/imports", status_code=202)
async def submit_import(
    request: Request,
    format: Literal["csv", "xlsx", "json", "xml", "zip"],
    as_of: str | None = None,
    user=Depends(require("data.import")),
):
    require_import_runtime()
    # CPU parsing and filesystem work run off the HTTP event loop.
    from starlette.concurrency import run_in_threadpool

    content = await telemetry_body(request)
    state = await run_in_threadpool(imports().submit, content, format, as_of, user.id)
    audit(user.id, "telemetry_import", {k: v for k, v in state.items() if k != "result"})
    return state


@app.get("/api/imports")
def import_list(user=Depends(current_user)):
    return imports().list()


@app.get("/api/imports/{job_id}")
def import_result(job_id: str, user=Depends(current_user)):
    return imports().get(job_id)


@app.get("/api/imports/{job_id}/forecast/{object_id}/{kind}")
def import_forecast(
    job_id: str,
    object_id: int,
    kind: Literal["fault", "fire", "flood", "access"],
    user=Depends(current_user),
):
    return imports().detail(job_id, object_id, kind, analytics())


async def telemetry_body(request):
    from moscollector.importing import MAX_BYTES

    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, f"Файл превышает {MAX_BYTES // 1024 // 1024} МБ")
    return bytes(content)


@app.post("/api/stream/events", status_code=202)
async def stream_events(
    request: Request,
    as_of: str,
    format: Literal["csv", "xlsx", "json", "xml", "zip"] = "json",
    user=Depends(require("data.import")),
):
    require_import_runtime()
    from starlette.concurrency import run_in_threadpool

    content = await telemetry_body(request)
    return await run_in_threadpool(stream().ingest, content, format, as_of, user.id)


@app.post("/api/stream/forecast", status_code=202)
def stream_forecast(as_of: str, user=Depends(require("data.import"))):
    require_import_runtime()
    state = stream().forecast(as_of, user.id)
    audit(user.id, "stream_forecast_requested", {"job_id": state["id"], "as_of": as_of})
    return state


@app.get("/api/stream")
def stream_status(user=Depends(current_user)):
    return stream().status()


def import_limits():
    from moscollector.importing import MAX_BYTES, MAX_ROWS

    return MAX_ROWS, MAX_BYTES


@app.get("/api/integrations")
def integrations(user=Depends(current_user)):
    imports_enabled = not SERVERLESS_MODE
    return {
        "sources": [
            {
                "id": "provided_archive",
                "label": "Архив заказчика",
                "status": "available",
                "access": "read_only",
            },
            {
                "id": "file_ingestion",
                "label": "Импорт телеметрии",
                "status": "available" if imports_enabled else "requires_worker_container",
                "access": "read_only",
            },
            {
                "id": "stream_gateway",
                "label": "Накопительный приём потока по API",
                "status": "available" if imports_enabled else "requires_worker_container",
                "access": "read_only_source",
            },
            {
                "id": "operational_monitoring",
                "label": "Система мониторинга заказчика",
                "status": "not_connected",
                "access": "read_only",
            },
            {
                "id": "work_orders",
                "label": "Система учёта заявок",
                "status": "emulated",
                "access": "submit_drafts_read_statuses",
            },
            {
                "id": "directory",
                "label": "Каталог LDAP/AD",
                "status": "configured" if os.getenv("CONTOUR_LDAP_URL") else "not_configured",
                "access": "read_only",
            },
        ],
        "equipment_commands": False,
        "serverless_mode": SERVERLESS_MODE,
        "imports_enabled": imports_enabled,
        "import_formats": ["csv", "xlsx", "json", "xml", "zip"],
        "max_import_rows": import_limits()[0],
        "max_import_bytes": import_limits()[1],
        "hosting_notice": (
            "Архивный прогноз, объяснения и решения доступны. Импорт запускайте в Docker-версии с постоянным worker."
            if SERVERLESS_MODE
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Technical staff: equipment condition, tasks from dispatcher decisions, work orders.
# ---------------------------------------------------------------------------


@app.get("/api/equipment")
def equipment(as_of: str | None = None, user=Depends(require("equipment.view"))):
    result = analytics().equipment(as_of, thresholds())
    parent = scope_parent(user)
    if parent is not None:
        result["objects"] = [r for r in result["objects"] if r["parent_id"] == parent]
    with SessionFactory() as db:
        open_orders = {}
        for order in db.scalars(select(WorkOrder).where(WorkOrder.status != "done")):
            open_orders.setdefault(order.object_id, []).append(order.number)
    for row in result["objects"]:
        row["open_orders"] = open_orders.get(row["object_id"], [])
    return result


@app.get("/api/tasks")
def tasks(user=Depends(require("work.view"))):
    """Decisions that require field work, with the work order they produced (if any)."""
    return [row for row in decision_rows() if row["requires_work"]]


class WorkOrderInput(BaseModel):
    decision_id: int | None = None
    object_id: int | None = None
    kind: Literal["fault", "fire", "flood", "access"] | None = None
    as_of: str | None = Field(default=None, max_length=40)


class WorkOrderUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    priority: Literal["urgent", "high", "normal"] | None = None


def order_payload(order):
    from moscollector.workorders import serialize

    return serialize(order, object_name(order.object_id))


@app.get("/api/work-orders")
def work_orders(user=Depends(require("work.view"))):
    with SessionFactory() as db:
        return [
            order_payload(o) for o in db.scalars(select(WorkOrder).order_by(WorkOrder.id.desc()).limit(500))
        ]


@app.post("/api/work-orders", status_code=201)
def create_work_order(body: WorkOrderInput, user=Depends(require("work.edit"))):
    """Draft built from the forecast card: decision task or equipment condition."""
    from moscollector.workorders import draft_text, order_number, priority_for

    service = analytics()
    action = None
    with SessionFactory() as db:
        if body.decision_id is not None:
            decision = db.get(Decision, body.decision_id)
            if decision is None:
                raise HTTPException(404, "Решение не найдено")
            if decision.action not in WORK_ACTIONS:
                raise HTTPException(422, "Решение диспетчера не требует работ")
            if db.scalar(
                select(WorkOrder).where(WorkOrder.decision_id == decision.id, WorkOrder.status != "done")
            ):
                raise HTTPException(409, "По этому решению уже есть открытая заявка")
            object_id, kind, as_of, action = (
                decision.object_id,
                decision.kind,
                decision.forecast_at,
                decision.action,
            )
            prediction = decision.prediction_id
        else:
            if body.object_id is None or body.kind is None:
                raise HTTPException(422, "Укажите решение диспетчера или объект и тип риска")
            object_id, kind, as_of, prediction = body.object_id, body.kind, body.as_of, None
        if prediction and prediction.startswith("batch:"):
            # Stream forecast: the snapshot keeps its own probability and explanation.
            forecast = imports().detail(prediction.split(":")[1], object_id, kind, service)
            forecast["risk"] = risk_level(forecast["probability"], forecast["threshold"])
            recommendations, factors = forecast["recommendations"], forecast["explanation"]
        else:
            forecast = service.detail(object_id, kind, as_of, thresholds())
            recommendations, factors = forecast["recommendations"], forecast["explanation"]
            prediction = prediction or forecast["id"]
        title, description = draft_text(forecast, action, recommendations, factors)
        created = datetime.now(UTC).replace(tzinfo=None)
        order = WorkOrder(
            number="pending",
            decision_id=body.decision_id,
            prediction_id=prediction,
            object_id=object_id,
            kind=kind,
            forecast_at=forecast["as_of"],
            title=title,
            description=description,
            priority=priority_for(forecast.get("risk")),
            status="draft",
            created_by=user.id,
            created_at=created,
            updated_at=created,
        )
        db.add(order)
        db.flush()
        order.number = order_number(order.id, created)
        db.add(
            AuditLog(
                user_id=user.id,
                action="work_order_drafted",
                detail=json.dumps(
                    {"number": order.number, "decision_id": body.decision_id}, ensure_ascii=False
                ),
            )
        )
        db.commit()
        return order_payload(order)


@app.patch("/api/work-orders/{order_id}")
def update_work_order(order_id: int, body: WorkOrderUpdate, user=Depends(require("work.edit"))):
    with SessionFactory() as db:
        order = db.get(WorkOrder, order_id)
        if order is None:
            raise HTTPException(404, "Заявка не найдена")
        if order.status != "draft":
            raise HTTPException(409, "Изменять можно только черновик; статус ведёт система учёта заявок")
        for field in ("title", "description", "priority"):
            value = getattr(body, field)
            if value is not None:
                setattr(order, field, value)
        order.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.add(
            AuditLog(user_id=user.id, action="work_order_edited", detail=json.dumps({"number": order.number}))
        )
        db.commit()
        return order_payload(order)


@app.post("/api/work-orders/{order_id}/submit")
def submit_work_order(order_id: int, user=Depends(require("work.edit"))):
    """Hand the draft to the external work-management system (emulated, see workorders.py)."""
    with SessionFactory() as db:
        order = db.get(WorkOrder, order_id)
        if order is None:
            raise HTTPException(404, "Заявка не найдена")
        if order.status != "draft":
            raise HTTPException(409, "Заявка уже передана")
        moment = datetime.now(UTC).replace(tzinfo=None)
        order.status = "submitted"
        order.submitted_at = moment
        order.synced_at = moment
        order.updated_at = moment
        order.external_id = f"HD-{order.id:06d}"
        db.add(
            AuditLog(
                user_id=user.id,
                action="work_order_submitted",
                detail=json.dumps({"number": order.number, "external_id": order.external_id}),
            )
        )
        db.commit()
        return order_payload(order)


def resolve_outcome(order):
    """Outcome the emulated system reports when an order closes: taken from the archive."""
    from moscollector.workorders import archive_outcome

    if not order.forecast_at or (order.prediction_id or "").startswith("batch:"):
        return None
    service = analytics()
    try:
        occurred = service.retrospective(order.object_id, order.kind, order.forecast_at)["occurred"]
        states = service.channel_states(service.resolve_time(order.forecast_at).isoformat())
    except (KeyError, ValueError):
        return None
    own = states[states.object_id.eq(order.object_id)]
    return archive_outcome(order.kind, occurred, bool(own.current_fault.any()))


@app.post("/api/work-orders/sync")
def sync_work_orders(user=Depends(require("work.view"))):
    """Read statuses back from the work-management system (read-only integration)."""
    from moscollector.workorders import emulated_status

    changed = []
    with SessionFactory() as db:
        moment = datetime.now(UTC).replace(tzinfo=None)
        for order in db.scalars(
            select(WorkOrder).where(WorkOrder.status.in_(["submitted", "accepted", "in_progress"]))
        ):
            status = emulated_status(order.submitted_at, moment)
            order.synced_at = moment
            if status != order.status:
                order.status = status
                order.updated_at = moment
                if status == "done":
                    order.outcome = resolve_outcome(order)
                changed.append({"number": order.number, "status": status, "outcome": order.outcome})
        if changed:
            db.add(AuditLog(user_id=user.id, action="work_orders_synced", detail=json.dumps(changed)))
        db.commit()
    return {"changed": changed, "source": "emulated_work_order_system"}


# ---------------------------------------------------------------------------
# Analyst: data verification for retraining, threshold proposals.
# ---------------------------------------------------------------------------


def label_candidates():
    from moscollector.workorders import suggested_label

    service = analytics()
    rows = []
    for row in decision_rows(limit=2000):
        archive = None
        if not row["prediction_id"].startswith("batch:"):
            try:
                archive = service.retrospective(row["object_id"], row["kind"], row["forecast_at"])["occurred"]
            except (KeyError, ValueError):
                archive = None
        order = row["work_order"] or {}
        label, basis = suggested_label(row["kind"], row["action"], order.get("outcome"), archive)
        rows.append({**row, "archive_occurred": archive, "suggested_label": label, "label_basis": basis})
    return rows


@app.get("/api/labels")
def labels(user=Depends(require("labels.verify"))):
    rows = label_candidates()
    with SessionFactory() as db:
        requests = db.scalars(select(RetrainRequest).order_by(RetrainRequest.id.desc()).limit(20)).all()
        history = [
            {
                "id": r.id,
                "created_at": utc_iso(r.created_at),
                "labels": r.labels,
                "positives": r.positives,
                "file_name": r.file_name,
                "status": r.status,
                "note": r.note,
            }
            for r in requests
        ]
    return {
        "rows": rows,
        "accepted": sum((r["label_review"] or {}).get("verdict") == "accepted" for r in rows),
        "rejected": sum((r["label_review"] or {}).get("verdict") == "rejected" for r in rows),
        "retraining": history,
    }


class LabelInput(BaseModel):
    verdict: Literal["accepted", "rejected"]
    label: Literal[0, 1] | None = None
    note: str = Field(default="", max_length=1000)


@app.put("/api/labels/{decision_id}")
def review_label(decision_id: int, body: LabelInput, user=Depends(require("labels.verify"))):
    if body.verdict == "accepted" and body.label is None:
        raise HTTPException(422, "Для принятой метки укажите значение 0 или 1")
    if body.verdict == "rejected" and not body.note.strip():
        raise HTTPException(422, "Укажите, почему пара исключается из обучения")
    with SessionFactory() as db:
        if db.get(Decision, decision_id) is None:
            raise HTTPException(404, "Решение не найдено")
        review = db.scalar(select(LabelReview).where(LabelReview.decision_id == decision_id))
        previous = {"verdict": review.verdict, "label": review.label} if review else None
        if review is None:
            review = LabelReview(decision_id=decision_id, verdict=body.verdict, user_id=user.id)
            db.add(review)
        review.verdict = body.verdict
        review.label = body.label if body.verdict == "accepted" else None
        review.note = body.note
        review.user_id = user.id
        review.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.add(
            AuditLog(
                user_id=user.id,
                action="label_reviewed",
                detail=json.dumps(
                    {
                        "decision_id": decision_id,
                        "previous": previous,
                        "verdict": body.verdict,
                        "label": body.label,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        db.commit()
    return {"saved": True}


def verified_labels_csv():
    import csv

    with SessionFactory() as db:
        reviews = {
            r.decision_id: r for r in db.scalars(select(LabelReview).where(LabelReview.verdict == "accepted"))
        }
    rows = [r for r in label_candidates() if r["id"] in reviews]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        ["prediction_id", "object_id", "kind", "as_of", "label", "dispatcher_action", "work_outcome"]
    )
    for r in rows:
        writer.writerow(
            [
                r["prediction_id"],
                r["object_id"],
                r["kind"],
                r["forecast_at"],
                reviews[r["id"]].label,
                r["action"],
                (r["work_order"] or {}).get("outcome") or "",
            ]
        )
    return stream.getvalue(), rows, reviews


@app.get("/api/labels/export.csv")
def export_labels(user=Depends(require("labels.verify"))):
    content, _, _ = verified_labels_csv()
    return Response(
        "﻿" + content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=verified-labels.csv"},
    )


@app.post("/api/retraining", status_code=201)
def request_retraining(user=Depends(require("retrain.run"))):
    """Save the analyst-verified labels as an immutable set.

    The training pipeline (train.py) learns from sensor-journal episodes and does not read
    these sets yet; they are kept so that verified outcomes accumulate for a future model.
    """
    content, rows, reviews = verified_labels_csv()
    if not rows:
        raise HTTPException(422, "Нет принятых меток: сначала верифицируйте решения")
    folder = RUNTIME / "retraining"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = f"labels-{stamp}.csv"
    (folder / name).write_text(content, encoding="utf-8")
    positives = sum(reviews[r["id"]].label == 1 for r in rows)
    note = f"Сохранено меток: {len(rows)}, из них «событие было»: {positives}."
    with SessionFactory() as db:
        request = RetrainRequest(
            created_by=user.id,
            labels=len(rows),
            positives=positives,
            file_name=name,
            status="prepared",
            note=note,
        )
        db.add(request)
        db.add(
            AuditLog(
                user_id=user.id,
                action="retraining_requested",
                detail=json.dumps({"labels": len(rows), "positives": positives, "file": name}),
            )
        )
        db.commit()
        return {
            "id": request.id,
            "labels": len(rows),
            "positives": positives,
            "file_name": name,
            "note": note,
        }


@app.get("/api/thresholds/preview")
def threshold_preview(
    kind: Literal["fault", "fire", "flood", "access"],
    threshold: float,
    as_of: str | None = None,
    user=Depends(current_user),
):
    if not (allowed(user.role, "thresholds.propose") or allowed(user.role, "thresholds.approve")):
        raise HTTPException(403, DENIED["thresholds.propose"])
    if not 0.001 <= threshold <= 1.01:
        raise HTTPException(422, "Порог должен быть от 0,001 до 1,01")
    return analytics().threshold_preview(kind, threshold, as_of, thresholds())


class ProposalInput(BaseModel):
    kind: Literal["fault", "fire", "flood", "access"]
    threshold: float
    rationale: str = Field(min_length=10, max_length=2000)


class ProposalDecision(BaseModel):
    note: str = Field(default="", max_length=2000)


def proposal_payload(p, users):
    return {
        "id": p.id,
        "kind": p.kind,
        "kind_label": KIND_LABELS[p.kind],
        "current_value": p.current_value,
        "proposed_value": p.proposed_value,
        "rationale": p.rationale,
        "preview": json.loads(p.preview_json),
        "status": p.status,
        "proposed_by": users.get(p.proposed_by),
        "created_at": utc_iso(p.created_at),
        "decided_by": users.get(p.decided_by) if p.decided_by else None,
        "decided_at": utc_iso(p.decided_at),
        "decision_note": p.decision_note,
    }


@app.get("/api/thresholds/proposals")
def proposals(user=Depends(current_user)):
    if not any(allowed(user.role, p) for p in ("thresholds.propose", "thresholds.approve", "audit.view")):
        raise HTTPException(403, DENIED["thresholds.propose"])
    with SessionFactory() as db:
        users = {u.id: u.display_name for u in db.scalars(select(User))}
        rows = db.scalars(select(ThresholdProposal).order_by(ThresholdProposal.id.desc()).limit(100)).all()
        return [proposal_payload(p, users) for p in rows]


@app.post("/api/thresholds/proposals", status_code=201)
def propose_threshold(body: ProposalInput, user=Depends(require("thresholds.propose"))):
    validate_thresholds({body.kind: body.threshold})
    service = analytics()
    current = service.thresholds(thresholds())[body.kind]
    if abs(current - body.threshold) < 1e-9:
        raise HTTPException(422, "Предложенный порог совпадает с действующим")
    preview = service.threshold_preview(body.kind, body.threshold, None, thresholds())
    with SessionFactory() as db:
        if db.scalar(
            select(ThresholdProposal).where(
                ThresholdProposal.kind == body.kind, ThresholdProposal.status == "pending"
            )
        ):
            raise HTTPException(409, "По этому типу уже есть предложение на рассмотрении")
        proposal = ThresholdProposal(
            kind=body.kind,
            current_value=current,
            proposed_value=body.threshold,
            rationale=body.rationale,
            preview_json=json.dumps(
                {k: preview[k] for k in ("current", "proposed", "period_label")}, default=str
            ),
            proposed_by=user.id,
        )
        db.add(proposal)
        db.flush()
        db.add(
            AuditLog(
                user_id=user.id,
                action="threshold_proposed",
                detail=json.dumps(
                    {"id": proposal.id, "kind": body.kind, "from": current, "to": body.threshold}
                ),
            )
        )
        db.commit()
        users = {u.id: u.display_name for u in db.scalars(select(User))}
        return proposal_payload(proposal, users)


def decide_proposal(proposal_id, approve, body, user):
    with SessionFactory() as db:
        proposal = db.get(ThresholdProposal, proposal_id)
        if proposal is None:
            raise HTTPException(404, "Предложение не найдено")
        if proposal.status != "pending":
            raise HTTPException(409, "Предложение уже рассмотрено")
        if not approve and not body.note.strip():
            raise HTTPException(422, "Укажите причину отклонения")
        proposal.status = "approved" if approve else "rejected"
        proposal.decided_by = user.id
        proposal.decided_at = datetime.now(UTC).replace(tzinfo=None)
        proposal.decision_note = body.note
        if approve:
            store_thresholds(db, user.id, {proposal.kind: proposal.proposed_value}, f"proposal:{proposal.id}")
        db.add(
            AuditLog(
                user_id=user.id,
                action="threshold_approved" if approve else "threshold_rejected",
                detail=json.dumps(
                    {"id": proposal.id, "kind": proposal.kind, "value": proposal.proposed_value}
                ),
            )
        )
        db.commit()
        users = {u.id: u.display_name for u in db.scalars(select(User))}
        return proposal_payload(proposal, users)


@app.post("/api/thresholds/proposals/{proposal_id}/approve")
def approve_proposal(proposal_id: int, body: ProposalDecision, user=Depends(require("thresholds.approve"))):
    return decide_proposal(proposal_id, True, body, user)


@app.post("/api/thresholds/proposals/{proposal_id}/reject")
def reject_proposal(proposal_id: int, body: ProposalDecision, user=Depends(require("thresholds.approve"))):
    return decide_proposal(proposal_id, False, body, user)


# ---------------------------------------------------------------------------
# Unit head: summary and management reports.
# ---------------------------------------------------------------------------


def workload():
    rows = decision_rows(limit=5000)
    with SessionFactory() as db:
        orders = [order_payload(o) for o in db.scalars(select(WorkOrder).order_by(WorkOrder.id.desc()))]
        pending = db.scalar(
            select(ThresholdProposal.id).where(ThresholdProposal.status == "pending").limit(1)
        )
    actions = {k: sum(r["action"] == k for r in rows) for k in ACTIONS}
    return (
        rows,
        orders,
        {
            "decisions": len(rows),
            "actions": [{"id": k, "label": ACTIONS[k], "count": v} for k, v in actions.items()],
            "false_alarm_share": actions["false_alarm"] / len(rows) if rows else None,
            "orders_open": sum(o["status"] != "done" for o in orders),
            "orders_done": sum(o["status"] == "done" for o in orders),
            "outcomes": {
                k: sum(o["outcome"] == k for o in orders)
                for k in ("confirmed", "not_confirmed", "sensor_fault")
            },
            "pending_proposals": pending is not None,
        },
    )


@app.get("/api/summary")
def summary(as_of: str | None = None, user=Depends(require("summary.view"))):
    result = analytics().summary(as_of, thresholds(), scope_parent(user))
    _, _, result["workload"] = workload()
    return result


def report_data(as_of, user):
    from moscollector.reports import build_report

    service = analytics()
    summary_data = service.summary(as_of, thresholds(), scope_parent(user))
    forecasts = service.forecast_rows(summary_data["as_of"], thresholds())
    rows, orders, _ = workload()
    return build_report(summary_data, forecasts, rows, orders)


@app.get("/api/reports/summary.xlsx")
def report_xlsx(as_of: str | None = None, user=Depends(require("reports.export"))):
    from moscollector.reports import to_xlsx

    content = to_xlsx(report_data(as_of, user))
    audit(user.id, "report_exported", {"format": "xlsx", "as_of": as_of})
    return Response(
        content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=contour-report.xlsx"},
    )


@app.get("/api/reports/summary.pdf")
def report_pdf(as_of: str | None = None, user=Depends(require("reports.export"))):
    from moscollector.reports import to_pdf

    content = to_pdf(report_data(as_of, user))
    audit(user.id, "report_exported", {"format": "pdf", "as_of": as_of})
    return Response(
        content,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=contour-report.pdf"},
    )


# ---------------------------------------------------------------------------
# Administrator: users and roles.
# ---------------------------------------------------------------------------


def user_admin_payload(u):
    return {
        "id": u.id,
        "username": u.username,
        "name": u.display_name,
        "role": u.role,
        "role_label": ROLES.get(u.role, u.role),
        "scope": u.scope or "district",
        "active": u.active,
    }


@app.get("/api/users")
def users(user=Depends(require("users.manage"))):
    with SessionFactory() as db:
        return {
            "users": [user_admin_payload(u) for u in db.scalars(select(User).order_by(User.id))],
            "roles": [{"id": k, "label": v, "permissions": permissions_for(k)} for k, v in ROLES.items()],
            "ldap_configured": bool(os.getenv("CONTOUR_LDAP_URL")),
        }


class UserCreate(BaseModel):
    username: str = Field(pattern=r"^[A-Za-z0-9_.@-]{3,80}$")
    name: str = Field(min_length=2, max_length=120)
    role: Literal["dispatcher", "technician", "analyst", "manager", "admin"]
    scope: str = Field(default="district", pattern=r"^(district|node:\d+)$")
    password: str | None = Field(default=None, max_length=256)
    ldap: bool = False


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    role: Literal["dispatcher", "technician", "analyst", "manager", "admin"] | None = None
    scope: str | None = Field(default=None, pattern=r"^(district|node:\d+)$")
    active: bool | None = None


@app.post("/api/users", status_code=201)
def create_user(body: UserCreate, user=Depends(require("users.manage"))):
    if not body.ldap and len(body.password or "") < 12:
        raise HTTPException(422, "Пароль локальной учётной записи — не короче 12 символов")
    password = secrets.token_urlsafe(40) if body.ldap else body.password
    with SessionFactory() as db:
        if db.scalar(select(User).where(User.username == body.username)):
            raise HTTPException(409, "Пользователь с таким логином уже есть")
        created = User(
            username=body.username,
            display_name=body.name,
            role=body.role,
            scope=body.scope,
            password_hash=hash_password(password),
            active=True,
        )
        db.add(created)
        db.flush()
        db.add(
            AuditLog(
                user_id=user.id,
                action="user_created",
                detail=json.dumps(
                    {"username": body.username, "role": body.role, "scope": body.scope, "ldap": body.ldap}
                ),
            )
        )
        db.commit()
        return user_admin_payload(created)


@app.patch("/api/users/{user_id}")
def update_user(user_id: int, body: UserUpdate, user=Depends(require("users.manage"))):
    with SessionFactory() as db:
        target = db.get(User, user_id)
        if target is None:
            raise HTTPException(404, "Пользователь не найден")
        if target.id == user.id and (body.active is False or (body.role and body.role != "admin")):
            raise HTTPException(422, "Нельзя заблокировать себя или снять с себя роль администратора")
        previous = user_admin_payload(target)
        if body.name is not None:
            target.display_name = body.name
        if body.role is not None:
            target.role = body.role
        if body.scope is not None:
            target.scope = body.scope
        if body.active is not None:
            target.active = body.active
            if not body.active:
                for session in db.scalars(select(LoginSession).where(LoginSession.user_id == target.id)):
                    db.delete(session)
        db.add(
            AuditLog(
                user_id=user.id,
                action="user_updated",
                detail=json.dumps(
                    {"previous": previous, "new": user_admin_payload(target)}, ensure_ascii=False
                ),
            )
        )
        db.commit()
        return user_admin_payload(target)


web_dist = ROOT / "web" / "dist"
if web_dist.exists():
    app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "API endpoint not found")
        if path == "favicon.svg":
            return FileResponse(web_dist / "favicon.svg", media_type="image/svg+xml")
        return FileResponse(web_dist / "index.html")
