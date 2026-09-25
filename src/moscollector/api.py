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
    LoginSession,
    Setting,
    User,
    make_database,
    seed_users,
)
from moscollector.domain import ACTIONS, KIND_LABELS, REASONS
from moscollector.paths import ARTIFACTS, ROOT
from moscollector.service import AnalyticsService, prediction_id

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


def can_edit(user=Depends(current_user)):
    if user.role not in ("dispatcher", "admin"):
        raise HTTPException(403, "Решения доступны диспетчеру")
    return user


def admin(user=Depends(current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Нужна роль администратора")
    return user


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
        return {"user": {"id": user.id, "name": user.display_name, "role": user.role}, "demo_mode": DEMO_MODE}


@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return {"id": user.id, "name": user.display_name, "role": user.role, "demo_mode": DEMO_MODE}


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
def overview(as_of: str | None = None, user=Depends(current_user)):
    result = analytics().overview(as_of, thresholds())
    with SessionFactory() as db:
        decisions = db.scalars(select(Decision)).all()
        by_id = {
            d.prediction_id: {"action": d.action, "reason": d.reason, "comment": d.comment, "id": d.id}
            for d in decisions
        }
        for row in result["forecasts"]:
            row["decision"] = by_id.get(row["id"])
    return result


@app.get("/api/topology")
def topology(as_of: str | None = None, user=Depends(current_user)):
    return analytics().topology(as_of, thresholds())


@app.get("/api/forecast/{object_id}/{kind}")
def forecast(
    object_id: int,
    kind: Literal["fault", "fire", "flood", "access"],
    as_of: str | None = None,
    user=Depends(current_user),
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
def evaluation(user=Depends(current_user)):
    service = analytics()
    research_path = ARTIFACTS / "research_report.json"
    research = json.loads(research_path.read_text()) if research_path.exists() else None
    quality_path = ARTIFACTS / "operational_quality.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
    return {
        **service.report,
        "uncertainty": getattr(service, "uncertainty", None),
        "research": research,
        "operational_quality": quality,
    }


@app.get("/api/evaluation/matches/{kind}")
def evaluation_matches(kind: Literal["fault", "fire", "flood", "access"], user=Depends(current_user)):
    return json.loads((ARTIFACTS / "predictions" / f"test-matches-{kind}.json").read_text())


@app.get("/api/quality")
def quality(user=Depends(current_user)):
    return analytics().quality()


@app.get("/api/reasons")
def reasons(user=Depends(current_user)):
    return {"reasons": REASONS, "actions": ACTIONS}


class DecisionInput(BaseModel):
    prediction_id: str = Field(max_length=100)
    object_id: int
    kind: Literal["fault", "fire", "flood", "access"]
    as_of: str = Field(max_length=40)
    action: Literal["dispatch", "monitor", "false_alarm", "maintenance"]
    reason: str = Field(max_length=80)
    comment: str = Field(default="", max_length=2000)
    batch_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


@app.post("/api/decisions")
def decide(body: DecisionInput, user=Depends(can_edit)):
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
def decisions(user=Depends(current_user)):
    service = analytics()
    with SessionFactory() as db:
        rows = db.scalars(select(Decision).order_by(Decision.updated_at.desc()).limit(1000)).all()
        return [
            {
                "id": d.id,
                "prediction_id": d.prediction_id,
                "object_id": d.object_id,
                "object_name": service.object_map.get(d.object_id, {}).get("object_name", str(d.object_id)),
                "kind": d.kind,
                "forecast_at": d.forecast_at,
                "action": d.action,
                "action_label": ACTIONS[d.action],
                "reason": d.reason,
                "comment": d.comment,
                "created_at": d.created_at.isoformat(),
                "updated_at": d.updated_at.isoformat(),
                "user_id": d.user_id,
            }
            for d in rows
        ]


class SettingsInput(BaseModel):
    thresholds: dict[str, float]


@app.get("/api/settings")
def settings(user=Depends(current_user)):
    return {
        "thresholds": analytics().thresholds(thresholds()),
        "defaults": analytics().thresholds(),
        "horizon_hours": 24,
        "cooldown_hours": 24,
        "mode": "historical_replay",
    }


@app.put("/api/settings")
def update_settings(body: SettingsInput, user=Depends(admin)):
    if any(k not in KIND_LABELS or not 0.001 <= v <= 1.01 for k, v in body.thresholds.items()):
        raise HTTPException(422, "Порог должен быть от 0,001 до 1,01; 1,01 отключает предупреждения")
    with SessionFactory() as db:
        previous = thresholds()
        row = db.get(Setting, "thresholds")
        if row:
            row.value = json.dumps(body.thresholds)
        else:
            db.add(Setting(key="thresholds", value=json.dumps(body.thresholds)))
        db.add(
            AuditLog(
                user_id=user.id,
                action="thresholds_updated",
                detail=json.dumps({"previous": previous, "new": body.thresholds}),
            )
        )
        db.commit()
    return {"saved": True, "thresholds": analytics().thresholds(body.thresholds)}


@app.get("/api/audit")
def audit_records(user=Depends(admin)):
    with SessionFactory() as db:
        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "action": r.action,
                "detail": json.loads(r.detail),
                "created_at": r.created_at.isoformat(),
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
    request: Request, as_of: str, format: Literal["csv", "xlsx", "json", "xml"], user=Depends(can_edit)
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
            raise HTTPException(413, "Файл превышает 20 МБ")
    return bytes(content)


@app.post("/api/stream/events", status_code=202)
async def stream_events(
    request: Request,
    as_of: str,
    format: Literal["csv", "xlsx", "json", "xml"] = "json",
    user=Depends(can_edit),
):
    require_import_runtime()
    from starlette.concurrency import run_in_threadpool

    content = await telemetry_body(request)
    return await run_in_threadpool(stream().ingest, content, format, as_of, user.id)


@app.post("/api/stream/forecast", status_code=202)
def stream_forecast(as_of: str, user=Depends(can_edit)):
    require_import_runtime()
    state = stream().forecast(as_of, user.id)
    audit(user.id, "stream_forecast_requested", {"job_id": state["id"], "as_of": as_of})
    return state


@app.get("/api/stream")
def stream_status(user=Depends(current_user)):
    return stream().status()


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
        ],
        "equipment_commands": False,
        "serverless_mode": SERVERLESS_MODE,
        "imports_enabled": imports_enabled,
        "import_formats": ["csv", "xlsx", "json", "xml"],
        "max_import_rows": 100000,
        "max_import_bytes": 20 * 1024 * 1024,
        "hosting_notice": (
            "Архивный прогноз, объяснения и решения доступны. Импорт запускайте в Docker-версии с постоянным worker."
            if SERVERLESS_MODE
            else None
        ),
    }


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
