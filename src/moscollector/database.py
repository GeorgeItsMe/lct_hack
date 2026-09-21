"""Application-owned data only. Customer monitoring sources remain read-only."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from moscollector.paths import DATA


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    display_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(30))
    password_hash: Mapped[str] = mapped_column(String(256))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class LoginSession(Base):
    __tablename__ = "login_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class Decision(Base):
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    object_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(20))
    forecast_at: Mapped[str] = mapped_column(String(40))
    action: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str] = mapped_column(String(80))
    comment: Mapped[str] = mapped_column(Text, default="")
    user_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(160))
    detail: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class StreamEvent(Base):
    """Application-owned copy; source systems are never modified."""

    __tablename__ = "stream_events"
    event_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    channel_id: Mapped[int] = mapped_column(Integer, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    value: Mapped[str] = mapped_column(Text)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    alarm: Mapped[bool] = mapped_column(Boolean)


class StreamBatch(Base):
    __tablename__ = "stream_batches"
    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    as_of: Mapped[datetime] = mapped_column(DateTime)
    input_rows: Mapped[int] = mapped_column(Integer)
    inserted_rows: Mapped[int] = mapped_column(Integer)
    duplicate_rows: Mapped[int] = mapped_column(Integer)
    user_id: Mapped[int] = mapped_column(Integer)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )


class WarningLock(Base):
    __tablename__ = "warning_lock"
    id: Mapped[int] = mapped_column(primary_key=True)
    last_as_of: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class WarningPolicyState(Base):
    __tablename__ = "warning_policy_states"
    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    state_json: Mapped[str] = mapped_column(Text)


class WarningBatch(Base):
    __tablename__ = "warning_batches"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    as_of: Mapped[datetime] = mapped_column(DateTime)
    model_version: Mapped[str] = mapped_column(String(40))
    decisions_json: Mapped[str] = mapped_column(Text)


def hash_password(password: str, salt: str | None = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 310000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, stored: str):
    try:
        _, salt, _ = stored.split("$")
        return hmac.compare_digest(hash_password(password, salt), stored)
    except (ValueError, TypeError):
        return False


def make_database(url: str | None = None):
    runtime = DATA / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    url = url or os.getenv("DATABASE_URL", f"sqlite:///{runtime / 'contour.db'}")
    kwargs = (
        {"connect_args": {"check_same_thread": False, "timeout": 30}}
        if url.startswith("sqlite")
        else {"pool_pre_ping": True}
    )
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def configure_sqlite(conn, _):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)


def seed_users(factory, demo_mode: bool):
    with factory() as db:
        from sqlalchemy import select

        if db.scalar(select(User.id).limit(1)):
            if not demo_mode and any(
                verify_password("contour-demo", u.password_hash) for u in db.scalars(select(User))
            ):
                raise RuntimeError("Demo accounts found. Use a new production database and provision users.")
            return
        if demo_mode:
            accounts = [
                ("dispatcher", "Диспетчер ОДС", "dispatcher", "contour-demo"),
                ("analyst", "Аналитик", "analyst", "contour-demo"),
                ("admin", "Администратор", "admin", "contour-demo"),
            ]
        else:
            password = os.getenv("CONTOUR_ADMIN_PASSWORD", "")
            if len(password) < 12:
                raise RuntimeError("Set CONTOUR_ADMIN_PASSWORD with at least 12 characters")
            accounts = [("admin", "Администратор", "admin", password)]
        for username, name, role, password in accounts:
            db.add(
                User(username=username, display_name=name, role=role, password_hash=hash_password(password))
            )
        db.commit()
