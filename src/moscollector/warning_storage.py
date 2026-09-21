"""Transactional in-app warning decisions for accumulated stream snapshots."""

import json

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from moscollector.database import WarningBatch, WarningLock, WarningPolicyState
from moscollector.warning_policy import HOUR_NS, PendingWarningState


def decide_stream_warnings(factory, job_id, as_of, model_version, forecasts, episodes, policies):
    """Commit states and replayable decisions together; never send external messages."""
    at = pd.Timestamp(as_of)
    with factory() as db:
        insert = pg_insert if db.bind.dialect.name == "postgresql" else sqlite_insert
        # The write serializes SQLite transactions. PostgreSQL additionally locks
        # the common row so concurrent workers cannot issue duplicate warnings.
        db.execute(
            insert(WarningLock).values(id=1, last_as_of=None).on_conflict_do_nothing(index_elements=["id"])
        )
        lock = db.scalar(select(WarningLock).where(WarningLock.id == 1).with_for_update())
        cached = db.get(WarningBatch, job_id)
        if cached:
            if cached.as_of != at.to_pydatetime() or cached.model_version != model_version:
                raise ValueError("Warning job identity changed")
            return json.loads(cached.decisions_json)
        stale = lock.last_as_of is not None and at <= pd.Timestamp(lock.last_as_of)
        hourly = at == at.floor("h")
        decisions = []
        for forecast in forecasts:
            kind, obj = forecast["kind"], int(forecast["object_id"])
            if kind not in policies:
                continue
            policy = policies[kind]
            if policy["type"] != "pending_count":
                raise ValueError("Unsupported operational warning policy")
            key = f"{obj}:{kind}"
            row = db.get(WarningPolicyState, key)
            state = (
                PendingWarningState.from_dict(json.loads(row.state_json)) if row else PendingWarningState()
            )
            issued = False
            if not stale and hourly:
                events = episodes.loc[episodes.object_id.eq(obj) & episodes.kind.eq(kind), "start_ts"]
                events = (
                    events[events.ge(at - pd.Timedelta(hours=24))]
                    .to_numpy(dtype="datetime64[ns]")
                    .astype("int64")
                )
                issued = state.step(
                    at,
                    forecast["probability"],
                    forecast["expected_episodes"],
                    events,
                    policy["margin"],
                    policy["probability_floor"],
                )
                encoded = json.dumps(state.to_dict())
                if row:
                    row.state_json = encoded
                else:
                    db.add(WarningPolicyState(key=key, state_json=encoded))
            decisions.append(
                {
                    "object_id": obj,
                    "kind": kind,
                    "notification_due": issued,
                    "notification_status": "older_or_same_snapshot"
                    if stale
                    else "between_hourly_checks"
                    if not hourly
                    else "issued"
                    if issued
                    else "suppressed_or_below_policy",
                    "pending_warnings": len([t for t in state.pending if at.value - t < 24 * HOUR_NS]),
                }
            )
        if not stale:
            lock.last_as_of = at.to_pydatetime()
        db.add(
            WarningBatch(
                job_id=job_id,
                as_of=at.to_pydatetime(),
                model_version=model_version,
                decisions_json=json.dumps(decisions),
            )
        )
        db.commit()
        return decisions
