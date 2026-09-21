"""Exercise two real telemetry batches and an exact retry on a local demo API.

This adds explicitly archived records to the application's stream copy. It never
changes source Parquet files or the frozen test. Run after starting the service.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import httpx
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--report", default="artifacts/stream_report.json")
args = parser.parse_args()
if urlparse(args.url).hostname not in {"localhost", "127.0.0.1", "::1"}:
    raise SystemExit("This demo verification only supports local hosts")

con = duckdb.connect()
con.execute("SET threads=2")
con.read_parquet("data/processed/events-2026.parquet").create_view("events")
con.read_parquet("data/processed/channels.parquet").create_view("channels")
report = {"base_url": args.url, "source": "provided_archive_replay", "batches": []}
source_path = Path("data/processed/events-2026.parquet")
original_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
with httpx.Client(base_url=args.url, timeout=120) as client:
    client.post(
        "/api/auth/login", json={"username": "dispatcher", "password": "contour-demo"}
    ).raise_for_status()
    for cutoff in [pd.Timestamp("2026-06-15T12:00"), pd.Timestamp("2026-06-15T12:05")]:
        frame = con.execute(
            "SELECT e.channel_id, e.ts, e.value, e.alarm FROM events e JOIN channels c USING(channel_id) "
            "WHERE e.ts >= ? AND e.ts < ? ORDER BY e.ts, e.channel_id LIMIT 100000",
            [(cutoff - pd.Timedelta(minutes=5)).to_pydatetime(), cutoff.to_pydatetime()],
        ).df()
        assert not frame.empty
        payload = frame.to_json(orient="records", date_format="iso").encode()
        started = time.monotonic()
        response = client.post(
            "/api/stream/events", params={"as_of": cutoff.isoformat(), "format": "json"}, content=payload
        )
        response.raise_for_status()
        accepted = response.json()
        assert accepted["job"], accepted
        job_id = accepted["job"]["id"]
        while True:
            job = client.get(f"/api/imports/{job_id}").raise_for_status().json()
            if job["status"] in {"complete", "failed"}:
                break
            if time.monotonic() - started > 300:
                raise TimeoutError("Stream prediction took longer than 300 seconds")
            time.sleep(2)
        assert job["status"] == "complete", job
        assert len(job["result"]["forecasts"]) == 312
        prediction = job["result"]["forecasts"][0]
        detail = (
            client.get(f"/api/imports/{job_id}/forecast/{prediction['object_id']}/{prediction['kind']}")
            .raise_for_status()
            .json()
        )
        assert len(detail["explanation"]) == 8
        assert all(pd.Timestamp(e["ts"]) < cutoff for e in detail["source_events"])
        decision = (
            client.post(
                "/api/decisions",
                json={
                    "prediction_id": detail["id"],
                    "batch_id": job_id,
                    "object_id": detail["object_id"],
                    "kind": detail["kind"],
                    "as_of": detail["as_of"],
                    "action": "monitor",
                    "reason": "maintenance_test",
                    "comment": "Демонстрационная проверка потока: архивные пакеты, команды оборудованию не отправлялись.",
                },
            )
            .raise_for_status()
            .json()
        )
        assert decision["saved"]
        count_before = client.get("/api/stream").raise_for_status().json()["events"]
        repeated = (
            client.post(
                "/api/stream/events", params={"as_of": cutoff.isoformat(), "format": "json"}, content=payload
            )
            .raise_for_status()
            .json()
        )
        count_after = client.get("/api/stream").raise_for_status().json()["events"]
        assert repeated["receipt"]["replayed"]
        assert repeated["job"]["id"] == job_id
        assert count_before == count_after
        item = {
            "as_of": cutoff.isoformat(),
            "input_rows": len(frame),
            "stored_stream_rows": count_after,
            "job_id": job_id,
            "mode": job["result"]["mode"],
            "stream_revision": job["result"]["stream_revision"],
            "forecast_count": len(job["result"]["forecasts"]),
            "inference_seconds": job["result"]["elapsed_seconds"],
            "receipt_to_result_seconds": round(time.monotonic() - started, 2),
            "retry_idempotent": True,
            "causal_explanation_checked": True,
            "dispatcher_decision_saved": True,
        }
        report["batches"].append(item)
        print(json.dumps(item), flush=True)
    client.post("/api/auth/logout").raise_for_status()
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_digest
report["source_unchanged"] = True
report["scope"] = "Local replay; real customer source and network SLA not tested."
Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
