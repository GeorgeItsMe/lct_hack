"""Check real archived stream snapshots and warning idempotency on a local API."""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import httpx
import numpy as np
import pandas as pd

from moscollector.prepare import sha256, write_json

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--report", type=Path, default=Path("artifacts/count_stream_report.json"))
p.add_argument("--resume", action="store_true")
args = p.parse_args()
if urlparse(args.url).hostname not in ("127.0.0.1", "localhost", "::1"):
    raise SystemExit("Only local demo APIs are supported")
source = Path("data/processed/events-2026.parquet")
checksum = sha256(source)
with httpx.Client(base_url=args.url, timeout=120) as client:
    client.post(
        "/api/auth/login", json={"username": "dispatcher", "password": "contour-demo"}
    ).raise_for_status()
    if args.resume:
        report = json.loads(args.report.read_text())
        last = report["snapshots"][-1]
        job = client.post("/api/stream/forecast", params={"as_of": last["as_of"]}).raise_for_status().json()
        assert job["id"] == last["job_id"]
        result = client.get(f"/api/imports/{job['id']}").raise_for_status().json()["result"]
        assert result == report["last_result"]
        report["result_and_retry_unchanged_after_restart"] = True
    else:
        state = client.get("/api/stream").raise_for_status().json()
        current = pd.Timestamp(state["as_of"])
        hour = current.floor("h") + pd.Timedelta(hours=1)
        report = {
            "scope": "provided_archive_inference_and_storage_check_not_quality_evaluation",
            "url": args.url,
            "snapshots": [],
        }
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.read_parquet(str(source)).create_view("events")
        con.read_parquet("data/processed/channels.parquet").create_view("channels")
        for cutoff in (hour, hour + pd.Timedelta(minutes=5)):
            raw = con.execute(
                "SELECT e.channel_id,e.ts,e.value,e.alarm FROM events e JOIN channels c USING(channel_id) WHERE e.ts>=? AND e.ts<? ORDER BY e.ts,e.channel_id LIMIT 100000",
                [(cutoff - pd.Timedelta(minutes=5)).to_pydatetime(), cutoff.to_pydatetime()],
            ).df()
            assert not raw.empty
            payload = raw.to_json(orient="records", date_format="iso").encode()
            receipt = (
                client.post(
                    "/api/stream/events",
                    params={"as_of": cutoff.isoformat(), "format": "json"},
                    content=payload,
                )
                .raise_for_status()
                .json()
            )
            job_id = receipt["job"]["id"]
            started = time.monotonic()
            while True:
                job = client.get(f"/api/imports/{job_id}").raise_for_status().json()
                if job["status"] in ("complete", "failed"):
                    break
                if time.monotonic() - started > 300:
                    raise TimeoutError("Count model exceeded the inference budget")
                time.sleep(2)
            assert job["status"] == "complete", job
            result = job["result"]
            access = [r for r in result["forecasts"] if r["kind"] == "access"]
            assert len(result["forecasts"]) == 312 and len(access) == 78
            assert all(np.isfinite(r["expected_episodes"]) and r["expected_episodes"] >= 0 for r in access)
            if cutoff.minute:
                assert all(
                    not r["notification_due"] and r["notification_status"] == "between_hourly_checks"
                    for r in access
                )
            repeated = (
                client.post("/api/stream/forecast", params={"as_of": cutoff.isoformat()})
                .raise_for_status()
                .json()
            )
            assert repeated["id"] == job_id
            item = {
                "as_of": cutoff.isoformat(),
                "job_id": job_id,
                "model_version": result["model_version"],
                "new_access_warnings": sum(r["notification_due"] for r in access),
                "above_policy": sum(r["above_threshold"] for r in access),
                "worker_seconds": result["elapsed_seconds"],
                "retry_idempotent": True,
            }
            report["snapshots"].append(item)
            report["last_result"] = result
            print(item, flush=True)
        con.close()
    client.post("/api/auth/logout").raise_for_status()
assert sha256(source) == checksum
report["source_unchanged"] = True
write_json(args.report, report)
