"""Verify a local model upgrade on the existing stream without changing telemetry."""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd

from moscollector.model_registry import active_version, load_bundle

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--report", default="artifacts/model_upgrade_report.json")
args = parser.parse_args()
if urlparse(args.url).hostname not in {"localhost", "127.0.0.1", "::1"}:
    raise SystemExit("Only local demo verification is supported")
version = active_version()
changed = set(load_bundle(version))
with httpx.Client(base_url=args.url, timeout=90) as client:
    client.post(
        "/api/auth/login", json={"username": "dispatcher", "password": "contour-demo"}
    ).raise_for_status()
    ready = client.get("/api/ready").raise_for_status().json()
    assert ready["operational_model_version"] == version
    before = client.get("/api/stream").raise_for_status().json()
    states = client.get("/api/imports").raise_for_status().json()
    old_state = next(
        s
        for s in states
        if s.get("mode") == "accumulated_stream"
        and s["status"] == "complete"
        and s["as_of"] == before["as_of"]
        and s.get("model_version", "legacy") != version
    )
    old_id = old_state["id"]
    old = client.get(f"/api/imports/{old_id}").raise_for_status().json()
    kind = sorted(changed)[0]
    sample = max((r for r in old["result"]["forecasts"] if r["kind"] == kind), key=lambda r: r["probability"])
    old_detail_path = f"/api/imports/{old_id}/forecast/{sample['object_id']}/{kind}"
    old_detail = client.get(old_detail_path).raise_for_status().json()
    started = time.monotonic()
    job = client.post("/api/stream/forecast", params={"as_of": before["as_of"]}).raise_for_status().json()
    job_id = job["id"]
    assert job_id != old_id
    while True:
        job = client.get(f"/api/imports/{job_id}").raise_for_status().json()
        if job["status"] in ("complete", "failed"):
            break
        if time.monotonic() - started > 300:
            raise TimeoutError("Model upgrade calculation exceeded 300 seconds")
        time.sleep(2)
    assert job["status"] == "complete", job
    assert job["result"]["model_version"] == version
    previous = {(r["object_id"], r["kind"]): r for r in old["result"]["forecasts"]}
    assert len(job["result"]["forecasts"]) == len(previous) == 312
    unchanged, changed_rows = 0, 0
    for row in job["result"]["forecasts"]:
        prior = previous[row["object_id"], row["kind"]]
        delta = abs(row["probability"] - prior["probability"])
        if row["kind"] not in changed:
            assert delta < 1e-12
            unchanged += 1
        elif delta > 1e-9:
            changed_rows += 1
    assert changed_rows > 0
    new_detail = (
        client.get(f"/api/imports/{job_id}/forecast/{sample['object_id']}/{kind}").raise_for_status().json()
    )
    assert new_detail["model_version"] == version and len(new_detail["explanation"]) == 8
    assert all(pd.Timestamp(row["ts"]) < pd.Timestamp(job["as_of"]) for row in new_detail["source_events"])
    assert client.get(old_detail_path).raise_for_status().json() == old_detail
    assert client.get(f"/api/imports/{old_id}").raise_for_status().json() == old
    repeated = (
        client.post("/api/stream/forecast", params={"as_of": before["as_of"]}).raise_for_status().json()
    )
    assert repeated["id"] == job_id
    after = client.get("/api/stream").raise_for_status().json()
    assert after["events"] == before["events"] and after["as_of"] == before["as_of"]
    report = {
        "url": args.url,
        "model_version": version,
        "old_job_id": old_id,
        "new_job_id": job_id,
        "changed_kinds": sorted(changed),
        "changed_probability_rows": changed_rows,
        "unchanged_probability_rows": unchanged,
        "old_result_and_explanation_unchanged": True,
        "versioned_retry_idempotent": True,
        "stream_events_unchanged": after["events"],
        "source_events_cutoff_verified": True,
        "forecast_count": 312,
        "worker_seconds": job["result"]["elapsed_seconds"],
        "new_explanation": new_detail["explanation"],
    }
    client.post("/api/auth/logout").raise_for_status()
Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "new_explanation"}, ensure_ascii=False), flush=True)
