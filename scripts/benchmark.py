"""Local HTTP concurrency check. Does not send telemetry outside the local app."""

import asyncio
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np


async def main():
    base = "http://127.0.0.1:8000"
    async with httpx.AsyncClient(base_url=base, timeout=120) as client:
        t = time.perf_counter()
        r = await client.get("/api/ready")
        r.raise_for_status()
        cold = time.perf_counter() - t
        r = await client.post("/api/auth/login", json={"username": "analyst", "password": "contour-demo"})
        r.raise_for_status()
        overview = (await client.get("/api/overview")).json()
        targets = overview["forecasts"][:20]

    async def user(i):
        durations = []
        async with httpx.AsyncClient(base_url=base, timeout=120) as client:
            (
                await client.post("/api/auth/login", json={"username": "analyst", "password": "contour-demo"})
            ).raise_for_status()
            f = targets[i]
            paths = [
                "/api/overview",
                "/api/topology",
                f"/api/forecast/{f['object_id']}/{f['kind']}?as_of={f['as_of']}",
                "/api/quality",
                "/api/decisions",
            ]
            for path in paths:
                begin = time.perf_counter()
                res = await client.get(path)
                durations.append(
                    {
                        "path": path.split("?")[0],
                        "status": res.status_code,
                        "ms": round((time.perf_counter() - begin) * 1000, 2),
                    }
                )
            (await client.post("/api/auth/logout")).raise_for_status()
        return durations

    started = time.perf_counter()
    rows = sum(await asyncio.gather(*(user(i) for i in range(20))), [])
    values = [r["ms"] for r in rows]
    report = {
        "platform": platform.platform(),
        "concurrent_sessions": 20,
        "requests": len(rows),
        "errors": sum(r["status"] != 200 for r in rows),
        "cold_readiness_seconds": round(cold, 3),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "max_ms": max(values),
        "requests_detail": rows,
        "scope": "Local macOS, SQLite, one API worker; not a production or PostgreSQL SLA test.",
    }
    Path("artifacts/performance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print({k: v for k, v in report.items() if k != "requests_detail"})


asyncio.run(main())
