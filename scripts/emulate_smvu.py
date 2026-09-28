"""Emulate the monitoring system (СМВУ) pushing telemetry to the stream API.

The organizers confirmed that real integration is impossible and asked for a plausible
emulation. This script replays real archived events for a time window through
POST /api/stream/events — the same entry point a monitoring gateway would use — and
waits for the forecast. The resulting warnings appear in the dispatcher's queue.

Usage: python scripts/emulate_smvu.py --as-of 2026-06-15T12:05 --minutes 60
In a pilot the gateway uses its own service account; the demo signs in as the analyst,
the role allowed to load data.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT / "data" / "processed", ROOT / "vercel_runtime" / "data" / "processed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--as-of", default="2026-06-15T12:05", help="Moment of the pushed snapshot, MSK")
    parser.add_argument("--minutes", type=int, default=60, help="Length of the replayed window")
    parser.add_argument("--username", default="analyst")
    parser.add_argument("--password", default="contour-demo")
    args = parser.parse_args()

    folder = next((c for c in CANDIDATES if (c / "events-2026.parquet").exists()), None)
    if folder is None:
        raise SystemExit("events-2026.parquet not found in data/processed or vercel_runtime/data/processed")
    cutoff = pd.Timestamp(args.as_of)
    con = duckdb.connect()
    frame = con.execute(
        "SELECT channel_id, ts, value, alarm FROM read_parquet(?) WHERE ts >= ? AND ts < ? ORDER BY ts LIMIT 100000",
        [
            str(folder / "events-2026.parquet"),
            (cutoff - pd.Timedelta(minutes=args.minutes)).to_pydatetime(),
            cutoff.to_pydatetime(),
        ],
    ).df()
    con.close()
    if frame.empty:
        raise SystemExit("No archived events in this window")
    payload = frame.to_json(orient="records", date_format="iso").encode()

    with httpx.Client(base_url=args.url, timeout=120) as client:
        client.post(
            "/api/auth/login", json={"username": args.username, "password": args.password}
        ).raise_for_status()
        started = time.monotonic()
        response = client.post(
            "/api/stream/events", params={"as_of": cutoff.isoformat(), "format": "json"}, content=payload
        )
        response.raise_for_status()
        accepted = response.json()
        receipt = accepted["receipt"]
        print(
            f"Принято событий: {receipt.get('inserted_rows')} (повторов {receipt.get('duplicate_rows', 0)})"
        )
        job = accepted.get("job")
        if not job:
            raise SystemExit(accepted.get("forecast_error") or "Расчёт не запущен")
        while True:
            state = client.get(f"/api/imports/{job['id']}").raise_for_status().json()
            if state["status"] in ("complete", "failed"):
                break
            if time.monotonic() - started > 300:
                raise SystemExit("Прогноз не готов за 300 секунд")
            time.sleep(2)
        if state["status"] == "failed":
            raise SystemExit(f"Расчёт не выполнен: {state.get('error')}")
        forecasts = state["result"]["forecasts"]
        warnings = [f for f in forecasts if f["above_threshold"]]
        print(
            f"Прогноз на {state['as_of']} готов за {time.monotonic() - started:.1f} с: "
            f"{len(forecasts)} прогнозов, выше порога {len(warnings)}"
        )


if __name__ == "__main__":
    main()
