"""Isolated batch worker: raw telemetry -> identical causal features -> frozen models."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
from catboost import CatBoostClassifier

from moscollector import features, prepare
from moscollector.domain import KIND_LABELS, RECOMMENDATIONS
from moscollector.inference import calibrated, model_input
from moscollector.model_registry import load_bundle
from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.service import clean

MIN_HISTORY_HOURS = 24


def run(directory: Path, as_of: str):
    started = time.monotonic()
    cutoff = pd.Timestamp(as_of)
    status_file = directory / "status.json"
    state = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
    version = state.get("model_version", "legacy")
    heads = load_bundle(version) if version != "legacy" else {}
    aligned = cutoff.floor("h")
    offset = cutoff - aligned
    year = cutoff.year
    processed = directory / "processed"
    artifacts = directory / "artifacts"
    processed.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    for name in ("channels.parquet", "objects.parquet"):
        shutil.copy2(PROCESSED / name, processed / name)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.read_parquet(str(directory / "input.parquet")).create_view("incoming")
    archive = PROCESSED / f"events-{year}.parquet"
    if archive.exists():
        con.read_parquet(str(archive)).create_view("archive")
        # Cut off future rows before any aggregation or feature derivation.
        con.execute(
            "CREATE TABLE combined AS SELECT event_id,channel_id,ts,value,numeric_value,alarm FROM archive WHERE ts < ? UNION ALL SELECT event_id,channel_id,ts,value,numeric_value,alarm FROM incoming",
            [cutoff.to_pydatetime()],
        )
    else:
        con.execute("CREATE TABLE combined AS SELECT * FROM incoming")
    begin, end, rows = con.execute("SELECT min(ts),max(ts),count(*) FROM combined").fetchone()
    if not begin or (cutoff - pd.Timestamp(begin)).total_seconds() < MIN_HISTORY_HOURS * 3600:
        raise ValueError(
            f"Для прогноза нужны данные хотя бы за {MIN_HISTORY_HOURS} часа до момента прогноза. "
            "Загрузите журнал за более длинный период"
        )
    if (cutoff - pd.Timestamp(end)).total_seconds() > 3600:
        raise ValueError(
            f"Последняя запись ({pd.Timestamp(end):%d.%m.%Y %H:%M}) раньше момента прогноза больше чем на час. "
            "Оставьте момент пустым, чтобы взять его по последней записи"
        )
    # Rebase the clock so aggregation uses trailing [t-h,t) windows at any
    # five-minute watermark, without introducing a second feature encoder.
    con.execute("UPDATE combined SET ts = ts - ? * INTERVAL '1 second'", [offset.total_seconds()])
    con.sql(
        "SELECT DISTINCT channel_id,ts,value,numeric_value,alarm,min(event_id) OVER(PARTITION BY channel_id,ts,value,alarm) AS event_id FROM combined"
    ).write_parquet(str(processed / f"events-{year}.parquet"), compression="zstd")
    con.close()
    # Globals change only within this dedicated subprocess, never inside API workers.
    features.PROCESSED = prepare.PROCESSED = processed
    features.ARTIFACTS = prepare.ARTIFACTS = artifacts
    prepare.write_json(
        artifacts / f"audit-{year}.json",
        {"summary": {"start": str(pd.Timestamp(begin) - offset), "end": aligned.isoformat()}},
    )
    features.aggregate_year(year, force=True)
    features.build_dataset([year], step_hours=1)
    data = pd.read_parquet(processed / "features.parquet")
    frame = data[data.as_of.eq(aligned)].copy()
    frame["as_of"] = cutoff
    hourly = pd.read_parquet(processed / f"hourly-{year}.parquet")
    hours = pd.date_range(aligned - pd.Timedelta(hours=168), aligned - pd.Timedelta(hours=1), freq="h")
    covered = hourly.groupby("hour").events.sum().reindex(hours, fill_value=0).gt(0)
    covered_hours = int(covered.sum())
    if covered_hours < MIN_HISTORY_HOURS:
        raise ValueError(
            f"За неделю до момента прогноза есть данные только за {covered_hours} ч из 168. "
            f"Нужно хотя бы {MIN_HISTORY_HOURS} ч: загрузите журнал за более длинный период"
        )
    # Models were trained on a full week of history. A shorter one still gives a forecast,
    # flagged in the result: weekly counters are then lower than in training.
    history = {
        "covered_hours": covered_hours,
        "required_hours": 168,
        "complete": covered_hours == 168,
        "first_covered_hour": covered[covered].index.min().isoformat() if covered_hours else None,
    }
    names = pd.read_parquet(processed / "objects.parquet").set_index("object_id").object_name.to_dict()
    forecasts = []
    for kind in ("fault", "fire", "flood", "access"):
        expected_counts = None
        if kind in heads:
            meta = heads[kind].meta
            probabilities = heads[kind].probability(frame)
            expected_counts = heads[kind].expected_count(frame)
        else:
            meta = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text(encoding="utf-8"))
            model = CatBoostClassifier()
            model.load_model(str(ARTIFACTS / "models" / f"{kind}.cbm"))
            probabilities = calibrated(
                model.predict(model_input(frame, meta["features"]), prediction_type="RawFormulaVal"),
                meta["calibration"],
            )
        for i, (obj, p) in enumerate(zip(frame.object_id, probabilities, strict=True)):
            policy = meta.get("alert_policy")
            count = float(expected_counts[i]) if expected_counts is not None else None
            above = bool(p >= meta["threshold"] and (policy is None or count >= policy["margin"]))
            forecasts.append(
                {
                    "object_id": int(obj),
                    "object_name": names[int(obj)],
                    "kind": kind,
                    "kind_label": KIND_LABELS[kind],
                    "probability": float(p),
                    "threshold": meta["threshold"],
                    "above_threshold": above,
                    **(
                        {
                            "expected_episodes": count,
                            "notification_due": False,
                            "notification_status": "preview",
                        }
                        if count is not None
                        else {}
                    ),
                    "recommendation": RECOMMENDATIONS[kind][0],
                }
            )
    forecasts.sort(key=lambda r: (not r["above_threshold"], -r["probability"] / max(r["threshold"], 0.001)))
    policies = {
        kind: head.meta["alert_policy"] for kind, head in heads.items() if "alert_policy" in head.meta
    }
    if policies and state.get("mode") == "accumulated_stream":
        from moscollector.database import make_database
        from moscollector.warning_storage import decide_stream_warnings

        observed = pd.read_parquet(processed / "episodes.parquet")
        observed["start_ts"] += offset
        engine, factory = make_database()
        try:
            warning_identity = hashlib.sha256(
                f"{state.get('sha256', directory.name)}:{version}".encode()
            ).hexdigest()
            decisions = decide_stream_warnings(
                factory, warning_identity, cutoff, version, forecasts, observed, policies
            )
            lookup = {(d["object_id"], d["kind"]): d for d in decisions}
            for row in forecasts:
                row.update(lookup.get((row["object_id"], row["kind"]), {}))
        finally:
            engine.dispose()
    result = {
        "as_of": cutoff.isoformat(),
        "model_version": version,
        "timezone": "Europe/Moscow",
        "horizon_hours": 24,
        "label_type": "proxy_sensor_episode",
        "mode": state.get("mode", "import_preview"),
        "stream_revision": state.get("stream_revision"),
        "history_rows": rows,
        "history": history,
        "history_start": str(begin),
        "last_observation": str(end),
        "source_cutoff": "strictly_before_as_of",
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "forecasts": forecasts,
        "notice": (
            "Расчёт по неизменяемому снимку накопленного потока и архиву."
            if state.get("mode") == "accumulated_stream"
            else "Независимый расчёт по пакету и архиву."
        )
        + " Исходный архив и тестовые метрики не изменены.",
    }
    prepare.write_json(directory / "result.json", clean(result))
    # Keep only the final feature snapshot and provenance, not duplicate historical telemetry.
    frame.to_parquet(directory / "feature_snapshot.parquet", index=False)
    shutil.rmtree(processed)
    shutil.rmtree(artifacts)
    return result


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    try:
        run(folder, sys.argv[2])
    except Exception as error:
        prepare.write_json(folder / "error.json", {"error": str(error)})
        raise
