"""Isolated batch worker: raw telemetry -> identical causal features -> frozen models."""

from __future__ import annotations

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
from moscollector.paths import ARTIFACTS, PROCESSED
from moscollector.service import clean
from moscollector.train import calibrated, model_input


def run(directory: Path, as_of: str):
    started = time.monotonic()
    cutoff = pd.Timestamp(as_of)
    status_file = directory / "status.json"
    state = json.loads(status_file.read_text()) if status_file.exists() else {}
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
    if not begin or (cutoff - pd.Timestamp(begin)).total_seconds() < 168 * 3600:
        raise ValueError("Для прогноза нужна как минимум неделя истории. Загрузите предысторию")
    if (cutoff - pd.Timestamp(end)).total_seconds() > 3600:
        raise ValueError("Последние события старше часа. Уточните момент прогноза или источник")
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
    if not covered.all():
        raise ValueError(
            f"В предыдущей неделе отсутствуют {int((~covered).sum())} часов общего потока. Прогноз не выдан"
        )
    names = pd.read_parquet(processed / "objects.parquet").set_index("object_id").object_name.to_dict()
    forecasts = []
    for kind in ("fault", "fire", "flood", "access"):
        meta = json.loads((ARTIFACTS / "models" / f"{kind}.json").read_text())
        model = CatBoostClassifier()
        model.load_model(str(ARTIFACTS / "models" / f"{kind}.cbm"))
        probabilities = calibrated(
            model.predict(model_input(frame, meta["features"]), prediction_type="RawFormulaVal"),
            meta["calibration"],
        )
        for obj, p in zip(frame.object_id, probabilities, strict=True):
            forecasts.append(
                {
                    "object_id": int(obj),
                    "object_name": names[int(obj)],
                    "kind": kind,
                    "kind_label": KIND_LABELS[kind],
                    "probability": float(p),
                    "threshold": meta["threshold"],
                    "above_threshold": bool(p >= meta["threshold"]),
                    "recommendation": RECOMMENDATIONS[kind][0],
                }
            )
    forecasts.sort(key=lambda r: (not r["above_threshold"], -r["probability"] / r["threshold"]))
    result = {
        "as_of": cutoff.isoformat(),
        "timezone": "Europe/Moscow",
        "horizon_hours": 24,
        "label_type": "proxy_sensor_episode",
        "mode": state.get("mode", "import_preview"),
        "stream_revision": state.get("stream_revision"),
        "history_rows": rows,
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
