"""Replay both count controls on the new input builder before neural training."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from moscollector.count_research import episode_counts
from moscollector.experiments.event_sequence_inputs import period_rows
from moscollector.experiments.fresh_counts_research import source_files
from moscollector.experiments.goal90_research import read
from moscollector.experiments.minute_cadence_research import forecasts
from moscollector.experiments.onset_channel_features import CATS, FOLDER, KEYS
from moscollector.experiments.onset_training_data import attach, model_input
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.train import calibrate, calibrated

torch.set_num_threads(4)
paths = [
    PROCESSED / name
    for name in (
        "features-channel-novelty.parquet",
        "features-dense-channel-novelty.parquet",
        "episodes.parquet",
    )
]
frame, dense = (pd.read_parquet(path) for path in paths[:2])
episodes = pd.read_parquet(paths[2], filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))])
triggers = pd.read_parquet(FOLDER / "triggers.parquet")
sources = {*paths, FOLDER / "triggers.parquet", FOLDER / "context.parquet"}
records = []
for kind in ("access", "fire", "fault"):
    eps = episodes.loc[episodes.kind.eq(kind)]
    for fold in ("screen_1", "screen_2"):
        metadata = source_files(kind, fold)[1]
        original = read(metadata)
        fresh, old = (
            Path("artifacts/research-v33") / kind / fold,
            Path("artifacts/research-v34") / kind / fold,
        )
        fresh_fit = read(fresh / "onset/fit.json")
        model = CatBoostRegressor()
        model.load_model(str(fresh / "onset/model.cbm"))
        sources.update(
            {
                metadata,
                fresh / "onset/fit.json",
                fresh / "onset/model.cbm",
                fresh / "result.json",
                old / "result.json",
            }
        )
        periods = original["periods"]
        context = pd.read_parquet(
            FOLDER / "context.parquet",
            filters=[
                ("as_of", ">=", pd.Timestamp(periods["calibration"][0])),
                ("as_of", "<", pd.Timestamp(periods["test"][1])),
            ],
            read_dictionary=CATS,
        )
        fresh_tables, exposure = {}, {}
        for part in ("calibration", "policy", "test"):
            base, slots = period_rows(frame, dense, triggers, periods, part)
            x = model_input(attach(base, slots, context, original["features"]), fresh_fit["features"])
            fresh_tables[part] = (
                slots[KEYS]
                .copy()
                .assign(raw=model.predict(x, prediction_type="RawFormulaVal", thread_count=2))
            )
            exposure[part] = len(base) / 24
        cal = fresh_tables["calibration"]
        counts = episode_counts(cal, eps)
        calibration = calibrate(cal.raw.to_numpy(), counts > 0)
        scale = float(counts.sum() / np.exp(np.clip(cal.raw, -20, 20)).sum())
        fresh_record = read(fresh / "result.json")["arms"]["onset_candidate"]
        assert calibration == fresh_record["calibration"] and scale == fresh_record["rate_scale"]
        old_tables, old_cal, old_exposure, _ = forecasts(kind, fold, frame, dense, triggers, eps)
        assert old_exposure == exposure == read(old / "result.json")["exposure"]
        assert all(
            read(old / "result.json")["arms"]["minute_candidate"][k] == v
            for k, v in old_cal["minute_candidate"].items()
        )
        for part in fresh_tables:
            pred = fresh_tables[part]
            pred["probability"] = calibrated(pred.raw, calibration)
            pred["expected_count"] = np.exp(np.clip(pred.raw, -20, 20)) * scale
            for path, actual in (
                (fresh / f"onset_candidate-{part}.parquet", pred),
                (old / f"minute_candidate-{part}.parquet", old_tables["minute_candidate"][part]),
            ):
                sources.add(path)
                pd.testing.assert_frame_equal(
                    actual, pd.read_parquet(path).drop(columns="alert", errors="ignore"), check_exact=True
                )
        records.append({"kind": kind, "fold": fold, "fresh_and_old_exact_parity": True, "exposure": exposure})
        print("CONTROL preflight", kind, fold, "all3parts exact", flush=True)
        del context, fresh_tables, old_tables
proof = {
    "scope": "All6fresh and6old count control periods replayed from frozen weights with the new period builder;36 prediction tables and calibration/count scales/exposure match exactly. No neural task model or policy evaluated.",
    "torch": str(torch.__version__),
    "periods": records,
    "source_hashes": {str(p): sha256(p) for p in sorted(sources)},
    "code_hashes": {
        str(p): sha256(p)
        for p in (
            Path(__file__),
            Path(period_rows.__code__.co_filename),
            Path(attach.__code__.co_filename),
            Path(forecasts.__code__.co_filename),
            Path(calibrate.__code__.co_filename),
        )
    },
}
write_json(Path("artifacts/research-v37/control-preflight.json"), proof)
