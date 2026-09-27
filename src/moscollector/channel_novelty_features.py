"""Channel-relative onset summaries, independent of future episode labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.ordered_state_features import BEFORE, canonical_transitions
from moscollector.paths import PROCESSED
from moscollector.prepare import connection, sha256, write_json

SIGNALS = ("fault", "unknown", "power", "fire", "access", "conflict")
STATS = (
    "channels_6h",
    "channels_24h",
    "new_channels_24h",
    "max_onsets_6h",
    "max_excess_6h",
    "max_log_ratio_6h",
    "max_log_ratio_24h",
    "concentration_6h",
)
COLUMNS = [f"nov_{signal}_{stat}" for signal in SIGNALS for stat in STATS] + [
    "nov_observed_channels",
    "nov_established_channels",
]


def extraction_query():
    predicates = {
        "fault": "{state} LIKE 'Неисправен|alarm=%'",
        "unknown": "{state} IN ('Неопределен|alarm=false','Неопределен|alarm=true','Не определено|alarm=false','Не определено|alarm=true')",
        "power": "split_part({state},'|alarm=',1) IN ('Обесточен','Питание от батарей','Батарея разряжена','Батарея неисправна','Отключено устройство')",
        "fire": "(sensor_type='Датчик дыма' AND {state} LIKE 'Обнаружен дым|alarm=%') OR (sensor_type='Тепловой датчик' AND {state} LIKE 'Температура выше 40ºC|alarm=%')",
        "access": "sensor_type IN ('КД Дверь','КД Люк','КД АВ','9-секционный люк','Датчик движения','Стекло') AND {state} IN ('Не замкнут|alarm=true','Обнаружено движение|alarm=true','Стекло|alarm=true')",
        "conflict": "split_part({state},'|alarm=',1)='__CONFLICT__'",
    }
    pieces = []
    for name, predicate in predicates.items():
        current = predicate.format(state="state")
        previous = predicate.format(state="previous")
        pieces.append(
            f"SELECT channel_id,object_id,ts,'{name}' AS signal FROM typed "
            f"WHERE previous<>'__INITIAL_OBSERVATION__' AND ({current}) AND NOT ({previous})"
        )
    return " UNION ALL ".join(pieces) + " ORDER BY object_id,channel_id,signal,ts"


def extract(year, folder):
    target = folder / f"onsets-{year}.parquet"
    first_path = folder / f"first-observed-{year}.parquet"
    meta_path = folder / f"onsets-{year}.json"
    inputs = [
        PROCESSED / f"events-{year}.parquet",
        PROCESSED / "channels.parquet",
        Path(__file__),
        Path(canonical_transitions.__code__.co_filename),
    ]
    seed = PROCESSED / "ordered-states-v17" / f"last-state-{year - 1}.parquet" if year > 2022 else None
    if seed:
        inputs.append(seed)
    hashes = {str(p): sha256(p) for p in inputs}
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["inputs"] != hashes:
            raise ValueError("Channel-onset cache inputs changed")
        if any(sha256(Path(p)) != v for p, v in meta["outputs"].items()):
            raise ValueError("Channel-onset cache changed")
        return target, first_path
    folder.mkdir(parents=True, exist_ok=True)
    c = connection()
    c.read_parquet(str(inputs[0])).filter(f"ts < TIMESTAMP '{BEFORE}'").create_view("source")
    if seed:
        c.read_parquet(str(seed)).create_view("seed")
    else:
        c.execute(
            "CREATE VIEW seed AS SELECT NULL::BIGINT channel_id,NULL::TIMESTAMP ts,NULL::VARCHAR state WHERE false"
        )
    canonical_transitions(c)
    c.read_parquet(str(inputs[1])).create_view("channels")
    c.execute(
        "CREATE VIEW typed AS SELECT e.*,ch.object_id,ch.sensor_type FROM changes e JOIN channels ch USING(channel_id)"
    )
    c.sql(extraction_query()).write_parquet(str(target), compression="zstd")
    c.sql("""SELECT channel_id,object_id,min(ts) AS first_ts FROM canonical
        JOIN channels USING(channel_id) GROUP BY 1,2 ORDER BY 1""").write_parquet(
        str(first_path), compression="zstd"
    )
    rows = c.read_parquet(str(target)).count("*").fetchone()[0]
    c.close()
    write_json(
        meta_path,
        {
            "before": BEFORE,
            "inputs": hashes,
            "rows": rows,
            "outputs": {str(p): sha256(p) for p in (target, first_path)},
        },
    )
    print("Extracted channel onsets", year, rows, flush=True)
    return target, first_path


def transform(frame, onsets, first_observed):
    """Object summaries of rates measured separately for each channel.

    Established means first observed at least30d ago, not continuous coverage.
    Ratios compare disjoint past windows; no parametric probability is claimed.
    """
    frame = frame.reset_index(drop=True)
    first = first_observed.groupby(["object_id", "channel_id"]).first_ts.min()
    result = pd.DataFrame(0.0, index=frame.index, columns=COLUMNS, dtype=np.float32)
    hour = 3_600_000_000_000
    for obj, rows in frame.groupby("object_id", sort=False):
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        initial = (
            first.loc[obj] if obj in first.index.get_level_values(0) else pd.Series(dtype="datetime64[ns]")
        )
        first_values = initial.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        result.loc[rows.index, "nov_observed_channels"] = np.searchsorted(
            np.sort(first_values), at, side="left"
        )
        result.loc[rows.index, "nov_established_channels"] = np.searchsorted(
            np.sort(first_values), at - 720 * hour, side="right"
        )
        local = onsets[onsets.object_id.eq(obj)]
        for signal in SIGNALS:
            totals = {key: np.zeros(len(at), dtype=np.float64) for key in STATS}
            count6 = np.zeros(len(at), dtype=np.float64)
            for channel, events in local[local.signal.eq(signal)].groupby("channel_id"):
                if channel not in initial.index:
                    raise ValueError("Onset channel missing first observed timestamp")
                ts = events.ts.sort_values().to_numpy(dtype="datetime64[ns]").astype(np.int64)
                end = np.searchsorted(ts, at, side="left")
                n6, n24, n720 = [end - np.searchsorted(ts, at - h * hour, side="left") for h in (6, 24, 720)]
                established = initial.loc[channel].value <= at - 720 * hour
                prior6, prior24 = n720 - n6, n720 - n24
                expected6 = prior6 * 6 / 714
                expected24 = prior24 * 24 / 696
                totals["channels_6h"] += n6 > 0
                totals["channels_24h"] += n24 > 0
                totals["new_channels_24h"] += established & (n24 > 0) & (prior24 == 0)
                totals["max_onsets_6h"] = np.maximum(totals["max_onsets_6h"], n6)
                totals["max_excess_6h"] = np.maximum(
                    totals["max_excess_6h"], np.where(established, n6 - expected6, 0)
                )
                for hours, values, expected in ((6, n6, expected6), (24, n24, expected24)):
                    name = f"max_log_ratio_{hours}h"
                    ratio = np.where(established, np.log1p(values / (expected + 0.1)), 0)
                    totals[name] = np.maximum(totals[name], ratio)
                count6 += n6
            totals["concentration_6h"] = np.divide(
                totals["max_onsets_6h"], count6, out=np.zeros(len(at)), where=count6 > 0
            )
            for key, values in totals.items():
                result.loc[rows.index, f"nov_{signal}_{key}"] = values.astype(np.float32)
        print("Channel novelty", obj, len(at), flush=True)
    if not np.isfinite(result.to_numpy()).all() or (result.to_numpy() < 0).any():
        raise ValueError("Invalid channel novelty values")
    return result.astype(np.float32)


def build(folder):
    files = [extract(year, folder) for year in range(2022, 2027)]
    onsets = pd.concat([pd.read_parquet(p) for p, _ in files], ignore_index=True)
    first = pd.concat([pd.read_parquet(p) for _, p in files], ignore_index=True)
    sources = [PROCESSED / n for n in ("features.parquet", "features-dense-round4.parquet")]
    original = pd.read_parquet(sources[0], filters=[("as_of", "<", pd.Timestamp(BEFORE))])
    dense = pd.read_parquet(
        sources[1], columns=original.columns.tolist(), filters=[("as_of", "<", pd.Timestamp(BEFORE))]
    )
    keys = ["object_id", "as_of"]
    unique = (
        pd.concat([original[keys], dense[keys]])
        .drop_duplicates(keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )
    extra = pd.concat([unique, transform(unique, onsets, first)], axis=1)
    outputs = []
    for frame, name in (
        (original, "features-channel-novelty.parquet"),
        (dense, "features-dense-channel-novelty.parquet"),
    ):
        result = frame.merge(extra, on=keys, how="left", validate="one_to_one", sort=False)
        pd.testing.assert_frame_equal(result[frame.columns], frame.reset_index(drop=True))
        if result[COLUMNS].isna().any().any():
            raise ValueError("Missing channel features")
        target = PROCESSED / name
        result.to_parquet(target, index=False, compression="zstd")
        outputs.append(target)
        print("Saved", target, len(result), flush=True)
    write_json(
        folder / "build.json",
        {
            "before": BEFORE,
            "features": COLUMNS,
            "inputs": {
                str(p): sha256(p) for p in (*sources, Path(__file__), *(p for pair in files for p in pair))
            },
            "outputs": {str(p): sha256(p) for p in outputs},
            "original_columns_unchanged": True,
            "same_time_parity_by_construction": True,
            "history": "All onsets strictly before t. Last6/24h vs disjoint rest of last30d, per channel before aggregation. First observation excluded as onset. Cross-year carry from verified v17 canonical seeds. Established refers only to time since first observed; no continuous-coverage assumption or statistical significance claim.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=PROCESSED / "channel-novelty-v18")
    args = parser.parse_args()
    build(args.folder)
