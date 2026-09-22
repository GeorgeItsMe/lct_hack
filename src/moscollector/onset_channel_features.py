"""v33: causal channel identities and recent onset context at arbitrary queries."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from moscollector.alert_diagnostics import HOUR_NS
from moscollector.cadence_capacity import subdivide_evaluation_slots
from moscollector.channel_novelty_features import SIGNALS
from moscollector.goal90_research import read
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

FOLDER = PROCESSED / "onset-context-v33"
KEYS = ["object_id", "as_of"]
CATS = [f"evt_{signal}_{name}" for signal in SIGNALS for name in ("channel", "sensor_type")]
NUMERIC = [
    f"evt_{signal}_{name}"
    for signal in SIGNALS
    for name in (
        "age_h",
        "gap_h",
        "channel_n_1h",
        "channel_n_24h",
        "object_n_1m",
        "object_n_1h",
        "object_n_6h",
        "object_n_24h",
    )
]
COLUMNS = [*CATS, *NUMERIC]
MAX_AGE = 720


def minute_onsets(onsets):
    raw = onsets.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    minute = HOUR_NS // 60
    return (
        onsets[["object_id"]]
        .assign(as_of=pd.to_datetime((raw // minute + 1) * minute))
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
        .reset_index(drop=True)
    )


def channel_context(queries, onsets, catalog):
    """No labels, future durations, or future values enter a query's features.

    For each signal retain its latest channel within30d (ties by channel ID),
    that channel's same-signal gap and counts, plus all-channel signal counts.
    Lack of recorded onsets is not a healthy-state or continuous-coverage claim.
    """
    queries = queries[KEYS].reset_index(drop=True)
    if queries.duplicated(KEYS).any() or queries.isna().any().any():
        raise ValueError("Duplicate or missing context query")
    if catalog.channel_id.duplicated().any():
        raise ValueError("Duplicate catalog channel")
    source = onsets[["object_id", "channel_id", "signal", "ts"]].copy()
    if source.isna().any().any() or source.duplicated(["channel_id", "signal", "ts"]).any():
        raise ValueError("Invalid canonical onset rows")
    typed = source.merge(
        catalog[["channel_id", "object_id", "sensor_type"]].rename(columns={"object_id": "catalog_object"}),
        on="channel_id",
        how="left",
        validate="many_to_one",
    )
    if typed.catalog_object.isna().any() or not typed.object_id.eq(typed.catalog_object).all():
        raise ValueError("Onset object differs from catalog")
    typed["sensor_type"] = typed.sensor_type.fillna("__UNKNOWN__").astype(str)
    result = {name: np.full(len(queries), "__NONE__", dtype=object) for name in CATS}
    result.update({name: np.zeros(len(queries), dtype=np.float32) for name in NUMERIC})
    for signal in SIGNALS:
        result[f"evt_{signal}_age_h"].fill(MAX_AGE + 1)
        result[f"evt_{signal}_gap_h"].fill(np.nan)
    for obj, rows in queries.groupby("object_id", sort=False):
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        dest = rows.index.to_numpy()
        for signal in SIGNALS:
            history = typed.loc[typed.object_id.eq(obj) & typed.signal.eq(signal)].sort_values(
                ["ts", "channel_id"], kind="stable"
            )
            if history.empty:
                continue
            ts = history.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            ids = history.channel_id.to_numpy(dtype=np.int64)
            end = np.searchsorted(ts, at, side="left")
            for name, window in (
                ("1m", HOUR_NS // 60),
                ("1h", HOUR_NS),
                ("6h", 6 * HOUR_NS),
                ("24h", 24 * HOUR_NS),
            ):
                result[f"evt_{signal}_object_n_{name}"][dest] = end - np.searchsorted(
                    ts, at - window, side="left"
                )
            index = np.maximum(end - 1, 0)
            age = (at - ts[index]) / HOUR_NS
            valid = (end > 0) & (age <= MAX_AGE)
            if not valid.any():
                continue
            selected, position = dest[valid], index[valid]
            chosen = ids[position]
            previous = history.groupby("channel_id").ts.shift(1)
            gap = (history.ts - previous).dt.total_seconds().to_numpy() / 3600
            result[f"evt_{signal}_channel"][selected] = chosen.astype(str)
            result[f"evt_{signal}_sensor_type"][selected] = history.sensor_type.to_numpy()[position]
            result[f"evt_{signal}_age_h"][selected] = age[valid]
            result[f"evt_{signal}_gap_h"][selected] = np.minimum(gap[position], MAX_AGE + 1)
            query = at[valid]
            for channel in np.unique(chosen):
                local = chosen == channel
                times = ts[ids == channel]
                for hours in (1, 24):
                    count = np.searchsorted(times, query[local], side="left") - np.searchsorted(
                        times, query[local] - hours * HOUR_NS, side="left"
                    )
                    result[f"evt_{signal}_channel_n_{hours}h"][selected[local]] = count
    return pd.DataFrame(result, columns=COLUMNS)


def build(folder=FOLDER):
    manifest = folder / "build.json"
    paths = [PROCESSED / "channel-novelty-v18" / f"onsets-{year}.parquet" for year in range(2022, 2027)]
    catalog_path = PROCESSED / "channels.parquet"
    original = PROCESSED / "features-channel-novelty.parquet"
    dense_path = PROCESSED / "features-dense-channel-novelty.parquet"
    inputs = {catalog_path, original, dense_path, *paths}
    for year in range(2022, 2027):
        source = PROCESSED / "channel-novelty-v18" / f"onsets-{year}.json"
        inputs.add(source)
        meta = read(source)
        for category in ("inputs", "outputs"):
            for path, value in meta[category].items():
                if sha256(Path(path)) != value:
                    raise ValueError(f"Changed onset provenance: {path}")
                inputs.add(Path(path))
    code = {Path(__file__), Path(subdivide_evaluation_slots.__code__.co_filename)}
    hashes = {
        "inputs": {str(p): sha256(p) for p in sorted(inputs)},
        "code_hashes": {str(p): sha256(p) for p in sorted(code)},
    }
    if manifest.exists():
        old = read(manifest)
        if any(old[k] != v for k, v in hashes.items()) or any(
            sha256(Path(p)) != v for p, v in old["outputs"].items()
        ):
            raise ValueError("Onset context build changed")
        return old
    folder.mkdir(parents=True, exist_ok=True)
    onsets = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    assert onsets.ts.lt(pd.Timestamp("2026-06-01")).all()
    catalog = pd.read_parquet(catalog_path)
    triggers = minute_onsets(onsets)
    triggers = triggers.loc[triggers.as_of.lt(pd.Timestamp("2026-06-01"))].reset_index(drop=True)
    base = pd.read_parquet(original, columns=KEYS)
    dense = pd.read_parquet(dense_path, columns=KEYS)
    quarters = subdivide_evaluation_slots(dense)
    queries = (
        pd.concat([base, quarters, triggers], ignore_index=True)
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    assert queries.as_of.lt(pd.Timestamp("2026-06-01")).all()
    schema = pa.schema(
        [
            ("object_id", pa.int64()),
            ("as_of", pa.timestamp("ns")),
            *[(c, pa.string()) for c in CATS],
            *[(c, pa.float32()) for c in NUMERIC],
        ]
    )
    target = folder / "context.parquet"
    with pq.ParquetWriter(target, schema, compression="zstd", use_dictionary=True) as writer:
        for obj, group in queries.groupby("object_id", sort=True):
            group = group.reset_index(drop=True)
            local = onsets.loc[onsets.object_id.eq(obj)]
            extra = channel_context(group, local, catalog)
            output = pd.concat([group, extra], axis=1)
            writer.write_table(pa.Table.from_pandas(output, schema=schema, preserve_index=False))
            print("CONTEXT", obj, len(group), flush=True)
    trigger_path = folder / "triggers.parquet"
    triggers.to_parquet(trigger_path, index=False, compression="zstd")
    report = {
        **hashes,
        "outputs": {str(p): sha256(p) for p in (target, trigger_path)},
        "rows": len(queries),
        "raw_onsets": len(onsets),
        "minute_triggers": len(triggers),
        "categorical": CATS,
        "numeric": NUMERIC,
        "columns": COLUMNS,
        "before": "2026-06-01",
        "history": "Strictly past unpruned onsets from2022; latest channel per signal up to30d, channel-specific preceding gap and1/24h counts, object1min/1/6/24h counts. First observed state is not an onset. All source rows remain unchanged. Static catalog and unknown heartbeat limitations remain; tied channel IDs are a deterministic convention, not physical ordering.",
    }
    write_json(manifest, report)
    return report


if __name__ == "__main__":
    build()
