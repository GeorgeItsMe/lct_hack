"""Causal co-activity within literal catalog-tag groups, not inferred wiring."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from moscollector.experiments.channel_novelty_features import extraction_query
from moscollector.experiments.goal90_research import read
from moscollector.experiments.ordered_state_features import canonical_transitions
from moscollector.paths import PROCESSED
from moscollector.prepare import connection, sha256, write_json

FOLDER = PROCESSED / "channel-tags-v41"
CUTOFF = pd.Timestamp("2026-06-01")
KEYS = ["object_id", "as_of"]
SIGNALS = ("fault", "unknown", "power", "fire", "access", "conflict", "flood", "pump")
STATS = (
    "groups_6h",
    "groups_24h",
    "max_channels_6h",
    "max_channels_24h",
    "max_fraction_6h",
    "max_fraction_24h",
    "max_other_channels_6h",
    "max_signal_types_6h",
)
COLUMNS = [f"tag_{signal}_{stat}" for signal in SIGNALS for stat in STATS]
HOUR = 3_600_000_000_000


def catalog_groups(catalog):
    """Same object and full parent tag; unknown syntax gets its own singleton."""
    if catalog.channel_id.duplicated().any() or catalog[["channel_id", "object_id"]].isna().any().any():
        raise ValueError("Invalid tag catalog identity")
    out = catalog[["channel_id", "object_id", "sensor_type", "system_tag"]].copy()
    text = out.system_tag.fillna("").astype(str).str.strip().str.replace(r"\.$", "", regex=True)
    valid = text.str.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:\.[0-9]+)+")
    parent = text.str.rsplit(".", n=1).str[0]
    out["tag_valid"] = valid
    out["tag_group"] = np.where(valid, "tag:" + parent, "channel:" + out.channel_id.astype(str))
    out["group_size"] = out.groupby(["object_id", "tag_group"]).channel_id.transform("size")
    return out


def extra_query():
    flood = "(sensor_type='Состояние насоса' AND {s} LIKE 'Затоплен|alarm=%') OR (sensor_type='Датчик затопления' AND {s}='Не замкнут|alarm=true')"
    return f"""SELECT channel_id,object_id,ts,'flood' AS signal FROM typed
        WHERE previous<>'__INITIAL_OBSERVATION__' AND ({flood.format(s="state")})
          AND NOT ({flood.format(s="previous")})
        UNION ALL SELECT channel_id,object_id,ts,'pump' AS signal FROM typed
        WHERE previous<>'__INITIAL_OBSERVATION__' AND sensor_type='Состояние насоса'
          AND split_part(state,'|alarm=',1) IN ('Включен','Выключен','Работают все насосы в АНС')
          AND split_part(state,'|alarm=',1)<>split_part(previous,'|alarm=',1)"""


def extract(year, folder=FOLDER):
    target, manifest = folder / f"onsets-{year}.parquet", folder / f"onsets-{year}.json"
    original = PROCESSED / "channel-novelty-v18" / f"onsets-{year}.parquet"
    source = PROCESSED / f"events-{year}.parquet"
    catalog = PROCESSED / "channels.parquet"
    seed = PROCESSED / "ordered-states-v17" / f"last-state-{year - 1}.parquet" if year > 2022 else None
    inputs = {
        source,
        catalog,
        original,
        Path(__file__),
        Path(canonical_transitions.__code__.co_filename),
        Path(extraction_query.__code__.co_filename),
    }
    if seed:
        inputs.add(seed)
    hashes = {str(p): sha256(p) for p in sorted(inputs)}
    if manifest.exists():
        meta = read(manifest)
        if hashes != meta["inputs"] or sha256(target) != meta["output_sha256"]:
            raise ValueError("Tag onset cache changed")
        return target
    folder.mkdir(parents=True, exist_ok=True)
    c = connection()
    c.read_parquet(str(source)).filter(f"ts<TIMESTAMP '{CUTOFF}'").create_view("source")
    if seed:
        c.read_parquet(str(seed)).create_view("seed")
    else:
        c.execute(
            "CREATE VIEW seed AS SELECT NULL::BIGINT channel_id,NULL::TIMESTAMP ts,NULL::VARCHAR state WHERE false"
        )
    canonical_transitions(c)
    c.read_parquet(str(catalog)).create_view("channels")
    c.execute(
        "CREATE VIEW typed AS SELECT e.*,ch.object_id,ch.sensor_type FROM changes e JOIN channels ch USING(channel_id)"
    )
    c.execute("CREATE TABLE original_signals AS " + extraction_query())
    c.read_parquet(str(original)).create_view("old")
    for first, second in (("old", "original_signals"), ("original_signals", "old")):
        if c.execute(
            f"SELECT count(*) FROM (SELECT * FROM {first} EXCEPT ALL SELECT * FROM {second})"
        ).fetchone()[0]:
            raise ValueError("Original six onset families differ")
    c.execute("CREATE TABLE extra_signals AS " + extra_query())
    c.sql(
        "SELECT * FROM original_signals UNION ALL SELECT * FROM extra_signals ORDER BY object_id,channel_id,signal,ts"
    ).write_parquet(str(target), compression="zstd")
    counts = c.execute(
        "SELECT signal,count(*) AS rows FROM (SELECT * FROM original_signals UNION ALL SELECT * FROM extra_signals) GROUP BY 1 ORDER BY 1"
    ).fetchdf()
    c.close()
    write_json(
        manifest,
        {
            "inputs": hashes,
            "output_sha256": sha256(target),
            "original_six_signals_exact": True,
            "signals": counts.to_dict("records"),
        },
    )
    print("TAG ONSETS", year, counts.to_dict("records"), flush=True)
    return target


def transform(queries, onsets, catalog):
    queries = queries[KEYS].reset_index(drop=True)
    if queries.isna().any().any() or queries.duplicated(KEYS).any():
        raise ValueError("Invalid tag query keys")
    if (
        onsets[["channel_id", "object_id", "signal", "ts"]].isna().any().any()
        or onsets.duplicated(["channel_id", "signal", "ts"]).any()
        or not onsets.signal.isin(SIGNALS).all()
    ):
        raise ValueError("Invalid tag onset stream")
    groups = catalog_groups(catalog)
    source = onsets.merge(
        groups[["channel_id", "object_id", "tag_group"]].rename(columns={"object_id": "catalog_object"}),
        on="channel_id",
        how="left",
        validate="many_to_one",
    )
    if source.tag_group.isna().any() or not source.object_id.eq(source.catalog_object).all():
        raise ValueError("Tag onset identity differs from catalog")
    result = pd.DataFrame(0.0, index=queries.index, columns=COLUMNS, dtype=np.float32)
    for obj, rows in queries.groupby("object_id", sort=False):
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        local = source.loc[source.object_id.eq(obj)]
        sizes = groups.loc[groups.object_id.eq(obj)].groupby("tag_group").channel_id.size()
        output = {s: {k: np.zeros(len(rows)) for k in STATS} for s in SIGNALS}
        for group_name, history in local.groupby("tag_group", sort=False):
            activity6 = np.zeros((len(SIGNALS), len(rows)), dtype=np.int32)
            activity24 = np.zeros_like(activity6)
            other6 = np.zeros_like(activity6)
            for _, channel in history.groupby("channel_id", sort=False):
                flags6, flags24 = [], []
                for signal in SIGNALS:
                    ts = np.sort(
                        channel.loc[channel.signal.eq(signal), "ts"]
                        .to_numpy(dtype="datetime64[ns]")
                        .astype(np.int64)
                    )
                    end = np.searchsorted(ts, at, side="left")
                    flags6.append(end > np.searchsorted(ts, at - 6 * HOUR, side="left"))
                    flags24.append(end > np.searchsorted(ts, at - 24 * HOUR, side="left"))
                flags6, flags24 = np.asarray(flags6), np.asarray(flags24)
                activity6 += flags6
                activity24 += flags24
                any_signal = flags6.any(axis=0)
                # A channel with this signal is never called a distinct other peer.
                other6 += any_signal[None, :] & ~flags6
            signal_types = (activity6 > 0).sum(axis=0)
            for i, signal in enumerate(SIGNALS):
                stats, active = output[signal], activity6[i] > 0
                stats["groups_6h"] += active
                stats["groups_24h"] += activity24[i] > 0
                for hours, counts in ((6, activity6[i]), (24, activity24[i])):
                    for key, values in (
                        (f"max_channels_{hours}h", counts),
                        (f"max_fraction_{hours}h", counts / sizes[group_name]),
                    ):
                        stats[key] = np.maximum(stats[key], values)
                stats["max_other_channels_6h"] = np.maximum(
                    stats["max_other_channels_6h"], np.where(active, other6[i], 0)
                )
                stats["max_signal_types_6h"] = np.maximum(
                    stats["max_signal_types_6h"], np.where(active, signal_types, 0)
                )
        for signal in SIGNALS:
            for stat in STATS:
                result.loc[rows.index, f"tag_{signal}_{stat}"] = output[signal][stat]
    if not np.isfinite(result.to_numpy()).all() or (result.to_numpy() < 0).any():
        raise ValueError("Invalid group co-activity")
    return result.astype(np.float32)


def support(catalog):
    tagged = catalog_groups(catalog)
    g = tagged.groupby(["object_id", "tag_group"]).agg(
        channels=("channel_id", "size"), types=("sensor_type", "nunique"), parsed=("tag_valid", "all")
    )
    mixed = g.channels.gt(1) & g.types.gt(1)
    return {
        "channels": len(tagged),
        "valid_tag_channels": int(tagged.tag_valid.sum()),
        "singleton_fallback_channels": int((~tagged.tag_valid).sum()),
        "groups": len(g),
        "mixed_type_groups": int(mixed.sum()),
        "channels_in_mixed_groups": int(g.loc[mixed, "channels"].sum()),
        "group_size_quantiles": g.channels.quantile([0, 0.5, 0.9, 0.99, 1]).to_dict(),
        "scope": "Literal same-object parent tags only. No physical adjacency, shared controller, installation dates or historical topology is established by this grouping.",
    }


def build(folder=FOLDER):
    paths = [extract(y, folder) for y in range(2022, 2027)]
    sources = [
        PROCESSED / n
        for n in (
            "features.parquet",
            "features-hourly-research.parquet",
            "features-channel-novelty.parquet",
            "features-dense-channel-novelty.parquet",
        )
    ]
    catalog_path = PROCESSED / "channels.parquet"
    catalog = pd.read_parquet(catalog_path)
    hashes = {
        "inputs": {str(p): sha256(p) for p in (*sources, *paths, catalog_path)},
        "code_hashes": {str(Path(__file__)): sha256(Path(__file__))},
    }
    target, manifest = folder / "context.parquet", folder / "build.json"
    if manifest.exists():
        old = read(manifest)
        if any(old[k] != v for k, v in hashes.items()) or sha256(target) != old["output_sha256"]:
            raise ValueError("Tag feature cache changed")
        return old
    queries = (
        pd.concat(
            [pd.read_parquet(p, columns=KEYS, filters=[("as_of", "<", CUTOFF)]) for p in sources],
            ignore_index=True,
        )
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
    )
    schema = pa.schema(
        [("object_id", pa.int64()), ("as_of", pa.timestamp("ns")), *[(c, pa.float32()) for c in COLUMNS]]
    )
    n = 0
    with pq.ParquetWriter(target, schema, compression="zstd") as writer:
        for obj, q in queries.groupby("object_id", sort=True):
            q = q.reset_index(drop=True)
            onsets = pd.read_parquet(paths, filters=[("object_id", "==", int(obj))])
            out = pd.concat([q, transform(q, onsets, catalog)], axis=1)
            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
            n += len(q)
            print("TAG CONTEXT", obj, len(q), flush=True)
    report = {
        **hashes,
        "rows": n,
        "features": COLUMNS,
        "before": str(CUTOFF),
        "output_sha256": sha256(target),
        "catalog_support": support(catalog),
        "extraction_manifests": {
            str(folder / f"onsets-{y}.json"): sha256(folder / f"onsets-{y}.json") for y in range(2022, 2027)
        },
    }
    write_json(manifest, report)
    write_json(
        Path("artifacts/channel_tag_support_audit.json"),
        {
            "catalog_support": support(catalog),
            "feature_rows": n,
            "features": COLUMNS,
            "build_sha256": sha256(manifest),
            "source_hashes": {str(catalog_path): sha256(catalog_path), str(manifest): sha256(manifest)},
            "code_hashes": hashes["code_hashes"],
            "event_metrics_evaluated": False,
            "goal_achieved": False,
        },
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    print(build(parser.parse_args().folder)["rows"])
