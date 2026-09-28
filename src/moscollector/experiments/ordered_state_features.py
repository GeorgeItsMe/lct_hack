"""Past-only categorical channel transitions, preserving order and identity.

No episode labels, episode ends or future duration filters enter these features.
All observations at a forecast's exact timestamp are excluded. Numeric readings
share a token; their measurement magnitudes remain in the original features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.paths import PROCESSED
from moscollector.prepare import connection, sha256, write_json

BEFORE = "2026-06-01"
LAGS = 4
MAX_AGE_HOURS = 30 * 24
CATS = [
    f"order_{scope}_{lag}_{field}"
    for scope in ("own", "sibling")
    for lag in range(1, LAGS + 1)
    for field in ("channel", "transition")
]
AGES = [f"order_{scope}_{lag}_age_h" for scope in ("own", "sibling") for lag in range(1, LAGS + 1)]


def canonical_transitions(c):
    """Read registered source/seed views; leave canonical and changes tables.

    seed contains the last observed state per channel from the preceding year.
    Ambiguous duplicate timestamps become a distinct observed unknown token.
    """
    c.execute("""CREATE TABLE canonical AS
        SELECT channel_id, ts,
          CASE WHEN count(DISTINCT value)>1 THEN '__CONFLICT__'
               WHEN count(DISTINCT value)=0 THEN '__MISSING__'
               WHEN bool_and(numeric_value IS NOT NULL) THEN '__NUMERIC__'
               ELSE min(value) END || '|alarm=' || CAST(bool_or(alarm) AS VARCHAR) AS state
        FROM source WHERE ts IS NOT NULL AND channel_id IS NOT NULL AND alarm IS NOT NULL
        GROUP BY channel_id,ts""")
    c.execute("""CREATE TABLE changes AS WITH ordered AS (
        SELECT *, lag(state) OVER(PARTITION BY channel_id ORDER BY ts) AS previous
        FROM (SELECT *,false AS seed_row FROM canonical
              UNION ALL SELECT *,true AS seed_row FROM seed))
        SELECT channel_id,ts,state,coalesce(previous,'__INITIAL_OBSERVATION__') AS previous
        FROM ordered WHERE NOT seed_row AND state IS DISTINCT FROM previous""")


def extract(year: int, folder: Path, previous: Path | None):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"transitions-{year}.parquet"
    seed_target = folder / f"last-state-{year}.parquet"
    manifest = folder / f"transitions-{year}.json"
    inputs = [
        PROCESSED / f"events-{year}.parquet",
        PROCESSED / "channels.parquet",
        PROCESSED / "objects.parquet",
        Path(__file__),
    ]
    if previous:
        inputs.append(previous)
    hashes = {str(p): sha256(p) for p in inputs}
    if manifest.exists():
        import json

        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["inputs"] != hashes or old["before"] != BEFORE:
            raise ValueError("Transition cache inputs changed")
        for path in (target, seed_target):
            if sha256(path) != old["outputs"][str(path)]:
                raise ValueError("Transition cache changed")
        return target, seed_target
    c = connection()
    c.read_parquet(str(inputs[0])).filter(f"ts < TIMESTAMP '{BEFORE}'").create_view("source")
    if previous:
        c.read_parquet(str(previous)).create_view("seed")
    else:
        c.execute(
            "CREATE VIEW seed AS SELECT NULL::BIGINT channel_id,NULL::TIMESTAMP ts,NULL::VARCHAR state WHERE false"
        )
    canonical_transitions(c)
    c.read_parquet(str(inputs[1])).create_view("channels")
    c.read_parquet(str(inputs[2])).create_view("objects")
    c.sql("""SELECT e.channel_id,e.ts,e.state,e.previous,ch.object_id,o.parent_id
        FROM changes e JOIN channels ch USING(channel_id) JOIN objects o USING(object_id)
        QUALIFY row_number() OVER(PARTITION BY ch.object_id,date_trunc('hour',e.ts)
                                  ORDER BY e.ts DESC,e.channel_id DESC)<=4
        ORDER BY e.ts,e.channel_id""").write_parquet(str(target), compression="zstd")
    # Preserve states of channels silent all year as well as recent readings.
    c.sql("""SELECT channel_id,ts,state FROM
        (SELECT *,row_number() OVER(PARTITION BY channel_id ORDER BY ts DESC) n
         FROM (SELECT * FROM canonical UNION ALL SELECT * FROM seed)) WHERE n=1""").write_parquet(
        str(seed_target), compression="zstd"
    )
    rows = c.execute("SELECT count(*) FROM changes").fetchone()[0]
    stored = c.read_parquet(str(target)).count("*").fetchone()[0]
    c.close()
    write_json(
        manifest,
        {
            "inputs": hashes,
            "before": BEFORE,
            "rows": rows,
            "stored_rows": stored,
            "pruning": "Last4 changes per object per clock hour preserve last4 own/sibling changes at whole-hour forecasts only.",
            "outputs": {str(p): sha256(p) for p in (target, seed_target)},
        },
    )
    print("Extracted ordered transitions", year, rows, flush=True)
    return target, seed_target


def transform(frame: pd.DataFrame, history: pd.DataFrame):
    """At most four strictly past changes of own and other sibling channels.

    Simultaneous changes have a deterministic channel-ID order; it does not
    imply a physically observed order within the source's one-second precision.
    Unknown parents are never combined into an artificial shared network.
    """
    frame = frame.reset_index(drop=True)
    history = history.sort_values(["ts", "channel_id"], kind="stable").reset_index(drop=True)
    if not frame.as_of.eq(frame.as_of.dt.floor("h")).all():
        raise ValueError("Pruned transition history requires whole-hour forecasts")
    # Catalog IDs are integers; original model tables store parent IDs as text.
    frame = frame.assign(parent_id=pd.to_numeric(frame.parent_id, errors="coerce"))
    history = history.assign(parent_id=pd.to_numeric(history.parent_id, errors="coerce"))
    if history.duplicated(["channel_id", "ts"]).any():
        raise ValueError("Canonical channel timestamps must be unique")
    if frame.groupby("object_id").parent_id.nunique(dropna=False).gt(1).any():
        raise ValueError("An object has inconsistent parents")
    result = {name: np.full(len(frame), "__NONE__", dtype=object) for name in CATS}
    result.update({name: np.full(len(frame), MAX_AGE_HOURS + 1, dtype=np.float32) for name in AGES})
    # Factorization shares immutable strings instead of allocating millions of
    # duplicate Python strings. Output columns use categorical Parquet encoding.
    channel = history.channel_id.astype(str).to_numpy()
    transition = (history.previous + " -> " + history.state).to_numpy()
    ts = history.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for obj, rows in frame.groupby("object_id", sort=False):
        parent = rows.parent_id.iloc[0]
        sets = {"own": history.object_id.eq(obj).to_numpy()}
        sets["sibling"] = (
            (history.parent_id.eq(parent) & history.object_id.ne(obj)).to_numpy()
            if pd.notna(parent)
            else np.zeros(len(history), dtype=bool)
        )
        at = rows.as_of.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        for scope, selected in sets.items():
            ids = np.flatnonzero(selected)
            times = ts[ids]
            count = np.searchsorted(times, at, side="left")
            for lag in range(1, LAGS + 1):
                positions = count - lag
                valid = positions >= 0
                take = ids[positions[valid]]
                age = (at[valid] - ts[take]) / 3.6e12
                recent = age <= MAX_AGE_HOURS
                take = take[recent]
                dest = rows.index.to_numpy()[valid][recent]
                result[f"order_{scope}_{lag}_channel"][dest] = channel[take]
                result[f"order_{scope}_{lag}_transition"][dest] = transition[take]
                result[f"order_{scope}_{lag}_age_h"][dest] = age[recent]
    out = pd.DataFrame(result)
    for name in CATS:
        out[name] = out[name].astype("category")
    return out


def build(folder=PROCESSED / "ordered-states"):
    paths, seed = [], None
    for year in range(2022, 2027):
        path, seed = extract(year, folder, seed)
        paths.append(path)
    history = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    source = PROCESSED / "features.parquet"
    original = pd.read_parquet(
        source, filters=[("as_of", ">=", pd.Timestamp("2022-01-01")), ("as_of", "<", pd.Timestamp(BEFORE))]
    )
    dense_source = PROCESSED / "features-dense-round4.parquet"
    dense = pd.read_parquet(
        dense_source, columns=original.columns.tolist(), filters=[("as_of", "<", pd.Timestamp(BEFORE))]
    )
    # Compute identical inputs only once across coarse and hourly opportunities.
    keys = ["object_id", "as_of"]
    unique = pd.concat([original[keys + ["parent_id"]], dense[keys + ["parent_id"]]])
    unique = unique.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    extra = pd.concat([unique[keys], transform(unique, history)], axis=1)
    targets = []
    for frame, name in ((original, "features-ordered.parquet"), (dense, "features-dense-ordered.parquet")):
        result = frame.merge(extra, on=keys, how="left", validate="one_to_one", sort=False)
        pd.testing.assert_frame_equal(result[frame.columns], frame.reset_index(drop=True))
        if result[CATS + AGES].isna().any().any():
            raise ValueError("Missing ordered features")
        target = PROCESSED / name
        result.to_parquet(target, index=False, compression="zstd")
        targets.append(target)
        print("Saved ordered features", target, len(result), flush=True)
    write_json(
        folder / "build.json",
        {
            "before": BEFORE,
            "categorical": CATS,
            "age_columns": AGES,
            "inputs": {str(p): sha256(p) for p in (source, dense_source, *paths, Path(__file__))},
            "outputs": {str(p): sha256(p) for p in targets},
            "original_columns_unchanged": True,
            "same_time_parity_by_construction": True,
            "history": "Strict ts<as_of; last4 own and last4 other-sibling channel state changes within30d. Cross-year state carry; first2022 observation explicitly marked. No episode information. Ties ordered by channel ID, not a claim of within-second temporal order.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=PROCESSED / "ordered-states")
    args = parser.parse_args()
    build(args.folder)
