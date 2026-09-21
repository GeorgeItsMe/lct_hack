"""Enrich hourly policy/test opportunities, with exact 3h feature parity checks."""

from pathlib import Path

import pandas as pd

from moscollector.device_features import transform as device_transform
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.sequence_features import object_sequence_features

cutoff = pd.Timestamp("2026-06-01")
frame = pd.read_parquet(PROCESSED / "features-hourly-research.parquet")
keep = (
    (frame.as_of.ge("2025-10-01") & frame.as_of.lt("2025-12-01"))
    | (frame.as_of.ge("2026-01-01") & frame.as_of.lt("2026-03-01"))
    | (frame.as_of.ge("2026-04-01") & frame.as_of.lt(cutoff))
)
frame = frame[keep].reset_index(drop=True)
hourly = pd.concat(
    [
        pd.read_parquet(PROCESSED / f"hourly-{y}.parquet", filters=[("hour", "<", cutoff)])
        for y in (2025, 2026)
    ]
)
devices = pd.concat(
    [pd.read_parquet(PROCESSED / f"device-hourly-{y}-2026-06-01.parquet") for y in (2025, 2026)]
)
assert devices.hour.max() < cutoff
episodes = pd.read_parquet(PROCESSED / "episodes.parquet", filters=[("start_ts", "<", cutoff)])
extra = []
for obj, rows in frame.groupby("object_id", sort=True):
    at = pd.DatetimeIndex(rows.as_of)
    sequence = object_sequence_features(
        at, hourly[hourly.object_id.eq(obj)], episodes[episodes.object_id.eq(obj)]
    )
    device = device_transform(at, devices[devices.object_id.eq(obj)])
    enriched = pd.concat([sequence, device], axis=1)
    enriched.index = rows.index
    extra.append(enriched)
    print(f"dense features object={obj}, rows={len(rows)}", flush=True)
frame = pd.concat([frame, pd.concat(extra).reindex(frame.index)], axis=1)
old = pd.read_parquet(PROCESSED / "features-device.parquet")
old = old[old.as_of.lt(cutoff - pd.Timedelta(hours=25))]
keys = old[["object_id", "as_of"]].merge(frame[["object_id", "as_of"]], validate="one_to_one")
old = old.merge(keys, validate="one_to_one").sort_values(["as_of", "object_id"]).reset_index(drop=True)
shared = (
    frame.merge(keys, validate="one_to_one")[old.columns]
    .sort_values(["as_of", "object_id"])
    .reset_index(drop=True)
)
pd.testing.assert_frame_equal(old, shared)
output = PROCESSED / "features-dense-rich.parquet"
frame.to_parquet(output, index=False, compression="zstd")
write_json(
    Path("artifacts/dense_rich_feature_parity.json"),
    {
        "rows": len(frame),
        "shared_rows": len(shared),
        "columns": len(frame.columns),
        "exact_parity": True,
        "sha256": sha256(output),
        "corrected_three_hour_features_sha256": sha256(PROCESSED / "features-device.parquet"),
        "before": "2026-06-01",
        "cadence_hours": 1,
        "horizon_hours": 24,
    },
)
print("Verified dense rich parity", len(shared), "rows", flush=True)
