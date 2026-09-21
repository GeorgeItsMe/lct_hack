"""Add predeclared Dec/Mar opportunities without modifying frozen v8 features."""

from pathlib import Path

import pandas as pd

from moscollector.device_features import transform as device_transform
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json
from moscollector.sequence_features import object_sequence_features

source = PROCESSED / "features-dense-rich.parquet"
digest = sha256(source)
old = pd.read_parquet(source)
frame = pd.read_parquet(PROCESSED / "features-hourly-research.parquet")
frame = frame[
    (frame.as_of.ge("2025-12-01") & frame.as_of.lt("2026-01-01"))
    | (frame.as_of.ge("2026-03-01") & frame.as_of.lt("2026-04-01"))
].reset_index(drop=True)
cutoff = pd.Timestamp("2026-06-01")
hourly = pd.concat(
    [
        pd.read_parquet(PROCESSED / f"hourly-{y}.parquet", filters=[("hour", "<", cutoff)])
        for y in (2025, 2026)
    ]
)
devices = pd.concat(
    [pd.read_parquet(PROCESSED / f"device-hourly-{y}-2026-06-01.parquet") for y in (2025, 2026)]
)
episodes = pd.read_parquet(PROCESSED / "episodes.parquet", filters=[("start_ts", "<", cutoff)])
parts = []
for obj, rows in frame.groupby("object_id", sort=True):
    at = pd.DatetimeIndex(rows.as_of)
    seq = object_sequence_features(at, hourly[hourly.object_id.eq(obj)], episodes[episodes.object_id.eq(obj)])
    dev = device_transform(at, devices[devices.object_id.eq(obj)])
    extra = pd.concat([seq, dev], axis=1)
    extra.index = rows.index
    parts.append(extra)
new = pd.concat([frame, pd.concat(parts).reindex(frame.index)], axis=1)
assert list(new.columns) == list(old.columns)
reference = pd.read_parquet(PROCESSED / "features-device.parquet")
keys = reference[["object_id", "as_of"]].merge(new[["object_id", "as_of"]], validate="one_to_one")
shared = (
    new.merge(keys, validate="one_to_one")[reference.columns]
    .sort_values(["object_id", "as_of"])
    .reset_index(drop=True)
)
expected = (
    reference.merge(keys, validate="one_to_one").sort_values(["object_id", "as_of"]).reset_index(drop=True)
)
pd.testing.assert_frame_equal(shared, expected)
combined = pd.concat([old, new], ignore_index=True).sort_values(["object_id", "as_of"]).reset_index(drop=True)
assert not combined.duplicated(["object_id", "as_of"]).any()
recovered = (
    combined.merge(old[["object_id", "as_of"]], validate="one_to_one")
    .sort_values(["object_id", "as_of"])
    .reset_index(drop=True)
)
pd.testing.assert_frame_equal(recovered, old.sort_values(["object_id", "as_of"]).reset_index(drop=True))
output = PROCESSED / "features-dense-round4.parquet"
combined.to_parquet(output, index=False, compression="zstd")
assert sha256(source) == digest
report = {
    "old_rows": len(old),
    "new_rows": len(new),
    "shared_new_rows": len(shared),
    "columns": len(new.columns),
    "new_3h_features_exact": True,
    "old_hourly_features_exact": True,
    "old_source_sha256": digest,
    "output_sha256": sha256(output),
    "periods_added": ["2025-12", "2026-03"],
    "old_source_unchanged": True,
}
write_json(Path("artifacts/round4_feature_parity.json"), report)
print(report, flush=True)
