"""Archive v1 research features and repair mixed-censoring state summaries.

Never alters original features.parquet, labels, operational weights or June report.
The old models using the flawed feature are marked invalid for promotion.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from moscollector.experiments.sequence_features import recurrence_features
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

report_path = Path("artifacts/censoring_repair.json")
if report_path.exists():
    raise SystemExit("Repair already recorded; refusing to overwrite provenance")
episodes = pd.read_parquet(
    PROCESSED / "episodes.parquet", filters=[("start_ts", "<", pd.Timestamp("2026-06-01"))]
)
sequence_path = PROCESSED / "features-sequence.parquet"
seq = pd.read_parquet(sequence_path)
columns = [c for c in seq if c.endswith("_active_confirmed")]
original = seq[columns].copy()
untouched = sha256(PROCESSED / "features.parquet")
for obj, rows in seq.groupby("object_id", sort=True):
    values = recurrence_features(pd.DatetimeIndex(rows.as_of), episodes[episodes.object_id.eq(obj)])
    seq.loc[rows.index, columns] = values[columns].to_numpy()
changed = {c: int((~np.isclose(original[c], seq[c], equal_nan=True)).sum()) for c in columns}
repair = {
    "created_at": datetime.now(UTC).isoformat(),
    "changed_rows": changed,
    "reason": "A grouped episode with at least one unobserved member end has no observed group completion, even if other members ended.",
    "mixed_groups": int((episodes.right_censored & episodes.end_ts.notna()).sum()),
    "unchanged_original_features_sha256": untouched,
    "files": {},
}
for stem in ("features-sequence", "features-device"):
    path = PROCESSED / f"{stem}.parquet"
    before_sha = sha256(path)
    frame = seq if stem == "features-sequence" else pd.read_parquet(path)
    if not frame[["object_id", "as_of"]].equals(seq[["object_id", "as_of"]]):
        raise ValueError("Device and sequence rows are not identical")
    frame[columns] = seq[columns]
    tmp = path.with_name(stem + ".repaired.parquet")
    frame.to_parquet(tmp, index=False, compression="zstd")
    archived = path.with_name(stem + "-invalid-v1.parquet")
    path.replace(archived)
    tmp.replace(path)
    metadata = path.with_suffix(".json")
    previous_metadata = json.loads(metadata.read_text())
    metadata.replace(metadata.with_name(stem + "-invalid-v1.json"))
    write_json(
        metadata,
        {
            **previous_metadata,
            "source_sha256": (
                sha256(sequence_path) if stem == "features-device" else previous_metadata["source_sha256"]
            ),
            "sha256": sha256(path),
            "censoring_version": 2,
            "repair_report": str(report_path),
        },
    )
    repair["files"][stem] = {
        "before_sha256": before_sha,
        "after_sha256": sha256(path),
        "archived_source": str(archived),
    }
    if stem == "features-device":
        del frame
for folder in ("artifacts/research-v6", "artifacts/research-v7"):
    write_json(
        Path(folder) / "INVALIDATED.json",
        {
            "reason": repair["reason"],
            "repair_report": str(report_path),
            "exception": "v6 recent_reference inputs do not include any changed column; these fits may be reused after verifying input equality.",
            "eligible_for_promotion": False,
        },
    )
# Verify unchanged model inputs on the ENTIRE table, not a sample.
old = pd.read_parquet(PROCESSED / "features-sequence-invalid-v1.parquet")
base_columns = [c for c in seq if not c.startswith("seq_")]
pd.testing.assert_frame_equal(seq[base_columns], old[base_columns])
repair["reference_all_input_rows_identical"] = True
repair["reused_reference_folders"] = []
for old_folder in Path("artifacts/research-v6").glob("*/recent_reference"):
    destination = Path("artifacts/research-v6b") / old_folder.parent.name / "recent_reference"
    if destination.exists():
        raise ValueError("Destination already exists")
    shutil.copytree(old_folder, destination)
    repair["reused_reference_folders"].append(str(destination))
write_json(report_path, repair)
assert sha256(PROCESSED / "features.parquet") == untouched
print(json.dumps(repair, ensure_ascii=False, indent=2))
