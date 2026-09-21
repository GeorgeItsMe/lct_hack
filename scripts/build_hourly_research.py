"""Build a separate hourly evaluation table; verify parity with frozen features."""

import logging
from pathlib import Path

import pandas as pd

from moscollector.features import build_dataset
from moscollector.paths import PROCESSED
from moscollector.prepare import sha256, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
output = PROCESSED / "features-hourly-research.parquet"
originals = [
    PROCESSED / "features.parquet",
    PROCESSED / "episodes.parquet",
    Path("artifacts/evaluation_report.json"),
]
before_hashes = {str(p): sha256(p) for p in originals}
build_dataset(
    [2025, 2026],
    step_hours=1,
    output_path=output,
    episode_output_path=PROCESSED / "episodes-hourly-research.parquet",
    audit_output_path=Path("artifacts/hourly_feature_audit.json"),
    before="2026-06-01",
)
old = pd.read_parquet(
    PROCESSED / "features.parquet",
    filters=[("as_of", ">=", pd.Timestamp("2025-01-01")), ("as_of", "<", pd.Timestamp("2026-06-01"))],
)
new = pd.read_parquet(output)
# The final 25h intentionally lose label observability under the June cutoff.
cutoff = pd.Timestamp("2026-06-01") - pd.Timedelta(hours=25)
old = old[old.as_of.lt(cutoff)].sort_values(["as_of", "object_id"]).reset_index(drop=True)
shared = new.merge(old[["object_id", "as_of"]], on=["object_id", "as_of"], validate="one_to_one")
shared = shared[old.columns].sort_values(["as_of", "object_id"]).reset_index(drop=True)
pd.testing.assert_frame_equal(old, shared)
assert all(sha256(p) == before_hashes[str(p)] for p in originals)
write_json(
    Path("artifacts/hourly_feature_parity.json"),
    {
        "shared_rows": len(shared),
        "shared_columns": len(shared.columns),
        "exact_parity": True,
        "hourly_rows": len(new),
        "horizon_hours": 24,
        "step_hours": 1,
        "before": "2026-06-01",
        "source_files_unchanged": before_hashes,
        "output_sha256": sha256(output),
    },
)
print("Verified exact hourly/3h feature parity", len(shared), "rows", flush=True)
