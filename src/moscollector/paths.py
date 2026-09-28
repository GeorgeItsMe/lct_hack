"""Project paths; production can mount data and artifacts outside the source tree.

Without explicit settings the full prepared dataset (data/, artifacts/) is used when it is
present; a fresh clone falls back to the committed runtime bundle (vercel_runtime/), which
holds the frozen models and the 2026 archive slice the service needs to run.
"""

import os
from pathlib import Path

ROOT = Path(os.getenv("CONTOUR_ROOT", Path(__file__).resolve().parents[2]))
BUNDLE = ROOT / "vercel_runtime"


def _default(full: Path, marker: str, bundled: Path) -> Path:
    return full if (full / marker).exists() or not (bundled / marker).exists() else bundled


DATA = Path(
    os.getenv("CONTOUR_DATA_DIR") or _default(ROOT / "data", "processed/features.parquet", BUNDLE / "data")
)
ARTIFACTS = Path(
    os.getenv("CONTOUR_ARTIFACT_DIR")
    or _default(ROOT / "artifacts", "models/access.json", BUNDLE / "artifacts")
)
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
# Application state never goes into the read-only bundle.
RUNTIME = Path(os.getenv("CONTOUR_RUNTIME_DIR") or ROOT / "data" / "runtime")
