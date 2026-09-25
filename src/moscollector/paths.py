"""Project paths; production can mount data and artifacts outside the source tree."""

import os
from pathlib import Path

ROOT = Path(os.getenv("CONTOUR_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.getenv("CONTOUR_DATA_DIR", ROOT / "data"))
ARTIFACTS = Path(os.getenv("CONTOUR_ARTIFACT_DIR", ROOT / "artifacts"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNTIME = Path(os.getenv("CONTOUR_RUNTIME_DIR", DATA / "runtime"))
