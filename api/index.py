"""Vercel entrypoint.

Vercel Functions have an immutable project filesystem. Runtime state therefore
uses ``/tmp`` while analytical data is loaded from the committed slim bundle.
Persistent sessions and dispatcher decisions require ``DATABASE_URL``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The package lives in src/; do not rely on the builder installing the project itself.
sys.path.insert(0, str(ROOT / "src"))

if os.getenv("VERCEL"):
    bundle = ROOT / "vercel_runtime"
    os.environ.setdefault("CONTOUR_DATA_DIR", str(bundle / "data"))
    os.environ.setdefault("CONTOUR_ARTIFACT_DIR", str(bundle / "artifacts"))
    os.environ.setdefault("CONTOUR_RUNTIME_DIR", "/tmp/contour-runtime")
    os.environ.setdefault("CONTOUR_SERVERLESS", "true")

from moscollector.api import app  # noqa: E402,F401
