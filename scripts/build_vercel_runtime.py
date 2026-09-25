"""Build the small, Git-friendly analytical bundle used by Vercel.

The Vercel demo exposes replay from May 2026 onward. Only the preceding day of
raw events is needed for the detail cards, so the large January-April event
archive is intentionally omitted. Training data and the frozen June evaluation
are never recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "vercel_runtime"
EVENT_BEGIN = pd.Timestamp("2026-04-30")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_filtered(source: Path, target: Path, column: str, begin: pd.Timestamp):
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source, filters=[(column, ">=", begin)])
    frame.to_parquet(target, index=False, compression="zstd")
    return len(frame)


def build(target: Path = DEFAULT_TARGET):
    target = target.resolve()
    if target == ROOT or ROOT not in target.parents:
        raise ValueError("Vercel runtime target must be a child of the project root")
    if target.exists():
        shutil.rmtree(target)

    processed = ROOT / "data" / "processed"
    artifacts = ROOT / "artifacts"
    target_processed = target / "data" / "processed"
    target_artifacts = target / "artifacts"

    for name in ("channels.parquet", "objects.parquet", "hourly-2026.parquet"):
        copy(processed / name, target_processed / name)

    rows = {
        "events_2026": write_filtered(
            processed / "events-2026.parquet",
            target_processed / "events-2026.parquet",
            "ts",
            EVENT_BEGIN,
        ),
        "features": write_filtered(
            processed / "features.parquet",
            target_processed / "features.parquet",
            "as_of",
            pd.Timestamp("2026-01-01"),
        ),
        "episodes": write_filtered(
            processed / "episodes.parquet",
            target_processed / "episodes.parquet",
            "start_ts",
            pd.Timestamp("2026-01-01"),
        ),
        "predictions": write_filtered(
            artifacts / "predictions" / "all.parquet",
            target_artifacts / "predictions" / "all.parquet",
            "as_of",
            pd.Timestamp("2026-01-01"),
        ),
    }

    for folder in ("models",):
        for source in sorted((artifacts / folder).glob("*")):
            if source.suffix in {".cbm", ".json"}:
                copy(source, target_artifacts / source.relative_to(artifacts))

    active = json.loads((artifacts / "active_model.json").read_text())
    operational = artifacts / "operational" / active["version"]
    for source in sorted(operational.glob("*")):
        if source.suffix in {".cbm", ".json", ".parquet"}:
            copy(source, target_artifacts / source.relative_to(artifacts))

    required_json = {
        "active_model.json",
        "catalog_audit.json",
        "evaluation_report.json",
        "feature_audit.json",
        "operational_quality.json",
        "research_report.json",
        "uncertainty_report.json",
    }
    required_json.update(f"audit-{year}.json" for year in range(2019, 2027))
    required_json.update(f"episode-audit-{year}.json" for year in range(2022, 2027))
    for name in sorted(required_json):
        source = artifacts / name
        if source.exists():
            copy(source, target_artifacts / name)

    for source in sorted((artifacts / "predictions").glob("test-matches-*.json")):
        copy(source, target_artifacts / "predictions" / source.name)

    files = sorted(path for path in target.rglob("*") if path.is_file())
    manifest = {
        "scope": "vercel_archive_demo_runtime",
        "raw_event_begin": EVENT_BEGIN.isoformat(),
        "active_model": active["version"],
        "rows": rows,
        "files": {str(path.relative_to(target)): sha256(path) for path in files},
    }
    (target / "MANIFEST.sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    print(f"Vercel runtime: {len(files) + 1} files, {size / 1024 / 1024:.1f} MiB")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    build(args.target)
