"""Package source and minimal runtime separately; no credentials or user sessions."""

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
RUNTIME = OUT / "runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)


def copy(relative):
    source = ROOT / relative
    target = RUNTIME / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


for name in ("channels.parquet", "objects.parquet", "events-2026.parquet", "hourly-2026.parquet"):
    copy(f"data/processed/{name}")
for name, column in [("features.parquet", "as_of"), ("episodes.parquet", "start_ts")]:
    frame = pd.read_parquet(
        ROOT / "data/processed" / name, filters=[(column, ">=", pd.Timestamp("2026-01-01"))]
    )
    frame.to_parquet(RUNTIME / "data/processed" / name, index=False, compression="zstd")
(RUNTIME / "artifacts/predictions").mkdir(parents=True, exist_ok=True)
pred = pd.read_parquet(
    ROOT / "artifacts/predictions/all.parquet", filters=[("as_of", ">=", pd.Timestamp("2026-01-01"))]
)
pred.to_parquet(RUNTIME / "artifacts/predictions/all.parquet", index=False, compression="zstd")
for file in (ROOT / "artifacts/models").glob("*"):
    if file.suffix in (".cbm", ".json"):
        copy(file.relative_to(ROOT))
for file in (ROOT / "artifacts/operational").rglob("*"):
    if file.is_file() and file.suffix in (".cbm", ".json"):
        copy(file.relative_to(ROOT))
for file in (ROOT / "artifacts/predictions").glob("test-matches-*.json"):
    copy(file.relative_to(ROOT))
for file in (ROOT / "artifacts").glob("*.json"):
    copy(file.relative_to(ROOT))
manifest = {
    str(file.relative_to(RUNTIME)): hashlib.sha256(file.read_bytes()).hexdigest()
    for file in sorted(RUNTIME.rglob("*"))
    if file.is_file() and file.name != "MANIFEST.sha256.json"
}
(RUNTIME / "MANIFEST.sha256.json").write_text(json.dumps(manifest, indent=2))
with tarfile.open(OUT / "contour-runtime.tar.gz", "w:gz", compresslevel=2) as archive:
    for file in sorted(RUNTIME.rglob("*")):
        if file.is_file():
            archive.add(file, arcname=str(file.relative_to(RUNTIME)))
files = []
for directory in ("src", "tests", "scripts", "docs", "examples", "deploy", "web/src", "web/public"):
    files.extend(
        p
        for p in (ROOT / directory).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and "deploy/certs/" not in str(p.relative_to(ROOT))
        and not any(part.endswith(".egg-info") for part in p.parts)
    )
for name in (
    "README.md",
    "TASK_ANALYSIS.md",
    "pyproject.toml",
    "requirements.lock",
    "requirements-neural.txt",
    "requirements-neural.lock",
    "Dockerfile",
    "compose.yaml",
    ".dockerignore",
    ".gitignore",
    ".env.example",
    ".vercelignore",
    "api/index.py",
    "vercel.json",
    "web/package.json",
    "web/package-lock.json",
    "web/tsconfig.json",
    "web/vite.config.ts",
    "web/index.html",
):
    files.append(ROOT / name)
files.extend((ROOT / "artifacts").glob("*.json"))
with zipfile.ZipFile(OUT / "contour-source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(set(files)):
        archive.write(file, file.relative_to(ROOT))
checksums = []
for name in (
    "contour-source.zip",
    "contour-runtime.tar.gz",
    "Контур_техническая_документация.docx",
    "Контур_техническая_документация.pdf",
):
    file = OUT / name
    checksums.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {name}")
(OUT / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
print(
    "\n".join(
        f"{name}: {(OUT / name).stat().st_size / 1024 / 1024:.1f} MiB"
        for name in ("contour-source.zip", "contour-runtime.tar.gz")
    )
)
