#!/usr/bin/env bash
# Creates NEW artifacts; never overwrites the submitted models or opens June.
set -euo pipefail
cd "$(dirname "$0")/.."
research_dir=${1:-artifacts/reproduction}
if [ -e "$research_dir" ]; then echo 'Use a new output directory.' >&2; exit 2; fi
mkdir -p "$research_dir/v1" "$research_dir/v2" "$research_dir/v3"
CONTOUR_ARTIFACT_DIR="$research_dir/v1" .venv/bin/python -m moscollector.train --feature-set base
CONTOUR_ARTIFACT_DIR="$research_dir/v2" .venv/bin/python -m moscollector.train --feature-set extended
CONTOUR_ARTIFACT_DIR="$research_dir/v3" .venv/bin/python -m moscollector.train --feature-set extended --train-since 2025-01-01
