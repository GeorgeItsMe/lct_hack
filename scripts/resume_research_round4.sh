#!/usr/bin/env bash
# Run serially in a full research checkout; never changes the active bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/build_round4_features.py
.venv/bin/python -m moscollector.count_extension_research --stage screen
.venv/bin/python -m moscollector.count_extension_research --stage confirm
.venv/bin/python -m moscollector.count_extension_research --stage stress
.venv/bin/python scripts/summarize_round4.py
.venv/bin/python scripts/summarize_round4_uncertainty.py
