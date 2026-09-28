#!/usr/bin/env bash
# Resume serially from completed checkpoints. Do not run beside another round-3 fit.
# Feature generation and causality checks are described in docs/research/RESEARCH_ROUND3.md.
# This script does not activate a model and never opens the frozen June test.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m moscollector.precision_research --stage screen
.venv/bin/python -m moscollector.experiments.device_research --stage screen
.venv/bin/python -m moscollector.cadence_research --stage screen
.venv/bin/python -m moscollector.count_research --stage screen
.venv/bin/python -m moscollector.precision_research --stage confirm
.venv/bin/python -m moscollector.experiments.device_research --stage confirm
.venv/bin/python -m moscollector.cadence_research --stage confirm
.venv/bin/python -m moscollector.count_research --stage confirm
.venv/bin/python scripts/research/summarize_round3.py
