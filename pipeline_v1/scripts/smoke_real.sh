#!/usr/bin/env bash
# Real smoke run for pipeline_v1: real YOLO detection + real OpenRouter
# character detection (paid, user-funded) + real Spark FLUX.2 Klein 9B + LoRA
# colorization, on ONE page (chapter page 4, i.e. --skip-first 3 --limit 1).
#
# Prerequisites:
#   - Spark inference server running: curl http://spark:3000/healthz
#   - OPENROUTER_API_KEY in the repo .env
#   - .venv with pipeline_v1 requirements installed
#
# Expect: first FLUX request pays the model-load cost (~1-3 min); each panel
# colorization at 20 steps takes a few minutes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --endpoint http://spark:3000 \
  --skip-first 3 \
  --limit 1 \
  "$@"
