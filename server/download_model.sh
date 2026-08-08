#!/usr/bin/env bash
# Download the gated FLUX.2 Klein 9B weights to ./models/FLUX.2-klein-9B.
# This is the "external module": a host directory mounted into the container at
# /models/flux2-klein, so the model never has to live inside the docker image.
#
# Prerequisites:
#   1. Accept the FLUX Non-Commercial License on the model page:
#      https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
#   2. Create a read token (hf.co/settings/tokens) and export it:
#      export HF_TOKEN=hf_xxx
#
# Usage:
#   HF_TOKEN=hf_xxx ./download_model.sh [target_dir]
#   (default target: ./models/FLUX.2-klein-9B next to this script)

set -euo pipefail

MODEL_ID="${FLUX2_MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/models/FLUX.2-klein-9B}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN is required." >&2
  echo "  Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B" >&2
  echo "  then create a read token and run: HF_TOKEN=hf_xxx ./download_model.sh" >&2
  exit 1
fi

# Use a throwaway venv so the host python is never touched.
VENV_DIR="$(mktemp -d)/venv"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade "huggingface_hub[cli]"

echo "Downloading $MODEL_ID -> $TARGET"
"$VENV_DIR/bin/python" - "$MODEL_ID" "$TARGET" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

model_id, target = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=model_id, local_dir=target, token=os.environ["HF_TOKEN"])
print(f"Done: {path}")
PY
