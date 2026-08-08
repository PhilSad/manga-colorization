#!/usr/bin/env bash
# Download the manga-colorization-by-reference LoRA for FLUX.2 Klein 9B.
#
#   https://huggingface.co/thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA
#
# The repo is public (Apache-2.0), so no HF token is needed. The weights land
# in ./models/FLUX.2-klein-lora/ next to the script and are mounted into the
# container read-only (FLUX2_LORA_PATH), like the model weights.
#
# Trained on the undistilled black-forest-labs/FLUX.2-klein-base-9B (which IS
# gated: accept the FLUX Non-Commercial License first). Rank-32, transformer
# only, trigger word `mngclranm`, recommended LoRA weight 0.8-1.0.
#
# Usage:
#   ./download_lora.sh [target_dir]
#   (default target: ./models/FLUX.2-klein-lora next to this script)

set -euo pipefail

LORA_REPO="thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/models/FLUX.2-klein-lora}"

mkdir -p "$TARGET"
echo "Downloading $LORA_REPO -> $TARGET"
curl -fL --retry 3 -o "$TARGET/manga_colorization.safetensors" \
  "https://huggingface.co/$LORA_REPO/resolve/main/manga_colorization.safetensors"
curl -fL --retry 3 -o "$TARGET/README.md" \
  "https://huggingface.co/$LORA_REPO/raw/main/README.md"

echo "Done:"
ls -la "$TARGET"
