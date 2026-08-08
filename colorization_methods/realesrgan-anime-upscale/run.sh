#!/usr/bin/env bash
# Upscale manga pages with the Real-ESRGAN anime model (realesrgan-ncnn-vulkan).
#
# Usage:
#   run.sh --input-dir DIR [--bin /path/to/realesrgan-ncnn-vulkan] [--model NAME]
#          [--scale N] [--output-root DIR]
#
# --input-dir   directory containing the pages to upscale (required)
# --bin         path to the realesrgan-ncnn-vulkan executable
#               (default: $REALESRGAN_NCNN_BIN or realesrgan-ncnn-vulkan on PATH)
# --model       ncnn model name, default realesrgan-x4plus-anime
# --scale       upscale factor, default 4
# --output-root parent for the fresh timestamped output directory
#               (default: <this method dir>/output)
#
# The binary is NOT vendored here; download the portable Linux release from
# https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0
# (realesrgan-ncnn-vulkan-20220424-ubuntu.zip) and point --bin at it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
BIN="${REALESRGAN_NCNN_BIN:-realesrgan-ncnn-vulkan}"
MODEL=realesrgan-x4plus-anime
SCALE=4
INPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --bin) BIN="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --scale) SCALE="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$INPUT_DIR" || ! -d "$INPUT_DIR" ]]; then
    echo "error: --input-dir must be an existing directory" >&2
    exit 2
fi
if ! command -v "$BIN" >/dev/null 2>&1 && [[ ! -x "$BIN" ]]; then
    echo "error: realesrgan-ncnn-vulkan not found at '$BIN'" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
RUN_DIR="$OUTPUT_ROOT/$(date +%Y%m%d-%H%M%S)"
while [[ -e "$RUN_DIR" ]]; do sleep 1; RUN_DIR="$OUTPUT_ROOT/$(date +%Y%m%d-%H%M%S)"; done
mkdir -p "$RUN_DIR"
echo "output directory: $RUN_DIR"

INPUTS=()
shopt -s nullglob
for f in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.png "$INPUT_DIR"/*.webp; do
    INPUTS+=("$f")
done
shopt -u nullglob
if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "error: no jpg/jpeg/png/webp images found in $INPUT_DIR" >&2
    exit 2
fi

BIN_PATH="$(command -v "$BIN" || echo "$BIN")"
BIN_VERSION="$("$BIN" -v 2>&1 | head -1 || true)"
MODEL_VERSION="$(grep -o 'name=[^ ]*' "$(dirname "$(command -v "$BIN" || echo "$BIN")")/models/$MODEL.param" 2>/dev/null | head -1 || echo "unknown")"
START_EPOCH="$(date +%s)"
PAGES_JSON="["
FIRST=1
N=${#INPUTS[@]}
I=0

for f in "${INPUTS[@]}"; do
    I=$((I + 1))
    base="$(basename "$f")"
    out="$RUN_DIR/${base%.*}.png"
    echo "[$I/$N] upscaling $base"
    "$BIN" -i "$f" -o "$out" -n "$MODEL" -s "$SCALE" >/dev/null
    if [[ $FIRST -eq 1 ]]; then FIRST=0; else PAGES_JSON+=","; fi
    PAGES_JSON+=$(cat <<EOF
    {
      "input": "$f",
      "input_bytes": $(stat -c%s "$f"),
      "output": "$out",
      "output_bytes": $(stat -c%s "$out")
    }
EOF
)
done
PAGES_JSON+="]"

ELAPSED=$(( $(date +%s) - START_EPOCH ))
cat > "$RUN_DIR/manifest.json" <<EOF
{
  "method": "realesrgan-anime-upscale",
  "status": "completed",
  "started_at": "$(date -Iseconds)",
  "run_directory": "$RUN_DIR",
  "binary": "$BIN_PATH",
  "binary_version": "$BIN_VERSION",
  "model": "$MODEL",
  "model_version": "$MODEL_VERSION",
  "scale": $SCALE,
  "input_dir": "$INPUT_DIR",
  "pages": $PAGES_JSON,
  "totals": {
    "pages_upscaled": ${#INPUTS[@]},
    "elapsed_seconds": $ELAPSED,
    "cost_usd": 0.0,
    "cost_notes": "Local inference on CPU/GPU; zero per-image API cost (electricity only)."
  }
}
EOF
echo "done: ${#INPUTS[@]} pages upscaled into $RUN_DIR (${ELAPSED}s)"
