# Real-ESRGAN anime upscale

Post-processing method: upscales colorized manga pages with the Real-ESRGAN anime-tuned model (`RealESRGAN_x4plus_anime_6B`) using the portable `realesrgan-ncnn-vulkan` executable, following the official [anime model doc](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_model.md).

This is not a colorization method itself; it is used to recover resolution from low-output colorizers (e.g. Gemini at `--image-size 512`, which emits 416×624 pages). Inputs are never modified; every invocation writes a fresh timestamped `output/YYYYMMDD-HHMMSS/` directory with the upscaled pages and a `manifest.json`.

## Setup

The binary is not vendored (≈47 MB). Download the portable Linux release and unpack it once:

```bash
cd /tmp
curl -L -o realesrgan.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan.zip -d realesrgan
chmod +x realesrgan/realesrgan-ncnn-vulkan
```

The archive bundles the ncnn models (`models/realesrgan-x4plus-anime.param/.bin`). Requires a Vulkan-capable GPU or llvmpipe fallback; no Python/torch dependency.

## Usage

```bash
colorization_methods/realesrgan-anime-upscale/run.sh \
  --input-dir colorization_methods/gemini-reference-cache-sequential/output/<run> \
  --bin /tmp/realesrgan/realesrgan-ncnn-vulkan
```

Options: `--model` (default `realesrgan-x4plus-anime`), `--scale` (default 4), `--output-root`. Alternatively set `REALESRGAN_NCNN_BIN`.

## Representative run

[`output/20260808-171010/`](output/20260808-171010/): the 5-page `gemini-3.1-flash-image --image-size 512` run (`gemini-reference-cache-sequential/output/20260808-170138/`), 416×624 → **1664×2496** (4×), plus the reference atlas. 6 images in ~64 s on an Intel Arc Pro 130T GPU. Zero per-image API cost (local inference; electricity only).

## Quality and known failure cases

- Real-ESRGAN anime is strong on line cleanup and sharpening; it can slightly soften very fine screentone/hatching and may oversharpen flat color areas.
- It cannot invent detail that the 416×624 source lacks — small text and thin linework recover only partially, so the upscale is bounded by the colorizer's output.
- Output is PNG (lossless) regardless of input format.
- Each page is processed independently; no temporal consistency handling for sequences.
