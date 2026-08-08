# pipeline_v1 — panel-wise manga colorization

A full colorization pipeline for manga pages: instead of colorizing whole pages
(which makes models redraw linework), it detects the panels, colorizes **each
panel individually** with a reference atlas containing **only the characters
that appear in that panel**, then stitches the colorized panels back onto the
original page.

Pipeline stages (per page):

1. **Detect panels** — YOLO26n (`leoxs22/manga-panel-detector-yolo26n`, weights
   auto-downloaded to `models/`) → `1_panels/`
2. **Extract in Japanese reading order** — right-to-left, top-to-bottom
   banding; crops `panel_0001.png …` + `panels.json` (boxes + order) +
   `overlay.png` debug image → `1_panels/<page>/`
3. **Detect characters per panel** — OpenRouter `google/gemma-4-31b-it`
   (same prompt as the
   [OpenRouter VLM method](../research/character_detection_methods/character-detection-openrouter-vlm/))
   → `2_characters/<page>/<panel>.json`
4. **Colorize panel by panel** — self-hosted **step-distilled** FLUX.2 Klein 9B
   + thedeoxen manga-colorization-by-reference LoRA (`mngclranm`, **4 steps**)
   on the DGX Spark server (same server as the
   [LoRA method](../research/colorization_methods/flux-2-klein-9b-base-lora-edit-sequential/));
   the request is the panel (`#1`) + a labelled atlas of **only the detected
   characters** (`#2`); panels with no detected characters are colorized
   panel-only → `3_colorized/<page>/`
5. **Stitch** — each colorized panel is resized back to its original box and
   pasted onto the page; everything outside the panels stays black & white →
   `4_stitched/<page>.png`

Each invocation creates a fresh `output/YYYYMMDD-HHMMSS/` run directory (never
overwritten) with the four numbered intermediate directories and an incremental
`manifest.json` (command, configuration, per-step records, measured costs).

## Setup

```bash
.venv/bin/python -m pip install -r pipeline_v1/requirements.txt
```

- The YOLO weights (~15 MB) download automatically to `pipeline_v1/models/`
  on first use (Apache-2.0).
- The FLUX.2 Klein 9B (distilled) + LoRA server must be running
  (`curl http://spark:3000/healthz`); see [`server/README.md`](../server/README.md).
  The compose deployment serves the step-distilled checkpoint with the LoRA
  loaded at 4 steps (`FLUX2_STEPS=4`).
- `OPENROUTER_API_KEY` must be in the repo `.env` (paid tier).

## Usage

Real run — one page smoke (chapter page 4):

```bash
.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --endpoint http://spark:3000 \
  --skip-first 3 --limit 1
```

Full chapter:

```bash
.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 --refs-dir data/refs --endpoint http://spark:3000
```

Offline demo (mock backends, no API keys, no server):

```bash
.venv/bin/python pipeline_v1/run.py --mock --limit 1
```

Useful flags: `--skip-first N`, `--limit N`, `--steps panels,characters`,
`--from-step colorize`, `--resume <previous-run-dir>` (re-uses its step
outputs), `--atlas-columns N`, `--num-inference-steps` (4 for the
step-distilled model; 20–50 if the server runs the undistilled base),
`--lora-scale` (0.8–1.0), `--seed`.

## Output layout

```text
output/<YYYYMMDD-HHMMSS>/
├── 1_panels/<page>/        crops + panels.json + overlay.png
├── 2_characters/<page>/    one JSON per panel (characters, cost, latency)
├── 3_colorized/<page>/     colorized panels + per-panel atlas
├── 4_stitched/<page>.png   final page (panels colorized, rest B&W)
└── manifest.json
```

## Size policy

Each panel is colorized at the resolution **closest to its original size with
both axes multiples of 16** (FLUX VAE constraint), then resized back to the
exact panel box when stitching. Small panels therefore stay small — they may
colorize poorly (no upscaling). Pending a real-run assessment.

## Cost

- **Character detection** (OpenRouter `google/gemma-4-31b-it`, paid tier):
  measured per call via `usage.cost`, recorded in `2_characters/` and the
  manifest `totals.openrouter_cost_usd`. Reference: ~$0.00008/panel on
  `data/panels` (4 panels = $0.000333).
- **Colorization** (self-hosted FLUX on Spark): **$0 per call** (electricity
  only, ~350–400 W during inference). Step-distilled 9B + LoRA at 4 steps is
  fast (roughly the fal 4-step endpoint's timing); the undistilled base would
  be ~5× slower. Do not compare with paid API pricing.

## Testing

```bash
.venv/bin/pytest pipeline_v1/tests -q
```

Fully offline: unit tests per stage plus an end-to-end suite running the whole
pipeline with mock backends on a synthetic manga page. The real smoke test
(real YOLO + OpenRouter + Spark) is `pipeline_v1/scripts/smoke_real.sh` and is
**not** part of pytest.

## Reproducibility

The manifest records the command, configuration, prompt files, dependency
versions, source hashes, per-panel character records (with cost/latency) and
per-panel colorization records (sizes, seeds, timings, hashes), plus pricing
assumptions. Re-running with the same inputs creates a new timestamped run.

## Quality and failure cases

> To be filled from the first real smoke run (`scripts/smoke_real.sh`).
