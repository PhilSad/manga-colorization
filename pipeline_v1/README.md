# pipeline_v1 — panel-wise manga colorization

A full colorization pipeline for manga pages: instead of colorizing whole pages
(which makes models redraw linework), it detects the panels, colorizes **each
panel individually** with a reference atlas containing **only the characters
that appear in that panel**, then stitches the colorized panels back onto the
original page.

Pipeline stages (per page):

1. **Detect panels** — YOLO26n (`leoxs22/manga-panel-detector-yolo26n`, weights
   auto-downloaded to `models/`) → `1_panels/`. Zero-detection pages get a
   blank-ink check; sparse full-page art gets one synthetic full-page box
   (`provenance: full-page-fallback`), effectively blank pages are skipped
   (V1.1, task 0004).
2. **Extract in Japanese reading order** — right-to-left, top-to-bottom
   banding; crops `panel_0001.png …` + `panels.json` (boxes + order) +
   `overlay.png` debug image → `1_panels/<page>/`
3. **Detect characters per page** — OpenRouter `google/gemma-4-31b-it`.
   The default `panel-page` mode makes one call per panel and sends the full
   numbered page as context plus the target panel; missing/invalid/`uncertain`
   results get a cropped-panel fallback. `--detection-mode panel` keeps the V1
   panel-only behaviour; `--detection-mode page` makes one call per page;
   `--detection-mode panel-page` (V1.2) keeps one call per panel
   but sends the full page as global context plus the target panel, with the
   same cropped-panel fallback as page mode. An optional cached
   chapter cast shortlist (`--cast-key`)
   focuses the prompt; identity hints come from the shared character
   profiles (task 0002) → `2_characters/<page>/<panel>.json`
4. **Colorize panel by panel** — self-hosted **step-distilled** FLUX.2 Klein 9B
   + thedeoxen manga-colorization-by-reference LoRA (`mngclranm`, **4 steps**)
   on the DGX Spark server; the request is the panel (`#1`) + a labelled atlas
   of **only the detected characters** (`#2`) + an explicit canonical-palette
   instruction rendered from the character profiles (task 0002). Panels with
   no detected characters are colorized panel-only. Oversized inputs are
   scaled down to the megapixel cap (task 0004) → `3_colorized/<page>/`
5. **Stitch** — each colorized panel is resized back to its original box and
   pasted onto the page; everything outside the panels stays black & white →
   `4_stitched/<page>.png`

Each invocation creates a fresh `output/YYYYMMDD-HHMMSS/` run directory (never
overwritten) with the four numbered intermediate directories and an incremental
`manifest.json` (command, configuration, prompt/profile hashes, per-step
records, measured costs).

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
outputs; with `--from-step` only the earlier step outputs are copied, task
0001), `--atlas-columns N`, `--num-inference-steps` (4 for the
step-distilled model; 20–50 if the server runs the undistilled base),
`--lora-scale` (0.8–1.0), `--seed`, `--detection-mode page|panel|panel-page|panel-page-cast`
(panel-page = one call per panel with the full page as context,
`prompt_panel_page.txt`; panel-page-cast = same with an automatically derived
per-chapter cast shortlist from `chapter_page_map.json`, `--cast-key` wins),
`--cast-key c001` (chapter cast shortlist), `--no-full-page-fallback`,
`--max-megapixels 2.0` (FLUX request cap),
`--only-panel P003:panel_0006` (targeted rerun; repeatable),
`--force-characters P003:panel_0006=Frieren` (ground-truth identities, no
paid detection call; repeatable),
`--workers N` (parallel character-detection worker threads: pages are
processed concurrently, one page per thread; the per-panel progress bars and
the `--sleep` throttle are disabled when N > 1),
`--stitch-bw-fallback` (a panel whose colorized output is missing — e.g. a
FLUX call that errored — is stitched from its original black & white crop
instead of failing the stitch step; each fallback is logged to stderr and
recorded in the step record as `panels_bw_fallback` and in the manifest
`totals.panels_bw_fallback`).

## Output layout

```text
output/<YYYYMMDD-HHMMSS>/
├── 1_panels/<page>/        crops + panels.json + overlay.png
├── 2_characters/<page>/    one JSON per panel (characters, cost, latency)
├── 3_colorized/<page>/     colorized panels + per-panel atlas
├── 4_stitched/<page>.png   final page (panels colorized, rest B&W)
└── manifest.json
```

## Debug annotation of a run

`scripts/annotate_stitch.py` renders a debug copy of a completed run's
`4_stitched/` pages with a colored bounding box per panel and a label with the
panel name + the characters detected for it (from `2_characters/`); panels
that were stitched from their original B&W crop (`--stitch-bw-fallback`) get
an orange box and a `[B&W fallback]` tag (read from the run's `manifest.json`).
Output goes to `<run-dir>/5_debug/` with a per-page `summary.json`:

```bash
.venv/bin/python pipeline_v1/scripts/annotate_stitch.py \
    --run-dir pipeline_v1/output/20260809-125148
# options: --output-dir, --page SUBSTR (repeatable filter), --font-size,
#          --bbox-width
```

## Size policy

Each panel is colorized at the resolution **closest to its original size with
both axes multiples of 16** (FLUX VAE constraint), then resized back to the
exact panel box when stitching. Small panels therefore stay small — they may
colorize poorly (no upscaling). Since V1.1 (task 0004) oversized inputs
(`> --max-megapixels`, default 2.0 MP) are scaled down proportionally to the
cap (multiples of 16); the original/requested size, scale, and applied cap are
recorded per call.

## V1.1 evaluation (tasks 0001–0004)

- `evaluation/v1_1_cases.json` — the fixed failure set: character confusion
  (DET-001..004), out-of-vocabulary identity (OOV-001), palette adherence
  (COL-001..003), palette geography (COL-004, V1.2), zero-panel fallback
  (LAY-001), blank-page skip (LAY-002), oversized-input capping (SIZE-001).
  COL-004 is the colorization-step test for V1.2 problem 1 (ideas.md): p013
  panel_0002 with forced hero-party identities (Heiter, Himmel, Frieren,
  Eisen) must come out with distinct canonical palettes left to right
  (green/blue/white-pink/yellow) instead of the uniform blue wash seen in
  `output/20260809-091129/`.
- **Integration suite** (`pytest -m integration`, excluded from plain
  pytest) — the evaluation of `v1_1_cases.json`: stage-isolated, **no mocks,
  real API calls**, one timestamped output dir per session under
  `tests/output/YYYYMMDD-HHMMSS/` with a manifest of per-case records and
  measured `usage.cost`. Inputs are committed under `tests/data/`
  (pre-cropped panels + the one layout page; regenerate with
  `tests/prepare_integration_data.py`):

  - `test_integration_detection.py` (DET-001..004, OOV-001): the committed
    crop -> real OpenRouter `google/gemma-4-31b-it` panel detection ->
    assert the fixture's character set. Currently 1/5 pass — the four
    failures are the fixture's documented V1 baselines (flashback cast,
    multi-character collapse, missed Heiter, Clematis forced to Denken).
  - `test_integration_color.py` (COL-001..004, SIZE-001): the committed
    crop + `forced_characters` -> real FLUX.2 Klein 9B on Spark -> real
    `openai/gpt-5.6-luna` validation (palette adherence for COL-001..003,
    left-to-right for COL-004). Live run 2026-08-09: COL-003/COL-004 pass
    (the V1.2 palette-geography bug is gone at seed 1337), COL-001/002 fail
    (Frieren's hair renders lavender-purple, forbidden); SIZE-001 asserts
    the 1600x1248 request cap.
  - `test_integration_layout.py` (LAY-001..002): the committed page -> real
    YOLO26n, reusing the pipeline's blank-check/full-page fallback policy.

  The colorize tests need the Spark server up and `OPENROUTER_API_KEY`
  (detection + validation); missing prerequisites skip their tests with a
  printed reason. Measured suite cost ≈ $0.002–0.003 per full run.
- `character_profiles.json` + `profiles.py` — canonical names, identity cues
  (detection hints), palette descriptions (FLUX prompt conditioning),
  reference files, aliases, variants.
- `chapter_casts.json` — optional cached chapter cast shortlists
  (`--cast-key`); never fetched remotely.

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
.venv/bin/pytest pipeline_v1/tests -q            # offline unit tests (default)
.venv/bin/pytest pipeline_v1/tests -m integration   # real-network integration suite
```

Offline (no network, $0): unit tests per stage plus an end-to-end suite
running the whole pipeline with mock backends on a synthetic manga page,
including page-level character detection, targeted reruns (`--only-panel` /
`--resume --from-step`), forced ground-truth identities, full-page fallback,
blank-page skip, and the megapixel cap. Integration tests are marked
`integration` and excluded by default (`addopts = "-m 'not integration'"`).

**Integration suite** — no mocks, real API calls, one timestamped output
`tests/output/YYYYMMDD-HHMMSS/` per session (see the V1.1 evaluation section
above). Skipped whenever a prerequisite is missing (`OPENROUTER_API_KEY`,
Spark server), printed reason; plain `pytest` never touches the network.

The real smoke run (real YOLO + OpenRouter + Spark) is
`pipeline_v1/scripts/smoke_real.sh` and is **not** part of pytest.

## Reproducibility

The manifest records the command, configuration, prompt files, dependency
versions, source hashes, per-panel character records (with cost/latency) and
per-panel colorization records (sizes, seeds, timings, hashes), plus pricing
assumptions. Re-running with the same inputs creates a new timestamped run.

## Quality and failure cases

First real evaluation (see [`pipelines.md`](../pipelines.md) for the full review
with annotated debug views):

- **Works:** panel detection + Japanese reading order (verified by eye); per-panel
  character lists match the story; per-panel colorization is coherent and fast
  (3.8–13.5 s per panel, 71.7 s for a 2896×2256 spread); stitching is exact and
  the page outside the panels stays untouched.
- **Known limits:** characters **not in `data/refs/`** can never be detected;
  the wiki reference images are colorless so the atlas carries no canonical
  colors; colors are not harmonized across panels of the same page; a correctly
  detected character is sometimes given wrong colors anyway.
- **Edge cases seen:** a full-page illustration page is detected as 0 panels and
  stays black & white; a double-page spread is detected as one large panel.

### V1.1 (epic 002, tasks 0001–0004; runs 2026-08-09, seed 1337)

- **Fixed failure set:** [`evaluation/v1_1_cases.json`](evaluation/v1_1_cases.json)
  + the real-network integration suite (`pytest -m integration`) — no mocks,
  one timestamped run per session in `tests/output/`; the retired
  `evaluate.py` CLI (auto-scored detection + human `color_review.md`) was
  replaced by it.
- **Detection (page-level):** precision 1.0 / recall 0.78 on the fixed set
  (V1: 0.17 / 0.11); Heiter/Sein confusion fixed; one hero-party member still
  missed in the flashback panels; OOV-001 (Clematis) still fails.
- **Color:** explicit canonical palettes reach the FLUX prompt; COL-001..003
  outputs and reports are in the fixed-experiment run — verdicts pending user
  review (objective signals show no magenta-dominant color).
- **Full-page + caps:** p006 full-page fallback colorized (LAY-001 ✓); blank
  page skipped (LAY-002 ✓); spread capped to 1600×1248, 28.9 s vs 71.7 s
  (SIZE-001 ✓).
- **Cost:** 5-page comparison $0.00050690 (5 page calls) vs V1 $0.00137551
  (18 calls); ch134 smoke $0.00014994 (1 call) vs $0.00040965 (5 calls).
- **Runs:** [fixed COL experiment](output/20260809-005032/) ·
  [vol-1 5 pages](output/20260809-010458/) · [ch134 smoke](output/20260809-011110/) ·
  [full-roster detection](output/20260809-013917/) · [layout run](output/20260809-014132/) ·
  [spread cap A/B @3.5 MP](output/20260809-014331/). Full comparison in
  [`pipelines.md`](../pipelines.md).
