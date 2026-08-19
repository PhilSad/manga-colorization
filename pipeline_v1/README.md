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
   The default `panel-page-prev2-cast` mode makes one call per panel, sends the
   full numbered page as context plus the target panel and the two preceding
   pages as story context, and restricts the prompt to the page's chapter
   cast shortlist; missing/invalid/`uncertain`
   results get a cropped-panel fallback. `--detection-mode panel` keeps the V1
   panel-only behaviour; `--detection-mode page` makes one call per page;
   `--detection-mode page-cast` is the same page-level call restricted to an
   automatically derived per-chapter cast shortlist (same resolution order as
   panel-page-cast: `--cast-key` wins, then per-page derivation);
   `--detection-mode panel-page` (V1.2) keeps one call per panel
   but sends the full page as global context plus the target panel, with the
   same cropped-panel fallback as page mode. `--detection-mode panel-page-prev2`
   additionally sends the two preceding pages in reading order as story
   context (fewer when they do not exist; blank pages are skipped), so the
   model can use recent story events to disambiguate identity — expect
   ~2–3× the panel-page prompt tokens per call. `panel-page-cast` and
   `panel-page-prev2-cast` focus the prompt with an automatically derived
   per-chapter cast shortlist (from `chapter_page_map.json`, `--cast-key`
   wins); identity hints come from the shared character
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
   `4_stitched/<page>.png`. A panel whose colorized output is missing (e.g. a
   failed FLUX call) is **always** stitched from its original B&W crop (no
   flag to opt out): every fallback is logged to stderr, recorded per page
   (`panels_bw_fallback`) and in `totals.panels_bw_fallback`
6. **Debug annotation** — a pure-image copy of each stitched page with the
   detected panel bounding boxes (from `1_panels/`) and a label per panel
   with the characters detected for it (from `2_characters/`); panels
   stitched from their original B&W crop (step 5's fallback) get an
   orange box and a `[B&W fallback]` tag → `5_debug/<page>.png`
7. **PDF export** — packs every stitched page (from `4_stitched/`, filename
   order = reading order) into one multi-page PDF with Pillow's native PDF
   writer (no extra dependency): `--pdf-name` (default `colorized.pdf`) and
   `--pdf-dpi` (default 72, page size in points = pixels × 72 / dpi) →
   `6_pdf/colorized.pdf` + `summary.json`
8. **Sanity check** — pure local compute (numpy + OpenCV, no backends):
   compares every colorized panel with its black & white original through
   structural thin-stroke line maps (`sanity.py`) and flags panels whose
   line art drifted below the fidelity threshold → `7_sanity/<page>.json` +
   `summary.json`, plus a side-by-side contact sheet `7_sanity/<page>_flagged.png`
   for every page that has flagged panels

Each invocation creates a fresh `output/YYYYMMDD-HHMMSS/` run directory (never
overwritten) with the seven numbered intermediate directories and an
incremental `manifest.json` (command, configuration, prompt/profile hashes,
per-step records, measured costs).

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
- Full-page mode (`--full-page`) uses OpenAI `gpt-image-2` instead: it needs
  `OPENAI_API_KEY` in `.env` (paid tier) and does **not** need the Spark server
  or the YOLO weights.

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

## Profiles

The CLI has grown a lot of flags, so named default profiles let one flag stand
in for a whole set of them. Profiles live in `pipeline_v1/cli_profiles.json`
(profile name → `description` + `args` mapping of flag → value, keys without
the leading dashes) and are applied with `--profile NAME`:

```bash
.venv/bin/python pipeline_v1/run.py --profile full-page \
  --input-dir data/chapter_134 --refs-dir data/refs --skip-first 3 --limit 5
```

Precedence: **explicit command-line flags always win** over profile values
(profile flags are injected first, so argparse's last-wins semantics apply;
`--only-panel`/`--force-characters` accumulate instead). A profile cannot be
"unset" per flag — to run without a profile default, pass the value you want
or drop `--profile`. Boolean profile flags are always-on (e.g.
`"full-page": true`); `false`/`null` values emit nothing. `--help` lists the
available profile names, and the applied profile name is recorded in each
run's `manifest.json` under `configuration.profile` so runs stay reproducible.

Current profiles:

| Profile | Expands to | Notes |
|---|---|---|
| `full-page` | `--full-page --atlas-source detected --detection-mode page-cast --vlm-model openai/gpt-5.6-luna --worker-detection 8 --worker-colorization 4 --verify-attempts 0` | Full-page gpt-image-2 colorization; per page, one Luna (OpenRouter) call restricted to the auto-derived per-chapter cast (`page-cast`) picks the atlas characters; 8 parallel detection workers, 4 parallel colorization workers (kept at 4 to stay under the gpt-image-2 org limit of ~5 input-images/min); no verification loop. Needs `OPENAI_API_KEY` (colorization) + `OPENROUTER_API_KEY` (detection). |

To add a profile, add an entry to `cli_profiles.json`; unknown profile names
and unknown/unknown-valued flags in a profile fail loudly at parse time.

### Full-page gpt-image-2 mode

`--full-page` skips panel extraction entirely: the whole page is colorized in
one OpenAI `gpt-image-2` call with a labelled reference atlas + palette
instruction, at the smallest output size that keeps the page's aspect ratio
(see Size policy below). The six pipeline stages still run and write the same
output layout (one synthetic `panel_0001` per page).

```bash
# One VLM call per page to pick the atlas characters (default):
.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 --refs-dir data/refs \
  --full-page --atlas-source detected --skip-first 3 --limit 5

# Same, with cast-limited Luna detection + 8 workers, via the profile:
.venv/bin/python pipeline_v1/run.py --profile full-page \
  --input-dir data/chapter_134 --refs-dir data/refs --skip-first 3 --limit 5

# Zero VLM calls: full chapter cast for the atlas (no OpenRouter key needed):
.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 --refs-dir data/refs \
  --full-page --atlas-source cast --skip-first 3 --limit 5
```

Full-page flags: `--full-page`, `--atlas-source detected|cast` (cast requires
`--full-page`; detected forces `--detection-mode page`), `--gpt-model`
(default `gpt-image-2`), `--gpt-image-prompt-file`, `--gpt-size WxH` (override
the computed minimal size; must satisfy the API constraints), `--gpt-atlas-scale`
(downscale the atlas before upload), `--openai-api-key-env`.

Quality is fixed at `medium` (no flag) — the research-v2 measured sweet spot
(672×1008 @ medium ≈ $0.0499/page). gpt-image-2 calls retry transient errors
with exponential backoff (up to 3 retries) and then fail loudly
(`ColorizeRecord(status="error")`).

Useful flags: `--profile NAME` (named defaults, see [Profiles](#profiles)),
`--skip-first N`, `--limit N`, `--steps panels,characters`,
`--from-step colorize`, `--resume <previous-run-dir>` (re-uses its step
outputs; with `--from-step` only the earlier step outputs are copied, task
0001), `--atlas-columns N`, `--num-inference-steps` (4 for the
step-distilled model; 20–50 if the server runs the undistilled base),
`--lora-scale` (0.8–1.0), `--seed`, `--detection-mode page|page-cast|panel|panel-page|panel-page-cast|panel-page-prev2|panel-page-prev2-cast`
(page = one call per page with per-panel fallbacks, `prompt.txt`;
page-cast = page with the automatically derived per-chapter cast shortlist
from `chapter_page_map.json`, `--cast-key` wins;
panel-page = one call per panel with the full page as context,
`prompt_panel_page.txt`; panel-page-cast = same with an automatically derived
per-chapter cast shortlist from `chapter_page_map.json`, `--cast-key` wins;
panel-page-prev2 = panel-page that also sends the two preceding pages in
reading order as story context, `prompt_panel_page_prev2.txt`;
panel-page-prev2-cast = the prev2 variant with the automatically derived
per-chapter cast shortlist, same resolution as panel-page-cast),
`--cast-key c001` (chapter cast shortlist), `--no-full-page-fallback`,
`--max-megapixels 2.0` (FLUX request cap),
`--only-panel P003:panel_0006` (targeted rerun; repeatable),
`--force-characters P003:panel_0006=Frieren` (ground-truth identities, no
paid detection call; repeatable),
`--worker-detection N` (parallel character-detection worker threads: pages are
processed concurrently, one page per thread; the per-panel progress bars and
the `--sleep` throttle are disabled when N > 1),
`--worker-colorization N` (parallel colorization worker threads over pages —
in full-page mode this parallelizes the paid gpt-image-2 calls directly; the
per-panel progress bars are replaced by a single page bar when N > 1),
`--debug-font-size` / `--debug-bbox-width` (5_debug label font size and
bounding-box stroke width),
`--pdf-name` / `--pdf-dpi` (6_pdf PDF filename, default `colorized.pdf`;
embedding resolution, default 72 — page size in points = pixels × 72 / dpi),
`--sanity-threshold 0.45` / `--sanity-max-edge 1024` (7_sanity
line-fidelity flag threshold in (0, 1] and analysis-grid long-edge cap in
px — the B&W panel and its colorized output are resampled onto the same
grid before comparison),
`--verify-attempts N` (character-palette verification loop, 0 = off),
`--verify-model <openrouter-model>` (default `openai/gpt-5.6-luna`),
`--verify-prompt-file <path>` (default `verify_color_prompt.txt`),
`--verify-mode fix-prompt|bbox` (retry backend, default `fix-prompt`),
`--verify-bbox-prompt-file <path>` (bbox verdict prompt, default
`verify_bbox_prompt.txt`),
`--verify-reasoning-effort <effort>` (bbox mode only, default `high`),
`--region-edit-prompt-file <path>` (bbox edit template, default
`gpt_region_edit_prompt.txt`),
`--region-edit-model <model>` (bbox editor model, default reuses
`--gpt-model`).

### Character-palette verification loop (`--verify-attempts`)

After colorization, each panel is checked by a vision-language verifier
(Luna, `openai/gpt-5.6-luna` on OpenRouter) against the same inputs the
colorizer saw — colorized panel, original B&W crop, and the labelled
character atlas — using strict structured output (`json_schema` +
`provider.require_parameters`):

```json
{ "analyse": "...", "good_color": true, "fix_prompt": "" }
```

- `--verify-attempts 1` — check only: every panel is verified, and any
  mismatch is written to `<panel>.fix_prompt.txt` next to the colorized
  panel; no re-colorization.
- `--verify-attempts N` (N ≥ 2) — auto-retry: on a mismatch, the panel is
  re-colorized with the verifier's `fix_prompt` appended as an authoritative
  block to the palette instruction, up to N−1 retries. The first verified
  attempt (or the last attempt if attempts are exhausted) becomes the
  canonical `<panel>.png`; every attempt that is superseded is kept as
  `<panel>.attempt_<n>.png` — including the original as `attempt_1.png`
  when a retry wins, so the initial bad colorization is never lost.
- A verifier error (non-JSON response, failed call, `good_color: null`)
  stops the loop for that panel without burning a retry; the panel keeps its
  latest colorization and is counted under `verifier_error_panels`.
- A colorization error stops the loop before any verify call.

Everything is recorded: per panel, `3_colorized/<page>/<panel>.verify.json`
holds every attempt (colorize + verify records, verdicts, latencies,
measured `usage.cost`), and the manifest `totals` gain `verify_calls`,
`successful_verify_calls`, `verified_panels`, `mismatch_panels`,
`verifier_error_panels`, `colorization_retries` and `verify_cost_usd`.
Usage records also carry `completion_tokens_details` (e.g. Luna's
`reasoning_tokens`) whenever the provider returns it, so reasoning spend is
visible in the per-call records.
`fix_prompt` (authoritative retry block) is written per panel only when
non-empty. Verify calls are paid OpenRouter calls (Luna pricing, measured
per call); retries are extra FLUX calls on the Spark server (still $0/call).

### Bbox-guided region edits (`--verify-mode bbox`)

Full-page mode only (config validation rejects it without `--full-page`):
the retry path of the verification loop switches from re-colorizing the
**whole page** with the fix prompt to a **region-scoped gpt-image-2 edit**.
The verifier's bbox verdict schema (`analyse` / `good_color` / `fix_prompt` /
`regions[]` — one Luna call per retry, `reasoning: {effort: "high"}`, 8192
output tokens: the probe proved 2048 truncates and wastes the call) also
emits the bounding boxes of the palette-wrong regions (normalized 0–1000
coordinates). On a mismatch:

1. `regions` non-empty → the rejected page is copied with the boxes drawn on
   it as `<panel>.attempt_<n>.boxed.png` (drawn at the resolution actually
   sent to gpt-image-2), and `GptImage2RegionEditor` (region_edit.py,
   `images.edit`, no mask — the boxes are the only locator) recolors only
   the boxed regions with the atlas + canonical palette; the next verify
   call sees the edited page, so a localization recall miss on one pass is
   simply boxed again on the next (the probed failure mode).
2. `regions` empty (localization recall miss) → fall back to the fix-prompt
   full re-colorization (`retry_backend: "fix-prompt"`); the loop always
   has a retry path.

Per attempt, `3_colorized/<page>/<panel>.verify.json` records
`retry_backend` (`"gpt-image-2-region-edit"` | `"fix-prompt"`), the
consumed `regions`, and for region edits the `boxed_image` file record, the
rendered `edit_prompt` and `edit_cost_usd`; the boxed image is kept on disk
next to the attempt images. The manifest `totals` gain `region_edit_calls`
and `region_edit_cost_usd`; `pricing_assumptions.region_edit` documents the
editor pricing.

```bash
# full-page + bbox-guided retries (up to 2 retries per page):
.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 --refs-dir data/refs \
  --full-page --verify-mode bbox --verify-attempts 3 --skip-first 3 --limit 5
```

Measured (probe `20260816-111714-gpt-edit-bbox`): a region edit costs
≈ **$0.04593** @ 672×1008 medium, vs ≈ $0.0499 for a fix-prompt full
re-colorize — slightly cheaper *and* targeted, at the price of possible
extra iterations when localization recall misses. Per bbox retry: 1 Luna
bbox call ≈ $0.00176 + 1 gpt-image-2 edit ≈ $0.046 ≈ **$0.048/retry**.

```bash
# verify only, output fix prompts, no re-colorization:
.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 --refs-dir data/refs \
  --endpoint http://spark:3000 --verify-attempts 1 --skip-first 3 --limit 5

# verify + up to 2 fix retries per panel:
.venv/bin/python pipeline_v1/run.py \
  --input-dir data/chapter_134 --refs-dir data/refs \
  --endpoint http://spark:3000 --verify-attempts 3 --skip-first 3 --limit 5
```

## Output layout

```text
output/<YYYYMMDD-HHMMSS>/
├── 1_panels/<page>/        crops + panels.json + overlay.png
├── 2_characters/<page>/    one JSON per panel (characters, cost, latency)
├── 3_colorized/<page>/     colorized panels + per-panel atlas + verify
│                           records (<panel>.verify.json, fix prompts,
│                           attempt_<n> images for superseded attempts
│                           when --verify-attempts ≥ 2, incl. attempt_1;
│                           bbox mode additionally keeps
│                           attempt_<n>.boxed.png region-edit sources)
├── 4_stitched/<page>.png   final page (panels colorized, rest B&W)
├── 5_debug/<page>.png      stitched page + bbox + character label per panel
├── 6_pdf/colorized.pdf     all stitched pages as one multi-page PDF
├── 7_sanity/<page>.json    line-fidelity metrics + verdict per panel
│   + summary.json, <page>_flagged.png contact sheets
└── manifest.json
```

## Debug annotation of a run

The final pipeline stage (`debug` → `5_debug/`) renders a debug copy of each
`4_stitched/` page with a colored bounding box per panel and a label with the
panel name + the characters detected for it (from `2_characters/`); panels
that were stitched from their original B&W crop (the stitch step's always-on
fallback) get an orange box and a `[B&W fallback]` tag (from the stitch step
record in the run's `manifest.json`). It runs automatically at the end of
every run and writes a per-page `summary.json`.

`scripts/annotate_stitch.py` is the standalone, offline companion of that
stage: it re-annotates any *completed* run (same `steps.debug.run_debug_step`
implementation) with custom options, without re-running the pipeline:

```bash
.venv/bin/python pipeline_v1/scripts/annotate_stitch.py \
    --run-dir pipeline_v1/output/20260809-125148
# options: --output-dir, --page SUBSTR (repeatable filter), --font-size,
#          --bbox-width
```

## PDF export of a run

The final pipeline stage (`pdf` → `6_pdf/`) packs every stitched page from
`4_stitched/` into a single multi-page PDF using Pillow's native PDF writer
(`save_all=True`) — **no extra dependency** (no reportlab/fpdf). Page order is
the `4_stitched/` filename order, i.e. the volume reading order; each PDF page
is the stitched PNG at its original pixel size, embedded at `--pdf-dpi`
(default 72, so 1 px = 1 pt; 144 gives a half-size page in points). The stage
runs automatically at the end of every run and writes a per-page `summary.json`
(next to `colorized.pdf`, or `--pdf-name`).

`scripts/make_pdf.py` is the standalone, offline companion of that stage: it
re-exports any *completed* run (same `steps.pdf.run_pdf_step` implementation)
with custom options, without re-running the pipeline:

```bash
.venv/bin/python pipeline_v1/scripts/make_pdf.py \
    --run-dir pipeline_v1/output/20260815-124816
# options: --output-dir, --page SUBSTR (repeatable filter), --name,
#          --dpi
```

## Line-art fidelity sanity check

The final pipeline stage (`sanity` → `7_sanity/`) answers "did the colorizer
preserve the panel's line art?" It compares every colorized panel with its
black & white original through structural thin-stroke line maps
(`pipeline_v1/sanity.py`), independently of the colorizer's color choices:

- both images are resampled onto the same analysis grid (long edge
  `--sanity-max-edge`, default 1024 px) and binarized into ink masks;
- the **B&W ink mask is skeletonized to thin lines** (Zhang–Suen) and the
  colorized ink is dilated; then four scores are computed — **line IoU**
  (ink overlap on the B&W lines), **chamfer distance** (avg distance from
  each B&W line pixel to the nearest colorized ink, or its reciprocal
  similarity), **component similarity** (large connected components of the
  colorized ink matching B&W line components), and **drift** (phase-correlation
  shift of the ink distributions, in px and as % of the panel diagonal);
- a composite **`line_fidelity`** in (0, 1] combines them, and a panel is
  **flagged** when fidelity < `--sanity-threshold` (default 0.45) **or** any
  hard rule trips: line IoU < 0.25, chamfer > 4.0 px, or drift > 3% of the
  panel diagonal.

Panels that were stitched from their original B&W crop (the stitch step's
always-on fallback) are skipped — there is no colorized output to compare.
The stage runs automatically at the end of every run, needs no backends or
API keys, and writes:

```text
7_sanity/<page>.json         per-panel metrics + verdict (provenance, box,
                             iou/chamfer/components/drift, fidelity, reasons)
7_sanity/<page>_flagged.png  side-by-side contact sheet of the flagged panels
7_sanity/summary.json        per-run totals: pages/panels checked + flagged,
                             plus the same flags aggregated in the manifest
                             totals (panels_sanity_checked/_flagged)
```

`scripts/check_sanity.py` is the standalone, offline companion of that stage:
it re-checks any *completed* run (same `steps.sanity.run_sanity_step`
implementation) with custom options, without re-running the pipeline; it
reuses the run's recorded `sanity_threshold`/`sanity_max_edge` unless
overridden:

```bash
.venv/bin/python pipeline_v1/scripts/check_sanity.py \
    --run-dir pipeline_v1/output/20260819-202719
# options: --output-dir, --threshold, --max-edge, --page SUBSTR
#          (repeatable filter)
```

Interpretation: the scores are strict about *geometry*. A colorizer that
redraws the art (e.g. panel-wise FLUX.2 edits) loses thin lines and scores
well below the threshold on every panel, while an edit that preserves the
original pixels (e.g. the full-page gpt-image-2 mode) passes with margin. A
flagged panel is not necessarily a *wrong* color — it means the colorized
line art differs measurably from the original and deserves a visual review
(the contact sheets are the fastest way to eyeball it).

### Luna line-art check (paid, optional second opinion)

`scripts/check_luna_sanity.py` is the **semantic** counterpart of the
structural check: it sends each colorized panel side by side with its B&W
original to OpenRouter `openai/gpt-5.6-luna` (strict structured output,
`luna_sanity_prompt.txt` + `luna_sanity.py`, one paid call per panel) and
asks one question — *does the colorized line art match the B&W original?*
Unlike the IoU/chamfer geometry, Luna judges strokes semantically
(missing/added/redrawn lines, hatching, screentones), so it agrees on
straight pixel-preserving edits and can still call a structurally "flagged"
redraw acceptable when every stroke survived in substance. Records mirror
`luna_sanity.py`: `line_art_matches` + `analyse`, `status`
(`match`/`mismatch`/`error`), `analysis_size`, `attempts` (raw retries on
malformed output, no downgrade), `cost_usd` (`usage.cost` measured),
`model_returned`.

```bash
.venv/bin/python pipeline_v1/scripts/check_luna_sanity.py \
    --run-dir pipeline_v1/output/20260819-202719 \
    --page p003 --workers 4
# options: --output-dir (default <run-dir>/7_sanity_luna), --max-edge
#          (default 1536 px), --model, --workers, --page SUBSTR (repeatable)
```

Writes per-page `<page>.json`, the provenance pairs
`inputs/<page>/<panel>.bw.png|colorized.png` (analysis-grid resized), a
`<page>_mismatch.png` contact sheet only for pages with mismatches, and
`summary.json` with totals + measured cost. Needs `OPENROUTER_API_KEY` in
`.env`. B&W-fallback panels are skipped. Measured 2026-08-21: ~$0.0015/panel
at 1536 px long edge; two ch.134 panel-wise FLUX panels that the structural
check flagged (line IoU ≈ 0.09) were both judged `match` by Luna — a useful
reminder that the two checks measure different things.

## Size policy

Each panel is colorized at the resolution **closest to its original size with
both axes multiples of 16** (FLUX VAE constraint), then resized back to the
exact panel box when stitching. Small panels therefore stay small — they may
colorize poorly (no upscaling). Since V1.1 (task 0004) oversized inputs
(`> --max-megapixels`, default 2.0 MP) are scaled down proportionally to the
cap (multiples of 16); the original/requested size, scale, and applied cap are
recorded per call.

**Full-page mode** uses `config.minimal_gpt_image_size(w, h)` instead: the
smallest output size that keeps the page's exact aspect ratio while satisfying
the gpt-image-2 API constraints (edges multiples of 16, area in
[655,360, 8,294,400] px, max edge 3840, ratio ≤ 3:1). Measured examples:
1500×2250 (2:3) → **672×1008**; a 3000×2250 spread (4:3) → **960×720**; a
300 dpi B5 scan (2480×3508) has no exact-ratio size within the caps and is
rejected loudly (`ValueError`) rather than distorted. `--gpt-size WxH` overrides
the computed size for comparison runs.

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
  (pre-cropped panels, committed pages + complete per-page panel sets for
  the detection pages, the one layout page; regenerate with
  `tests/prepare_integration_data.py`). Tests consume committed inputs and
  call the real function directly — no panel detection inside the
  detection/color tests:

  - `test_integration_detection.py` (DET-001..010, OOV-001): six
    parametrized tests, one per detection mode/variant (`panel`, `panel-page`,
    `panel-page-cast`, `panel-page-prev2`, `panel-page-prev2-cast`, `page`),
    each over all 11 cases
    -> real OpenRouter `google/gemma-4-31b-it` -> assert the fixture's
    character set. The page-context modes use the committed full page + its
    complete panel set; `panel-page-cast` and `panel-page-prev2-cast` add the
    chapter-cast shortlist (`fixture["cast_keys"]`); `panel-page-prev2` and
    `panel-page-prev2-cast` add two preceding-page context images that reuse
    the case's own committed page (committed inputs only, no fabricated
    pages). Known-failing cases fail loudly and stay tracked; per-mode live
    verdicts are in pipelines.md.
  - `test_integration_color.py` (COL-001..004, SIZE-001): one parametrized
    test over all five cases: committed crop + `forced_characters` -> real
    FLUX.2 Klein 9B on Spark -> real `openai/gpt-5.6-luna` validation via one
    generic strict structured-output palette verdict (`analyse`/`good_color`,
    json_schema + `provider.require_parameters`, no fixture expectations in
    the prompt; the verdict also gets the reference atlas of the forced
    characters as context); SIZE-001 asserts the 1600x1248 request cap (no
    VLM verification).
  - `test_integration_layout.py` (LAY-001..002 + crop-stability tripwire):
    the committed page -> real YOLO26n, reusing the pipeline's
    blank-check/full-page fallback policy. The tripwire re-extracts the
    committed detection pages and asserts the crops still match
    byte-for-byte, so the eval cases' panel references cannot silently go
    stale.
  - `test_end_to_end_integration.py` (E2E-P130): the FULL pipeline — real
    YOLO + real OpenRouter `panel-page-prev2-cast` detection (prev2 with the
    chapter-cast shortlist; the cast key comes from the fixture, per the
    test's `DETECTION_MODE`/`CAST_KEY` constants) + real FLUX on Spark +
    stitching — on one real page (volume-1 p130, the DET-005..010 page).
    Deliberately NOT stage-isolated: runs the same real backends `run.py`
    builds and asserts the wiring end to end (panel crops reproduce the
    committed fixture set byte-for-byte, every panel gets a character
    record with no call errors and a colorized output differing from its
    B&W crop, the stitch preserves the B&W gutters pixel-exactly).
    Detection identities are recorded for provenance but not asserted —
    DET-006..008 are known Flamme/Frieren failures owned by the
    stage-isolated detection suite. Needs the Spark server + API key
    (skips otherwise). ~6 OpenRouter calls + 6 FLUX calls (first pays the
    model-load cost).

  The colorize tests need the Spark server up and `OPENROUTER_API_KEY`
  (detection + validation); missing prerequisites skip their tests with a
  printed reason. Measured suite cost ≈ $0.005–0.01 per full run (44
  detection calls + 5 FLUX + 4 VLM verifications).
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
- **Full-page mode** (OpenAI `gpt-image-2`, standard tier): paid per call.
  Projected from research-v2 measurements: ≈ **$0.0499/page** at the minimal
  672×1008 size (1029 output tokens; input floor ≈ $0.019/page with the
  4-character atlas), i.e. ≈ **$9.32 for volume 1** (187 pages) — a
  projection, not yet measured through the pipeline; the manifest records
  `totals.gpt_image_calls` / `totals.gpt_image_cost_usd` from each call's
  parsed usage (image input $8/1M, text input $5/1M, image+text output
  $30/1M).
- **Verification loop** (OpenRouter `openai/gpt-5.6-luna`, paid tier, only
  with `--verify-attempts ≥ 1`): measured per verify call via `usage.cost`,
  recorded in each `3_colorized/<page>/<panel>.verify.json` and the manifest
  `totals.verify_cost_usd` (plus `verify_calls`, `verified_panels`,
  `mismatch_panels`, `colorization_retries`, …). **Measured live:**
  ≈ **$0.00102/call** (run `output/20260815-165721`, 6 calls $0.00613).
  Retries are extra colorization calls — on FLUX they are still $0/call, but
  in full-page mode each gpt-image-2 retry costs ≈ $0.05/page (measured:
  one retry $0.05074 in that run).
- **Region edits** (OpenAI `gpt-image-2`, bbox mode only): probe-measured
  ≈ **$0.04593** @ 672×1008 medium (2026-08-16); recorded per edit in the
  verify attempt docs (`edit_cost_usd`) and the manifest
  `totals.region_edit_calls` / `totals.region_edit_cost_usd`.

## Testing

```bash
.venv/bin/pytest pipeline_v1/tests -q            # offline unit tests (default)
.venv/bin/pytest pipeline_v1/tests -m integration   # real-network integration suite
```

Offline (no network, $0): unit tests per stage plus an end-to-end suite
running the whole pipeline with mock backends on a synthetic manga page,
including page-level character detection, targeted reruns (`--only-panel` /
`--resume --from-step`), forced ground-truth identities, full-page fallback,
blank-page skip, and the megapixel cap. `tests/test_verify_loop.py` covers the
verification loop end to end with mock backends (verified-on-first-attempt,
fix-prompt retry, attempts-exhausted, verifier-error and colorize-error
outcomes, per-panel `verify.json` / `fix_prompt.txt` / `attempt_<n>` images,
and the manifest verify totals). `tests/test_full_page.py` covers the
`--full-page` gpt-image-2 mode end to end with mock backends (detected and
cast atlas sources, parallel colorization). Integration tests are marked
`integration` and excluded by default (`addopts = "-m 'not integration'"`).

**Integration suite** — no mocks, real API calls, one timestamped output
`tests/output/YYYYMMDD-HHMMSS/` per session (see the V1.1 evaluation section
above). Skipped whenever a prerequisite is missing (`OPENROUTER_API_KEY`,
Spark server), printed reason; plain `pytest` never touches the network.

The real smoke run (real YOLO + OpenRouter + Spark) is
`pipeline_v1/scripts/smoke_real.sh` (still handy for a quick manual run on
any input folder); the same full-pipeline path on a fixed page is also in
pytest as `test_end_to_end_integration.py` (real backends, `-m integration`).

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
