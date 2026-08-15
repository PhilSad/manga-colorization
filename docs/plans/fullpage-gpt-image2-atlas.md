# Plan: full-page gpt-image-2 atlas colorization in pipeline_v1

## Goal

Add the research-v2 `gpt-image-2` atlas colorization method to `pipeline_v1` as
a **full-page** pipeline mode:

1. **No panel extraction** — the whole page is colorized in a single call
   (no YOLO, no panel crops, no stitching of sub-regions).
2. **OpenAI gpt-image-2 + labelled atlas** — the page + a reference atlas
   (reusing `pipeline_v1/atlas.py`) are sent to `images/edit`.
3. **Output at the minimal size the API supports while keeping the page's
   aspect ratio** (currently the API requires: edges multiples of 16, area in
   [655,360, 8,294,400] px, max edge ≤ 3840, ratio ≤ 3:1 — the empirical
   result for a 1500×2250 page is **672×1008**, exactly what research-v2
   measured and the user already picked as the sweet spot).
4. **Two new flags** (per the request):
   - `--full-page`: enable the full-page mode (gpt-image-2 backend, no panel
     extraction).
   - `--atlas-source {detected,cast}`: where the atlas characters come from —
     `detected` = one VLM call per page listing the characters present,
     `cast` = the full chapter cast (auto-derived via
     `cast_key_for_page` / `--cast-key`), **zero VLM calls**.

This is a *mode* of pipeline_v1 (documented in `pipelines.md`), not a new
method entry in `methods.md` (same convention as the FLUX panel pipeline).

---

## Design overview

The pipeline skeleton (5 steps, `STEP_ORDER`, `STEP_DIRS`, resume/from-step,
manifest, `--only-panel`, `--mock`) is reused unchanged. Full-page mode makes
each step operate on **one synthetic "panel" per page** (`panel_0001` = the
whole page), so every downstream consumer (character records, atlas building,
manifest totals, debug annotation, resume) works with zero structural changes:

| step | panel mode (today) | full-page mode (new) |
|---|---|---|
| `panels` | YOLO26n detect + crop in reading order | **no YOLO**: write `panel_0001.png` = full page + `panels.json` with one full-page box, `provenance: full-page-mode`; blank-page ink check still applies (skips blank pages) |
| `characters` | OpenRouter VLM per detection mode | `--atlas-source detected` → force `detection_mode="page"` (1 call/page; with one panel its `panel_0001` mapping is exactly the page's character set). `--atlas-source cast` → **step is a no-op** (writes `summary.json` with zero calls; no API key needed) |
| `colorize` | FLUX per panel, filtered atlas | **gpt-image-2** per page: page + atlas (detected chars or chapter cast) + palette instruction; minimal aspect-preserving size; OpenAI usage/cost recorded |
| `stitch` | paste panels onto page | passthrough: copy `3_colorized/<page>/panel_0001.png` → `4_stitched/<page>.png` (keeps `pages_stitched` totals + debug uniform) |
| `debug` | bbox + characters annotation | unchanged — annotates the single full-page box + detected characters (useful provenance) |

Key property: `--resume`, `--from-step`, `--only-panel`, `--mock`,
`--skip-first/--limit` all keep working with identical semantics.

---

## New flags (all in `pipeline_v1/config.py`)

| flag | default | meaning |
|---|---|---|
| `--full-page` | off | full-page gpt-image-2 mode: no panel extraction, one colorization call per page, minimal aspect-preserving output size |
| `--atlas-source` | `detected` | `detected`: VLM per page (forces `detection_mode="page"`); `cast`: full chapter cast for the atlas, no VLM calls |
| `--gpt-model` | `gpt-image-2` | OpenAI image model |
| `--gpt-image-prompt-file` | `pipeline_v1/gpt_image_prompt.txt` | atlas prompt (new file, generalized from `research-v2/data/atlas/prompt.txt`) |
| `--gpt-size` | *minimal* | optional `WxH` override for comparison runs (default: computed minimal size) |
| `--gpt-atlas-scale` | `1.0` | downscale the atlas before upload (research-v2 `--atlas-scale`; cuts the fixed image-input cost) |
| `--openai-api-key-env` | `OPENAI_API_KEY` | env var for the OpenAI key |

Decisions (user-confirmed):
- Quality is **fixed at `medium`** — no `--gpt-quality` flag (constant
  `GPT_IMAGE_QUALITY = "medium"`; research-v2 measured 672×1008 @ medium ≈
  $0.0499/page).
- `--atlas-source cast` is **full-page mode only**: passing it without
  `--full-page` is a validation error.
- **Retries**: gpt-image-2 calls retry transient errors with exponential
  backoff; **after more than 3 retries, fail loudly** (record
  `ColorizeRecord(status="error")` with the last error — same shape as the
  FLUX retry loop, `retries=3`).

Validation:
- `--full-page` + `--atlas-source detected` → `detection_mode` is **forced to
  `page`** (warning printed if the user passed another mode; reject instead if
  it turns out cleaner).
- `--atlas-source cast` without `--full-page` → error
  ("--atlas-source cast requires --full-page").
- `--full-page` + `--atlas-source cast` → characters step skipped; requires
  `OPENAI_API_KEY` only, **not** `OPENROUTER_API_KEY`.
- `--gpt-size` must satisfy the API constraints (multiples of 16, area range,
  ratio ≤ 3:1) when given.

---

## Worker flags: rename + new parallel-colorization flag

**`--workers` → `--worker-detection`** (clean rename, no alias kept):

| old | new | config field | meaning |
|---|---|---|---|
| `--workers` | `--worker-detection` | `worker_detection: int = 1` | parallel character-detection worker threads (1 = sequential, current behavior) |
| — | `--worker-colorization` | `worker_colorization: int = 1` | parallel colorization worker threads (1 = sequential, current behavior) |

- Both validated ≥ 1 (`ValueError` otherwise, mirroring today's check).
- Both in `to_dict` → `"worker_detection"` / `"worker_colorization"`.
- Historical narrative in `pipelines.md` (lines ~389–400, which record actual
  `--workers 16` runs) is **left untouched** — it documents what was run, not
  the current CLI. `AGENTS.md` and `pipeline_v1/README.md` (current-flag
  docs) get the new names.
- **Out of scope:** `tests/sweep_detection_models.py --workers` and
  `research/character_detection_methods/character-detection-qwen3-vllm-ref-pair/run.py
  --workers` are separate tools with their own semantics — not renamed.

### Parallel colorization (`--worker-colorization`, default 1)

Same shape as the characters step's parallel path: pages are independent
units of work — each page writes only to its own `3_colorized/<page>/` dir
(panel outputs, `*_atlas.jpg`), so worker threads never race on files.
`run_colorize_step` is refactored: the per-page body becomes
`_process_page(ctx, config, colorizer, page_dir, profiles, profiles_sha,
extension) -> (records, totals_delta, fresh_stems)` (returns per-page records
instead of mutating shared lists; `_copy_resumed_panels` stays inside the
worker — it only touches that page's dirs).

- `worker_colorization <= 1` → the current sequential path unchanged
  (per-panel progress bars + `set_postfix`).
- `worker_colorization > 1` → `ThreadPoolExecutor(max_workers=...,
  thread_name_prefix="colorize")` over pages; records/totals merged back in
  the main thread via `as_completed` (same pattern as `steps/characters.py`
  lines 117–131); single `pages_bar` instead of per-page bars.
- **Thread safety of the shared colorizer:** `FluxColorizer.colorize()` is
  one stateless `requests.post` per call (no shared `Session`) — safe to
  share. The new `GptImage2Colorizer` uses the OpenAI SDK client, which is
  thread-safe for concurrent `images.edit` calls — also safe to share. No
  per-thread colorizer instances needed.
- In full-page mode this parallelizes the gpt-image-2 calls directly (one
  call per page) — the main win for the paid backend.

---

## Minimal output size algorithm (`config.minimal_gpt_image_size`)

Exact-ratio, floor-driven (reproduces both research-v2 measured sizes):

1. Reduce the page ratio to lowest terms: `g = gcd(w, h)`, `(w', h') = (w/g, h/g)`.
2. Smallest integer multiplier `k` such that both edges are multiples of 16:
   `step = lcm(16/gcd(w',16), 16/gcd(h',16))`.
3. Area floor: `k_min = ceil(sqrt(655_360 / (w'·h')))`.
4. `k = smallest multiple of step ≥ k_min`.
5. Return `(w'·k, h'·k)`; raise a clear `ValueError` if the page ratio is
   outside [1:3, 3:1] (the API rejects every size at that ratio — fail loudly
   rather than distort).

Worked examples (both match research-v2's measured runs exactly):
- **1500×2250 (2:3)** → reduced 2:3, `step=16`, `k=336` → **672×1008** (0.68 MP, $0.0499 @ medium).
- **3000×2250 spread (4:3)** → reduced 4:3, `step=16`, `k=240` → **960×720** (691,200 px, just above the 655,360 floor — NOT 672×1008, which would squeeze the spread into 2:3).

Constants live in `config.py` next to the FLUX ones
(`GPT_IMAGE_MIN_PIXELS = 655_360`, `GPT_IMAGE_MAX_PIXELS = 8_294_400`,
`GPT_IMAGE_MAX_EDGE = 3840`, `GPT_IMAGE_MAX_RATIO = 3.0`, multiple 16).

---

## File-by-file changes

**`pipeline_v1/config.py`**
- New dataclass fields (list above), `to_dict` entries, argparse flags,
  `_validate` rules.
- `minimal_gpt_image_size(width, height)` helper + constants.
- **Worker rename:** `workers: int` → `worker_detection: int = 1` (field,
  `to_dict` key, `_validate` check, argparse `--worker-detection`, the
  `PipelineConfig(workers=...)` construction, docstring comment).
- **New flag:** `worker_colorization: int = 1` + argparse
  `--worker-colorization` + `_validate` (≥ 1) + `to_dict` entry.
- `--steps`/`--from-step` unchanged (same 5 step names).

**`pipeline_v1/colorizer.py`**
- Extend `ColorizeRecord` with optional fields `model`, `quality`, `usage`
  (dict), `est_cost_usd` (all `None` for FLUX → `to_dict` omits them).
- `Colorizer` protocol unchanged.

**`pipeline_v1/gpt_colorizer.py`** (new module)
- `GptImage2Colorizer` implementing the `Colorizer` protocol:
  `colorize(page, atlas, output, palette_instruction="") -> ColorizeRecord`.
- Calls `openai.OpenAI(timeout=600).images.edit(model, image=[page, atlas],
  prompt, quality=GPT_IMAGE_QUALITY ("medium"), size=minimal_or_override,
  output_format, n=1)`; decodes `b64_json`; parses `usage` into the
  research-v2 token accounting (image input $8/1M, text input $5/1M, image
  output $30/1M, standard tier) and computes `est_cost_usd`;
  `requested_size` = the minimal size.
- **Retry policy**: mirror the FLUX retry loop — retry transient errors
  (connection errors, 429/5xx) with exponential backoff up to `retries=3`
  (4 attempts total); after the last failed attempt record
  `ColorizeRecord(status="error", error=<last error>)` (fail loudly, no
  silent partial state).
- Prompt rendering mirrors `FluxColorizer._prompt` with the same placeholders
  (`{width}`, `{height}`, `{atlas_instruction}`, `{character_profiles}`), so
  the FLUX atlas/palette instruction strings are reused.

**`pipeline_v1/gpt_image_prompt.txt`** (new)
- Generalized from `research-v2/data/atlas/prompt.txt`: no hardcoded character
  names (atlas labels are dynamic), keep the "use atlas colors, preserve the
  page's lineart/composition, do not copy atlas layout/labels/borders" core,
  plus the size and palette-instruction placeholders.

**`pipeline_v1/steps/panels.py`**
- `config.full_page` branch: skip the detector entirely; one `PanelBox(0, 0,
  w, h, 1.0)` per page, `provenance="full-page-mode"`, `full_page_fallback`
  semantics reused for blank-page skip (ink check unchanged). `overlay.png`
  still drawn (shows the single box).

**`pipeline_v1/steps/characters.py`**
- `--atlas-source cast`: detect early and write an empty `summary.json`
  (zero calls/cost) — the colorize step derives the names. No OpenRouter key
  required.
- `--atlas-source detected`: unchanged path, but `strategy_for` is handed
  `detection_mode="page"` in full-page mode.
- `config.workers` → `config.worker_detection` (docstring, `<= 1` check,
  `max_workers`).

**`pipeline_v1/steps/colorize.py`**
- Backend is `GptImage2Colorizer` (wired in `run.py`); loop is per page.
- `atlas-source=cast`: `names_by_panel = {panel_0001: cast_names}` where
  `cast_names` comes from `cast_key_for_page(page, chapter_casts_file,
  chapter_page_map_file)` (or `--cast-key`); no chapter cast derivable → full
  roster fallback (same convention as `characters.py`).
- Palette instruction still rendered from `character_profiles.json` for the
  chosen names.
- **Parallel path:** refactor per-page body into `_process_page(...)`; add
  `ThreadPoolExecutor` over pages when `config.worker_colorization > 1`
  (details in the worker-flags section above).

**`pipeline_v1/steps/stitch.py`**
- Full-page mode: copy the colorized `panel_0001` to `4_stitched/<page>.png`
  (passthrough so totals/debug/resume stay uniform).

**`pipeline_v1/steps/debug.py`** — no change (single box + label works).

**`pipeline_v1/characters.py`** (detector module)
- `OpenRouterCharacterDetector.__init__(workers=...)` →
  `worker_detection=...`; `self.workers` → `self.worker_detection`;
  `detector.workers > 1` (tqdm disable check) → `detector.worker_detection > 1`.

**`pipeline_v1/run.py`**
- `--full-page` → build `GptImage2Colorizer` (require `OPENAI_API_KEY`);
  `--atlas-source cast` → skip OpenRouter key check/character detector build.
- `workers=config.workers` → `worker_detection=config.worker_detection` when
  building the detector. `--worker-colorization` needs no run.py wiring — the
  colorize step reads `config.worker_colorization` directly.
- `--mock --full-page` → existing `MockColorizer` (works for any Colorizer).

**`pipeline_v1/orchestrator.py`**
- Totals: `gpt_image_calls`, `successful_gpt_image_calls`,
  `gpt_image_cost_usd` (mirrors the `flux_calls` pattern).
- `pricing_assumptions`: keep the FLUX block (still the panel-mode default),
  add an OpenAI gpt-image-2 block (model, quality, standard-tier rates,
  "paid API" note) — the manifest records which backend ran.

**`pipeline_v1/mock_backends.py`**
- `MockColorizer` already satisfies the protocol; add a `backend` marker
  ("flux"|"gpt-image-2") so tests can assert the right backend was selected.
  Optionally tint differently in gpt mode for visual distinction.

---

## Tests (all offline; existing 233 must stay green)

- `test_config.py`: new flags parse/validate (`--atlas-source cast` without
  `--full-page` raises; `--worker-detection`/`--worker-colorization` parse and
  validate ≥ 1; `--workers` is gone); `minimal_gpt_image_size` cases —
  1500×2250 → (672,1008), 3000×2250 → (960,720), 1200×1800 → (672,1008),
  exact ratio preserved, floor respected, ratio > 3:1 raises.
- `test_gpt_colorizer.py` (new): fake OpenAI client (pattern from
  `fake_flux_server.py`) — request carries the minimal size + `medium`
  quality + 2 images; usage parsed; `est_cost_usd` matches research-v2
  accounting; API error → `ColorizeRecord(status="error")`.
- `test_gpt_colorizer.py`: retry behavior — transient failures are retried
  (≤ 3 retries) and eventually succeed; persistent failures after 3 retries
  yield `status="error"` with the last error (fail loudly).
- `test_panels.py`: full-page mode writes one synthetic panel without calling
  the detector (detector that raises if invoked); blank page still skipped.
- `test_orchestrator.py` (or a new `test_full_page.py`): `--mock --full-page
  --atlas-source cast` full run — 1 colorize call per page, 0 VLM calls, stitch
  passthrough, debug annotated; `--atlas-source detected` forces page-mode
  detection.
- `test_characters.py`: `config.workers = N` → `config.worker_detection = N`
  in the parallelization test (line ~1251) + docstring.
- **New parallel-colorize test** (in `test_orchestrator.py` or a dedicated
  `test_colorize_step.py`): `--mock --worker-colorization 4` on a 3-page
  synthetic input — same records/totals as `=1` (merged in completion
  order), all pages present, no duplicate records; with a fake FLUX server
  that records concurrency, assert overlapping in-flight calls when
  `worker_colorization > 1`.
- Integration (optional, gated on `OPENAI_API_KEY` in `.env`): one real
  gpt-image-2 full-page run on a committed page, recorded as a new pipeline
  evaluation entry (COL-FP-001 style) with measured tokens/cost.

---

## Docs

- `pipeline_v1/README.md`: new flags, example commands
  (`--full-page --atlas-source detected`, `--full-page --atlas-source cast`),
  minimal-size table, cost notes; `--workers` → `--worker-detection` in the
  flags list, plus the new `--worker-colorization`.
- `AGENTS.md`: `--workers N` → `--worker-detection N` (line ~211).
- `pipelines.md`: new "Full-page gpt-image-2 atlas" section — projected
  per-page cost from research-v2 (672×1008 @ medium ≈ **$0.0499/page**
  measured, input floor ≈ $0.019/page; volume-1 projection ≈ $9.32/187 pages
  at the minimal size) labelled as projection until a real pipeline run lands
  (same "never fabricate numbers" convention).
- `pipeline_v1/ARCHITECTURE.md`: note the full-page mode branch in the step
  map.

---

## Verification plan (after implementation)

1. `.venv/bin/pytest pipeline_v1/tests -q` — offline suite green.
2. `pipeline_v1/run.py --mock --full-page --limit 1` — offline smoke.
3. Real run on `research-v2/data/pages` (p006–p008): one run with
   `--atlas-source detected`, one with `--atlas-source cast`; verify output
   sizes are exactly 672×1008 (or the computed minimal), record usage/cost in
   the manifest, update `pipelines.md` with measured numbers.
4. Confirm `--resume`/`--from-step` work across the two modes.
5. Confirm `--worker-colorization 4` (mock, 3+ pages) produces identical
   records/totals to `=1` and actually overlaps in-flight calls; and that
   `--worker-detection` still behaves like the old `--workers`.

---

## Decisions (user-confirmed, 2026-08-15)

1. Flag names `--full-page` + `--atlas-source {detected,cast}` — **OK**.
2. `--atlas-source cast` is **full-page mode only** (error otherwise).
3. Quality is **only `medium`** — no quality flag.
4. **Retry with backoff; after more than 3 retries fail loudly** (record the
   error, no silent partial state).
5. **Worker flags:** `--workers` is renamed to `--worker-detection`, and a new
   `--worker-colorization` flag parallelizes the colorize step over pages.

All five are now incorporated into the design above; the plan is ready to
implement.
