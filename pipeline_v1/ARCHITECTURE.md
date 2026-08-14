# pipeline_v1 — architecture

Panel-wise manga colorization pipeline: for each manga page, detect its panels,
extract them in Japanese reading order, detect which reference characters appear in
each panel, colorize each panel with FLUX.2 Klein 9B base + manga-colorization LoRA
using an atlas filtered to the detected characters, stitch the colorized panels
back onto the original page, and finally annotate a debug copy of each stitched
page with its panel boxes and detected characters.

## Module map

```text
pipeline_v1/
├── run.py                # entry point: config -> backends -> orchestrator
├── config.py             # PipelineConfig dataclass + argparse CLI + FLUX size helpers
├── run_context.py        # timestamped run dirs, atomic JSON, incremental manifest
├── detection.py          # PanelDetector protocol + YoloPanelDetector (ultralytics, lazy)
├── panel_ordering.py     # pure: Japanese reading order for detected boxes
├── extraction.py         # pure: crop panels, numbered files
├── steps/panels.py       # stage 1+2 -> 1_panels/ (crops + panels.json + overlay; blank check + full-page fallback)
├── characters.py         # CharacterDetector protocol + OpenRouter gemma-4-31b-it client (per-panel + page-level)
├── prompt.txt            # page-level character-detection prompt (numbered panels -> JSON mapping)
├── prompt_panel.txt      # per-panel fallback detection prompt
├── character_profiles.json  # canonical profiles: identity cues + palettes (task 0002)
├── profiles.py           # profile loading, validation, prompt rendering
├── chapter_casts.json    # optional cached chapter cast shortlists (task 0003)
├── selection.py          # --only-panel / --force-characters selectors (task 0001)
├── steps/characters.py   # stage 3 -> 2_characters/ (page calls + fallbacks + forced identities)
├── atlas.py              # labelled atlas filtered to detected characters only
├── colorizer.py          # Colorizer protocol + FluxColorizer (multipart POST /edit; palette + size cap)
├── colorizer_prompt.txt  # colorize-only prompt with mngclranm trigger + {character_profiles}
├── steps/colorize.py     # stage 4 -> 3_colorized/ (filtered atlas + palette + resume reuse)
├── stitching.py          # pure: paste colorized panels back at recorded boxes
├── steps/stitch.py       # stage 5 -> 4_stitched/
├── steps/debug.py        # stage 6 -> 5_debug/ (bbox + character annotation; shared with scripts/annotate_stitch.py)
├── orchestrator.py       # step sequencing, manifest aggregation, resume (copies steps before --from-step)
├── mock_backends.py      # fake detector / VLM / colorizer for offline runs & tests
├── evaluation/v1_1_cases.json  # fixed failure set (task 0001), run by the integration suite
├── verify_color.py       # real gpt-5.6-luna generic color verifier (strict structured output: analyse/good_color), integration suite
├── tests/                # offline unit tests + real-network integration suite (-m integration)
├── output/<ts>/          # per-run artifacts (gitignored)
│   ├── 1_panels/         # crops + panels.json (boxes + reading order + provenance) + overlay
│   ├── 2_characters/     # per-panel detection JSONs (source: page|fallback|forced) + summary
│   ├── 3_colorized/      # per-panel colorized outputs
│   ├── 4_stitched/       # final pages
│   ├── 5_debug/          # stitched pages + bbox + characters per panel (stage 6)
│   └── manifest.json
└── tests/                # unit tests + offline end-to-end suite
```

## Data flow

```text
page (input_dir)
  -> [detection.py + panel_ordering.py + extraction.py]
       -> 1_panels/<page>/panel_000N.*        (crops, reading order)
       -> 1_panels/<page>/panels.json         (boxes, confidence, order)
  -> [characters.py]  per page (page-level call) with per-panel fallbacks
       -> 2_characters/<panel>.json           (canonical names, source, cost, latency)
  -> [profiles.py + atlas.py]  per panel
       -> explicit palette instruction + filtered labelled atlas
       -> (detected characters only; None if none)
  -> [colorizer.py]   per panel  POST /edit [panel, atlas?, palette text]
       -> 3_colorized/<page>/panel_000N.png
  -> [stitching.py]
       -> 4_stitched/<page>.png               (panels colorized, rest B&W)
  -> [steps/debug.py]  per page (pure image processing)
       -> 5_debug/<page>.png                  (bbox + characters per panel)
       -> 5_debug/summary.json
```

## Key decisions

- **Library ports, not subprocesses**: the pipeline reuses the logic (prompt,
  JSON contract, retry/cost accounting) of the research methods
  `character-detection-openrouter-vlm` and
  `flux-2-klein-9b-base-lora-edit-sequential` as modules; those methods stay
  untouched.
- **Backends behind protocols** (`PanelDetector`, `CharacterDetector`,
  `Colorizer`): real implementations import heavy/paid dependencies lazily, so
  the whole test suite runs offline with mock backends.
- **Size policy (user-confirmed)**: each panel is colorized at the resolution
  closest to its original size with both axes multiples of 16
  (`nearest_multiple_of`), then resized back to the exact panel box when
  stitching. Since V1.1 oversized inputs are capped at `--max-megapixels`
  (default 2.0 MP, aspect-preserving, multiples of 16).
- **Empty character detection**: a panel with no detected reference characters
  is colorized with the panel only (no atlas).
- **Character profiles (V1.1)**: canonical identity cues and palettes live in
  `character_profiles.json`; detection prompts and the FLUX palette
  instruction render from the same source of truth.
- **Page-level detection (V1.1)**: one paid call per page; the annotated page
  carries the reading-order numbers used by extraction. Missing/invalid/
  uncertain panel entries trigger cropped-panel fallbacks.
- **Panel+page detection (V1.2)**: one call per panel; the numbered annotated
  page (target highlighted) is sent as global context plus the crop.
  `panel-page-prev2` additionally sends the two preceding non-blank pages in
  reading order as story context, found via the sibling page dirs in
  `1_panels/` (`panels.json` `page_path`); fewer are sent at the start of a
  book, degrading to plain `panel-page` shape. `panel-page-cast` and
  `panel-page-prev2-cast` resolve the per-chapter cast key the same way
  (explicit `cast_key` -> `--cast-key` -> `cast_key_for_page`) and render
  the shortlist into their prompt via `panel_page_*_prompt_for(key)`, so
  per-page casts stay thread-safe. The per-panel call/fallback loop is
  shared between the two modes.
- **Targeted reruns (V1.1)**: `--only-panel PAGE:PANEL` restricts processing;
  `--force-characters` injects ground-truth identities without paid calls;
  `--resume RUN --from-step STEP` copies only the outputs before STEP.
- **Numbered intermediate dirs**: `1_panels/`, `2_characters/`,
  `3_colorized/`, `4_stitched/` under the timestamped run dir; the stage-1
  detection positions are persisted in `1_panels/<page>/panels.json` for the
  final stitch stage.
- **Run conventions**: fresh `output/YYYYMMDD-HHMMSS/` dir per invocation
  (never overwritten), incremental `manifest.json` with command, configuration,
  inputs, per-step records, measured costs (OpenRouter from `usage.cost`;
  FLUX self-hosted = $0 + electricity note).
