# pipeline_v1 — architecture

Panel-wise manga colorization pipeline: for each manga page, detect its panels,
extract them in Japanese reading order, detect which reference characters appear in
each panel, colorize each panel with FLUX.2 Klein 9B base + manga-colorization LoRA
using an atlas filtered to the detected characters, and stitch the colorized panels
back onto the original page.

## Module map

```text
pipeline_v1/
├── run.py                # entry point: config -> backends -> orchestrator
├── config.py             # PipelineConfig dataclass + argparse CLI + FLUX size helpers
├── run_context.py        # timestamped run dirs, atomic JSON, incremental manifest
├── detection.py          # PanelDetector protocol + YoloPanelDetector (ultralytics, lazy)
├── panel_ordering.py     # pure: Japanese reading order for detected boxes
├── extraction.py         # pure: crop panels, numbered files
├── steps/panels.py       # stage 1+2 -> 1_panels/ (crops + panels.json + overlay)
├── characters.py         # CharacterDetector protocol + OpenRouter gemma-4-31b-it client
├── prompt.txt            # character-detection prompt (same as research method)
├── atlas.py              # labelled atlas filtered to detected characters only
├── colorizer.py          # Colorizer protocol + FluxColorizer (multipart POST /edit)
├── colorizer_prompt.txt  # colorize-only prompt with mngclranm trigger
├── stitching.py          # pure: paste colorized panels back at recorded boxes
├── steps/stitch.py       # stage 5 -> 4_stitched/
├── orchestrator.py       # step sequencing, manifest aggregation, resume
├── mock_backends.py      # fake detector / VLM / colorizer for offline runs & tests
├── output/<ts>/          # per-run artifacts (gitignored)
│   ├── 1_panels/         # crops + panels.json (boxes + reading order) + overlay
│   ├── 2_characters/     # per-panel detection JSONs + summary
│   ├── 3_colorized/      # per-panel colorized outputs
│   ├── 4_stitched/       # final pages
│   └── manifest.json
└── tests/                # unit tests + offline end-to-end suite
```

## Data flow

```text
page (input_dir)
  -> [detection.py + panel_ordering.py + extraction.py]
       -> 1_panels/<page>/panel_000N.*        (crops, reading order)
       -> 1_panels/<page>/panels.json         (boxes, confidence, order)
  -> [characters.py]  per panel
       -> 2_characters/<panel>.json           (canonical names, cost, latency)
  -> [atlas.py]       per panel
       -> filtered labelled atlas (detected characters only; None if none)
  -> [colorizer.py]   per panel  POST /edit [panel, atlas?]
       -> 3_colorized/<page>/panel_000N.png
  -> [stitching.py]
       -> 4_stitched/<page>.png               (panels colorized, rest B&W)
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
  stitching. No fixed upscaling target.
- **Empty character detection**: a panel with no detected reference characters
  is colorized with the panel only (no atlas).
- **Numbered intermediate dirs**: `1_panels/`, `2_characters/`,
  `3_colorized/`, `4_stitched/` under the timestamped run dir; the stage-1
  detection positions are persisted in `1_panels/<page>/panels.json` for the
  final stitch stage.
- **Run conventions**: fresh `output/YYYYMMDD-HHMMSS/` dir per invocation
  (never overwritten), incremental `manifest.json` with command, configuration,
  inputs, per-step records, measured costs (OpenRouter from `usage.cost`;
  FLUX self-hosted = $0 + electricity note).
