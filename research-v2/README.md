# research-v2

New experiment area for pipeline-v2 work (sibling of `research/` and
`pipeline_v1/`, documented here rather than in `methods.md` until it becomes a
formal method or pipeline).

## Layout

- `data/pages/`: committed input pages (volume 1, pages p004–p005 … p010,
  original filenames, reading order). The shared input set for experiments.
- `data/panels/`: committed panel crops extracted from `data/pages/` (run
  `20260815-014738`), one subdirectory per page, `panel_000N.png` numbered in
  reading order. Regenerate with `split_panels.py` and copy from the run dir.
- `output/`: timestamped run dirs (`YYYYMMDD-HHMMSS/`), gitignored.
- `split_panels.py`: panel extraction, pipeline_v1-style.
- `detect_characters.py`: character (body) detection on panels with
  `deepghs/manga109_yolo`.
- `detect_characters_yoloe.py`: cast-aware character detection with ultralytics
  YOLOE, prompted with the chapter-cast reference images.
- `models/`: downloaded model weights (gitignored).

## split_panels.py

Splits pages into panels using the exact pipeline_v1 implementation imported
as a library (same YOLO26n detector `leoxs22/manga-panel-detector-yolo26n` and
weights, same Japanese reading order, same blank-ink check and full-page
fallback). Only the output layout is research-v2's own:

```text
output/<YYYYMMDD-HHMMSS>/
├── manifest.json              # command, config, backend, totals, per-page records
└── <page_stem>/
    ├── panel_0001.png ...     # crops numbered in reading order
    ├── panels.json            # geometry + order (pipeline_v1 schema)
    └── overlay.png            # page with numbered boxes
```

Run:

```bash
.venv/bin/python research-v2/split_panels.py                                  # data/pages, all pages
.venv/bin/python research-v2/split_panels.py --confidence 0.3 --panel-inset 2
.venv/bin/python research-v2/split_panels.py --skip-first 1 --limit 3
```

Flags: `--input-dir`, `--output-root`, `--confidence`, `--panel-inset`,
`--blank-ink-threshold`, `--no-full-page-fallback`, `--skip-first`, `--limit`.
Cost: $0 per call (local inference; first run downloads the weights to
`pipeline_v1/models/` if missing).

## detect_characters.py

Runs the YOLO detector `deepghs/manga109_yolo` (variant `v2023.12.07_x`, 68.2M
params, F1 0.92, trained on Manga109, classes `body`/`face`/`frame`/`text`) on
each panel crop and keeps only the `body` detections — the character regions.
Draws a green bounding box + `body <conf>` label per detection. Weights are
ultralytics 8.3.70 checkpoints (AGPL-3.0), downloaded once into
`research-v2/models/` (gitignored).

Output (mirrors the input layout):

```text
output/<YYYYMMDD-HHMMSS>/
├── manifest.json          # command, config, model, totals, per-panel records
└── <page>/                # one dir per page of panels
    ├── panel_NNNN.json          # body detections: box (xyxy), confidence, label
    └── panel_NNNN_annotated.png # bboxes drawn on the panel
```

Run:

```bash
.venv/bin/python research-v2/detect_characters.py                          # data/panels, conf 0.355
.venv/bin/python research-v2/detect_characters.py --conf 0.25              # more permissive
.venv/bin/python research-v2/detect_characters.py --input-dir PATH --imgsz 1280
```

Flags: `--input-dir`, `--output-root`, `--model-path`, `--conf` (default
0.355, the model-card F1-optimal threshold), `--imgsz` (default 640, the
training size), `--device`, `--font-size`. Cost: $0 per call (local).

Note: `frame` and `text` detections are discarded — this same model also
outputs manga panel boxes (`frame`) and speech bubbles (`text`), so it could
replace the pipeline_v1 panel detector too.

## detect_characters_yoloe.py

Cast-aware character detection with **ultralytics YOLOE** (open-vocabulary
YOLO, `yoloe-26l-seg.pt` by default, AGPL-3.0) using **visual prompting**:
YOLOE conditions detection on binary masks over regions of the *same canvas*
it detects, so each panel is composited onto a canvas with the cast's
reference thumbnails (one per character, from `data/refs/`) below it, and the
references' bounding boxes are passed as the visual prompts (SAVPE).
Detections whose box center falls in the panel region are the panel's
characters; `ref_sheet.png` documents the prompts used.

The cast is taken from `pipeline_v1/chapter_casts.json` (`casts.<key>`,
default `c001` → Himmel, Frieren, Eisen, Heiter). Per-panel outputs mirror
`detect_characters.py`: `panel_NNNN.json` (label, box, confidence) and
`panel_NNNN_annotated.png`.

```bash
.venv/bin/python research-v2/detect_characters_yoloe.py                            # 26l, refs 300, imgsz 1280, conf 0.1
.venv/bin/python research-v2/detect_characters_yoloe.py --cast-key c002 --conf 0.05
.venv/bin/python research-v2/detect_characters_yoloe.py --model yoloe-26s-seg.pt    # faster, weaker
```

Flags: `--input-dir`, `--output-root`, `--model`, `--cast-key`, `--ref-size`,
`--imgsz`, `--conf`, `--device`, `--font-size`. Cost: $0 per call (local CPU).

### Quality assessment — experimental, NOT usable for cast identification

On the 21 c001 panels (run 20260815-021726, conf ≥ 0.1): **7 detections**, all
low-confidence and mostly spurious:

| example | label | conf | box |
|---|---|---|---|
| p008/panel_0003 | Himmel | 0.44 | [1,2,140,291] (flush to left edge) |
| p008/panel_0003 | Himmel | 0.37 | [1,1,410,330] |
| p009/panel_0003 | Eisen | 0.24 | [1,1,394,225] |
| p010/panel_0003 | Himmel | 0.13 | [2,0,314,212] |

Observed problems (measured, not speculation):
- **Identity is unstable**: in a config sweep on p008/panel_0002 the same
  character region was labeled Frieren, Himmel, or Heiter depending on model
  size / ref size / imgsz; per-panel labels contradict each other.
- **Boxes hug panel edges** (x=1–2, spanning full height) — the model fires on
  panel borders rather than character bodies.
- **Low confidence on genuine hits** (≤ 0.26 even for the strongest box); the
  references themselves are re-detected at ~0.78, so the prompt mechanism
  works but does not transfer across the **color-reference → B&W-manga** domain
  gap (color is the main identity cue for this cast; the panel has none).
- Sparse recall: only 7/21 panels get any detection (manga109_yolo found 34
  body boxes; pipeline_v1's VLM identifies the cast correctly).

Verdict: keep as an experimental open-vocabulary datapoint; do not use for
cast detection in the pipeline.

## Conventions

Same as the repo's methods: each run creates a fresh timestamped directory,
never overwrites a previous run, and records a manifest with input files,
configuration, and cost at run time.
