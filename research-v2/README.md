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
- `convert_refs_to_manga.py`: converts the color reference portraits into B&W
  manga line art via the Spark FLUX.2 Klein edit server (inputs for YOLOE).
- `data/refs_manga/`: the converted references (`manifest.json` records the
  prompt/seed/cost) — used automatically by `detect_characters_yoloe.py` when
  present.
- `data/patch/`: the OpenAI gpt-image-2 colorization test inputs — `orig.png`
  (the B&W panel), `patch.png` (the same panel with the character reference
  composited on top) and `prompt.txt`.
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

On the 21 c001 panels, **7 detections** at conf ≥ 0.1 in both reference
variants, all low-confidence and mostly spurious:

| reference variant | best conf | mean conf | example hits |
|---|---|---|---|
| color refs (run 20260815-021726) | 0.439 | ~0.235 | p008/panel_0003 Himmel (edge-hugging box) |
| manga-converted refs (run 20260815-022342) | 0.194 | ~0.137 | p008/panel_0004 Eisen, Heiter |

`convert_refs_to_manga.py` re-rendered each cast portrait into B&W line art
(FLUX.2 Klein edit, Spark, sat ≈3–6, 85–91% white — see `data/refs_manga/`),
then `detect_characters_yoloe.py --refs-dir data/refs_manga` re-ran the
prompts. The manga references did **not** improve detection — confidence
*dropped* (best 0.439 → 0.194).

Observed problems (measured, not speculation):
- **Identity is unstable**: in a config sweep on p008/panel_0002 the same
  character region was labeled Frieren, Himmel, or Heiter depending on model
  size / ref size / imgsz; per-panel labels contradict each other.
- **Boxes hug panel edges** (x=1–2, spanning full height) — the model fires on
  panel borders rather than character bodies.
- **Low confidence on genuine hits**; the references themselves are re-detected
  at ~0.78, so the prompt mechanism works but does not transfer to manga
  character line art, whether the reference is colored or itself line art.
  YOLOE's SAVPE is a one-shot "find more like this exact object" matcher; it
  does not bridge pose/style/scale variation between a portrait reference and
  a panel character.
- Sparse recall: only 7/21 panels get any detection (manga109_yolo found 34
  body boxes; pipeline_v1's VLM identifies the cast correctly).

Verdict: keep as an experimental open-vocabulary datapoint; do not use for
cast detection in the pipeline.

## gpt_image_colorize.py

One-shot manga panel colorization with **OpenAI gpt-image-2** (Image API
`/images/edits` endpoint, no mask — both input images act as references).

```bash
.venv/bin/python research-v2/gpt_image_colorize.py
# --quality low --quality medium --quality high   (defaults)
# --size 2880x2240                                (multiple of 16; source-panel aspect)
```

Inputs (from `data/patch/`): `orig.png` — the black & white panel to colorize,
and `patch.png` — the same panel with the character reference composited on
top. The prompt instructs the model to colorize the line art with the
reference character's colors while adapting the reference's orientation/pose
to match the B&W panel. Both images are sent together; the prompt supplies the
glue. Output: `output/<YYYYMMDD-HHMMSS>/quality_<low|medium|high>.png` (one
image per requested quality, same size) + `manifest.json` (prompt, config,
per-quality timestamps, API `usage` tokens and computed cost).

Flags: `--model`, `--quality` (repeatable; `low|medium|high|auto`),
`--size` (WxH, constraints: max edge ≤ 3840, both edges multiples of 16,
ratio ≤ 3:1, 655,360–8,294,400 px), `--output-format` (`png|jpeg|webp`),
`--input-dir`, `--prompt-file`, `--output-root`, `--env-file`. Cost: paid
OpenAI API; `gpt-image-2` bills image input $8/1M tokens, text input $5/1M,
image output $30/1M (standard tier) and always processes image inputs at high
fidelity.

### Quality sweep run 20260815-082623 (2880×2240, 2 reference images)

| quality | latency | output tokens | cost (measured) |
|---|---|---:|---:|
| low | 51 s | 406 | $0.0364 |
| medium | 67 s | 3753 | $0.1368 |
| high | 149 s | 15213 | $0.4806 |

Measured via API `usage` (input tokens 3042 = 2992 image + 50 text, constant
across the sweep; output tokens = total − 3042 — the input count was confirmed
by an identical-inputs probe call). Sweep total ≈ $0.65 (+ ≈$0.03 for the
probe). Input 2895×2250 + 2048×1591; output 2880×2240 PNG.

Observations (objective only — files in the run dir for visual inspection):
all three outputs are colored (mean saturation 0.18–0.22, Hasler–Süsstrunk
colorfulness ≈ 32–35 vs 0 for the gray input); file size grows with quality
(7.1/8.1/10.5 MB), consistent with higher detail. The model produced a
full-page colorization from the two reference images — no mask or atlas
needed. Known limitations from the docs: output >2560×1440 is "experimental",
so a 2K size (e.g. 2048×1600) is a cheaper, supported alternative; the model
cannot guarantee exact reference-color reproduction across generations.

## Conventions

Same as the repo's methods: each run creates a fresh timestamped directory,
never overwrites a previous run, and records a manifest with input files,
configuration, and cost at run time.
