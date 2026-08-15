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
- `data/atlas/`: prompt for the atlas method (atlas itself is built at run
  time into the run dir by `gpt_image_colorize.py --atlas-chars ...`).
- `data/noref/`: prompt for the no-reference baseline arm.
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
`/images/edits` endpoint, no mask — additional input images act as
references). Three input modes, all on the same `data/patch/orig.png`:

- **patch** (default): `orig.png` + `patch.png` — the same panel with the
  character reference composited on top; prompt from `data/patch/prompt.txt`.
- **atlas**: `--atlas-chars NAME...` builds the pipeline_v1 labelled reference
  atlas (360×480 labelled cells, `pipeline_v1/atlas.py`) from `data/refs/`
  for the given characters and sends `orig.png` + `atlas.jpg`; prompt from
  `data/atlas/prompt.txt` (defaults there automatically).
- **no-reference**: `--no-reference` sends only `orig.png` (model baseline,
  no reference conditioning); pass `--prompt-file` explicitly (e.g.
  `data/noref/prompt.txt`).

```bash
.venv/bin/python research-v2/gpt_image_colorize.py
# --quality low --quality medium --quality high   (defaults)
# --size 2880x2240                                (multiple of 16; source-panel aspect)
# --atlas-chars frieren himmel heiter eisen       (atlas method)
# --orig-file "research-v2/data/pages/<page>.png" (colorize any page/panel directly)
# --no-reference --prompt-file research-v2/data/noref/prompt.txt   (baseline)
```

The prompt instructs the model to colorize the line art with the reference
character colors while keeping the panel's own lineart, poses, orientation,
and composition. All input images are sent together; the prompt supplies the
glue. Output: `output/<YYYYMMDD-HHMMSS>/quality_<low|medium|high>.png` (one
image per requested quality, same size) + `manifest.json` (prompt, config,
mode, atlas provenance, per-quality timestamps, API `usage` tokens and
computed cost).

Flags: `--model`, `--quality` (repeatable; `low|medium|high|auto`),
`--size` (WxH, constraints: max edge ≤ 3840, both edges multiples of 16,
ratio ≤ 3:1, 655,360–8,294,400 px), `--output-format` (`png|jpeg|webp`),
`--input-dir`, `--orig-file` (default `<input-dir>/orig.png`; colorize any
page/panel directly), `--prompt-file`, `--output-root`, `--env-file`,
`--atlas-chars`, `--refs-dir`, `--no-reference`. Cost: paid OpenAI API; `gpt-image-2` bills
image input $8/1M tokens, text input $5/1M, image output $30/1M (standard
tier) and always processes image inputs at high fidelity.

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

### A/B test: patch vs atlas vs no-reference (runs 20260815-085914 / 20260815-090035)

Research question: was the good patch-method result caused by the *patch
method* (reference composited on the panel) or just by *gpt-image-2 itself*?
Same input `data/patch/orig.png` (2895×2250), same model, same size
2880×2240, same `medium` quality in all three arms:

| arm | reference conditioning | run | cost (measured) | mean sat. | colorfulness |
|---|---|---|---|---:|---:|---:|
| patch | panel + composited refs (2048×1591) | 20260815-082623 (medium row) | $0.1368 | 0.174 | 32.1 |
| atlas | panel + labelled 720×960 atlas (4 chars) | 20260815-085914 | $0.1313 | 0.220 | 41.4 |
| no-reference | panel only | 20260815-090035 | $0.1248 | 0.209 | 37.3 |

All three arms colorize the panel with similar latency (67–73 s) and cost
(≈$0.12–0.14/image at medium). The **no-reference baseline is nearly as
colorful as the reference-conditioned runs** (sat 0.209/colorfulness 37.3 vs
atlas 0.220/41.4), so a large part of the result is simply gpt-image-2's
built-in ability to colorize manga sensibly from a B&W panel plus a short
prompt. The **atlas is as good as (objectively slightly more saturated than)
the patch** — so the patch compositing method itself is *not* what made the
patch result good; the model handles either reference presentation, and
references mainly steer palette *choices* (character-accurate hues) rather
than overall colorfulness. Visual comparison sheet:
`output/20260815-085914/comparison_patch_vs_atlas_vs_noref.png`.

Caveats: subjective palette faithfulness (are Himmel's eyes actually blue,
Frieren's coat actually white?) needs visual inspection of the run outputs;
objective metrics here measure only how much/varied color was applied, not
whether it matches canonical character colors. The no-reference prompt
(`data/noref/prompt.txt`) asks for a "restrained anime palette consistent with
the series" — a stronger baseline prompt could close the gap further.

### Full-page atlas + size/cost sweep (p009, runs 20260815-090951 … 091610)

Cost-optimization datapoint: the atlas method applied to a **full page**
(p009, 1500×2250 B&W) at `medium` quality, keeping the page's 2:3 aspect at
four output sizes. All runs share the same input page + same 4-character
atlas (frieren/himmel/heiter/eisen); input tokens are constant (2416 = 2304
image + 112 text ≈ $0.019), so the cost ladder is output-token driven:

| output size | pixels | output tokens | cost (measured) | latency |
|---|---:|---:|---:|---:|
| 2240×3360 | 7.53 MP | 3659 | $0.1288 | 62 s |
| 1664×2496 | 4.15 MP | 2363 | $0.0899 | 64 s |
| 1280×1920 | 2.46 MP | 1712 | $0.0704 | 53 s |
| 1024×1536 | 1.57 MP | 1372 | $0.0602 | 56 s |
| 960×1440 | 1.38 MP | 1299 | $0.0580 | 58 s |
| 832×1248 | 1.04 MP | 1167 | $0.0540 | 53 s |
| 768×1152 | 0.88 MP | 1108 | $0.0522 | 54 s |
| 672×1008 | 0.68 MP | 1029 | $0.0499 | 59 s |

Side-by-side comparison sheets (all sizes scaled to the same height, labelled
with size + cost): `output/page9_atlas_medium_sizes_compare.jpg` (4 sizes)
and `output/page9_atlas_medium_sizes_compare_v2.jpg` (all 8 sizes); full-res
PNGs in each run dir. The input-cost floor (~$0.02) means halving the output
edge only halves the *output* component. The smaller runs show **diminishing
returns**: below 1024×1536 the output-token count plateaus (1372 → 1029
tokens from 1.57 → 0.68 MP) and cost only drops $0.060 → $0.050, while
resolution visibly suffers — the last four sizes cost ~90% of 1024×1536 for
less than half the pixels. Practical sweet spot: 1024×1536 ($0.060). Quality
differences are for the reader to judge visually; see the sheets.

#### Volume-1 projection (187 pages)

Volume 1 (`data/page_per_volume/… v01 …/`) has 187 pages, every one
1500×2250 — the same dimensions as p009, so the measured per-page cost
transfers directly. High-res poster with all 8 sizes + B&W, labelled with
cost/page and volume cost: `output/page9_atlas_medium_sizes_compare_v3_volume1cost.jpg`.

| output size | cost/page (measured p009) | volume 1 (187 pages) |
|---|---:|---:|
| 2240×3360 | $0.1288 | $24.08 |
| 1664×2496 | $0.0899 | $16.81 |
| 1280×1920 | $0.0704 | $13.16 |
| 1024×1536 | $0.0602 | $11.25 |
| 960×1440 | $0.0580 | $10.84 |
| 832×1248 | $0.0540 | $10.10 |
| 768×1152 | $0.0522 | $9.77 |
| 672×1008 | $0.0499 | $9.32 |

Estimate: per-page cost measured on p009 (atlas of the 4 party members +
`data/atlas/prompt.txt`); assumes other pages produce similar output tokens.
The input component (~$0.019/page) is fixed across sizes and dominates the
cheapest sizes, so volume cost compresses from $24.08 (7.53 MP output) to
$9.32 (0.68 MP) — only 2.6× cheaper for 11× fewer pixels.

#### All pages in `data/pages/` at the smallest size (20260815-093857 … 094401)

The user picked 672×1008 as the sweet spot; applied the atlas method to every
page in `data/pages/` (chapter 1 pages). Same atlas (frieren/himmel/heiter/
eisen), same `data/atlas/prompt.txt`, `medium` quality. The spread
p004-p005 (3000×2250, 4:3) uses the aspect-matched smallest valid size
960×720 (691,200 px, just above the 655,360 px floor) instead of 672×1008,
which would have squeezed it into 2:3.

| page | input | output size | cost (measured) | run dir |
|---|---|---|---:|---|
| p004-p005 (spread) | 3000×2250 | 960×720 | $0.0532 | 20260815-093857 |
| p006 | 1500×2250 | 672×1008 | $0.0499 | 20260815-093957 |
| p007 | 1500×2250 | 672×1008 | $0.0499 | 20260815-094048 |
| p008 | 1500×2250 | 672×1008 | $0.0499 | 20260815-094146 |
| p009 | 1500×2250 | 672×1008 | $0.0499 | 20260815-094303 |
| p010 | 1500×2250 | 672×1008 | $0.0499 | 20260815-094401 |

All 6 pages together: **$0.303**. p009 was re-run for a complete,
self-contained batch (identical inputs to 20260815-092433, same cost).
Grid overview (all pages at the same visual scale, labelled with size +
cost): `output/pages_all_small_grid.jpg`; full-res PNGs in each run dir.

## Conventions

Same as the repo's methods: each run creates a fresh timestamped directory,
never overwrites a previous run, and records a manifest with input files,
configuration, and cost at run time.
