# pipeline_v1 — first real evaluation

Hands-on review of the first real runs of the panel-wise colorization pipeline
([`pipeline_v1/`](pipeline_v1/README.md)): panels are detected with YOLO26n,
extracted in Japanese reading order, characters are detected per panel with
OpenRouter `google/gemma-4-31b-it`, each panel is colorized with the
step-distilled FLUX.2 Klein 9B + manga-colorization LoRA (4 steps) using an
atlas filtered to the detected characters, and the colorized panels are
stitched back onto the page.

## Experiments

### Chapter 134 (smoke test)

- **Run:** [`pipeline_v1/output/20260808-221331/`](pipeline_v1/output/20260808-221331/)
- **Input:** `data/chapter_134/0134-004.png` (1200×1800) — 1 page, 5 panels
- **Cost:** $0.00040965 OpenRouter (5 calls, all ok) + $0 FLUX (self-hosted);
  wall time 75 s

Debug view (bbox + reading order + detected characters on the colorized
stitched page):

![0134-004 debug](docs/pipeline_v1/ch134-0134-004-annotated.png)

### Volume 1, chapter 1 (5 pages)

- **Run:** [`pipeline_v1/output/20260808-223234/`](pipeline_v1/output/20260808-223234/)
- **Input:** `data/page_per_volume/… v01 …/`, pages p003–p008 (1500×2250,
  incl. a 3000×2250 spread), `--skip-first 3 --limit 5` — 5 pages, 18 panels
- **Cost:** $0.00137551 OpenRouter (18 calls, all ok) + $0 FLUX (self-hosted);
  wall time 358 s
- **Note:** p006 (a full-page illustration, ~2% ink, no panel frames) is
  detected as 0 panels and stays black & white; the p004-p005 spread is
  detected as a single large panel (2896×2256).

Debug views (bbox + reading order + detected characters on the colorized
stitched pages):

| p003 (6 panels) | p004-p005 (1 panel) | p006 (0 panels) | p007 (7 panels) | p008 (4 panels) |
|---|---|---|---|---|
| ![p003 debug](docs/pipeline_v1/vol1-p003.png) | ![p004-p005 debug](docs/pipeline_v1/vol1-p004-p005.png) | ![p006 debug](docs/pipeline_v1/vol1-p006.png) | ![p007 debug](docs/pipeline_v1/vol1-p007.png) | ![p008 debug](docs/pipeline_v1/vol1-p008.png) |

## What works

- **Panel detection** — the YOLO26n detector finds the panel layout reliably
  and the reading-order numbering matches how the pages read (verified by eye
  on 0134-004 and on the volume pages).
- **Character detection** — the calls are cheap and often identify distinctive
  characters correctly, but the lists do not always match the story context.
  For example, p003 panel 2 reports Fern/Stark during the hero-party flashback,
  and p008 panel 3 reports Sein for Heiter.
- **Colorization** — the distilled 9B + LoRA at 4 steps produces coherent
  colors per panel; each panel keeps its own composition and linework. It is
  also fast: 3.8–13.5 s per panel at chapter size, 71.7 s for the huge spread.
- **Stitching** — the colorized panels are pasted back at the right positions
  and the rest of the page (gutters, margins, text outside panels) stays
  untouched.

## What to improve

### Character detection misses or confuses characters

On chapter 134 some characters appearing in panels are **not in the reference
set** (`data/refs/` has only 17 canonical characters), so they can never be
detected. Even characters that are present in the reference set can be confused
when a crop removes page-level story context, as seen with Heiter/Sein and with
the chapter-1 flashback cast.

Ideas:
- Keep a complete character list for the series, but pass only a cached
  chapter-specific cast shortlist to each detection request:
  <https://frieren.fandom.com/wiki/Category:Characters>
- Detect all numbered panels in one page-level call so the VLM can use dialogue
  and neighboring panels as context; fall back to a per-panel call only for an
  uncertain result.
- Cache chapter descriptions or manually curated cast metadata as optional
  hints rather than downloading them on every run.

### Colored references are not followed reliably

The 18 files in `data/refs/` represent 17 canonical characters (Frieren has two
filenames) and are already colored anime character sheets. The atlas therefore
does carry canonical colors. The observed wrong colors — for example Frieren's
silver hair becoming magenta — are failures to follow the reference, not missing
reference information.

The implementation currently passes detected names to the atlas builder but not
to the FLUX text prompt. The prompt only asks the model to use the labelled atlas,
so the four-step model must read the labels, associate each sheet with a manga
figure, and infer all of the colors from the image.

Ideas:
- Store explicit canonical color profiles (hair, eyes, clothing, accessories,
  and outfit/era variants) next to the references and inject the detected
  characters' profiles directly into the FLUX prompt.
- Keep the colored atlas as visual evidence, but do not rely on its labels as
  the only connection between a character name and its palette.
- Compare prompt/atlas variants on a small fixed set of failing panels with a
  fixed seed before running another volume sample.

### Cross-panel consistency

Colors are chosen panel by panel; a character can end up with slightly
different colors in different panels of the same page (or across pages).

Ideas:
- First enforce the same explicit canonical color profiles in every panel and
  use fixed seeds for reproducible comparisons.
- Defer a generative page-level harmonization pass until prompt-level palette
  conditioning has been measured; an extra pass can redraw linework or propagate
  an already-wrong palette.

### Correctly detected character, wrong colors

Even when the character is correctly detected and its atlas entry is passed,
the model does not always apply the right colors to it.

Idea:
- If prompt-level conditioning is still insufficient, verify a contact sheet of
  all colorized panels once per page and re-colorize only flagged panels, with a
  one-retry limit. Avoid one paid verification call per panel.
