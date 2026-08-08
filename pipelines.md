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
- **Debug view (bbox + reading order + detected characters on the stitched
  page):** [`annotated_colorized/0134-004.png`](pipeline_v1/output/20260808-221331/annotated_colorized/0134-004.png)
- **Cost:** $0.00040965 OpenRouter (5 calls, all ok) + $0 FLUX (self-hosted);
  wall time 75 s

### Volume 1, chapter 1 (5 pages)

- **Run:** [`pipeline_v1/output/20260808-223234/`](pipeline_v1/output/20260808-223234/)
- **Input:** `data/page_per_volume/… v01 …/`, pages p003–p008 (1500×2250,
  incl. a 3000×2250 spread), `--skip-first 3 --limit 5` — 5 pages, 18 panels
- **Debug views (bbox + reading order + detected characters on the stitched
  pages):** [`annotated_colorized/`](pipeline_v1/output/20260808-223234/annotated_colorized/)
  (p003, p004-p005, p006, p007, p008)
- **Cost:** $0.00137551 OpenRouter (18 calls, all ok) + $0 FLUX (self-hosted);
  wall time 358 s
- **Note:** p006 (a full-page illustration, ~2% ink, no panel frames) is
  detected as 0 panels and stays black & white; the p004-p005 spread is
  detected as a single large panel (2896×2256).

## What works

- **Panel detection** — the YOLO26n detector finds the panel layout reliably
  and the reading-order numbering matches how the pages read (verified by eye
  on 0134-004 and on the volume pages).
- **Character detection** — per-panel character lists are sensible and match
  the story context (e.g. p003 detects Frieren/Fern/Himmel/Heiter/Sein/Stark,
  p007 detects Eisen/Heiter/Himmel/Frieren).
- **Colorization** — the distilled 9B + LoRA at 4 steps produces coherent
  colors per panel; each panel keeps its own composition and linework. It is
  also fast: 3.8–13.5 s per panel at chapter size, 71.7 s for the huge spread.
- **Stitching** — the colorized panels are pasted back at the right positions
  and the rest of the page (gutters, margins, text outside panels) stays
  untouched.

## What to improve

### Character detection misses characters

On chapter 134 some characters appearing in panels are **not in the reference
set** (`data/refs/` has only 17 characters), so they can never be detected.
The character detection is limited by the reference list.

Ideas:
- Use a complete character list for the series:
  <https://frieren.fandom.com/wiki/Category:Characters>
- Download the wiki pages describing each chapter and use them as hints for
  which characters appear where.

### Reference images have no colors

The wiki character images are mostly line art / colorless, so the atlas does
not actually tell the model the character's colors.

Idea:
- Colorize the references myself once (fix the canonical colors per character)
  so the atlas carries real color information.

### Cross-panel consistency

Colors are chosen panel by panel; a character can end up with slightly
different colors in different panels of the same page (or across pages).

Ideas:
- Add a second pass at the **page level** that harmonizes the colorized
  panels.
- Add the previously colorized panels to the model context (sequential
  continuity like the whole-page methods).

### Correctly detected character, wrong colors

Even when the character is correctly detected and its atlas entry is passed,
the model does not always apply the right colors to it.

Idea:
- Add a verification agent that looks at the colorized panel + the atlas and
  checks that each detected character received its canonical colors, and
  re-colorizes when it did not.
