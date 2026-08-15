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

---

## V1.1 — fixed evaluation set, explicit palettes, page-level detection

V1.1 (epic 002, tasks 0001–0004) targets the V1 failures: character confusion
(Heiter/Sein, flashback cast), out-of-vocabulary identity (Clematis), palette
adherence (Frieren's magenta hair), the p006 zero-panel page, and the oversized
spread. It adds a fixed evaluation contract
([`pipeline_v1/evaluation/v1_1_cases.json`](pipeline_v1/evaluation/v1_1_cases.json))
and the real-network **integration suite** (`pytest pipeline_v1/tests -m integration`,
no mocks) that runs the fixed set live against the real backends; the earlier
`evaluate.py` CLI (auto-scored detection + human `color_review.md`) has been
retired in its favour. On top of the stage-isolated DET/OOV/COL/SIZE/LAY
tests, a full end-to-end integration test (`test_end_to_end_integration.py`,
case E2E-P130) runs the entire pipeline — real YOLO + real OpenRouter
`panel-page-prev2-cast` detection (prev2 with the chapter-cast shortlist) +
real FLUX on Spark + stitching — on volume-1 p130
(the DET-005..010 page) and asserts the wiring end to end: panel crops
reproduce the committed fixture set byte-for-byte, every panel gets a
character record and a colorized output, and the stitch preserves the B&W
gutters. All V1.1 runs used a fixed seed
(`1337`); V1 used the default (random) seed.

Changed settings vs V1: **per-page character detection** (one paid call per
page + cropped fallbacks) instead of one call per panel; **explicit canonical
palettes** injected into the FLUX prompt (`character_profiles.json`);
**full-page fallback** for zero-panel pages; **2.0 MP request cap** for
oversized inputs; `max_tokens` 2048 → 1024; configurable inter-call delay.

### Runs (2026-08-09, all self-hosted FLUX, $0/call + electricity)

| Run | Input | Detection | Colorization | OpenRouter cost | Wall time |
|---|---|---|---|---|---|
| Fixed COL experiment ([20260809-005032](pipeline_v1/output/20260809-005032/)) | P003/P007/P008 COL panels, forced identities | 0 calls, $0 | 3 FLUX | $0 | 36.6 s |
| Volume 1, ch.1 5 pages ([20260809-010458](pipeline_v1/output/20260809-010458/)) | p003–p008, `--skip-first 3 --limit 5`, cast `c001` | 5 page calls + 0 fallbacks | 19 FLUX (incl. p006 fallback) | **$0.00050690** | 276.6 s |
| Chapter 134 smoke ([20260809-011110](pipeline_v1/output/20260809-011110/)) | 0134-004, cast `ch134` | 1 page call | 5 FLUX | $0.00014994 | 50.7 s |
| Full-roster detection ([20260809-013917](pipeline_v1/output/20260809-013917/)) | P003+P008 page-level, **no** cast shortlist | 2 page calls | 0 FLUX | $0.00032604 | — |
| Layout run ([20260809-014132](pipeline_v1/output/20260809-014132/)) | p006 + generated blank page | 0 calls | 1 FLUX (p006 only) | $0 | — |
| Spread cap A/B @3.5 MP ([20260809-014331](pipeline_v1/output/20260809-014331/)) | spread panel, seed 1337 | 0 | 1 FLUX | $0 | — |

### Character detection — page-level, measured on the task-0001 ground truth

Detection is scored automatically (set comparison, exact TP/FP/FN; the
evaluator never scores a forced ground-truth record as a detection):

| Case | V1 (per-panel) | V1.1 (page-level, shortlist `c001`) | V1.1 (page-level, full roster) |
|---|---|---|---|
| DET-001 (P003:2, expected Frieren,Himmel) | [Fern, Stark] (tp0 fp2 fn2) | [Himmel] (tp1 fp0 fn1) | [Frieren] (tp1 fp0 fn1) |
| DET-002 (P003:4, expected Frieren,Himmel,Heiter,Eisen) | [Sein] (tp0 fp1 fn4) | [Frieren,Heiter,Himmel] (tp3 fp0 fn1) | [Frieren] (tp1 fp0 fn3) |
| DET-003 (P008:3, expected Heiter) | [Sein] ✗ | [Heiter] ✓ | [Heiter] ✓ |
| DET-004 (P008:4, expected Frieren,Heiter) | [Frieren,Sein] (tp1 fp1 fn1) | [Heiter,Frieren] ✓ | [Heiter,Frieren] ✓ |
| **Precision / recall** | **0.167 / 0.111** | **1.000 / 0.778** | **1.000 / 0.778** |

The Heiter/Sein confusion is fixed by page-level context alone (the full-roster
run scores identically to the shortlist run — the shortlist is not the reason).
Remaining detection misses are era/under-detection: DET-001/002 still miss a
hero-party member (Himmel on P003:2, Eisen on P003:4) because the model
interprets the flashback cast inconsistently.

OOV-001 (chapter 134, panel 3) **still fails**: Clematis is not in the 17-name
reference vocabulary and the model forced her to **Wirbel** instead of leaving
her unknown. The page-level prompt instructs unknown characters to be reported
as `uncertain`, but the model does not comply reliably.

Run 20260809-091129 (panel-page detection over volume 1) exposed a second
look-alike confusion: on **p130** (ch. 5 "Killing Magic") 4 of 6 panels were
detected as **Flamme** — Frieren's look-alike master, who is not in chapter 5's
cast — where the cast is Frieren/Fern (panel 4 even kept Frieren but forced
Fern to Flamme). Recorded as the six-panel page set **DET-005..010** in the
evaluation fixture, with the observed misdetections as per-case baselines.

### Detection model sweep — panel-only mode (2026-08-09, live OpenRouter)

Three models × 4 reps over all 11 DET/OOV cases on the same committed
panel-only crops and V1 panel prompt as `test_integration_detection.py`
(temperature 0.2). Script: `pipeline_v1/tests/sweep_detection_models.py`
(`--re-render` regenerates the tables from a saved summary); per-call records
in `pipeline_v1/tests/output/20260809-204754/`. Total cost of the sweep:
$0.10. Pass = exact character set + `unknown_present` semantics.

| Case | expected | google/gemma-4-31b-it | openai/gpt-5.6-luna | xiaomi/mimo-v2.5 |
|---|---|---|---|---|
| DET-001 | Frieren, Himmel | 0/4 | 1/4 | 0/4 (3 parse-fail) |
| DET-002 | Frieren, Himmel, Heiter, Eisen | 0/4 | 2/4 (1 parse-fail) | 0/4 (4 parse-fail) |
| DET-003 | Heiter | 3/4 | 0/4 | 2/4 (1 parse-fail) |
| DET-004 | Frieren, Heiter | 1/4 | 0/4 (3 parse-fail) | 0/4 (4 parse-fail) |
| OOV-001 | — | 0/4 | 0/4 | 0/4 (4 parse-fail) |
| DET-005 | — | 4/4 | 4/4 | 4/4 |
| DET-006 | Frieren | 4/4 | 2/4 | 4/4 |
| DET-007 | Frieren | 4/4 | 4/4 | 4/4 |
| DET-008 | Frieren, Fern | 0/4 | 2/4 (1 parse-fail) | 0/4 (4 parse-fail) |
| DET-009 | Frieren, Fern | 2/4 | 2/4 (1 parse-fail) | 1/4 (3 parse-fail) |
| DET-010 | Frieren | 0/4 | 0/4 | 3/4 (1 parse-fail) |
| **Total** | | **18/44** | **17/44** | **18/44** |

Modal (most frequent) detection per model over the 4 reps:

| Case | expected | gemma-4-31b-it | gpt-5.6-luna | mimo-v2.5 |
|---|---|---|---|---|
| DET-001 | Frieren, Himmel | Fern, Stark (4/4) | Frieren, Wirbel (1/4) | ∅ (3/4) |
| DET-002 | Frieren, Himmel, Heiter, Eisen | Heiter, Himmel (2/4) | Eisen, Frieren, Heiter, Himmel (2/4) | ∅ (4/4) |
| DET-003 | Heiter | Heiter (3/4) | Denken (4/4) | Heiter (2/4) |
| DET-004 | Frieren, Heiter | Frieren (2/4) | ∅ (3/4) | ∅ (4/4) |
| OOV-001 | — | Heiter (4/4) | Heiter (3/4) | ∅ (4/4) |
| DET-005 | — | ∅ (4/4) | ∅ (4/4) | ∅ (4/4) |
| DET-006 | Frieren | Frieren (4/4) | Frieren (2/4) | Frieren (4/4) |
| DET-007 | Frieren | Frieren (4/4) | Frieren (4/4) | Frieren (4/4) |
| DET-008 | Frieren, Fern | Frieren, Serie (3/4) | Fern, Frieren (2/4) | ∅ (4/4) |
| DET-009 | Frieren, Fern | Fern, Frieren (2/4) | Fern, Frieren (2/4) | ∅ (3/4) |
| DET-010 | Frieren | Fern (2/4) | Serie (4/4) | Frieren (3/4) |

Aggregates over all case-reps (TP/FP/FN on known characters):

| model | pass | precision | recall | parse-fail | cost | avg latency |
|---|---|---|---|---|---|---|
| google/gemma-4-31b-it | 18/44 (41%) | 0.574 | 0.547 | 0 | $0.0031 | 2.8 s |
| openai/gpt-5.6-luna | 17/44 (39%) | 0.611 | 0.516 | 6 | $0.0135 | 6.0 s |
| xiaomi/mimo-v2.5 | 18/44 (41%) | 0.889 | 0.250 | 24 | $0.0838 | 13.3 s |

Findings:

- **Pass rates are tied (18/17/18 of 44), failure modes are not.** gemma is
  deterministic and stable on the p130 panels (DET-005/006/007 4/4) and
  cheapest (≈4× less than luna, ≈27× less than mimo per call). luna is the
  only model that sometimes produces the full hero party on DET-002 (modal
  `{Eisen, Frieren, Heiter, Himmel}` 2/4) but it fails the single-character
  Heiter close-up (DET-003: Denken 4/4) that gemma passes 3/4.
- **mimo-v2.5 is unusable in panel-only mode**: 24/44 calls returned
  unparseable JSON (it rarely emits the strict `{"characters": [...]}`
  format). Its apparent 0.889 precision is refusal-to-guess (modal ∅ on most
  cases) at the cost of the worst recall (0.250) and highest price/latency.
- **p130 stays hard for all three in panel-only mode** — and the ground truth
  holds up: a neutral free-form VLM description of the DET-008/010 crops
  (dark-haired girl + hexagonal barrier = Fern; adult white-haired elf with
  pointed ears = Frieren) confirms the fixture. The confusions
  Fern→Serie/Flamme and Frieren→Fern/Aura/Serie are look-alike errors driven
  by the manga's dark Fern hair vs the pale anime Fern in `data/refs/`.
- **OOV-001 remains unsolved by every model**: gemma/luna force Heiter/Wirbel
  instead of reporting the unknown Clematis; mimo returns ∅/unparseable,
  which fails the `unknown_present` assertion too.
- **Verdict**: keep `google/gemma-4-31b-it` as the production detector
  (equal pass rate, best price/latency/stability). luna is worth re-testing
  with page-level context for the flashback-era cases; mimo-v2.5 is out.

### Detection mode sweep — panel vs page vs panel-page (2026-08-09, live OpenRouter)

The follow-up sweep runs the two viable models (gemma-4-31b-it, gpt-5.6-luna)
× the pipeline's three detection modes × 4 reps over the same 11 DET/OOV
cases: panel (crop only), page (one call per page, numbered panels),
panel-page (one call per panel, full page highlighted + crop). Per-call
records: `pipeline_v1/tests/output/20260809-211355/`; total sweep cost
$0.056. Pass = exact character set + `unknown_present` semantics.

| Case | expected | panel·gemma | panel·luna | page·gemma | page·luna | panel-page·gemma | panel-page·luna |
|---|---|---|---|---|---|---|---|
| DET-001 | Frieren, Himmel | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 | 3/4 |
| DET-002 | Frieren, Himmel, Heiter, Eisen | 0/4 | 2/4 | 0/4 | 4/4 | 4/4 | 4/4 |
| DET-003 | Heiter | 2/4 | 1/4 | 4/4 | 0/4 (4 fbk) | 4/4 | 4/4 |
| DET-004 | Frieren, Heiter | 1/4 | 0/4 | 4/4 | 2/4 (4 fbk) | 4/4 | 2/4 (1 fbk) |
| OOV-001 | — | 0/4 | 0/4 | 0/4 | 0/4 (4 fbk) | 0/4 | 0/4 |
| DET-005 | — | 4/4 | 4/4 | 4/4 | 4/4 (2 fbk) | 4/4 | 4/4 |
| DET-006 | Frieren | 4/4 | 3/4 | 4/4 | 4/4 (2 fbk) | 4/4 | 4/4 |
| DET-007 | Frieren | 3/4 | 4/4 | 4/4 | 4/4 (2 fbk) | 4/4 | 4/4 |
| DET-008 | Frieren, Fern | 1/4 | 3/4 | 0/4 | 2/4 (2 fbk) | 3/4 | 1/4 (3 fbk) |
| DET-009 | Frieren, Fern | 1/4 | 3/4 | 0/4 | 3/4 (2 fbk) | 4/4 | 4/4 |
| DET-010 | Frieren | 3/4 | 0/4 | 4/4 | 2/4 (2 fbk) | 3/4 | 4/4 |
| **Total** | | **19/44** | **20/44** | **24/44** | **29/44** | **38/44** | **34/44** |

Modal (most frequent) detection per mode/model over the 4 reps:

| Case | expected | panel·gemma | panel·luna | page·gemma | page·luna | panel-page·gemma | panel-page·luna |
|---|---|---|---|---|---|---|---|
| DET-001 | Frieren, Himmel | Fern, Stark (3/4) | Uebel (1/4) | Himmel (4/4) | Frieren, Himmel (4/4) | Frieren, Himmel (4/4) | Frieren, Himmel (3/4) |
| DET-002 | Frieren, Himmel, Heiter, Eisen | Heiter, Himmel (2/4) | Eisen, Frieren, Heiter, Himmel (2/4) | Frieren, Heiter, Himmel (4/4) | Eisen, Frieren, Heiter, Himmel (4/4) | Eisen, Frieren, Heiter, Himmel (4/4) | Eisen, Frieren, Heiter, Himmel (4/4) |
| DET-003 | Heiter | Heiter (2/4) | Denken (3/4) | Heiter (4/4) | Denken (3/4) | Heiter (4/4) | Heiter (4/4) |
| DET-004 | Frieren, Heiter | Frieren (2/4) | Flamme, Heiter (2/4) | Frieren, Heiter (4/4) | Frieren, Heiter (2/4) | Frieren, Heiter (4/4) | Frieren, Heiter (2/4) |
| OOV-001 | — | Heiter (2/4) | Heiter (3/4) | Denken (3/4) | Wirbel (2/4) | Denken (4/4) | Wirbel (4/4) |
| DET-005 | — | ∅ (4/4) | ∅ (4/4) | ∅ (4/4) | ∅ (4/4) | ∅ (4/4) | ∅ (4/4) |
| DET-006 | Frieren | Frieren (4/4) | Frieren (3/4) | Frieren (4/4) | Frieren (4/4) | Frieren (4/4) | Frieren (4/4) |
| DET-007 | Frieren | Frieren (3/4) | Frieren (4/4) | Frieren (4/4) | Frieren (4/4) | Frieren (4/4) | Frieren (4/4) |
| DET-008 | Frieren, Fern | Aura, Frieren (2/4) | Fern, Frieren (3/4) | Frieren (4/4) | Frieren (2/4) | Fern, Frieren (3/4) | ∅ (3/4) |
| DET-009 | Frieren, Fern | Aura, Frieren (3/4) | Fern, Frieren (3/4) | Frieren (4/4) | Fern, Frieren (3/4) | Fern, Frieren (4/4) | Fern, Frieren (4/4) |
| DET-010 | Frieren | Frieren (3/4) | Serie (4/4) | Frieren (4/4) | Serie (2/4) | Frieren (3/4) | Frieren (4/4) |

Aggregates over all case-reps (TP/FP/FN on known characters; fallbacks =
reps resolved by a cropped-panel call; parse-fail = unparseable/error calls
or unparsed page answers; page-mode cost/latency shared across the page's
panels):

| mode · model | pass | precision | recall | fallbacks | parse-fail | cost | avg latency |
|---|---|---|---|---|---|---|---|
| panel · gemma-4-31b-it | 19/44 (43%) | 0.623 | 0.594 | 0 | 1 | $0.0031 | 2.9 s |
| panel · gpt-5.6-luna | 20/44 (45%) | 0.649 | 0.578 | 0 | 4 | $0.0123 | 5.6 s |
| page · gemma-4-31b-it | 24/44 (55%) | 0.906 | 0.750 | 0 | 0 | $0.0009 | 2.1 s |
| page · gpt-5.6-luna | 29/44 (66%) | 0.836 | 0.797 | 24 | 24 | $0.0178 | 7.8 s |
| panel-page · gemma-4-31b-it | 38/44 (86%) | 0.912 | 0.969 | 0 | 0 | $0.0052 | 5.5 s |
| panel-page · gpt-5.6-luna | 34/44 (77%) | 0.900 | 0.844 | 4 | 4 | $0.0162 | 8.0 s |

Findings:

- **The mode matters far more than the model.** panel-page (V1.2) nearly
  doubles panel-only pass rates for both models (gemma 43% → 86%, luna 45% →
  77%). Page context resolves the flashback-era confusion outright:
  panel-page·gemma gets DET-001 and DET-002 4/4 (hero party complete) and
  DET-003/004 (Heiter) 4/4 — the exact failures panel-only mode tracks.
- **gemma + panel-page is the best combination**: 38/44, precision 0.912,
  recall 0.969, zero fallbacks and zero parse-fails, ~$0.005 per 44
  case-reps. It fixes the p130 look-alike cases too: DET-009 4/4, DET-008
  3/4, DET-010 3/4 (panel-only modal for DET-008/009 was even `{Aura,
  Frieren}` — a demon forced into a Frieren/Fern panel).
- **page mode is a distant second and luna's 66% is misleading**: 24/44 of
  luna's page-mode case-reps came from cropped-panel fallbacks because its
  page-level answers fail the strict per-panel JSON format (24 parse-fail; 6
  of 16 page calls unparsed). When luna's page answer does parse it is
  excellent (DET-001/002 4/4), but it cannot be relied on. gemma's page
  mode parsed 44/44 and is the cheapest (one call serves the whole page) but
  under-detects on numbered-page-only prompts (DET-001 modal `{Himmel}`,
  DET-002 modal `{Frieren, Heiter, Himmel}` — misses a hero-party member).
- **OOV-001 stays unsolved in every mode/model**: Clematis is forced to
  Denken/Wirbel/Heiter and never reported unknown.
- **Default adopted**: the pipeline defaulted to `panel-page` with
  `google/gemma-4-31b-it` — panel-page at ~$0.0001/
  case-rep and ~5 s per panel is worth ~5× a page call, and the
  `panel-page-cast` mode additionally auto-restricts each page's prompt to
  its chapter cast (via `chapter_page_map.json`), which excludes look-alikes
  outside the chapter (e.g. Flamme is not in ch. 5's cast and could not be
  guessed on p130). luna remains the fallback candidate (34/44) if gemma
  regresses. The default has since moved to **`panel-page-prev2-cast`**
  (panel-page + the two preceding pages as story context + the per-chapter
  cast shortlist).

**panel-page-cast follow-up** (2026-08-09, gemma only, 4 reps, records in
`pipeline_v1/tests/output/20260809-222825/`): the new auto-cast mode scores
**39/44 (89%)** — one point above plain panel-page, and it directly kills the
Flamme failure from run 20260809-091129: DET-008 and DET-010 (p130 panels 4
and 6, where that run answered `Flamme`) go from 3/4 to **4/4** because
ch. 5's shortlist (Frieren, Fern, Himmel, Eisen, Heiter) excludes Flamme.
No Flamme/Serie/Aura/Denken appears in any rep; the only remaining miss on
p130 is one DET-007 rep read as Fern. OOV-001 stays unsolved — with the
c134 shortlist (Stark, Frieren, Fern) the model now forces **Stark** instead
of Denken/Wirbel, i.e. casting channels the guess into the shortlist's
closest member; the unknown-identity contract is untouched by casting.

| Case | expected | panel-page·gemma | panel-page-cast·gemma |
|---|---|---|---|
| DET-001 | Frieren, Himmel | 4/4 | 4/4 |
| DET-002 | Frieren, Himmel, Heiter, Eisen | 4/4 | 4/4 |
| DET-003 | Heiter | 4/4 | 4/4 |
| DET-004 | Frieren, Heiter | 4/4 | 4/4 |
| OOV-001 | — | 0/4 | 0/4 |
| DET-005 | — | 4/4 | 4/4 |
| DET-006 | Frieren | 4/4 | 4/4 |
| DET-007 | Frieren | 4/4 | 3/4 |
| DET-008 | Frieren, Fern | 3/4 | 4/4 |
| DET-009 | Frieren, Fern | 4/4 | 4/4 |
| DET-010 | Frieren | 3/4 | 4/4 |
| **Total** | | **38/44** | **39/44** |

**Temperature-0 follow-up** (2026-08-09, gemma, 4 reps, records in
`pipeline_v1/tests/output/20260809-232133/`): with detection temperature set
from 0.2 → **0.0**, plain panel-page scores **40/44 (91%)** — the best result
of any mode/model so far: all 10 character cases pass 4/4 (DET-008/010, which
still flipped at 0.2, are now deterministic), precision 0.941, **recall 1.0**
(zero false negatives over 44 case-reps), zero fallbacks, zero parse-fails,
$0.0042. Sampling noise, not mode weakness, was the remaining cause of the
DET-007/008/010 misses at 0.2. OOV-001 remains the sole failure (Denken
forced 4/4). Sweep runs are now threaded (`--workers`, default 8): the 4-rep
panel-page sweep dropped from ~15 min sequential to ~4.5 min at 16 workers.

**Concurrency finding** (2026-08-09): OpenRouter's gemma-4-31b-it is
**non-deterministic under concurrent load even at temperature 0**. The same
panel-page request on p130 panel 6 answered Frieren 14/14 sequentially but
flipped to Frieren 9, **Flamme 3**, Fern 3, Aura 1 under 16 concurrent calls
— which is exactly why a `--workers 16` pipeline run (20260809-232836, plain
panel-page, full roster) produced 44 Flamme false positives across ch2–7
while the low-concurrency sweeps saw none. `--workers 1` is stable;
**panel-page-cast confines the damage**: a fully parallel re-run
(`--parallel-reps --workers 16`, records in
`pipeline_v1/tests/output/20260810-002338/`) still scores **40/44 (91%)**
with **zero Flamme/Aura/Serie/Denken** across all 44 case-reps — the chapter
shortlist makes out-of-cast look-alikes un-answerable regardless of the
flip noise. The annotated-page write in `_annotated_page` is now atomic
(temp + rename) so concurrent same-page re-renders can't expose torn PNGs.

### Color — explicit palettes (human review pending)

The three COL cases were run with forced ground-truth identities (Run
20260809-005032, 0 paid detection calls) and also produced in the volume run
with the detected identities. The generated image, monochrome input, atlas and
fixture expectation for every variant are in the run's
[`evaluation/color_review.md`](pipeline_v1/output/20260809-005032/evaluation/color_review.md);
verdicts are **pending user review** by design (no automated color verdict).
Objective signals only: the dominant saturated color in all three outputs is
warm gold/white (no magenta-dominant color), and COL-002's hair region is very
light — consistent with the injected "silver-white hair" palette, but this is
not a pass/fail judgement.

The FLUX prompt now contains explicit canonical colors, e.g. for COL-001/002:
`Frieren: silver-white hair; green eyes; white coat with gold trim.` recorded
with the profile hash in the manifest (`palette_instruction`,
`profiles_sha256`). `--force-characters` made the fixed experiment free of paid
detection calls.

### Full-page art and oversized inputs

- **LAY-001** ✓ — p006 (full-page illustration, no panel frames) now produces
  one synthetic full-page panel (`provenance: full-page-fallback`), is
  colorized (capped 1152×1728, 22.3 s) and stitched back to exactly 1500×2250.
  Previously the page stayed black & white. See
  [v11-p006-fullpage.png](docs/pipeline_v1/v11-p006-fullpage.png).
- **LAY-002** ✓ — a generated all-white 1500×2250 page is skipped with an
  explicit `blank-page` record, zero character/FLUX calls, and an unchanged
  1500×2250 final page.
- **SIZE-001** ✓ — the 2895×2250 spread is capped to 1600×1248 (multiples of
  16, 1,996,800 px ≤ 2.0 MP; original/requested size, scale and applied cap
  recorded). The final stitched page stays exactly 3000×2250.

### Spread size-cap A/B (same seed 1337)

| Cap | Request size | Spread FLUX latency | vs V1 (2896×2256, 71.7 s) |
|---|---|---|---|
| 2.0 MP (default) | 1600×1248 | 28.9 s | −60% |
| 3.5 MP | 2112×1648 | 43.5 s | −39% |

Outputs for visual comparison:
[2.0 MP](docs/pipeline_v1/v11-spread-cap-2.0MP.png) ·
[3.5 MP](docs/pipeline_v1/v11-spread-cap-3.5MP.png). 2.0 MP stays the default
(SIZE-001 is the deterministic 2.0 MP fixture); the largest cap with no visible
regression is a user judgement pending on these two images.

### Cost summary

- V1 total (ch134 + 5 pages): **$0.00178516** OpenRouter.
- V1.1 total (all six runs above): **$0.00098288** — 45% less, and the
  5-page comparison alone is **$0.00050690** (2.7× under V1's $0.00137551 for
  the same input). Page-level detection used 5 paid calls instead of 18.
- FLUX stays $0/call (self-hosted); wall time for the 5-page run dropped from
  358 s to 276.6 s **while colorizing one more panel** (p006 fallback).

### Integration-suite evaluation (2026-08-09, live backends, seed 1337)

The evaluation of `evaluation/v1_1_cases.json` now runs as stage-isolated
real-network pytest (`-m integration`): committed inputs in
`pipeline_v1/tests/data/` (pre-cropped panels, plus committed pages and
complete per-page panel sets for the detection pages), real gemma-4-31b-it
detection, real FLUX on Spark + real gpt-5.6-luna color validation (one
generic strict structured-output verdict, see below), real YOLO;
one timestamped run per session in `pipeline_v1/tests/output/`.

**Suite restructured 2026-08-10:** the detection stage now runs **all five
detection modes** (`page`, `panel`, `panel-page`, `panel-page-cast`,
`panel-page-prev2`) as five parametrized tests over the same 11 DET/OOV
cases, consuming the committed per-page inputs — the tests call the real
detector functions directly and no longer run panel detection themselves. A
crop-stability tripwire in the layout stage re-extracts the committed pages
and asserts the crops still match byte-for-byte, so the eval cases' panel
references cannot silently go stale. `panel-page-prev2` (added 2026-08-14)
sends the two preceding pages as extra story-context images; its first live
verdicts are below. Per-mode verdicts for the other four modes come from the
4-rep sweeps above; the findings further down are the pre-restructure
baselines.

**Color verdict restructured 2026-08-14:** the color stage no longer has two
verifiers (palette adherence for COL-001..003, left-to-right geography for
COL-004). Every COL case is now judged by one generic `openai/gpt-5.6-luna`
prompt — "are all the characters' color palettes correct?" — answered as a
strict structured output (`analyse: str`, `good_color: bool`) via OpenRouter's
`json_schema` response_format with `provider.require_parameters: true`; no
fixture expectations are rendered into the prompt (they remain in
`v1_1_cases.json` as documentation). The verdict call also receives the
**reference atlas of the detected/forced characters** (the same contact sheet
the colorizer saw) as image #3, so luna checks each character's palette
against the characters that should appear rather than whoever it happens to
identify. The historical color verdicts below were produced with the previous
two-verifier scheme and are not directly comparable.

**panel-page-prev2 live run** (2026-08-14, 1 rep per case, temperature 0.0,
model chosen via `INTEGRATION_DETECTION_MODEL`; records:
[20260814-202749](pipeline_v1/tests/output/20260814-202749/) luna,
[20260814-203045](pipeline_v1/tests/output/20260814-203045/) gemma; total
OpenRouter cost $0.0149 + $0.0020): each call carries the numbered annotated
current page, the crop, and two preceding-page context images. The suite
lays out two preceding page dirs whose `page_path` reuses the case's own
committed page (committed inputs only, no fabricated pages — the
0/1-image degrade shapes are covered by the offline unit tests in
`test_characters.py`). Pass = exact character set + `unknown_present`.

| Case | expected | panel-page·gemma (4-rep) | prev2·gemma | prev2·luna | prev2-cast·gemma | prev2-cast·luna |
|---|---|---|---|---|---|---|
| DET-001 | Frieren, Himmel | 4/4 | ✓ `{Frieren, Himmel}` | ✗ `{Frieren, Stark}` | ✓ | ✓ |
| DET-002 | Frieren, Himmel, Heiter, Eisen | 4/4 | ✓ all four | ✓ all four | ✓ | ✓ |
| DET-003 | Heiter | 4/4 | ✓ | ✓ | ✓ | ✓ |
| DET-004 | Frieren, Heiter | 4/4 | ✓ | ✓ | ✓ | ✓ |
| OOV-001 | — (unknown) | 0/4 | ✗ `{Denken}` | ✗ `{Heiter}` | ✗ `{Wirbel}` | ✗ `{Himmel}` |
| DET-005 | — | 4/4 | ✓ ∅ | ✓ ∅ | ✓ ∅ | ✓ ∅ |
| DET-006 | Frieren | 4/4 | ✓ | ✓ | ✓ | ✓ |
| DET-007 | Frieren | 3/4 | ✓ | ✓ | ✓ | ✓ |
| DET-008 | Frieren, Fern | 3/4 | ✗ `{Frieren, Flamme}` | ✗ unparseable | ✓ `{Fern, Frieren}` | ✓ `{Fern, Frieren}` |
| DET-009 | Frieren, Fern | 4/4 | ✓ | ✓ | ✓ | ✓ |
| DET-010 | Frieren | 3/4 | ✓ | ✓ | ✓ | ✓ |
| **Pass** | | **38/44** | **9/11** | **8/11** | **10/11** | **10/11** |

Findings:

- **prev2 holds panel-page's level at one rep**: gemma 9/11, luna 8/11 —
  both models pass the flashback-era hero party (DET-002) and the p130
  look-alike page (DET-005..007, DET-009, DET-010), the cases panel-only
  mode fails; luna additionally misses DET-001 (Stark for Himmel).
- **prev2-cast (same mode, chapter shortlist rendered in the prompt) closes
  both remaining misses — 10/11 for both models** (sessions
  [20260814-204127](pipeline_v1/tests/output/20260814-204127/) luna,
  [20260814-204320](pipeline_v1/tests/output/20260814-204320/) gemma):
  luna's DET-001 (Stark) and both models' DET-008 (Flamme) pass because
  ch. 1/ch. 5 shortlists exclude Stark-era/out-of-cast guesses — same
  mechanism as the panel-page-cast follow-up above. The only remaining
  failure is **OOV-001**: Clematis is forced to Wirbel (gemma) / Himmel
  (luna) instead of reported unknown — the cast channels the guess into the
  shortlist's closest member, leaving the unknown-identity contract
  untouched.
- **Suite fix**: the cast tests crashed on `Path(None)` until
  `build_panel_detector` wired `chapter_casts_file` — the mode tests
  pass a cast key explicitly, so the file (never the auto-derivation) is
  required to render the shortlist; this also unblocks the pre-existing
  `panel-page-cast` integration test, whose live verdicts were still
  pending.
- **The two shared failures are the tracked ones**: OOV-001 (Clematis forced
  to Denken/Heiter, never reported unknown) and DET-008 (gemma answers
  Fern→Flamme; luna's prev2 answer fails the strict JSON format and its
  cropped-panel fallback comes back unparseable too). Same-page context
  images cannot rule out out-of-cast look-alikes — only the chapter
  shortlist (panel-page-cast / prev2-cast) excludes Flamme.
- **prev2 calls are the suite's most expensive per call** (~2× panel-page's
  token count for the two extra page images): luna ≈$0.0016–0.0019/call,
  gemma ≈$0.00014–0.00024/call.

Findings (measured `usage.cost` ≈ $0.0022/session, pre-restructure):

- **Detection (panel-only mode): 1/5 pass.** DET-003 (Heiter close-up) passes;
  DET-001 reproduces the exact V1 baseline `{Fern, Stark}` (flashback cast),
  DET-002 collapses to `{Fern, Heiter, Eisen}` (missing Himmel), DET-004
  misses Heiter, and OOV-001 forces Clematis to **Denken** (not reported
  unknown). Panel-only mode cannot meet the fixture's expectations for the
  flashback-era cases — the V1.1 page-level mode is why those were fixed
  before; the integration suite keeps the crop-only contract and tracks the
  failures. The fixture has since grown **DET-005..010** (volume-1 p130's six
  panels, Flamme/Frieren look-alike confusion observed in panel-page run
  20260809-091129); their panel-only verdicts are pending the next live run.
- **Color: 3/5 pass, with a confirmed real defect.** COL-003 (Heiter) passes.
  **COL-004 passes** — the live left-to-right palette is true at seed 1337
  (green / blue / pale white-lavender / golden yellow-brown), i.e. the V1.2
  palette-geography failure from run 20260809-091129 is gone with the current
  atlas + palette conditioning. COL-001 and COL-002 (Frieren, two crops) fail
  consistently: the hair renders **lavender-purple (forbidden)** instead of
  silver-white, eyes and white/gold outfit correct — confirmed by the real
  VLM, no longer "pending user review".
- **Layout 2/2, size 1/1:** LAY-001 full-page fallback (box 0,0,1500,2250),
  LAY-002 blank-page skip, SIZE-001 request capped 2895×2250 → 1600×1248.

### Remaining known failures (honest)

1. OOV-001: Clematis forced to a known character (Denken in panel mode, Wirbel
   in page mode) — unknown-identity handling needs a "refuse to guess"
   constraint; the test asserts it and fails on purpose.
2. DET-001/002/004: crop-only panel detection cannot resolve the flashback-era
   confusion (present-day cast, multi-character collapse, missed Heiter); the
   page-level mode that fixed DET-002 in V1.1 is outside the crop-only
   integration contract.
3. COL-001/002: Frieren's hair renders lavender-purple instead of the required
   silver-white — a consistent palette-adherence defect confirmed by the real
   VLM; candidate for the next colorization fix (V1.2).
4. Task 0005 (page-batched verification + bounded retries) is **gated**: it is
   implemented only if explicit palette conditioning still leaves material
   color-adherence failures after the integration-suite verdicts above.

Debug views (bbox + reading order + detected characters on the colorized
stitched page):

| vol1 p003 | vol1 p008 | ch134 0134-004 |
|---|---|---|
| ![p003](docs/pipeline_v1/v11-vol1-p003.png) | ![p008](docs/pipeline_v1/v11-vol1-p008.png) | ![0134-004](docs/pipeline_v1/v11-ch134-0134-004.png) |

---

## Full-page gpt-image-2 atlas mode (pipeline_v1 `--full-page`)

A second colorization backend for the same pipeline skeleton, ported from the
`research-v2` atlas method: **no panel extraction** — the whole page is
colorized in one OpenAI `gpt-image-2` `images.edit` call (page + labelled
reference atlas + explicit palette instruction from
`character_profiles.json`), at the smallest output size that preserves the
page's exact aspect ratio (`config.minimal_gpt_image_size`). The five stages
still run and write the same `output/<ts>/` layout with one synthetic
`panel_0001` per page (`provenance: full-page-mode`), so `--resume`,
`--from-step`, `--only-panel`, `--mock` and the debug annotation all keep
working unchanged.

- **Flags:** `--full-page`; `--atlas-source detected|cast` — `detected`
  (default) forces the page-level VLM detection mode (one
  `google/gemma-4-31b-it` call per page), `cast` skips character detection
  entirely (zero VLM calls, `OPENROUTER_API_KEY` not needed; the colorize
  step derives the chapter cast via `cast_key_for_page` / `--cast-key`, full
  canonical roster fallback when no cast is derivable). `--atlas-source cast`
  requires `--full-page`. Optional: `--gpt-size WxH` (override), `--gpt-model`,
  `--gpt-atlas-scale` (downscale the atlas before upload). Quality is fixed at
  `medium`; calls retry transient errors with backoff (≤ 3 retries) and then
  fail loudly.
- **Size policy:** minimal exact-ratio size — 1500×2250 (2:3) → **672×1008**,
  3000×2250 spread (4:3) → **960×720** (both match the research-v2 measured
  runs); 300 dpi B5 scans (2480×3508) are unsolvable within the API caps and
  are rejected loudly instead of distorted.
- **Cost (measured, run `output/20260815-152145`):** 5 pages, `--skip-first 3
  --limit 5` on volume 1 (p003–p008), **$0.24958 total** (std tier) =
  **$0.0499 avg/page**, matching the research-v2 projection:
  - p003 672×1008 $0.05074 · p004-p005 960×720 $0.05401 · p006 672×1008
    $0.04424 · p007 672×1008 $0.05074 · p008 672×1008 $0.04985.
  - Character detection added $0.00056 (5 page-level gemma calls).
  - Volume-1 projection (187 pages): **≈ $9.3** — the projection is now
    backed by measured per-call usage in the manifest
    (`steps.colorize.records[].est_cost_usd`, `totals.gpt_image_cost_usd`).
- **Atlas upload bug fixed:** the first real run (`output/20260815-151659`)
  failed 4/5 pages with `400 - Invalid file 'image[1]': unsupported mimetype
  ('application/octet-stream')` — the atlas `BytesIO` was uploaded bare, so
  httpx couldn't sniff a mimetype (the no-atlas page succeeded). `_scaled_atlas`
  now returns a `("atlas.jpg", buffer)` tuple (openai 3.x `FileTypes`),
  regression-tested in `tests/test_gpt_colorizer.py::test_atlas_upload_carries_filename`.
- **Atlas matters (quality finding):** in the same run, p006 (no detected
  characters → no atlas, no palette) came out essentially **black & white**
  (mean HSV saturation 2.8/255 vs 31–67 for the atlas=yes pages). The
  reference atlas + palette instruction is what drives the colorization;
  a no-atlas page is a de-facto no-op.
- **Backend comparison:** FLUX panel mode is $0/call + electricity on Spark
  (free but panel-wise, per-panel coherence); gpt-image-2 full-page mode is a
  paid API (~$0.05/page) with page-level coherence in a single call — the
  comparison is quality vs cost, and this mode is the pipeline-level
  counterpart of the `research-v2` atlas method.
- **Status:** implemented, offline-tested (`tests/test_full_page.py`, mock
  backends; full offline suite green), and verified with a real 5-page run
  (`output/20260815-152145`, 5/5 colorize calls OK, 0 errors, ~5 min wall
  time). See `docs/plans/fullpage-gpt-image2-atlas.md` for the original
  verification list.
