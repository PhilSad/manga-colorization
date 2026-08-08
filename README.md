# Manga Colorization — Methods Compared

A practical comparison of AI-assisted manga colorization services on **quality and cost**. Three commercial image-generation models are used to colorize the same source chapter with the same character reference atlas and the same sequential page-context strategy (each page sees the previously colorized page, so colors stay consistent across the chapter).

- **Input:** chapter 134 of a manga, 18 pages at 1200×1800 (JPEG data in `.png` files), kept unmodified under `data/`.
- **References:** labelled character reference atlas built from `data/refs/`.
- **Strategy:** colorize pages in filename order; from page 2 onward, feed the previous generated page back as context.
- **Outputs:** one fresh timestamped run directory per invocation; never overwritten.
- Full index and method details: [`methods.md`](methods.md). Reproduce instructions are in each method's `README.md`.

> Prices are USD paid-tier estimates, dated **2026-08-08**, excluding taxes. What is *measured* (metered usage × price) vs *estimated* (price-card math) is labeled in the tables.

## Methods

| Method | Model / service | Output | Cost / page (measured) | Full chapter |
|---|---|---|---|---|
| [Gemini reference atlas + sequential context](colorization_methods/gemini-reference-cache-sequential/) | Google Gemini 3.1 Flash Lite Image (Nano Banana 2 Lite) | 848×1264 JPEG | **$0.03451** (18 pages, measured) | **$0.62113** |
| [GPT Image 1 Mini low + sequential context](colorization_methods/openai-gpt-image-1-mini-low-sequential/) | OpenAI `gpt-image-1-mini`, `quality=low` | 1024×1536 JPEG | **$0.00827** (1-page smoke test only) | not measured — estimated ≈ $0.108 output + input tokens |
| [fal FLUX.2 Klein 9B Edit + sequential context](colorization_methods/fal-flux-2-klein-9b-edit-sequential/) | fal `fal-ai/flux-2/klein/9b/edit`, 4 steps | 1216×1824 PNG | **$0.05676** (pages 2–18, estimated) | **$1.07956** (18 pages + 1 recovery call, estimated) |

## Before / after — page 1

Page 1 is the same monochrome input for every method (no previous-page context, so this is where each model invents a palette for anything not covered by the reference atlas).

![All methods, page 1](docs/all-methods-page1.jpg)


## Before / after — page 2 (page-by-page consistency)

Page 2 is the first page to receive **previous-page context**: each method is given its own page-1 colorization and asked to keep colors consistent. It shows how well the sequential mechanism carries the palette (and its errors) forward. GPT Image 1 Mini has no page-2 output because only a 1-page smoke test was run.

![All methods, page 2](docs/all-methods-page2.jpg)


Page 2 and later pages colorized with this strategy can be inspected in the full run directories ([Gemini `20260808-003106`](colorization_methods/gemini-reference-cache-sequential/output/20260808-003106/), [fal `20260808-011051`](colorization_methods/fal-flux-2-klein-9b-edit-sequential/output/20260808-011051/)); note these are local, gitignored artifacts, so only the page-1/page-2 samples above are tracked in this repository.

## Quality notes (sampled pages, full chapter where noted)

| Method | Strengths | Known failures |
|---|---|---|
| **Gemini** (full 18-page run) | Coherent subdued palette; panels and linework mostly preserved; consistent silver/white hair and blue eyes; previous-page context worked pages 2–18. | Invented a hanging lamp on page 1; malformed chapter-title lettering; large near-monochrome areas; color bled into the outer margin of page 18. |
| **GPT Image 1 Mini** (1-page smoke test) | Coherent palette; two figures recognizable. | Substantially redrew/smoothed the linework, cropped margins and the printed page number, changed clothing/scene details, replaced the chapter title with malformed text. Failed the "add color only" requirement; continuity untested. |
| **fal FLUX.2 Klein** (full 18-page run) | Best structural preservation: panels, margins, page numbers, and dialogue kept; coherent detailed palette. | Regenerated/smoothed linework; changed faces, hair, clothing, small details; atlas colors unreliable (non-silver Frieren on page 18); false-positive safety block produced a black page 18, recovered separately with the checker disabled. |

Bottom line: **fal FLUX.2 Klein** is structurally the most faithful and visually polished but the most expensive and still redraws rather than strictly colorizes. **GPT Image 1 Mini low** is by far the cheapest per page but the least faithful at low quality. **Gemini 3.1 Flash Lite Image** sits in the middle on both axes. None of the three achieves strict "color only" processing on this test set.

## Comparability caveats

- **Different output resolutions:** Gemini 848×1264 JPEG, OpenAI 1024×1536 JPEG, fal 1216×1824 PNG. Not resolution-equivalent.
- **Different run scopes:** Gemini and fal are full 18-page runs; OpenAI is a 1-page smoke test (its full-chapter cost is an estimate, not measured).
- **Cost basis:** measured metered usage for Gemini and the OpenAI smoke test; price-card estimates for fal (the API does not return a billed amount). fal's full-chapter figure includes a 19th recovery request for the false-blocked page 18.

## Reproduce

Every method has its own directory with `README.md` (setup + exact commands), `run.py`, prompt, and requirements. See [`methods.md`](methods.md) for the full index. API keys are read from `.env` at runtime; the `.env` file itself is not committed.

## Volume tooling (`script/`)

Utility scripts for manga volumes (Python 3 stdlib only, runnable from any directory):

- `script/extract_pages.py` — unpack `.cbz` volumes from `data/volumes/` into per-volume page folders under `data/page_per_volume/<volume-name>/`. Pages keep their original filenames (zero-padded, so natural sort = reading order), extraction is resumable, and `--force` re-extracts. `--volume` filters by name substring.
- `script/merge_to_cbz.py` — pack a folder of pages back into a `.cbz`, natural-sorted. Only image files by default (`--all` to include everything); output defaults to `<folder-name>.cbz` in the current directory, override with `--output`.

Both `data/volumes/` and `data/page_per_volume/` are gitignored; extracted pages are local artifacts, not tracked in this repository.
