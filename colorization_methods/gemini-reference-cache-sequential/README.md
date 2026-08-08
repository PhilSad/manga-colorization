# Gemini reference atlas with sequential page context

This method colorizes a chapter in filename order. It builds one labelled atlas from `data/refs/`, supplies the current monochrome page on every request, and supplies the immediately preceding generated page from page 2 onward. On page 1, the prompt explicitly tells Gemini to use known character references and invent a coherent palette for everything else.

The source files under `data/` are never modified. Every invocation creates a fresh local-time `output/YYYYMMDD-HHMMSS/` directory containing the atlas, generated pages, and an incrementally updated `manifest.json`.

## Important model constraint

The requested Nano Banana 2 Lite model is `gemini-3.1-flash-lite-image`. As of 2026-08-08, Google documents image generation/editing and up to 14 reference images for this model, but explicitly marks context caching as unsupported. The verified workflow therefore sends the reference atlas inline on every page request, plus the previous generated page when available.

An attempted explicit-cache setup with `gemini-2.5-flash-image` returned `404 NOT_FOUND`: the model was not supported for `createCachedContent` on the Gemini Developer API. Although its capability page broadly says caching is supported, that can refer to implicit caching rather than a user-created cache. The failed setup is preserved in [`output/20260808-002322/`](output/20260808-002322/). Explicit `--reference-mode cache` is disabled until a compatible image-generation model is verified.

## Setup

Run these commands from the repository root. `run.py` loads `GEMINI_API_KEY` from the repository `.env` file, so the key does not need to be exported or placed on the command line.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r colorization_methods/gemini-reference-cache-sequential/requirements.txt
```

## Run Nano Banana 2 Lite

This is the closest supported implementation of the requested Lite workflow. It does not create an API cache because Lite does not support one.

```bash
.venv/bin/python colorization_methods/gemini-reference-cache-sequential/run.py \
  --model gemini-3.1-flash-lite-image \
  --reference-mode inline-atlas \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --aspect-ratio 2:3
```

## One-page smoke test

Use this command to make exactly one image-generation request:

```bash
.venv/bin/python colorization_methods/gemini-reference-cache-sequential/run.py \
  --model gemini-3.1-flash-lite-image \
  --reference-mode inline-atlas \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --aspect-ratio 2:3 \
  --limit 1
```

`--skip-first N` (default 0) skips the first N sorted pages before `--limit` is applied, e.g. `--skip-first 3 --limit 5` colorizes pages 4–8 of a volume, skipping cover/title pages.

The completed smoke test is preserved in [`output/20260808-002733/`](output/20260808-002733/), with the [generated first page](output/20260808-002733/0134-001.jpg) and manifest.

## Completed full-chapter run

The 18-page run is preserved in [`output/20260808-003106/`](output/20260808-003106/). It completed all pages in filename order, with each page after the first receiving the preceding generated page as context. See its [manifest](output/20260808-003106/manifest.json) for per-page hashes, usage, and cost.

## Volume 1 comparison runs (2026-08-08)

Two 5-page runs on Frieren vol. 1 (c001 p003–p008, `--skip-first 3 --limit 5`), identical settings except the model, for a head-to-head quality/cost comparison:

- [`output/20260808-164132/`](output/20260808-164132/): `gemini-3.1-flash-lite-image` — **$0.1723 total ($0.0345/page)**.
- [`output/20260808-165248/`](output/20260808-165248/): `gemini-3.1-flash-image` (Nano Banana 2) — **$0.3376 total ($0.0675/page)**, roughly 2× the Lite cost as expected from list pricing ($0.067 vs $0.0336 per 1K output image).
- [`output/20260808-170138/`](output/20260808-170138/): `gemini-3.1-flash-image` at `--image-size 0.5K` (sent as `512`) — **$0.2276 total ($0.0455/page)** at 416×624 output, a ~33% cost cut but ~4× fewer pixels than the 1K run. A preceding attempt that sent `0.5K` literally failed with `400 INVALID_ARGUMENT` (supported values: `1K, 2K, 4K, 512, 512P, 512PX`); that failure is preserved in [`output/20260808-170120/`](output/20260808-170120/).

`--image-size` (default `1K`) accepts `0.5K` (alias for `512`), `512`, `1K`, `2K`, `4K` and only applies to `gemini-3.1-*` models; per-size image pricing is picked from the model's `output_image_each_by_size` table.

## Reproducibility and cost

The manifest records the full command, model, effective reference mode, prompt and system instruction, dependency versions, cache metadata, all input/reference SHA-256 hashes and dimensions, output hashes, per-page usage metadata, and measured cost estimates. It is written after every successful page so partial runs remain inspectable.

Pricing assumptions dated 2026-08-08:

- `gemini-3.1-flash-image`: $0.067 per 1K output image ($60 per 1M image tokens) plus $0.50 per million input tokens and $3.00 per million text/thinking output tokens.
- `gemini-3.1-flash-lite-image`: $0.0336 per 1K output image plus $0.25 per million input tokens and any text/thinking output. The fixed output portion is $0.6048 for 18 pages.

These figures are paid-tier estimates, exclude taxes, and can change. Failed or retried API requests may incur charges not represented by a successful response's usage metadata.

## Evaluation status and likely failure cases

The full run completed on 2026-08-08 using `gemini-3.1-flash-lite-image`: 18 successful API calls, 65,310 prompt tokens, 30,224 candidate tokens, 95,534 total tokens, and a measured estimated cost of **$0.6211275** (**$0.0345071/page**). All outputs are 848×1264 JPEG. The earlier one-page smoke test cost $0.03424075.

Across sampled pages, the run produced coherent warm interiors, subdued city scenes, clothing colors, skin tones, silver/white hair, and blue eyes while mostly preserving panel geometry and fine linework. The immediately previous page successfully remained in the request chain for pages 2–18.

Observed failures include an invented hanging lamp on page 1, slightly malformed chapter-title lettering, large areas that remain near-monochrome, and color bleeding into a white outer margin on page 18. Known risks also include model redraws, color drift accumulating through the previous-page chain, missed characters in the atlas, reduced character fidelity from combining references into one atlas, and propagation of a bad color choice to later pages. Reference faithfulness and character consistency have not yet been scored systematically across every page.

## API references

- [Nano Banana image-generation guide](https://ai.google.dev/gemini-api/docs/generate-content/image-generation)
- [Gemini 3.1 Flash Lite Image capabilities](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-image)
- [Gemini 2.5 Flash Image capabilities](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-image)
- [Context caching](https://ai.google.dev/gemini-api/docs/generate-content/caching)
- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)
