# GPT Image 1 Mini low-quality sequential page context

This method mirrors the Gemini sequential-reference experiment using OpenAI's `gpt-image-1-mini` image-edit endpoint at `quality=low`.

For every page, the request supplies the current monochrome page as the primary first image and a labelled atlas built from `data/refs/` as the second image. From page 2 onward, the immediately preceding generated page is supplied as the third image for color continuity. Page 1 has no previous-page context and invents colors where the canonical atlas does not apply.

There is no explicit reusable image-context cache in this workflow, so the reference atlas is sent on every edit request. The source files under `data/` are never modified. Every invocation creates a new local-time `output/YYYYMMDD-HHMMSS/` directory with generated pages, the reference atlas, normalized request inputs, and an incremental `manifest.json`.

## Preprocessing

The chapter files have `.png` names but contain JPEG data. To avoid a multipart filename/content mismatch, each selected page is decoded and saved losslessly as an actual PNG under the timestamped run's `normalized-inputs/` directory. The original files remain unchanged.

The character references are arranged into one 1440×2400 labelled JPEG atlas. This reduces 18 separate references to one request image and makes the comparison structurally similar to the Gemini method.

## Setup

Run from the repository root. The runner loads `OPENAI_API_KEY` from the repository `.env` file.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r colorization_methods/openai-gpt-image-1-mini-low-sequential/requirements.txt
```

Your OpenAI organization may need API Organization Verification before GPT Image models are available.

## One-page smoke test

```bash
.venv/bin/python colorization_methods/openai-gpt-image-1-mini-low-sequential/run.py \
  --model gpt-image-1-mini \
  --quality low \
  --size 1024x1536 \
  --output-format jpeg \
  --output-compression 95 \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --limit 1
```

## Full chapter

```bash
.venv/bin/python colorization_methods/openai-gpt-image-1-mini-low-sequential/run.py \
  --model gpt-image-1-mini \
  --quality low \
  --size 1024x1536 \
  --output-format jpeg \
  --output-compression 95 \
  --input-dir data/chapter_134 \
  --refs-dir data/refs
```

## Reproducibility and cost

The manifest records the command, model, quality, size, format, prompt, image order, source/reference hashes, preprocessing, dependency versions, output hashes, request IDs, API usage fields when returned, and per-page cost calculations.

Official pricing checked 2026-08-08:

- text input: $2.00 per million tokens;
- image input: $2.50 per million tokens;
- low-quality 1024×1536 output: $0.006 per image.

The fixed output portion is **$0.006/page** or **$0.108 for 18 pages**, plus text and image input tokens. Because the atlas is resent on every request and pages 2–18 include a previous page, the final measured cost will be higher. Failed or retried requests may incur additional charges.

This is not directly resolution-equivalent to the Gemini run: GPT Image outputs 1024×1536, while the measured Gemini outputs were 848×1264.

## Measured smoke test

The one-page smoke test completed successfully on 2026-08-08:

- output: [`output/20260808-005423/0134-001.jpg`](output/20260808-005423/0134-001.jpg);
- request: 326 text-input tokens, 646 image-input tokens, and 408 output tokens;
- measured estimated cost: **$0.008267** for one page;
- latency: approximately 20 seconds from run start to completed manifest.

The color palette is coherent and the two figures remain recognizable, but preservation quality is unsuitable for a full chapter without further iteration. The model substantially redrew and smoothed the source linework, cropped the white margins and printed page number, changed clothing and small scene details, and replaced the exact chapter title with malformed text. It therefore failed the method's central "add color only" requirement. Page-to-page continuity and atlas faithfulness were not evaluated because the smoke test stopped after page 1.

Official OpenAI documentation notes that GPT Image models can struggle with exact text, recurring-character consistency, and precise structured composition. Low quality is optimized for fast drafts, so it may preserve manga linework and small lettering less reliably than medium or high quality. Other risks include atlas references being ignored, an error propagating through the previous-page chain, and input-token costs exceeding the fixed output charge.

## API references

- [GPT Image 1 Mini model and pricing](https://developers.openai.com/api/docs/models/gpt-image-1-mini)
- [Image generation and editing guide](https://developers.openai.com/api/docs/guides/image-generation)
