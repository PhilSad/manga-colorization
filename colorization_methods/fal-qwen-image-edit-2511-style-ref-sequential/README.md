# fal Qwen Image Edit 2511 with sequential page context and a style reference

This method is a port of the [`fal-flux-2-klein-9b-edit-style-ref-sequential`](../fal-flux-2-klein-9b-edit-style-ref-sequential/) pipeline to the fal **Qwen Image Edit 2511** endpoint (`fal-ai/qwen-image-edit-2511`, based on Alibaba's instruction-based Qwen-Image-Edit image editing model). It keeps the same sequential design: a labelled character-reference atlas, a colorized **style reference page** supplied on every request, and the previously generated page as color-continuity context from page 2 onward.

Each request supplies the current monochrome page as **image 1**, the labelled character-reference atlas as **image 2**, and the style reference as **image 3**. From page 2 onward, the preceding generated page is supplied as **image 4**. Page 1 invents colors where the atlas does not apply, guided by the style reference. The prompt refers to images by ordinal ("the first image", "the second image", …), which is how the Qwen edit model addresses multi-image input.

The atlas and the style reference are each uploaded once per run and their fal-hosted URLs are reused. Each invocation creates a fresh local-time `output/YYYYMMDD-HHMMSS/` directory with the output images, normalized request inputs, atlas, and incremental manifest.

## Differences from the FLUX.2 Klein variant

| Aspect | FLUX.2 Klein 9B Edit variant | This method |
|---|---|---|
| Endpoint | `fal-ai/flux-2/klein/9b/edit` | `fal-ai/qwen-image-edit-2511` |
| Image input | `image_urls` (list) | `image_urls` (list) — same name |
| Output size | Requested 1200×1800, actual 1216×1824 (FLUX VAE rounding) | Explicit `image_size` 1200×1800 requested; Qwen echoes input dims when `image_size` is omitted, but the inputs have mixed sizes (page 1200×1800, atlas 1440×2400, style 848×1264), so an explicit size is passed to force a 1200×1800 output |
| Inference steps | 4 | Qwen model default **28** (range 1–50) — Qwen needs far more steps than FLUX Klein |
| Guidance | — (not a FLUX Klein param) | `guidance_scale` (Qwen default 4.5) and `acceleration` (`none`/`regular`/`high`, default `regular`) |
| Local server | `--endpoint` can run against the self-hosted BentoML FLUX.2 Klein server (`server/` at repo root) | **Not available.** The local server only serves FLUX.2 Klein weights, so this method runs on fal only; `local_fal_client.py` was dropped |
| Safety checker | Optional, caused a false-positive black page 18 | Optional (`--disable-safety-checker`); blocked images are recorded via `has_nsfw_concepts` |
| Pricing | $0.011/MP, each input billed at 1 MP | **$0.03/MP** (fal listing); input billing basis undocumented — see [Cost](#cost) |

## Style reference

The default style reference is page 8 of the [Gemini full-chapter run](../gemini-reference-cache-sequential/output/20260808-003106/0134-008.jpg) — a 848×1264 JPEG. It is recorded in the manifest with its SHA-256 hash so results remain reproducible. Pass any colorized page to `--style-ref` to change the colorization style; the image is uploaded once per run and reused by URL, like the atlas.

The prompt instructs the model to match the style reference's overall look (palette mood and saturation, contrast, shading/rendering approach, screentone treatment, paper tone) while never copying its content, characters, or panel layout into the target page.

## Preprocessing

The chapter files have `.png` names but contain JPEG data. The runner decodes each selected page and writes a true lossless PNG under the run's `normalized-inputs/` directory. Original files under `data/` remain unchanged.

The references are combined into one 1440×2400 labelled JPEG atlas. The default output canvas is 1200×1800, matching the chapter sources. The style reference is uploaded as-is (no resizing or re-encoding).

## Setup

The runner loads `FAL_API_KEY` from the repository `.env`. It also accepts fal's standard `FAL_KEY` environment variable.

```bash
.venv/bin/python -m pip install \
  -r colorization_methods/fal-qwen-image-edit-2511-style-ref-sequential/requirements.txt
```

## One-page smoke test

```bash
.venv/bin/python colorization_methods/fal-qwen-image-edit-2511-style-ref-sequential/run.py \
  --model fal-ai/qwen-image-edit-2511 \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 28 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --limit 1
```

## Full chapter

```bash
.venv/bin/python colorization_methods/fal-qwen-image-edit-2511-style-ref-sequential/run.py \
  --model fal-ai/qwen-image-edit-2511 \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 28 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs
```

## Custom style reference

Any colorized manga page can act as the style reference; pass its path to `--style-ref`. The file must exist and be a supported image format; it is recorded (hash + metadata) in the manifest.

```bash
.venv/bin/python colorization_methods/fal-qwen-image-edit-2511-style-ref-sequential/run.py \
  --model fal-ai/qwen-image-edit-2511 \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 28 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --style-ref /path/to/any/colorized/page.jpg \
  --limit 1
```

## Resume or recover one page

Use `--start-at` to select a one-based chapter page and `--previous-page` to seed its first request with an existing colorized predecessor. The optional safety checker can be disabled for a known-safe page that was falsely blocked. A recovery invocation still creates a new timestamped run and never overwrites the blocked artifact.

```bash
.venv/bin/python colorization_methods/fal-qwen-image-edit-2511-style-ref-sequential/run.py \
  --model fal-ai/qwen-image-edit-2511 \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 28 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --start-at 18 \
  --limit 1 \
  --previous-page colorization_methods/fal-qwen-image-edit-2511-style-ref-sequential/output/<full-run>/0134-017.png \
  --disable-safety-checker
```

## Cost

fal lists the model at **$0.03 per megapixel**. Unlike the FLUX Klein endpoint, fal does not document how input images are billed (FLUX documents each input as resized to 1 MP for billing; Qwen does not). The manifest therefore records an estimate that assumes each uploaded input image is billed at its native megapixel count, and it records the sizes of every input so the assumption can be re-checked.

Per-request native megapixel totals (atlas 1440×2400 = 3.456 MP, style ref 848×1264 ≈ 1.072 MP, page/output 1200×1800 = 2.16 MP):

- **Page 1** (3 inputs + output): 6.688 + 2.16 = 8.848 MP → ≈ **$0.2654**
- **Pages 2–18** (4 inputs + output): 8.848 + 2.16 = 11.008 MP → ≈ **$0.3302**
- **Full 18-page run, native-MP billing: ≈ $5.88** (one page-1 + seventeen later-page costs)

If fal instead caps each input at 1 MP for billing (like the FLUX endpoint), page 1 would be (3 + 2.16) × $0.03 ≈ $0.1548 and later pages (4 + 2.16) × $0.03 ≈ $0.1848, for a full run of ≈ **$3.30**. Both are estimates: the endpoint response does not include the billed amount, and failed requests may still incur charges. Costs are labeled as estimates in the manifest, not measured.

## Reproducibility

The manifest records the command, prompt, model settings (steps, guidance scale, acceleration, image size, seed), source/reference hashes (including the style reference), preprocessing, dependency versions, fal upload and output URLs, request IDs, seeds, timings, safety results, output hashes, and the cost estimate.

## Status and expected quality

**Not yet run.** Quality and cost are unmeasured for this model on this task. Qwen-Image-Edit is an instruction-following edit model (a different family from FLUX Klein Edit), so expected differences versus the FLUX variant include: stronger text/instruction adherence (the "add color only, preserve everything" requirement should be followed more literally), better handling of the labelled atlas because it is a VLM-grounded edit model, and 28-step inference making each page slower and costlier. Expected risks carried over from the base method: style-reference content leaking into the target page, atlas colors being applied unreliably, and Qwen's tendency to soften linework on heavy screentone pages. These are hypotheses to test, not measured outcomes.

## API references

- [Qwen Image Edit 2511 model and pricing](https://fal.ai/models/fal-ai/qwen-image-edit-2511)
- [Endpoint API schema](https://fal.ai/models/fal-ai/qwen-image-edit-2511/api)
- [fal Python client](https://fal.ai/docs/reference/client-libraries/python)
