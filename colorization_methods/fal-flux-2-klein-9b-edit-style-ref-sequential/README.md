# fal FLUX.2 Klein 9B Edit with sequential page context and a style reference

This method is a variant of the [`fal-flux-2-klein-9b-edit-sequential`](../fal-flux-2-klein-9b-edit-sequential/) pipeline. It adds a **style reference image** — a colorized manga page that defines the target colorization look — to every request, so the palette mood, saturation, shading, and screentone treatment can be adjusted by swapping the reference.

Each request supplies the current monochrome page as `#1`, the labelled character-reference atlas as `#2`, and the style reference as `#3`. From page 2 onward, the preceding generated page is supplied as `#4` for color continuity. Page 1 invents colors where the atlas does not apply, guided by the style reference.

The endpoint accepts at most four edit images, so the four-image sequence on later pages is at its documented limit. There is no model-context cache: the atlas and the style reference are each uploaded once per run and their fal-hosted URLs are reused. Each invocation creates a fresh local-time `output/YYYYMMDD-HHMMSS/` directory with the output images, normalized request inputs, atlas, and incremental manifest.

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
  -r colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/requirements.txt
```

## Local inference server (optional)

A self-hosted BentoML server running the same `FLUX.2-klein-9B` weights locally on the DGX Spark can replace the paid fal endpoint entirely. Everything server-side lives in [`server/`](../../server/) at the repo root (docker-packaged; the ~35 GB weights are an "external module" downloaded to `models/` and mounted read-only, not baked into the image). See [`server/README.md`](../../server/README.md) for the full deployment guide.

The client side uses `local_fal_client.py`, a fal-compatible shim (`upload_file`/`submit`/`get`) that talks to the server's `POST /edit` multipart endpoint; no manifest/provenance logic changes. Use `--endpoint` to switch:

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/run.py \
  --model black-forest-labs/FLUX.2-klein-9B \
  --endpoint http://spark:3000 \
  --width 1216 \
  --height 1824 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --limit 1
```

Notes for local runs:
- Use `--width 1216 --height 1824`: the FLUX VAE needs multiples of 16 and diffusers floors non-compliant sizes (1200×1800 → 1200×1792); 1216×1824 also matches fal's actual outputs for direct comparison.
- Local runs record the self-hosted pricing block in the manifest ($0 per call, electricity estimate) instead of the fal $/MP pricing — do not compare local and fal costs as if identical (see `server/README.md` at the repo root).
- The local pipeline has no safety checker (fal's optional one falsely blocked page 18 of the base method).


## One-page smoke test

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/run.py \
  --model fal-ai/flux-2/klein/9b/edit \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --limit 1
```

## Full chapter

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/run.py \
  --model fal-ai/flux-2/klein/9b/edit \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs
```

## Custom style reference

Any colorized manga page can act as the style reference; pass its path to `--style-ref`. The file must exist and be a supported image format; it is recorded (hash + metadata) in the manifest.

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/run.py \
  --model fal-ai/flux-2/klein/9b/edit \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --style-ref /path/to/any/colorized/page.jpg \
  --limit 1
```

## Skip leading pages

The runner shows a tqdm progress bar while processing: it displays the current chapter page, the number of pages done, elapsed/remaining time, and the current source filename. The bar is suppressed automatically on non-interactive terminals; each completed page is still logged as `[n/N] wrote <output path>`. `tqdm` is installed via `requirements.txt` and its version is recorded in the manifest.

`--skip-first N` skips the first N pages of the input folder (a zero-based folder offset applied before `--start-at`; the two compose, so `--skip-first 8 --start-at 2` starts at chapter page 10). This is convenient when a previous run was interrupted partway: resume the remaining pages without reprocessing earlier ones. Each invocation still creates a fresh timestamped run directory and never overwrites previous outputs.

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/run.py \
  --model fal-ai/flux-2/klein/9b/edit \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --skip-first 10
```

Pages are selected as `folder[skip_first + start_at - 1 :][:limit]`; the manifest's per-page `sequence` and the seed offsets reflect the real chapter page numbers (e.g. with `--skip-first 8`, the first processed page is recorded as sequence 9). The `skip_first` value is stored under `configuration` in the manifest.

## Resume or recover one page

Use `--start-at` to select a one-based chapter page and `--previous-page` to seed its first request with an existing colorized predecessor. The optional safety checker can be disabled for a known-safe page that was falsely blocked. A recovery invocation still creates a new timestamped run and never overwrites the blocked artifact.

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/run.py \
  --model fal-ai/flux-2/klein/9b/edit \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --start-at 18 \
  --limit 1 \
  --previous-page colorization_methods/fal-flux-2-klein-9b-edit-style-ref-sequential/output/<full-run>/0134-017.png \
  --disable-safety-checker
```

## Reproducibility and cost

The manifest records the command, prompt, model settings, source/reference hashes (including the style reference), preprocessing, dependency versions, fal upload and output URLs, request IDs, seeds, timings, safety results, output hashes, and estimated cost.

Fal lists the model at **$0.011 per megapixel of input and output** and says each input image is resized to 1 MP for billing. This method sends **one more input image than the base method** (the style reference): page 1 uses 3 inputs, later pages 4. Before running, a 1200×1800 output (2.16 MP) would put page 1 at about $0.0568 for three inputs plus output and later pages at $0.0678 for four inputs plus output (using the observed 1216×1824 output size). These are estimates: the endpoint response does not include the billed amount, and failed requests may still incur charges.

## Status and measured results

**Not yet run.** Quality and cost for this variant are unmeasured. The base method (see [`fal-flux-2-klein-9b-edit-sequential/README.md`](../fal-flux-2-klein-9b-edit-sequential/README.md)) showed strong panel/margin/dialogue preservation but smoothed generated linework, changed details, and had unreliable atlas colors — the style reference is intended to tighten palette consistency and shading treatment, but this is a hypothesis to test, not a measured outcome. Expected additional failure cases include the model copying style-reference content or panel layout into the target page (the prompt explicitly forbids this) and the extra input pushing later pages to the four-image endpoint limit.

## API references

- [FLUX.2 Klein 9B Edit model and pricing](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit)
- [Endpoint API schema](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit/api)
- [fal Python client](https://fal.ai/docs/reference/client-libraries/python)
