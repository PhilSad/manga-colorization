# fal FLUX.2 Klein 9B Edit with sequential page context

This method tests [`fal-ai/flux-2/klein/9b/edit`](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit) with the same sequential-reference design as the Gemini and GPT Image experiments.

Each request supplies the current monochrome page as `#1` and a labelled character-reference atlas as `#2`. From page 2 onward, the preceding generated page is supplied as `#3` for color continuity. Page 1 invents colors where the atlas does not apply.

The endpoint accepts at most four edit images, so this three-image sequence is within its documented limit. There is no model-context cache: the atlas is uploaded once per run and its fal-hosted URL is reused for every page. Each invocation creates a fresh local-time `output/YYYYMMDD-HHMMSS/` directory with the output images, normalized request inputs, atlas, and incremental manifest.

## Preprocessing

The chapter files have `.png` names but contain JPEG data. The runner decodes each selected page and writes a true lossless PNG under the run's `normalized-inputs/` directory. Original files under `data/` remain unchanged.

The references are combined into one 1440×2400 labelled JPEG atlas. The default output canvas is 1200×1800, matching the chapter sources.

## Setup

The runner loads `FAL_API_KEY` from the repository `.env`. It also accepts fal's standard `FAL_KEY` environment variable.

```bash
.venv/bin/python -m pip install \
  -r colorization_methods/fal-flux-2-klein-9b-edit-sequential/requirements.txt
```

## Local inference server (optional)

A self-hosted BentoML server running the same `FLUX.2-klein-9B` weights locally on the DGX Spark can replace the paid fal endpoint entirely. Everything server-side lives in [`server/`](../../server/) at the repo root (docker-packaged; the ~35 GB weights are an "external module" downloaded to `models/` and mounted read-only, not baked into the image). See [`server/README.md`](../../server/README.md) for the full deployment guide.

The client side uses `local_fal_client.py`, a fal-compatible shim (`upload_file`/`submit`/`get`) that talks to the server's `POST /edit` multipart endpoint; no manifest/provenance logic changes. Use `--endpoint` to switch:

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-sequential/run.py \
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
- The local pipeline has no safety checker (fal's optional one falsely blocked page 18).


## One-page smoke test

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-sequential/run.py \
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
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-sequential/run.py \
  --model fal-ai/flux-2/klein/9b/edit \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs
```

## Skip leading pages

The runner shows a tqdm progress bar while processing: it displays the current chapter page, the number of pages done, elapsed/remaining time, and the current source filename. The bar is suppressed automatically on non-interactive terminals; each completed page is still logged as `[n/N] wrote <output path>`. `tqdm` is installed via `requirements.txt` and its version is recorded in the manifest.

`--skip-first N` skips the first N pages of the input folder (a zero-based folder offset applied before `--start-at`; the two compose, so `--skip-first 8 --start-at 2` starts at chapter page 10). This is convenient when a previous run was interrupted partway: resume the remaining pages without reprocessing earlier ones. Each invocation still creates a fresh timestamped run directory and never overwrites previous outputs.

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-sequential/run.py \
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
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-sequential/run.py \
  --model fal-ai/flux-2/klein/9b/edit \
  --width 1200 \
  --height 1800 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --start-at 18 \
  --limit 1 \
  --previous-page colorization_methods/fal-flux-2-klein-9b-edit-sequential/output/<full-run>/0134-017.png \
  --disable-safety-checker
```

## Reproducibility and cost

The manifest records the command, prompt, model settings, source/reference hashes, preprocessing, dependency versions, fal upload and output URLs, request IDs, seeds, timings, safety results, output hashes, and estimated cost.

Fal lists the model at **$0.011 per megapixel of input and output** and says each input image is resized to 1 MP for billing. Before running, a 1200×1800 output (2.16 MP) put page 1 at $0.04576 for two inputs plus output and later pages at $0.05676 for three inputs plus output.

## Measured smoke test

The one-page smoke test completed successfully on 2026-08-08:

- output: [`output/20260808-010413/0134-001.png`](output/20260808-010413/0134-001.png);
- request ID: `019fde78-5ffc-7c60-ad11-443f43a816ac`;
- seed: `1499077118`;
- wall time: approximately 11 seconds; reported model inference: 1.515 seconds;
- output: 1216×1824 PNG, despite requesting 1200×1800;
- estimated cost: **$0.04639782** (2 input MP plus 2.217984 output MP).

The full page composition, white margins, chapter title, and printed page number were retained, making structural preservation substantially better than the GPT Image 1 Mini low smoke test. The palette is coherent and detailed. However, the result still regenerated and smoothed much of the source linework, changed facial/hair/clothing details, and did not preserve the exact requested dimensions. It is a polished reinterpretation rather than strict color-only processing. Atlas faithfulness and page-to-page continuity were not established on page 1.

Using the observed 1216×1824 output size, later three-input pages would be estimated at $0.05739782 each and an 18-page run at approximately **$1.02216083**. These remain pricing estimates because the response does not expose the final billed amount; failed requests may still incur charges.

Likely additional failure cases include copied atlas/previous-page content and propagation of a bad color or redraw through the sequential chain.

## Measured full chapter

The full chapter run [`output/20260808-011051/`](output/20260808-011051/) issued 18 API calls and finished in approximately 2 minutes 37 seconds. All outputs were 1216×1824 PNGs. The estimated request cost was **$1.02216076**. Mean reported model inference time was 1.870 seconds per page (33.656 seconds total), excluding uploads, queueing, and downloads.

Pages 1–17 produced usable images. Page 18 was falsely marked by the endpoint with `has_nsfw_concepts: true` and returned an effectively black PNG. The blocked output and its request metadata remain in the full-run directory for provenance.

Page 18 was recovered in the separate non-overwriting run [`output/20260808-011534/`](output/20260808-011534/) using page 17 as continuity context and `--disable-safety-checker`. That request cost an additional estimated **$0.05739782**, bringing the actual 19-request experiment estimate to **$1.07955858**. The usable 18-page set is pages 1–17 from the full run plus page 18 from the recovery run.

Across sampled pages, panel geometry, margins, page numbers, and dialogue were preserved well, and recurring characters generally kept a coherent palette. The model consistently replaced the original manga rendering with smoother generated linework and changed faces, hair, clothing, and small details. Canonical atlas faithfulness is unreliable: on recovered page 18, Frieren's expected silver-white hair is rendered brown/olive, and other small appearances use inconsistent colors. The method is therefore visually polished and structurally strong, but not faithful enough to count as color-only processing.

## API references

- [FLUX.2 Klein 9B Edit model and pricing](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit)
- [Endpoint API schema](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit/api)
- [fal Python client](https://fal.ai/docs/reference/client-libraries/python)
