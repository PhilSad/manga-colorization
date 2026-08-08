# fal Qwen Image Edit 2511 with sequential page context and a style reference

This method is a port of the [`fal-flux-2-klein-9b-edit-style-ref-sequential`](../fal-flux-2-klein-9b-edit-style-ref-sequential/) pipeline to the fal **Qwen Image Edit 2511** endpoint (`fal-ai/qwen-image-edit-2511`, based on Alibaba's instruction-based Qwen-Image-Edit image editing model). It keeps the same sequential design: a labelled character-reference atlas, a colorized **style reference page** supplied on every request, and the previously generated page as color-continuity context from page 2 onward.

Each request supplies the current monochrome page as **Picture 1**, the labelled character-reference atlas as **Picture 2**, and the style reference as **Picture 3**. From page 2 onward, the preceding generated page is supplied as **Picture 4**. Page 1 invents colors where the atlas does not apply, guided by the style reference.

**Prompting is designed around Qwen-Image-Edit's documented behavior.** The model's multi-image mode is a *composition* feature (see [Prompting and best practices](#prompting-and-best-practices)), so the prompt states in no uncertain terms that only Picture 1 may appear in the output and Pictures 2–4 are reference-only, and a `negative_prompt` is sent to suppress atlas/portrait leakage. The prompt uses the model's own vocabulary ("Picture N") — the diffusers pipeline prepends `Picture N: <image>` before the instruction.

The atlas and the style reference are each uploaded once per run and their fal-hosted URLs are reused. Each invocation creates a fresh local-time `output/YYYYMMDD-HHMMSS/` directory with the output images, normalized request inputs, atlas, and incremental manifest.

## Differences from the FLUX.2 Klein variant

| Aspect | FLUX.2 Klein 9B Edit variant | This method |
|---|---|---|
| Endpoint | `fal-ai/flux-2/klein/9b/edit` | `fal-ai/qwen-image-edit-2511` |
| Image input | `image_urls` (list) | `image_urls` (list) — same name |
| Output size | Requested 1200×1800, actual 1216×1824 (FLUX VAE rounding) | Requested 1200×1800; Qwen's resolution bucketing (multiples of 16) yields **1200×1792** in practice. The inputs have mixed sizes (page 1200×1800, atlas 1440×2400, style 848×1264) and the endpoint resizes them to a common size, so an explicit `image_size` is passed for deterministic output |
| Inference steps | 4 | **40** (Qwen's official 2511 examples; range 1–50). Cost is per megapixel, so more steps are free |
| Guidance | — (not a FLUX Klein param) | `guidance_scale` (fal default 4.5) and `acceleration` (`none`/`regular`/`high`, default `regular`) |
| Negative prompt | — | `--negative-prompt` (default targets multi-image composition leakage: portraits, atlas labels, pasted/collaged elements, reference content) |
| Local server | `--endpoint` can run against the self-hosted BentoML FLUX.2 Klein server (`server/` at repo root) | **Not available.** The local server only serves FLUX.2 Klein weights, so this method runs on fal only; `local_fal_client.py` was dropped |
| Safety checker | Optional, caused a false-positive black page 18 | Optional (`--disable-safety-checker`); blocked images are recorded via `has_nsfw_concepts` |
| Pricing | $0.011/MP, each input billed at 1 MP | **$0.03/MP** (fal listing); input billing basis undocumented — see [Cost](#cost) |

## Prompting and best practices

Research into the official Qwen docs (model cards, GitHub README, diffusers pipeline source) shows Qwen-Image-Edit behaves very differently from FLUX Klein Edit, and the prompt design here follows those findings:

1. **Multi-image input is a composition feature, not a reference feature.** The official Qwen-Image-Edit-2509/2511 docs describe multi-image editing as "person + person", "person + product", "person + scene" fusion, and the 2511 blog highlights "high-fidelity fusion of two separate person images into a coherent group shot". Feeding the model a labelled character atlas therefore *invites* it to paste atlas characters onto the page — this was observed in the first test run (atlas characters composited on top of the colorized page). The mitigation is a prompt that states only Picture 1 may appear in the output, plus the `negative_prompt`.
2. **Optimal input count is 1–3 images** (2509 doc). Page 1 uses 3 (within range); later pages use 4 (page, atlas, style, previous) which is beyond the documented optimum — accepted as the price of sequential continuity, and mitigated by prompt + negative prompt.
3. **Prompt rewriting is officially recommended.** The Qwen team notes edit results become unstable without it and ships a `polish_edit_prompt` VLM helper. The rules it encodes are applied statically here: a direct, specific imperative at the top; explicit statement of *which* image is modified; reference images described by role rather than relied on as-is; positive framing ("keep X unchanged") over negations where possible. A static prompt file can't adapt to the actual reference images, so if instability persists, pre-rewriting `prompt.txt` with a VLM (e.g. Qwen's `polish_edit_prompt` pattern) is the documented next step.
4. **The model addresses images as "Picture 1/2/3/4".** The diffusers pipeline prepends `Picture N: <image>` tokens before the instruction; the prompt uses that vocabulary.
5. **Single-image mode is the strict mode.** With one input image, Qwen-Image-Edit performs appearance editing where "all other regions of the image remain completely unchanged". Multi-image mode relaxes this. If the reference approach keeps leaking, the fallback is a single-image variant (current page only) with the atlas/style described in text.
6. **Official 2511 generation settings:** 40 inference steps, guidance 1.0 + true-CFG 4.0 (the fal endpoint exposes a single `guidance_scale` knob, default 4.5). Outputs are resolution-bucketed to multiples of 16 → 1200×1792.

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
  --num-inference-steps 40 \
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
  --num-inference-steps 40 \
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
  --num-inference-steps 40 \
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
  --num-inference-steps 40 \
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

Per-request native megapixel totals (atlas 1440×2400 = 3.456 MP, style ref 848×1264 ≈ 1.072 MP, page 1200×1800 = 2.16 MP, observed output 1200×1792 = 2.1504 MP):

- **Page 1** (3 inputs + output): 6.688 + 2.16 = 8.848 MP → ≈ **$0.2654**
- **Pages 2–18** (4 inputs + output): 8.848 + 2.16 = 11.008 MP → ≈ **$0.3302**
- **Full 18-page run, native-MP billing: ≈ $5.88** (one page-1 + seventeen later-page costs)

If fal instead caps each input at 1 MP for billing (like the FLUX endpoint), page 1 would be (3 + 2.16) × $0.03 ≈ $0.1548 and later pages (4 + 2.16) × $0.03 ≈ $0.1848, for a full run of ≈ **$3.30**. Both are estimates: the endpoint response does not include the billed amount, and failed requests may still incur charges. Costs are labeled as estimates in the manifest, not measured.

## Reproducibility

The manifest records the command, prompt, negative prompt, model settings (steps, guidance scale, acceleration, image size, seed), source/reference hashes (including the style reference), preprocessing, dependency versions, fal upload and output URLs, request IDs, seeds, timings, safety results, output hashes, and the cost estimate.

## Status and measured results

**First test run (20260808-104057, 2 pages, pre-fix prompt): failed the "add color only" requirement.** Pages 1–2 completed (1200×1792 PNG, $0.5948 total for 2 pages — the estimated cost landed between the two billing scenarios below because outputs were 2.1504 MP and inputs are billed at native MP), but the model pasted characters from the reference atlas on top of the colorized page — the documented multi-image *composition* behavior. The current prompt + `negative_prompt` design was introduced to suppress that. **The fix is untested as of this update**; quality and cost after the fix are unmeasured.

Expected differences versus the FLUX variant: stronger text/instruction adherence (Qwen is a VLM-grounded instruction edit model), 40-step inference (free, since cost is per megapixel), and output at 1200×1792. Remaining risks: residual atlas/style/previous-content leakage despite the prompt, unreliable atlas colors, softened linework on screentone-heavy pages, and the 4-image requests on later pages being beyond Qwen's documented 1–3 image sweet spot.

## API references

- [Qwen Image Edit 2511 model and pricing](https://fal.ai/models/fal-ai/qwen-image-edit-2511)
- [Endpoint API schema](https://fal.ai/models/fal-ai/qwen-image-edit-2511/api)
- [fal Python client](https://fal.ai/docs/reference/client-libraries/python)
