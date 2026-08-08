# FLUX.2 Klein 9B base + manga-colorization LoRA, sequential page context

Variant of the [fal FLUX.2 Klein 9B Edit method](../fal-flux-2-klein-9b-edit-sequential/)
that targets the **self-hosted** BentoML server (see [`server/`](../../server/)),
which loads [thedeoxen's manga-colorization-by-reference LoRA](https://huggingface.co/thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA)
(trigger word **`mngclranm`**) on top of the undistilled
`black-forest-labs/FLUX.2-klein-base-9B` checkpoint.

Each request supplies the current monochrome page as `#1` and a labelled
character-reference atlas as `#2`. From page 2 onward, the preceding generated
page is supplied as `#3` for color continuity. The prompt starts with the
LoRA's trigger word `mngclranm`, which activates the adapter.

The base model is **NOT step-distilled**: use `--num-inference-steps 20-50` and
`--guidance-scale 4-5` (the defaults here are 20 steps / 1216×1824). The LoRA
weight is tunable per request with `--lora-scale` (0.8–1.0 recommended; `0`
disables the adapter for a single call). Compared with the distilled 4-step
fal method, expect roughly **5× longer per page**.

Each invocation creates a fresh local-time `output/YYYYMMDD-HHMMSS/`
directory with the output images, normalized request inputs, atlas, and
incremental manifest. Never overwrites previous runs.

## Setup

Dependencies are the same as the fal method (the local server replaces the
fal endpoint client-side via `local_fal_client.py`, a fal-compatible shim):

```bash
.venv/bin/python -m pip install \
  -r colorization_methods/flux-2-klein-9b-base-lora-edit-sequential/requirements.txt
```

No API key is required (no fal calls). The server deployment is described in
[`server/README.md`](../../server/README.md); the compose deployment there
serves exactly the base model + LoRA this method targets.

## Run (self-hosted, Spark backend)

```bash
.venv/bin/python colorization_methods/flux-2-klein-9b-base-lora-edit-sequential/run.py \
  --model black-forest-labs/FLUX.2-klein-base-9B \
  --endpoint http://spark:3000 \
  --width 1216 \
  --height 1824 \
  --num-inference-steps 20 \
  --guidance-scale 4.0 \
  --lora-scale 1.0 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs
```

Notes for local runs:

- `--width 1216 --height 1824`: the FLUX VAE needs multiples of 16 (diffusers
  floors non-compliant sizes) and matches fal's actual outputs for comparison.
- `--guidance-scale` and `--lora-scale` are forwarded to the server only when
  `--endpoint` is set (fal's schema has no such fields).
- The first request per container pays the model-loading cost (~1–3 min) and
  the first inference recompiles Triton kernels (~1 min; persisted in the
  server's `triton-cache/` volume afterwards).
- Local runs record the self-hosted pricing block in the manifest ($0 per
  call, electricity estimate) instead of fal's $/MP pricing — do not compare
  local and fal costs as if identical (see `server/README.md`).

## One-page smoke test

```bash
.venv/bin/python colorization_methods/flux-2-klein-9b-base-lora-edit-sequential/run.py \
  --model black-forest-labs/FLUX.2-klein-base-9B \
  --endpoint http://spark:3000 \
  --width 1216 \
  --height 1824 \
  --num-inference-steps 20 \
  --guidance-scale 4.0 \
  --lora-scale 1.0 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --limit 1
```

## Reproducibility and cost

The manifest records the command, prompt (including the `mngclranm` trigger
word), model settings, source/reference hashes, preprocessing, dependency
versions, seeds, timings, output hashes, and cost.

**$0 per call** (self-hosted; no fal pricing). Electricity estimate: DGX Spark
~350–400 W during inference; an 18-page run at ~10–20 min is roughly
$0.01–0.02 at $0.15/kWh. One-time disk: ~35 GB base weights + ~165 MB LoRA.
Licensing: FLUX Non-Commercial License for the base weights (accepted on the
HF model page); the LoRA is Apache-2.0.

## API references

- [thedeoxen manga-colorization-by-reference LoRA](https://huggingface.co/thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (gated)
- [Server deployment guide](../../server/README.md)
