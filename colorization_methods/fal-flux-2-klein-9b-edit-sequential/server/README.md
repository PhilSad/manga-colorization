# Local FLUX.2 Klein 9B inference server (BentoML) for the DGX Spark

Self-hosted replacement for the paid fal endpoint `fal-ai/flux-2/klein/9b/edit`,
serving the same multi-image editing workflow used by `run.py` (current B&W page
+ labelled character-reference atlas + previous colorized page → colorized page).

- Framework: **BentoML** 1.4.x (type-hinted `@bentoml.service` + `@bentoml.api`).
- Model: `black-forest-labs/FLUX.2-klein-9B` (gated, FLUX Non-Commercial License)
  via diffusers `Flux2KleinPipeline` 0.39 + torch 2.13.0+cu130 (aarch64).
- Packaging: everything server-side lives in this directory and is built into a
  **docker image**; the model weights are the only thing kept outside the image
  (an "external module" downloaded to `models/` and mounted read-only).

```
local machine (repo)                          DGX Spark 192.168.1.40
  run.py --endpoint http://spark:3000    -->   docker container flux2-klein:latest
    local_fal_client.py (fal shim)              bentoml serve  (port 3000)
      POST /edit (multipart images+params)        Flux2KleinPipeline (BF16, 4 steps)
                                                    weights mounted from models/
```

## Prerequisites (once)

1. Accept the FLUX Non-Commercial License on
   https://huggingface.co/black-forest-labs/FLUX.2-klein-9B (click
   "Agree and access repository"), then create a **read token** at
   https://huggingface.co/settings/tokens.
2. The DGX Spark must have docker with GPU support (already the case:
   docker 29 + nvidia-container-toolkit, `phil` in the `docker` group).

## Copy to the server

This directory is meant to be moved to the server (e.g. under the workspace):

```bash
scp -r colorization_methods/fal-flux-2-klein-9b-edit-sequential/server \
    spark:/home/phil/agent_workspace/flux2-klein-server
```

All steps below run on the server (`ssh spark`), inside
`/home/phil/agent_workspace/flux2-klein-server`.

## 1. Download the model (external module)

```bash
export HF_TOKEN=hf_xxx
./download_model.sh                 # -> models/FLUX.2-klein-9B/ (~35 GB BF16)
```

## 2. Build and run the container

```bash
docker build -t flux2-klein:latest .          # arm64 build on the server
docker compose up -d                          # or the docker run equivalent below
```

Equivalent plain docker run:

```bash
docker run -d --name flux2-klein --restart unless-stopped --gpus all \
  --shm-size 8g -p 3000:3000 \
  -v "$PWD/models/FLUX.2-klein-9B:/models/flux2-klein:ro" \
  flux2-klein:latest
```

First start loads the model into the GB10's 128 GB unified memory
(≈ 1–3 min); subsequent restarts are faster.

## 3. Verify

```bash
docker ps                              # STATUS should show (healthy) eventually
curl -s http://localhost:3000/healthz  # -> {"status":"ok",...}
curl -s http://localhost:3000/         # Swagger UI with the /edit schema
```

## 4. Smoke test from the client machine (repo)

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-sequential/run.py \
  --model black-forest-labs/FLUX.2-klein-9B \
  --endpoint http://spark:3000 \
  --width 1216 --height 1824 \
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --limit 1
```

`--width 1216 --height 1824` is important: the FLUX VAE requires multiples of
16, and diffusers silently floors non-compliant sizes (1200×1800 → 1200×1792).
1216×1824 is also exactly the size the fal endpoint produced, so outputs are
directly comparable with the existing runs in `output/20260808-011051/`.

## API contract (what local_fal_client.py speaks)

`POST /edit` with `multipart/form-data`:

| field | type | notes |
|---|---|---|
| `images` | repeated file parts, all named `images` | order = `[current, atlas, previous?]`, same as fal |
| `prompt` | text | |
| `width`, `height` | integer text | defaults 1216 / 1824 |
| `num_inference_steps` | integer text | default 4 (Klein is step-distilled) |
| `seed` | integer text | omit for random |
| `output_format` | text | `png` \| `jpeg` \| `webp` |

Response: raw image bytes in the requested format.

## Cost and license

- **$0 per call** (no fal pricing). Electricity estimate: DGX Spark ~350–400 W
  during inference; an 18-page run at ~10–20 min is roughly $0.01–0.02 at
  $0.15/kWh. One-time disk: ~35 GB for the weights + ~15–20 GB image.
- fal comparison (measured 2026-08-08): 18-page chapter + page-18 recovery =
  $1.07955858 across 19 calls; local replaces that with ≈ $0 + energy.
- **License**: FLUX Non-Commercial License — personal/local experimentation is
  fine; commercial use is not. Keep the repo's `LICENSE.md` with the weights.

## Known differences vs the fal endpoint

- fal silently rounded 1200×1800 requests to 1216×1824; locally diffusers
  floors to the nearest multiple of 16 — pass 1216×1824 explicitly for parity.
- fal's optional safety checker is absent locally (the cloud false-positive
  that blacked out page 18 does not exist here; there is also no safety net).
- fal "acceleration" and any endpoint-side instruction tuning are not
  replicated: this is the raw `FLUX.2-klein-9B` checkpoint with the same
  3-image input order. Validate parity by comparing page 1 against the fal
  smoke test (`output/20260808-010413/0134-001.png`, seed 1499077118).
- No auth on the endpoint: bind to the LAN as-is, or use an SSH tunnel
  (`ssh -N -L 3000:localhost:3000 spark`) and bind only locally.

## Troubleshooting / tuning

- **Slow page times**: expected on a single GB10 (roughly 20–60 s/page at
  1216×1824, 4 steps, BF16). Later optimizations: load the transformer in FP8
  (`torch_dtype={"transformer": torch.float8_e4m3fn, "default": torch.bfloat16}`
  in `service.py`) or use `Flux2KleinKVPipeline` with a persistent KV cache of
  the constant reference atlas.
- **OOM**: not expected on 128 GB unified memory (BF16 ≈ 35 GB). If it happens,
  switch the transformer to FP8 (above) or use `FLUX2_WIDTH/HEIGHT` env vars to
  lower the default canvas.
- **Model not found at startup**: check the mount (`docker inspect flux2-klein`
  → Mounts) and that `models/FLUX.2-klein-9B/model_index.json` exists.
- **GPU not used**: confirm `docker exec flux2-klein nvidia-smi` shows the GB10.
- **Rebuild**: `docker compose down && docker compose up -d --build`.

## Verified facts (2026-08-08)

- `Flux2KleinPipeline` exists in diffusers ≥ 0.37.0 (latest 0.39.0); multi-image
  editing is native (`image=[target, ref1, ...]`, order matters); Klein is
  step-distilled to 4 steps and ignores `guidance_scale`; dims must be multiples
  of 16 (VAE packs latents into 2×2 patches); no safety checker in the pipeline.
- PyTorch publishes official aarch64 CUDA wheels (cu130 index, up to 2.13.0,
  cp312) — no NGC container needed for the app layer. GB10 is sm_121
  (a.k.a. "sm_103" in some docs); prebuilt cu130 wheels support it; do not
  install flash-attn (fails on GB10).
- BentoML 1.4.39 is a pure-Python wheel (aarch64 OK). `DiffusersRunner` is
  deprecated; the current pattern is `@bentoml.service` + `@bentoml.on_startup`
  + type-hinted `@bentoml.api`. Multipart lists use repeated field names
  (`form.getlist`), non-file fields are JSON-parsed form values.
- `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` has an arm64 manifest and the
  host driver (CUDA 13.0) is ≥ 12.8. GPU passthrough was verified on the server.
- Model repos are gated: `black-forest-labs/FLUX.2-klein-9B` (complete diffusers
  layout: Qwen3 8B text encoder + 9B transformer + VAE, ≈ 35 GB BF16) and
  `black-forest-labs/FLUX.2-klein-9b-fp8` (transformer-only single file).
- The fal endpoint schema (for parity): `prompt`, `image_urls` (≤ 4),
  `image_size`, `num_inference_steps` (min 4), `num_images`, `seed`,
  `output_format`, `enable_safety_checker`, `acceleration`; output `images[]`,
  `seed`, `timings`, `has_nsfw_concepts`, `prompt`.
