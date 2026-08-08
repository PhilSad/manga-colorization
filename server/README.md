# Local FLUX.2 Klein 9B inference server (BentoML) for the DGX Spark

Self-hosted replacement for the paid fal endpoint `fal-ai/flux-2/klein/9b/edit`,
serving the same multi-image editing workflow used by `run.py` (current B&W page
+ labelled character-reference atlas + previous colorized page → colorized page).
Optionally loads a LoRA (`manga_colorization.safetensors` by thedeoxen) on top of
the undistilled **base** model for reference-driven manga colorization.

- Framework: **BentoML** 1.4.x (type-hinted `@bentoml.service` + `@bentoml.api`).
- Model: `black-forest-labs/FLUX.2-klein-9B` (gated, FLUX Non-Commercial License)
  or, for the LoRA, `black-forest-labs/FLUX.2-klein-base-9B` (same license),
  via diffusers `Flux2KleinPipeline` 0.39 + torch 2.13.0+cu130 (aarch64).
- Packaging: everything server-side lives in this directory and is built into a
  **docker image**; the model weights and the LoRA are the only things kept
  outside the image (external modules downloaded to `models/` and mounted
  read-only).

```
local machine (repo)                          DGX Spark 192.168.1.40
  run.py --endpoint http://spark:3000    -->   docker container flux2-klein:latest
    local_fal_client.py (fal shim)              bentoml serve  (port 3000)
      POST /edit (multipart images+params)        Flux2KleinPipeline (BF16)
                                                    weights mounted from models/
                                                    optional LoRA (FLUX2_LORA_PATH)
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
scp -r server spark:/home/phil/agent_workspace/flux2-klein-server
```

(`server/` is at the root of this repository.)

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

Equivalent plain docker run (LoRA deployment, base model):

```bash
docker run -d --name flux2-klein --restart unless-stopped --gpus all \
  --shm-size 8g -p 3000:3000 \
  -v "$PWD/models/FLUX.2-klein-base-9B:/models/flux2-klein:ro" \
  -v "$PWD/models/FLUX.2-klein-lora:/models/flux2-klein-lora:ro" \
  -e FLUX2_LORA_PATH=/models/flux2-klein-lora/manga_colorization.safetensors \
  -e FLUX2_LORA_SCALE=1.0 \
  -e FLUX2_GUIDANCE_SCALE=4.0 \
  -e FLUX2_STEPS=20 \
  flux2-klein:latest
```

First start loads the model into the GB10's 128 GB unified memory
(≈ 1–3 min); subsequent restarts are faster. The base model + LoRA needs more
steps than the distilled 4-step model: expect ≈ 5× slower per page at 20 steps.

## 2b. Optional: manga-colorization-by-reference LoRA

[`thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA`]
(https://huggingface.co/thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA)
(public, Apache-2.0) colorizes B&W manga pages using a color-reference image;
rank-32, transformer-only, trigger word **`mngclranm`**.

- **Trained on the undistilled base model** `black-forest-labs/FLUX.2-klein-base-9B`
  (gated — accept the FLUX Non-Commercial License on the model page first). It is
  NOT step-distilled: use ~20–50 `num_inference_steps` and `guidance_scale` ~4–5
  (the ComfyUI workflow the author ships uses 20 steps / CFG 5 / LoRA weight 1.0).
  The step-distilled `FLUX.2-klein-9B` ignores `guidance_scale` and runs 4 steps;
  the LoRA technically loads there too (same transformer architecture) but is
  off-label — the author's settings assume the base model.
- Download the weights once on the host (no HF token needed):

  ```bash
  ./download_lora.sh                        # -> models/FLUX.2-klein-lora/
  FLUX2_MODEL_ID=black-forest-labs/FLUX.2-klein-base-9B \
    HF_TOKEN=hf_xxx ./download_model.sh     # -> models/FLUX.2-klein-base-9B/ (~35 GB)
  ```

- The `docker-compose.yml` in this repo is already wired for the LoRA deployment
  (base model + LoRA mounts, `FLUX2_LORA_PATH`/`FLUX2_LORA_SCALE=1.0`/
  `FLUX2_GUIDANCE_SCALE=4.0`/`FLUX2_STEPS=20`). Make sure BOTH model dirs exist
  before `docker compose up -d`, otherwise the container will not start.
- **Choosing whether the LoRA is loaded** — this is decided when the container
  starts, via the `FLUX2_LORA_PATH` env var. No rebuild needed, just restart
  with the var set, empty, or removed:

  ```bash
  # LoRA loaded (base model + adapter)
  docker compose up -d                            # compose already sets FLUX2_LORA_PATH

  # LoRA NOT loaded — plain checkpoint (empty string is treated as unset)
  FLUX2_LORA_PATH= docker compose up -d

  # Same for the plain docker run: just omit the -e FLUX2_LORA_PATH flag
  ```

  The startup log confirms which mode is active: `Loading LoRA <path> at scale
  ...` when the adapter is applied, nothing when it is skipped. If
  `FLUX2_LORA_PATH` points at a missing file, the service fails fast at startup
  (`FileNotFoundError`) instead of silently serving without the LoRA.

  **Per-request override without restart:** every `/edit` call accepts
  `lora_scale` even when a LoRA is loaded — `lora_scale=0` disables the adapter
  for that single call (output = pure base model), `0.8–1.0` tunes the
  color-transfer strength. Useful for A/B comparisons against the plain model.

- **Caveat:** steps/guidance are *model*-dependent, not LoRA-dependent. The
  compose defaults (`FLUX2_STEPS=20`, `FLUX2_GUIDANCE_SCALE=4.0`) are right for
  the base model with or without the LoRA. If you switch the mount back to a
  step-distilled checkpoint (`FLUX.2-klein-9B`/`-4B`), set `FLUX2_STEPS=4`
  (guidance is ignored there). Clients can override both per request anyway.
- Clients: `example_client.py` takes `--guidance-scale` and `--lora-scale`;
  `run.py` accepts `--guidance-scale`/`--lora-scale`, forwarded to the server
  only when `--endpoint` is set (fal's schema has no such fields). Example:

  ```bash
  python server/example_client.py \
    --endpoint http://spark:3000 \
    --images data/chapter_134/0134-001.png data/refs/frieren_reference.webp \
    --prompt "mngclranm, flat anime-style color, silver-white hair, emerald green eyes" \
    --width 1216 --height 1824 --steps 20 --guidance-scale 4.0 --lora-scale 1.0 \
    --output lora-page1.png
  ```

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

## 5. Standalone example client

[`example_client.py`](example_client.py) is a minimal, dependency-free
(stdlib + Pillow) script that calls the server directly — useful for quick
experiments without the full sequential pipeline:

```bash
python server/example_client.py \
  --endpoint http://spark:3000 \
  --images data/chapter_134/0134-001.png data/refs/frieren_reference.webp \
  --prompt "Add flat anime-style color to the black-and-white manga page. \
            Frieren has silver-white hair and emerald green eyes." \
  --width 1216 --height 1824 --steps 4 --seed 42 \
  --output colorized-page1.png
```

It normalizes each input to a true PNG before upload (source files may carry
misleading extensions, e.g. JPEG data in `.png` containers) and supports
`--seed`, `--output-format`, and `--prompt-file`. The docstring also shows the
equivalent `curl` command for the same request. Verified 2026-08-08 against the
running server: produced a 1216×1824 RGB PNG in ~13 s.

## API contract (what local_fal_client.py speaks)

`POST /edit` with `multipart/form-data`:

| field | type | notes |
|---|---|---|
| `images` | repeated file parts, all named `images` | order = `[current, atlas, previous?]`, same as fal |
| `prompt` | text | include the trigger word `mngclranm` for the LoRA |
| `width`, `height` | integer text | defaults 1216 / 1824 |
| `num_inference_steps` | integer text | default 4 (distilled) / 20 (LoRA compose); base model wants 20–50 |
| `guidance_scale` | float text | default `FLUX2_GUIDANCE_SCALE` (4.0); ignored by the distilled model, ~4–5 for the LoRA base model |
| `lora_scale` | float text | optional per-request LoRA weight override (0.8–1.0); ignored when no LoRA is loaded |
| `seed` | integer text | omit for random |
| `output_format` | text | `png` \| `jpeg` \| `webp` |

Response: raw image bytes in the requested format.

## Verified on 2026-08-08 (end-to-end run on the DGX Spark)

- Docker image built successfully on the server (arm64, 9.45 GB; weights not
  baked in). `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` + Python 3.12 venv
  + torch 2.13.0+cu130 aarch64 + diffusers 0.39.0 + transformers 4.57.6.
- Three runtime requirements were found by testing and are already in the
  Dockerfile/`requirements.txt`: `build-essential` (gcc) **and
  `python3.12-dev` (Python.h)** — torch 2.13 JIT-compiles Triton CUDA kernels
  on first inference and fails without both ("Failed to find C compiler", then
  "fatal error: Python.h") — **and `peft`**: diffusers 0.39's
  `load_lora_weights(..., adapter_name=...)` raises
  `ValueError: PEFT backend is required for this method.` without it (hit on
  the first 9B+LoRA start, 2026-08-08; without `peft` the container loops
  "Application startup failed" and stays `health: starting`).
- Verified with the ungated `black-forest-labs/FLUX.2-klein-4B` stand-in
  (identical diffusers layout): container healthy, `healthz` 200, model
  resident on GPU (~15 GB BF16), and a full client round-trip
  (`run.py --endpoint http://spark:3000 --limit 1`) produced a real 1216×1824
  RGB PNG in ~17.4 s wall time at $0 cost — output kept in
  `colorization_methods/.../output/20260808-095523/`.
- 9B deployment verified 2026-08-08 after accepting the FLUX Non-Commercial
  License (base repo `black-forest-labs/FLUX.2-klein-base-9B`):
  `./download_model.sh` (with `FLUX2_MODEL_ID=black-forest-labs/FLUX.2-klein-base-9B`)
  pulled all 29 files (~35 GB BF16, incl. the monolithic
  `flux-2-klein-base-9b.safetensors`), then `docker compose up -d --build`
  replaced the 4B stand-in. Container healthy, healthz 200; a client smoke
  test (`example_client.py --steps 20 --guidance-scale 4.0 --lora-scale 1.0`,
  prompt with the `mngclranm` trigger word) produced a real 1216×1824 RGB PNG
  in ~253 s wall (incl. ~1 min first-inference Triton recompile), 2026-08-08.
  Note: `models/FLUX.2-klein-9B` is the step-distilled checkpoint and remains
  unauthorized on this account (403) — not needed by the compose deployment.
- First inference per container recompiles Triton kernels (~1 min); the
  `triton-cache/` volume in docker-compose.yml persists that cache.

## Cost and license

- **$0 per call** (no fal pricing). Electricity estimate: DGX Spark ~350–400 W
  during inference; an 18-page run at ~10–20 min is roughly $0.01–0.02 at
  $0.15/kWh. One-time disk: ~35 GB for the base weights + ~165 MB for the LoRA
  + ~15–20 GB image. The base model at 20–50 steps draws longer than the
  distilled 4-step model (~20–60 s/page at 4 steps; ~5× at 20 steps).
- fal comparison (measured 2026-08-08): 18-page chapter + page-18 recovery =
  $1.07955858 across 19 calls; local replaces that with ≈ $0 + energy.
- **License**: FLUX Non-Commercial License — personal/local experimentation is
  fine; commercial use is not. Keep the repo's `LICENSE.md` with the weights.
  The LoRA itself is Apache-2.0 (public).

## Known differences vs the fal endpoint

- fal silently rounded 1200×1800 requests to 1216×1824; locally diffusers
  floors to the nearest multiple of 16 — pass 1216×1824 explicitly for parity.
- fal's optional safety checker is absent locally (the cloud false-positive
  that blacked out page 18 does not exist here; there is also no safety net).
- fal "acceleration" and any endpoint-side instruction tuning are not
  replicated: this is the raw `FLUX.2-klein-9B` checkpoint with the same
  3-image input order. Validate parity by comparing page 1 against the fal
  smoke test (`output/20260808-010413/0134-001.png`, seed 1499077118).
- The LoRA deployment replaces the distilled checkpoint with the undistilled
  **base** model (as the LoRA author requires), so step count and guidance
  behavior differ from the fal endpoint; outputs are not directly comparable
  to the 4-step runs.
- No auth on the endpoint: bind to the LAN as-is, or use an SSH tunnel
  (`ssh -N -L 3000:localhost:3000 spark`) and bind only locally.

## Troubleshooting / tuning

- **Slow page times**: expected on a single GB10 (roughly 20–60 s/page at
  1216×1824, 4 steps, BF16; ~5× longer for the base model at 20 steps). Later
  optimizations: load the transformer in FP8
  (`torch_dtype={"transformer": torch.float8_e4m3fn, "default": torch.bfloat16}`
  in `service.py`) or use `Flux2KleinKVPipeline` with a persistent KV cache of
  the constant reference atlas.
- **LoRA not loaded**: check `docker inspect flux2-klein` → Mounts for
  `models/FLUX.2-klein-lora`, and the container log for `Loading LoRA ...`;
  `FLUX2_LORA_PATH` must be a .safetensors **file** path. diffusers 0.39
  auto-converts the ai-toolkit key layout (`diffusion_model.double_blocks.*`)
  to the Flux2 transformer — no manual remap needed.
- **Distilled model ignoring guidance**: expected — diffusers logs a warning
  and disables CFG for step-distilled checkpoints; the LoRA's base model uses
  `guidance_scale` normally.
- **OOM**: not expected on 128 GB unified memory (BF16 ≈ 35 GB). If it happens,
  switch the transformer to FP8 (above) or use `FLUX2_WIDTH/HEIGHT` env vars to
  lower the default canvas.
- **Model not found at startup**: check the mount (`docker inspect flux2-klein`
  → Mounts) and that the mounted `model_index.json` exists (e.g.
  `models/FLUX.2-klein-base-9B/model_index.json`).
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
