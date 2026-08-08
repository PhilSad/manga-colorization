# Local FLUX.2 Klein 9B inference server on the DGX Spark — setup plan

Status: **plan (not yet implemented)** · Created: 2026-08-08 · Framework: **BentoML**
Goal: replace the paid fal endpoint `fal-ai/flux-2/klein/9b/edit` with a self-hosted
inference server on the DGX Spark (192.168.1.40), serving the same multi-image
editing workflow used by `run.py` (current B&W page + labelled reference atlas +
previous colorized page → colorized page), and let the existing sequential
colorization pipeline target it with minimal changes.

---

## 1. Architecture

```
┌──────────────────────────── local machine (repo) ────────────────────────────┐
│  run.py --endpoint http://spark:3000                                          │
│    └─ local_fal_client.py  (fal-compatible shim: upload_file/submit/get)      │
│         │  HTTP multipart: images[1..3] + prompt + params                     │
└─────────┼────────────────────────────────────────────────────────────────────┘
          ▼
┌──────────────────────────── DGX Spark 192.168.1.40 ──────────────────────────┐
│  Docker container: bento-flux2-klein:latest                                   │
│    BentoML service  (bentoml 1.4.39, Python 3.12, CUDA 12.8.1 base image)     │
│      Flux2KleinPipeline (diffusers 0.39.0, torch 2.13.0+cu130 aarch64)        │
│      weights: black-forest-labs/FLUX.2-klein-9B (gated, non-commercial)       │
│    GPU: GB10 Blackwell, 128 GB unified memory (model ≈ 35 GB BF16)            │
└───────────────────────────────────────────────────────────────────────────────┘
```

Facts verified 2026-08-08 (see research notes in §8):

- `Flux2KleinPipeline` exists in diffusers ≥ 0.37.0 (latest 0.39.0). Multi-image
  editing is native: `pipe(prompt=..., image=[target, ref1, ref2], ...)` — the
  image list order is the same one `run.py` already uses (#1 current, #2 atlas,
  #3 previous page).
- Klein is step-distilled to **4 inference steps**; `guidance_scale` is ignored
  for the distilled model, so the local service does not need it.
- diffusers' FLUX.2 pipelines have **no safety checker** (fal's did, and it
  false-positived page 18). Locally the false-positive path disappears entirely.
- PyTorch publishes official **aarch64 CUDA wheels** (`torch 2.13.0+cu130`,
  manylinux_2_28_aarch64, cp312) — no NGC container required for the app layer.
- BentoML 1.4.39 is a pure-Python wheel (aarch64 OK). The recommended pattern is
  `@bentoml.service` + `HuggingFaceModel` + `@bentoml.on_startup` +
  type-hinted `@bentoml.api` (`DiffusersRunner` is deprecated since 1.4).
- `bentoml containerize` on the arm64 host builds `linux/arm64` natively; the
  default CUDA base `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` has an arm64
  manifest and the host driver (13.0) is ≥ 12.8. GPU passthrough verified working
  on spark (`docker run --gpus all nvidia/cuda:... nvidia-smi` ✓).

---

## 2. Prerequisites

1. **HuggingFace access to the gated model** (required; both the full model and
   the fp8 variant are gated under the FLUX Non-Commercial License):
   - In a browser, log in to huggingface.co, open
     https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and click
     **Agree and access repository**.
   - Create a read token: Settings → Access Tokens → New token (type "read") →
     copy it. Keep it in a safe place; it will be exported as `HF_TOKEN` on the
     server. (The user already has accounts for fal/Gemini/OpenAI in the repo
     `.env`, but there is no HF token yet — `huggingface-cli` is not installed
     locally and no `HF_TOKEN` is set.)
2. **SSH access to spark** — already working (`ssh spark`), user `phil` is in the
   `docker` group.
3. **Disk**: model ≈ 35 GB (BF16) + docker image ≈ 15–20 GB + build cache.
   1.6 TB free on spark — fine.
4. **License note**: the FLUX Non-Commercial License permits personal/local
   experimentation but **not commercial use**. Fine for this project; keep the
   `LICENSE.md` from the repo next to the weights and record it in the manifest
   when runs are logged.

---

## 3. Server-side setup, step by step

All commands below run on the DGX Spark (`ssh spark`).

### 3.1 One-time environment

```bash
# optional but recommended: put the token in ~/.bashrc or a root-only file
export HF_TOKEN=hf_xxx
```

### 3.2 (Recommended) Pre-download weights into the HF cache

This makes the first `bentoml serve` / `bentoml containerize` deterministic and
lets the Bento reuse the cache instead of re-downloading:

```bash
cd /home/phil/agent_workspace
python3 -m venv hf-dl-venv && . hf-dl-venv/bin/activate
pip install -U "huggingface_hub[cli]"
HF_TOKEN=$HF_TOKEN huggingface-cli download black-forest-labs/FLUX.2-klein-9B
```

(≈ 35 GB; alternatively download inside the Bento build/run step — see §3.5.)

### 3.3 Create the BentoML service project

Directory to create on the server (mirroring the repo, kept out of git):

```text
/home/phil/agent_workspace/flux2-klein-server/
├── service.py          # BentoML service (sketch in §4.1)
├── bentofile.yaml      # build manifest (sketch in §4.2)
├── requirements.txt    # deps (sketch in §4.3)
└── README.md           # copy of the run instructions in this plan
```

### 3.4 Dev run (fast iteration, no image build)

```bash
cd /home/phil/agent_workspace/flux2-klein-server
python3 -m venv .venv && . .venv/bin/activate
pip install -U bentoml
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
HF_TOKEN=$HF_TOKEN bentoml serve service.py:Flux2Klein --port 3000
```

- Binds `0.0.0.0:3000` (Swagger UI at `http://spark:3000`).
- First start downloads/loads the model (several minutes), then it is resident
  in the 128 GB unified memory. Startup takes ~1–3 min per subsequent start.
- Smoke-test with the one-page request (§6) before going further.

### 3.5 Containerized run (persistent server)

```bash
cd /home/phil/agent_workspace/flux2-klein-server
. .venv/bin/activate
HF_TOKEN=$HF_TOKEN bentoml build
HF_TOKEN=$HF_TOKEN bentoml containerize flux2-klein-server:latest \
  --image-tag bento-flux2-klein:latest
docker run -d --name flux2-klein --restart unless-stopped \
  --gpus all --shm-size=8g -p 3000:3000 \
  -e HF_TOKEN=$HF_TOKEN \
  bento-flux2-klein:latest
docker logs -f flux2-klein   # watch startup
```

Notes:
- `bentoml containerize` bundles the model weights into the image via
  `HuggingFaceModel` (HF token needed at build time).
- If the default CUDA base image misbehaves on arm64 (fallback path, unlikely):
  build with a custom Dockerfile `FROM nvcr.io/nvidia/pytorch:25.10-py3`
  (NGC PyTorch container, CUDA 13 for GB10) and run `bentoml serve` inside it.
- To upgrade: rebuild and `docker rm -f flux2-klein` before `docker run`.

### 3.6 Health checks

```bash
curl -s http://localhost:3000/healthz | head -c 200; echo
docker exec flux2-klein nvidia-smi | head -15     # confirm GPU residency
```

---

## 4. Proposed server files (to be created in the implementation step)

### 4.1 `service.py` (sketch — exact multipart mapping to be verified in §7)

```python
import bentoml, torch
from bentoml.models import HuggingFaceModel
from diffusers import Flux2KleinPipeline
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

@bentoml.service(
    image=bentoml.images.Image(python_version="3.12")
              .requirements_file("requirements.txt"),
    resources={"gpu": 1},
    traffic={"timeout": 900, "concurrency": 1},   # single GPU: serialize
    envs=[{"name": "HF_TOKEN"}],
)
class Flux2Klein:
    model_path = HuggingFaceModel(MODEL_ID)

    @bentoml.on_startup
    def load(self):
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.model_path,
            torch_dtype={"transformer": torch.float8_e4m3fn,
                         "default": torch.bfloat16},   # or plain torch.bfloat16
            use_safetensors=True,
        ).to("cuda")

    @bentoml.api
    def edit(self, images: list[Image], prompt: str,
             height: int = 1216, width: int = 1824,
             num_inference_steps: int = 4,
             seed: int | None = None,
             output_format: str = "png") -> bytes:
        gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        out = self.pipe(prompt=prompt, image=images,
                        height=height, width=width,
                        num_inference_steps=num_inference_steps,
                        generator=gen, output_type="pil").images[0]
        fmt = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[output_format]
        import io
        buf = io.BytesIO()
        out.save(buf, format=fmt, quality=95 if fmt != "PNG" else None)
        return buf.getvalue()
```

Notes on this sketch:
- **Image list order must match `run.py`**: `[current_page, reference_atlas]` or
  `[current_page, reference_atlas, previous_page]` — the same order fal receives.
- **Dimensions**: fal's endpoint returned **1216×1824** for a 1200×1800 request
  (fal rounds to VAE-friendly buckets; 1800 is not divisible by 16). Start with
  1216×1824 to be bit-for-bit comparable with the existing fal outputs, and test
  whether 1200×1800 works locally (see §7, open question O-2).
- **Precision**: start with plain `torch.bfloat16` (≈ 35 GB, simplest). The fp8
  transformer swap (`torch_dtype` dict, ≈ 26 GB) is a documented diffusers
  feature and can be added later; the gated `-9b-fp8` repo is transformer-only
  (single-file, no configs) so it is easier to keep the full repo and only switch
  the transformer dtype.
- **Return type**: returning `bytes` is the intent; if the type-hinted API does
  not map it cleanly, fall back to `input=bentoml.io.Multipart(...)` +
  `output=bentoml.io.Binary()` descriptors (bentoml.io still ships in 1.4.x).
- **Optional future optimization**: `Flux2KleinKVPipeline` (diffusers ≥ 0.37)
  KV-caches reference images — for our workflow the atlas is constant across the
  18 pages, so a persistent KV cache of atlas (+ previous page) would cut a large
  chunk of the per-page compute. Defer to a follow-up; the KV variant also has
  its own gated weights repo (`FLUX.2-klein-9b-kv`).

### 4.2 `bentofile.yaml`

```yaml
service: "service.py:Flux2Klein"
name: flux2-klein-server
python:
  requirements_txt: requirements.txt
docker:
  cuda_version: "12.8"        # arm64 manifest verified; host driver 13.0 ≥ 12.8
```

### 4.3 `requirements.txt`

```text
--extra-index-url https://download.pytorch.org/whl/cu130
bentoml>=1.4.39
diffusers>=0.39.0,<0.40
torch>=2.13.0
transformers>=4.51
accelerate>=0.31
safetensors>=0.8
pillow
huggingface-hub>=0.34
```

(Do **not** install flash-attn — it fails on GB10; diffusers' native SDPA suffices.)

---

## 5. Client integration (repo side, implementation step)

Goal: `run.py` targets the local server with a tiny change; all manifest,
provenance, and sequential-context logic stays intact.

1. New file `colorization_methods/fal-flux-2-klein-9b-edit-sequential/local_fal_client.py`
   — a fal-compatible shim:
   - `upload_file(path) -> str` returns an internal id (`local://<n>`) and keeps a
     `path -> id` registry (no real upload; the atlas/previous-page reuse pattern
     is preserved in code).
   - `submit(model, arguments) -> Handler` collects `arguments["image_urls"]`
     (all `local://` ids), loads the images, and issues one HTTP POST to
     `{endpoint}/edit` (multipart `images` + `prompt`, `width`, `height`,
     `num_inference_steps`, `seed`, `output_format`).
   - `Handler.request_id` = uuid; `Handler.get()` writes the returned bytes to a
     temp file and returns a fal-shaped dict:
     `{"images": [{"url": "file:///tmp/..."}], "seed": ..., "timings": {...},
     "has_nsfw_concepts": [False], "prompt": ...}` — `run.py`'s existing
     `download_file()` works with `file://` URLs via `urllib`.
2. Patch `run.py`:
   - add `--endpoint <url>` (default: unset → fal);
   - when set: `import local_fal_client as fal_client`, skip the API-key check,
     record `configuration.model` as
     `black-forest-labs/FLUX.2-klein-9B via <endpoint>`, and switch the manifest's
     `pricing_assumptions` to a "self-hosted" block ($0 per call, electricity
     estimate, license note) instead of the fal $/MP pricing. The `estimate_cost`
     totals then reflect local cost (≈ 0) + timing, per AGENTS.md ("label
     estimates clearly; do not present different setups as comparable").
3. Update the method `README.md` with a "Local inference server" section pointing
   to `server/`, the run command, and a note that outputs are local (1216×1824
   vs fal's same 1216×1824 — sizes will match if §4.1's dimensions are used).

No changes are needed to `data/`; inputs stay untouched. Each local run still
creates a fresh timestamped output dir exactly as today.

---

## 6. Validation and parity test

1. **Smoke test** (1 page, after server is up) — reproduces the fal smoke test
   `output/20260808-010413/`:

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

   Compare against fal's page-1 output on the same seed (1499077118): panel
   geometry, line preservation, palette coherence, artifact level. Expected
   differences: exact 1216×1824 output, possibly no safety-checker interference,
   and whatever divergence the fal "edit" serving layer (acceleration/prompt
   handling) introduces vs the raw base weights.
2. **Full chapter** (18 pages, sequential): run the same command without
   `--limit`, then diff/eyeball all pages against
   `output/20260808-011051/` + the page-18 recovery. Record objective notes
   (dimensions, hashes, timings) and subjective notes in the run manifest, per
   AGENTS.md.
3. **Continuity check**: verify the atlas + previous-page context behaves like
   fal's (the local pipeline receives the same three images in the same order).
4. **Failure modes to probe**: page 18 (fal false-positive safety block — local
   should just work); a page with heavy screentones (line smoothing behavior).

---

## 7. Open questions to verify during implementation

- **O-1 Multipart contract**: exact field naming for `list[Image]` with
  BentoML's type-hinted API (N files under one field vs `images1..N`); fallback
  is explicit `bentoml.io.Multipart` descriptors. Confirm with the smoke test.
- **O-2 Output size**: does diffusers generate 1200×1800 directly, or does it
  need 1216×1824 (or 1200×1792) for the FLUX VAE? Compare 1200×1800 vs
  1216×1824 outputs before committing.
- **O-3 Base image on arm64**: confirm `bentoml containerize` default CUDA image
  builds/runs on arm64 (fallback: custom Dockerfile from NGC PyTorch container).
- **O-4 Parity with fal's "edit"**: assume fal serves the base
  `FLUX.2-klein-9B` weights for its `/edit` endpoint (no secret LoRA); validate
  by comparing outputs. If divergence is large, reconsider
  community edit LoRAs (several exist on HF, e.g. `dumplingtoto/...-lora-edit`).
- **O-5 fp8 / KV pipeline**: decide after the BF16 baseline runs (speed/memory
  measurements).

---

## 8. Research sources (verified 2026-08-08)

- diffusers: `Flux2KleinPipeline` since 0.37.0, latest 0.39.0 —
  https://huggingface.co/docs/diffusers/api/pipelines/flux2 ;
  `pipeline_flux2_klein.py` (image list = editing references, order matters;
  distilled → 4 steps, guidance ignored; no safety checker).
- PyTorch aarch64 CUDA wheels (2.9+ announcement, cu130 wheels incl. 2.13.0):
  https://pytorch.org/blog/pytorch-2.9/ ; wheel index
  https://download.pytorch.org/whl/cu130/ (manylinux_2_28_aarch64, cp312) —
  GB10 reported as sm_121 (a.k.a. "sm_103" in some docs); prebuilt cu130 wheels
  support it out of the box; skip flash-attn.
- Model repo (gated, non-commercial): https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
  (complete diffusers layout: Qwen3 8B text encoder, 9B transformer, VAE).
  fp8 variant (transformer-only single file, also gated):
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8
- BentoML 1.4.39: `DiffusersRunner` deprecated; current pattern is
  `@bentoml.service` + `HuggingFaceModel` + `@bentoml.on_startup` + type-hinted
  `@bentoml.api` — see `bentoml/BentoDiffusion/flux-timestep-distilled/service.py` ;
  containerize default CUDA base 12.8.1 (arm64 manifest verified on Docker Hub).
- fal endpoint schema (for parity): `fal-ai/flux-2/klein/9b/edit` — input
  `prompt`, `image_urls` (≤ 4), `image_size`, `num_inference_steps` (min 4),
  `num_images`, `seed`, `output_format`, `enable_safety_checker`, `acceleration`;
  output `images[]`, `seed`, `timings`, `has_nsfw_concepts`, `prompt`.
- NVIDIA DGX Spark GB10 playbooks (ComfyUI/llama.cpp examples confirm the cu130
  aarch64 + CUDA 13 driver stack): https://github.com/NVIDIA/dgx-spark-playbooks
- DGX Spark on-site facts: GPU GB10, driver 580.159.03, CUDA 13.0, 121 GB RAM,
  docker 29.2.1 + nvidia-container-toolkit 1.19.1 active, phil ∈ docker group,
  internet OK, 1.6 TB free, LAN IP 192.168.1.40.

## 9. Cost comparison summary (for the eventual methods.md note)

| Item | fal endpoint (measured 2026-08-08) | Local server (estimate) |
|---|---|---|
| Full 18-page chapter | $1.02216076 (19 calls incl. recovery: $1.07955858) | **$0 per call** |
| Electricity per run | — | ≈ 350–400 W × ~10–20 min ≈ $0.01–0.02 @ $0.15/kWh (estimate) |
| Hardware | — | already owned (DGX Spark); one-time disk ≈ 35 GB |
| Setup effort | none | one-time ~1–2 h (this plan) |
| Speed (per page, 4 steps) | ~1.5–1.9 s model time (fal datacenter GPU) | expect 20–60 s on GB10 (measure; 4-step distilled) |
| License | fal terms | FLUX Non-Commercial License (personal use OK) |

Do not present fal and local results as directly comparable without the
dimension/serving differences noted (§6, §7 O-2/O-4).
