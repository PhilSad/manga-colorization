#!/usr/bin/env python3
"""BentoML inference service for FLUX.2 Klein 9B (manga colorization workflow).

Replicates the multi-image editing workflow of fal's
`fal-ai/flux-2/klein/9b/edit` endpoint: one target image plus up to three
reference images plus a text prompt, returning a generated image.

The model weights are NOT bundled in the container image. They are downloaded
once on the host (`./download_model.sh`) and mounted into the container at
`/models/flux2-klein` (override with the FLUX2_MODEL_PATH env var).

HTTP contract (POST /edit, multipart/form-data):
  images                 repeated file parts, all named "images", in the order
                         [current_page, reference_atlas, previous_page?]
                         (the same order run.py sends to fal)
  prompt                 str field
  width, height          int fields
  num_inference_steps    int field (Klein distilled default 4; the LoRA's
                         base model wants ~20-50)
  guidance_scale         float field (default FLUX2_GUIDANCE_SCALE; ~4-5 for
                         the LoRA's base model, ignored by the distilled one)
  lora_scale             float field, optional (override the LoRA weight per
                         request, e.g. 0.8-1.0; ignored when no LoRA is loaded)
  seed                   int field (omit for random)
  output_format          "png" | "jpeg" | "webp"

Response: raw image bytes in the requested format.

HTTP contract (POST /edit2, multipart/form-data) — concurrency-2 batch variant:
  images1, images2       repeated file parts, all named "images1" (resp.
                         "images2"), in the same order as /edit's "images"
  prompt1, prompt2       str fields, one per job
  width, height, num_inference_steps, guidance_scale, lora_scale,
  output_format          shared by both jobs (same semantics as /edit)
  seed1, seed2           int fields, one per job (omit for random)
Response: JSON {"images": [base64, base64], "job_latency_s": [s, s],
"output_format": ...}. The two jobs run concurrently, each on its own
pipeline instance (FLUX2_NUM_PIPES, default 2; see the concurrency note).

Optional LoRA (manga colorization by reference, thedeoxen on HF):
  FLUX2_LORA_PATH        path to a .safetensors LoRA file to load at startup
  FLUX2_LORA_SCALE       default LoRA weight (default 1.0)
The LoRA is trained on the undistilled `FLUX.2-klein-base-9B`; point
FLUX2_MODEL_PATH at that base model, pass num_inference_steps ~20-50 and
guidance_scale ~4-5, and include the trigger word `mngclranm` in the prompt.

Notes:
- Dimensions must be multiples of 16 for the FLUX VAE; diffusers silently
  floors them down to the nearest multiple of 16 (e.g. 1200x1800 becomes
  1200x1792). Use 1216x1824 for exact parity with the fal endpoint's outputs.
- `guidance_scale` is ignored by the step-distilled Klein model (diffusers
  warns and disables classifier-free guidance), and used normally by the base.
- The diffusers FLUX.2 pipeline has no safety checker (fal's optional one
  falsely blocked page 18 in the cloud run).
- Concurrency: the service loads `FLUX2_NUM_PIPES` (default 1 — the original
  single-pipe, concurrency-1 behavior) pipeline instances and hands one out
  per in-flight request from a pool (traffic concurrency == FLUX2_NUM_PIPES,
  so requests never share a pipe). `/edit2` (two concurrent jobs on two
  pipes) requires FLUX2_NUM_PIPES=2 — the opt-in concurrency-2 mode
  benchmarked in docs/color_concurency.md. Each pipe owns its own mutable
  scheduler/adapter state — the reason the LoRA scale override in `_run_edit`
  is thread-safe. Optional `torch.compile` for the single-request path:
  FLUX2_COMPILE=1 compiles the transformer, =2 also the VAE
  (lazy, first call pays ~1-3 min of inductor compilation; dynamic shapes by
  default so per-panel-size recompiles are avoided).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import bentoml
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("FLUX2_MODEL_PATH", "/models/flux2-klein")
# Optional LoRA: thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA.
# Must point at a .safetensors file; loaded at startup and applied to the
# transformer (rank-32, ai-toolkit format, trigger word `mngclranm`).
LORA_PATH = os.environ.get("FLUX2_LORA_PATH") or None
LORA_SCALE = float(os.environ.get("FLUX2_LORA_SCALE", "1.0"))
_ADAPTER_NAME = "manga_colorization"
DEFAULT_WIDTH = int(os.environ.get("FLUX2_WIDTH", "1216"))
DEFAULT_HEIGHT = int(os.environ.get("FLUX2_HEIGHT", "1824"))
DEFAULT_STEPS = int(os.environ.get("FLUX2_STEPS", "4"))
# The undistilled base model (used by the LoRA) wants ~4-5; the step-distilled
# model ignores guidance entirely, so this default is safe for both.
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("FLUX2_GUIDANCE_SCALE", "4.0"))
# Number of independent pipeline instances handed out to in-flight requests
# (traffic concurrency == this value). Default 1 = the original single-pipe,
# concurrency-1 server. Set FLUX2_NUM_PIPES=2 to re-enable the concurrency-2
# mode (/edit2) benchmarked in docs/color_concurency.md; two bf16 instances
# are ~67 GB resident, within the GB10's 120 GB unified memory.
_NUM_PIPES = int(os.environ.get("FLUX2_NUM_PIPES", "1"))

# Optional torch.compile for the single-request path (concurrency 1):
#   FLUX2_COMPILE=1  compile the transformer
#   FLUX2_COMPILE=2  compile the transformer and the VAE encode/decode
# Compilation happens lazily on the first inference call (~1-3 min for the
# transformer, cached by the Triton/inductor cache dirs), so the first
# request after enabling is slow. Dynamic shapes avoid recompiling per panel
# size; set FLUX2_COMPILE_DYNAMIC=0 to use static reduce-overhead graphs
# (each new panel size then triggers a recompile).
FLUX2_COMPILE = int(os.environ.get("FLUX2_COMPILE", "0"))
FLUX2_COMPILE_DYNAMIC = os.environ.get("FLUX2_COMPILE_DYNAMIC", "1") != "0"

_OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


@bentoml.service(
    resources={"gpu": 1},
    # In-flight requests can exceed 1: each request takes a pipeline instance
    # from the pool (one per worker), so concurrent requests never share a
    # pipe. With FLUX2_NUM_PIPES=2 the service can process 2 requests at a
    # time (the concurrency-2 mode this service was built to benchmark).
    traffic={"timeout": 900, "concurrency": _NUM_PIPES},
)
class Flux2Klein:
    """Self-hosted FLUX.2 Klein 9B image-editing endpoint."""

    @bentoml.on_startup
    def load(self) -> None:
        # One pipeline instance per concurrent worker. Each pipe owns its own
        # mutable scheduler/adapter state, so concurrent jobs never share a
        # pipe: that is what makes /edit and /edit2 thread-safe. Requests
        # acquire pipes from a pool; with traffic concurrency == len(pipes)
        # every in-flight request holds a distinct pipe.
        self.pipes = [self._load_pipe() for _ in range(_NUM_PIPES)]
        self._pipe_pool: queue.Queue = queue.Queue()
        for pipe in self.pipes:
            self._pipe_pool.put(pipe)
        self.pipe = self.pipes[0]  # legacy alias used by /edit

    def _load_pipe(self) -> Flux2KleinPipeline:
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        if LORA_PATH:
            if not os.path.isfile(LORA_PATH):
                raise FileNotFoundError(
                    f"FLUX2_LORA_PATH is not a file: {LORA_PATH}"
                )
            logger.info(
                "Loading LoRA %s at scale %.3f (adapter=%s)",
                LORA_PATH,
                LORA_SCALE,
                _ADAPTER_NAME,
            )
            # ai-toolkit format (`diffusion_model.double_blocks.*`): diffusers
            # 0.39 auto-converts these keys to the Flux2 transformer layout.
            pipe.load_lora_weights(LORA_PATH, adapter_name=_ADAPTER_NAME)
            pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[LORA_SCALE])
        if FLUX2_COMPILE:
            compile_kwargs = {"dynamic": FLUX2_COMPILE_DYNAMIC}
            if not FLUX2_COMPILE_DYNAMIC:
                compile_kwargs["mode"] = "reduce-overhead"
            # Compile the forward *method*, not the module: replacing
            # pipe.transformer wholesale hides the LoRA adapter registry
            # (set_adapters -> get_list_adapters finds nothing -> 500).
            logger.info(
                "torch.compile transformer.forward (dynamic=%s)",
                FLUX2_COMPILE_DYNAMIC,
            )
            pipe.transformer.forward = torch.compile(
                pipe.transformer.forward, **compile_kwargs
            )
            if FLUX2_COMPILE >= 2:
                logger.info("torch.compile vae encode/decode")
                pipe.vae.encode = torch.compile(pipe.vae.encode, **compile_kwargs)
                pipe.vae.decode = torch.compile(pipe.vae.decode, **compile_kwargs)
        return pipe

    @bentoml.api
    def edit(
        self,
        images: list[Image.Image],
        prompt: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        lora_scale: float | None = None,
        seed: int | None = None,
        output_format: str = "png",
    ) -> bytes:
        """Edit the first image using `images[1:]` as references."""
        # Each in-flight request holds a distinct pipe from the pool, so the
        # per-request adapter-scale mutation inside _run_edit is thread-safe.
        pipe = self._pipe_pool.get()
        try:
            return self._run_edit(
                pipe, images, prompt, width, height, num_inference_steps,
                guidance_scale, lora_scale, seed, output_format,
            )
        finally:
            self._pipe_pool.put(pipe)

    def _run_edit(
        self,
        pipe: Flux2KleinPipeline,
        images: list[Image.Image],
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        lora_scale: float | None,
        seed: int | None,
        output_format: str,
    ) -> bytes:
        """Run one edit job on a specific pipeline instance."""
        if not images:
            raise ValueError("at least one image is required")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if guidance_scale <= 0:
            raise ValueError("guidance_scale must be > 0")
        if lora_scale is not None and lora_scale < 0:
            raise ValueError("lora_scale must be >= 0")
        fmt = _OUTPUT_FORMATS.get(output_format, "PNG")

        # Per-request LoRA weight override. `pipe` is exclusively owned by the
        # calling job (see edit2), so this mutation is thread-safe.
        if LORA_PATH and lora_scale is not None:
            pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[lora_scale])

        generator = (
            torch.Generator(device="cuda").manual_seed(seed)
            if seed is not None
            else None
        )
        out = pipe(
            prompt=prompt,
            image=images,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        ).images[0]

        buffer = io.BytesIO()
        if fmt == "JPEG":
            out = out.convert("RGB")
        out.save(buffer, format=fmt, quality=95 if fmt in ("JPEG", "WEBP") else None)
        return buffer.getvalue()

    @bentoml.api
    def edit2(
        self,
        images1: list[Image.Image],
        prompt1: str,
        images2: list[Image.Image],
        prompt2: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        lora_scale: float | None = None,
        seed1: int | None = None,
        seed2: int | None = None,
        output_format: str = "png",
    ) -> dict:
        """Concurrency-2 variant of /edit: two independent edit jobs in one
        request, run concurrently on two pipeline instances from the pool
        (each job owns its pipe, so there is no shared mutable state). Returns
        {"images": [base64, base64], "job_latency_s": [s, s], ...}."""
        if len(self.pipes) < 2:
            raise ValueError(
                f"edit2 needs >= 2 pipeline instances, got {len(self.pipes)}"
            )
        # Take two pipes out of the pool for the duration of the call; the
        # service's traffic concurrency (== len(pipes)) guarantees at most
        # len(pipes) in-flight get() calls, so this cannot deadlock.
        pipe_a = self._pipe_pool.get()
        pipe_b = self._pipe_pool.get()
        try:
            def run_job(
                pipe, images: list[Image.Image], prompt: str, seed: int | None
            ) -> tuple[bytes, float]:
                started = time.monotonic()
                out = self._run_edit(
                    pipe, images, prompt, width, height,
                    num_inference_steps, guidance_scale, lora_scale, seed,
                    output_format,
                )
                return out, time.monotonic() - started

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(run_job, pipe_a, images1, prompt1, seed1),
                    pool.submit(run_job, pipe_b, images2, prompt2, seed2),
                ]
                results = [future.result() for future in futures]
        finally:
            self._pipe_pool.put(pipe_a)
            self._pipe_pool.put(pipe_b)

        return {
            "images": [
                base64.b64encode(image_bytes).decode("ascii")
                for image_bytes, _ in results
            ],
            "job_latency_s": [round(latency, 3) for _, latency in results],
            "output_format": output_format,
        }
