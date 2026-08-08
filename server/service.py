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
"""

from __future__ import annotations

import io
import logging
import os

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

_OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 900, "concurrency": 1},
)
class Flux2Klein:
    """Self-hosted FLUX.2 Klein 9B image-editing endpoint."""

    @bentoml.on_startup
    def load(self) -> None:
        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
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
            self.pipe.load_lora_weights(LORA_PATH, adapter_name=_ADAPTER_NAME)
            self.pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[LORA_SCALE])

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

        # Per-request LoRA weight override (concurrency is 1, so mutating the
        # pipe's active adapter scale here is safe).
        if LORA_PATH and lora_scale is not None:
            self.pipe.set_adapters([_ADAPTER_NAME], adapter_weights=[lora_scale])

        generator = (
            torch.Generator(device="cuda").manual_seed(seed)
            if seed is not None
            else None
        )
        out = self.pipe(
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
