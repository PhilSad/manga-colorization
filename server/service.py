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
  num_inference_steps    int field (Klein is step-distilled; 4 is the default)
  seed                   int field (omit for random)
  output_format          "png" | "jpeg" | "webp"

Response: raw image bytes in the requested format.

Notes:
- Dimensions must be multiples of 16 for the FLUX VAE; diffusers silently
  floors them down to the nearest multiple of 16 (e.g. 1200x1800 becomes
  1200x1792). Use 1216x1824 for exact parity with the fal endpoint's outputs.
- `guidance_scale` is ignored by the step-distilled Klein model.
- The diffusers FLUX.2 pipeline has no safety checker (fal's optional one
  falsely blocked page 18 in the cloud run).
"""

from __future__ import annotations

import io
import os

import bentoml
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

MODEL_PATH = os.environ.get("FLUX2_MODEL_PATH", "/models/flux2-klein")
DEFAULT_WIDTH = int(os.environ.get("FLUX2_WIDTH", "1216"))
DEFAULT_HEIGHT = int(os.environ.get("FLUX2_HEIGHT", "1824"))
DEFAULT_STEPS = int(os.environ.get("FLUX2_STEPS", "4"))

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

    @bentoml.api
    def edit(
        self,
        images: list[Image.Image],
        prompt: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_inference_steps: int = DEFAULT_STEPS,
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
        fmt = _OUTPUT_FORMATS.get(output_format, "PNG")

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
            generator=generator,
            output_type="pil",
        ).images[0]

        buffer = io.BytesIO()
        if fmt == "JPEG":
            out = out.convert("RGB")
        out.save(buffer, format=fmt, quality=95 if fmt in ("JPEG", "WEBP") else None)
        return buffer.getvalue()
