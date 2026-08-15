"""Panel colorization client for the self-hosted FLUX.2 Klein server.

Posts one panel (+ optional filtered atlas) to `POST /edit` (multipart), the
same HTTP contract as `server/service.py` in this repo: the first image is the
edit target, further images are references. The request size is the resolution
closest to the panel's original size with both axes multiples of 16 (V1 size
policy), bounded by the configurable megapixel cap (task 0004): oversized
inputs are scaled down proportionally, never upscaled. Two exceptions forced
by the server contract (FLUX.2 Klein edit pipeline on Spark):

- the server rejects any input image with an axis below `min_axis` (64 px),
  so degenerate panels/atlases are upscaled client-side to the floor (recorded
  as `upscaled` in the colorize record);
- transient HTTP 5xx responses (e.g. cuBLAS hiccups on the GPU) are retried
  with exponential backoff before the panel is recorded as failed.

The prompt can carry an explicit canonical-palette instruction for the
detected characters (task 0002) in addition to the atlas.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests
from PIL import Image

from config import FLUX_MIN_AXIS, bounded_requested_size
from util import file_record

_TIMEOUT_SECONDS = 1800
# Status codes worth retrying on a self-hosted server: transient GPU/cuBLAS
# failures surface as 500, and 502/503/504 cover proxy/gateway hiccups.
_RETRYABLE_STATUS = {500, 502, 503, 504}

ATLAS_INSTRUCTION = (
    "Use the labelled character reference atlas in #2 for canonical hair, eye, "
    "skin, clothing, and accessory colors whenever a referenced character appears."
)
NO_ATLAS_INSTRUCTION = (
    "No reference atlas is provided: invent a coherent, restrained anime palette "
    "consistent with the series."
)
NO_PROFILE_INSTRUCTION = (
    "No explicit character palette profiles are provided; derive canonical colors "
    "from the atlas and invent coherent colors consistent with the series."
)


@dataclass
class ColorizeRecord:
    status: str                    # ok | error
    output: Path | None
    requested_size: tuple[int, int]
    latency_s: float
    error: str | None = None
    seed: int | None = None
    original_size: tuple[int, int] | None = None
    scale: float | None = None
    cap_applied: bool = False
    max_megapixels: float | None = None
    upscaled: bool = False          # input was below the server min axis and
                                    # had to be upscaled client-side
    # Paid-backend provenance (all None for the self-hosted FLUX path, in
    # which case to_dict omits them): the backend model id (e.g. gpt-image-2),
    # the quality setting, the API's usage details, and the estimated USD cost.
    model: str | None = None
    quality: str | None = None
    usage: dict | None = None
    est_cost_usd: float | None = None

    def to_dict(self, panel: Path, atlas: Path | None) -> dict:
        doc = {
            "status": self.status,
            "panel": panel.name,
            "panel_sha256": file_record(panel)["sha256"],
            "atlas": atlas.name if atlas else None,
            "original_size": {"width": self.original_size[0],
                              "height": self.original_size[1]}
            if self.original_size else None,
            "requested_size": {"width": self.requested_size[0],
                               "height": self.requested_size[1]},
            "scale": round(self.scale, 4) if self.scale is not None else None,
            "cap_applied": self.cap_applied,
            "max_megapixels": self.max_megapixels,
            "upscaled": self.upscaled,
            "latency_s": round(self.latency_s, 3),
            "seed": self.seed,
            "error": self.error,
        }
        for key, value in (
            ("model", self.model),
            ("quality", self.quality),
            ("usage", self.usage),
            ("est_cost_usd", self.est_cost_usd),
        ):
            if value is not None:
                doc[key] = value
        if self.output is not None:
            doc["output"] = file_record(self.output)
        return doc


class Colorizer(Protocol):
    """Interface for anything that colorizes one panel with an optional
    reference atlas."""

    def colorize(
        self,
        panel: Path,
        atlas: Path | None,
        output: Path,
        palette_instruction: str = "",
    ) -> ColorizeRecord:
        ...


class FluxColorizer:
    """Client for the BentoML `POST /edit` endpoint (self-hosted, no auth)."""

    def __init__(
        self,
        endpoint: str,
        prompt_template: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        lora_scale: float = 1.0,
        seed: int | None = None,
        output_format: str = "png",
        timeout: float = _TIMEOUT_SECONDS,
        max_megapixels: float = 2.0,
        min_axis: int = FLUX_MIN_AXIS,
        retries: int = 2,
        retry_backoff_s: float = 3.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.prompt_template = prompt_template
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.lora_scale = lora_scale
        self.seed = seed
        self.output_format = output_format
        self.timeout = timeout
        self.max_megapixels = max_megapixels
        self.min_axis = min_axis
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s

    def _prompt(
        self,
        width: int,
        height: int,
        atlas: Path | None,
        palette_instruction: str = "",
    ) -> str:
        instruction = ATLAS_INSTRUCTION if atlas else NO_ATLAS_INSTRUCTION
        palette = palette_instruction or NO_PROFILE_INSTRUCTION
        return self.prompt_template.format(
            width=width,
            height=height,
            atlas_instruction=instruction,
            character_profiles=palette,
        )

    def colorize(
        self,
        panel: Path,
        atlas: Path | None,
        output: Path,
        palette_instruction: str = "",
    ) -> ColorizeRecord:
        with Image.open(panel) as image:
            original = (image.width, image.height)
        requested = bounded_requested_size(
            original[0], original[1], self.max_megapixels,
            min_axis=self.min_axis,
        )
        width, height = requested
        original_area = original[0] * original[1]
        cap_applied = original_area > self.max_megapixels * 1_000_000
        scale = (
            math.sqrt((width * height) / original_area)
            if cap_applied and original_area
            else 1.0
        )
        # The server rejects input images with an axis below the floor, so a
        # degenerate panel is upscaled to the requested (already floored) size
        # before upload; a degenerate atlas is upscaled to the floor too.
        upscaled = original[0] < self.min_axis or original[1] < self.min_axis
        panel_payload = _image_payload(
            panel, target=(width, height) if upscaled else None
        )
        atlas_payload = (
            _image_payload(atlas, min_axis=self.min_axis) if atlas is not None
            else None
        )

        fields = {
            "prompt": self._prompt(width, height, atlas, palette_instruction),
            "width": str(width),
            "height": str(height),
            "num_inference_steps": str(self.num_inference_steps),
            "guidance_scale": str(self.guidance_scale),
            "output_format": self.output_format,
        }
        if self.lora_scale is not None:
            fields["lora_scale"] = str(self.lora_scale)
        if self.seed is not None:
            fields["seed"] = str(self.seed)

        started = time.monotonic()
        last_error: str | None = None
        response: requests.Response | None = None
        for attempt in range(self.retries + 1):
            # Fresh byte streams per attempt (requests consumes them).
            files = [
                (
                    "images",
                    (panel_payload[0], io.BytesIO(panel_payload[1]),
                     panel_payload[2]),
                )
            ]
            if atlas_payload is not None:
                files.append(
                    (
                        "images",
                        (atlas_payload[0], io.BytesIO(atlas_payload[1]),
                         atlas_payload[2]),
                    )
                )
            try:
                response = requests.post(
                    f"{self.endpoint}/edit",
                    data=fields,
                    files=files,
                    timeout=self.timeout,
                )
            except Exception as error:  # noqa: BLE001 - connection errors etc.
                last_error = f"{type(error).__name__}: {error}"
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                break
            if response.status_code == 200:
                last_error = None
                break
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if (
                response.status_code not in _RETRYABLE_STATUS
                or attempt >= self.retries
            ):
                break
            time.sleep(self.retry_backoff_s * (2 ** attempt))

        latency = time.monotonic() - started
        if response is not None and response.status_code == 200:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(response.content)
        return ColorizeRecord(
            status="ok" if last_error is None else "error",
            output=output if last_error is None else None,
            requested_size=(width, height),
            latency_s=latency,
            error=last_error,
            seed=self.seed,
            original_size=original,
            scale=scale,
            cap_applied=cap_applied,
            max_megapixels=self.max_megapixels,
            upscaled=upscaled,
        )


def _image_payload(
    path: Path,
    target: tuple[int, int] | None = None,
    min_axis: int | None = None,
) -> tuple[str, bytes, str]:
    """Read an image file for multipart upload.

    If `target` is given the image is resized to exactly that size; else if
    `min_axis` is given and the image has an axis below it, the image is
    upscaled proportionally so both axes reach the floor. Resized images are
    re-encoded as PNG; otherwise the original bytes are passed through.
    """
    with Image.open(path) as image:
        below_floor = (
            min_axis is not None
            and (image.width < min_axis or image.height < min_axis)
        )
        if target is None and below_floor:
            scale = max(min_axis / image.width, min_axis / image.height)
            target = (
                max(min_axis, round(image.width * scale)),
                max(min_axis, round(image.height * scale)),
            )
        if target is not None:
            buffer = io.BytesIO()
            image.convert("RGB").resize(
                target, Image.Resampling.LANCZOS
            ).save(buffer, format="PNG")
            return path.name, buffer.getvalue(), "image/png"
    return path.name, path.read_bytes(), _mime(path)


def _mime(path: Path) -> str:
    return Image.MIME.get((Image.open(path).format or "").upper(), "application/octet-stream")
