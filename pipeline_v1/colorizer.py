"""Panel colorization client for the self-hosted FLUX.2 Klein server.

Posts one panel (+ optional filtered atlas) to `POST /edit` (multipart), the
same HTTP contract as `server/service.py` in this repo: the first image is the
edit target, further images are references. The request size is the resolution
closest to the panel's original size with both axes multiples of 16
(user-confirmed size policy); stitching resizes the output back to the exact
panel box afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests
from PIL import Image

from config import requested_panel_size
from util import file_record

_TIMEOUT_SECONDS = 1800

ATLAS_INSTRUCTION = (
    "Use the labelled character reference atlas in #2 for canonical hair, eye, "
    "skin, clothing, and accessory colors whenever a referenced character appears."
)
NO_ATLAS_INSTRUCTION = (
    "No reference atlas is provided: invent a coherent, restrained anime palette "
    "consistent with the series."
)


@dataclass
class ColorizeRecord:
    status: str                    # ok | error
    output: Path | None
    requested_size: tuple[int, int]
    latency_s: float
    error: str | None = None
    seed: int | None = None

    def to_dict(self, panel: Path, atlas: Path | None) -> dict:
        doc = {
            "status": self.status,
            "panel": panel.name,
            "panel_sha256": file_record(panel)["sha256"],
            "atlas": atlas.name if atlas else None,
            "requested_size": {"width": self.requested_size[0],
                               "height": self.requested_size[1]},
            "latency_s": round(self.latency_s, 3),
            "seed": self.seed,
            "error": self.error,
        }
        if self.output is not None:
            doc["output"] = file_record(self.output)
        return doc


class Colorizer(Protocol):
    """Interface for anything that colorizes one panel with an optional
    reference atlas."""

    def colorize(self, panel: Path, atlas: Path | None, output: Path) -> ColorizeRecord:
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
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.prompt_template = prompt_template
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.lora_scale = lora_scale
        self.seed = seed
        self.output_format = output_format
        self.timeout = timeout

    def _prompt(self, width: int, height: int, atlas: Path | None) -> str:
        instruction = ATLAS_INSTRUCTION if atlas else NO_ATLAS_INSTRUCTION
        return self.prompt_template.format(
            width=width, height=height, atlas_instruction=instruction
        )

    def colorize(
        self, panel: Path, atlas: Path | None, output: Path
    ) -> ColorizeRecord:
        with Image.open(panel) as image:
            width, height = requested_panel_size(image.width, image.height)
        fields = {
            "prompt": self._prompt(width, height, atlas),
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

        files = [("images", (panel.name, open(panel, "rb"), _mime(panel)))]
        if atlas is not None:
            files.append(("images", (atlas.name, open(atlas, "rb"), _mime(atlas))))

        started = time.monotonic()
        try:
            response = requests.post(
                f"{self.endpoint}/edit",
                data=fields,
                files=files,
                timeout=self.timeout,
            )
        except Exception as error:  # noqa: BLE001 - connection errors etc.
            return ColorizeRecord(
                status="error",
                output=None,
                requested_size=(width, height),
                latency_s=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
                seed=self.seed,
            )
        finally:
            for _, (_, handle, _) in files:
                handle.close()

        latency = time.monotonic() - started
        if response.status_code != 200:
            return ColorizeRecord(
                status="error",
                output=None,
                requested_size=(width, height),
                latency_s=latency,
                error=f"HTTP {response.status_code}: {response.text[:500]}",
                seed=self.seed,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(response.content)
        return ColorizeRecord(
            status="ok",
            output=output,
            requested_size=(width, height),
            latency_s=latency,
            error=None,
            seed=self.seed,
        )


def _mime(path: Path) -> str:
    return Image.MIME.get((Image.open(path).format or "").upper(), "application/octet-stream")
