"""Bbox-guided region editing for the verify loop (`--verify-mode bbox`).

When Luna's bbox verdict flags palette-wrong regions on a rejected
colorization, the boxes are drawn on the rejected image and the boxed image
(plus the labelled atlas) is handed to gpt-image-2 (`images.edit`, no mask —
the boxes are the only locator) for a region-scoped recolor: only the red
rectangles change, everything outside stays byte-stable per the prompt.

This is the library implementation of the probed experiment
(scripts/probe_luna_bboxes.py + scripts/probe_gpt_edit_bbox.py, write-up in
pipelines.md §"BBox probe"). Per user decision 3 (docs/plans/
verify-bbox-region-edit.md, 2026-08-16) the committed probe scripts stay
standalone: `draw_boxes` / `region_instruction` are duplicated here instead
of imported, keeping the probe behavior untouched (accepted duplication).

Resolution rule: boxes are drawn on the image at the resolution actually
sent to gpt-image-2 (`size`). Luna's bbox coordinates are normalized 0-1000
so they scale exactly; upscale-first-then-draw keeps the boxes pixel-correct
for the edit request.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from colorizer import ColorizeRecord
from config import GPT_IMAGE_QUALITY, minimal_gpt_image_size
from gpt_colorizer import (
    _RETRYABLE_EXCEPTION_SUFFIXES,
    _RETRYABLE_STATUS,
    _TIMEOUT_SECONDS,
    _parse_usage,
)

# High-contrast cycling colors for the drawn boxes (same palette as the
# probe, so boxed outputs look identical between probe and pipeline).
BOX_COLORS = [
    (255, 0, 0),      # red
    (0, 0, 255),      # blue
    (0, 200, 0),      # green
    (255, 165, 0),    # orange
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
]

_DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ---------------------------------------------------------------------------
# Box drawing / prompt rendering (duplicated from the probes, see module doc)

def draw_boxes(
    source: Path,
    regions: list[dict[str, Any]],
    out: Path,
    size: tuple[int, int] | None = None,
) -> Path:
    """Overlay each region's bbox (normalized 0-1000) on the image.

    `source` is the colorized image; `regions` entries carry a `bbox`
    `[x1, y1, x2, y2]` in normalized 0-1000 coordinates (missing/malformed
    bboxes are skipped, the region text still logged). When `size` is given
    and differs from the source dimensions, the image is resized to `size`
    first — boxes are then drawn at the resolution actually sent to
    gpt-image-2, so Luna's normalized boxes stay pixel-correct. Returns the
    annotated image path. Requires Pillow."""
    image = Image.open(source).convert("RGB")
    if size is not None and tuple(size) != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype(_DEFAULT_FONT, 20)
    except Exception:  # noqa: BLE001 - fall back to the default bitmap font
        font = ImageFont.load_default()

    for index, region in enumerate(regions):
        bbox = region.get("bbox")
        if bbox is None or len(bbox) != 4:
            print(f"  [verify] region {index} has no usable bbox: {region!r}",
                  flush=True)
            continue
        x1, y1, x2, y2 = [max(0, min(1000, int(v))) for v in bbox]
        color = BOX_COLORS[index % len(BOX_COLORS)]
        rect = (
            int(x1 / 1000 * width),
            int(y1 / 1000 * height),
            int(x2 / 1000 * width),
            int(y2 / 1000 * height),
        )
        draw.rectangle(rect, outline=color, width=4)
        label = f"{index}: {region.get('character') or '?'}"
        draw.text((rect[0], max(0, rect[1] - 24)), label, fill=color, font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def region_instruction(regions: list[dict[str, Any]]) -> str:
    """Render the numbered region fixes for the edit prompt's
    `{region_instruction}` slot, matching the boxes drawn on the image."""
    lines = ["Regions to fix (numbered in order, matching the red rectangles):"]
    for i, region in enumerate(regions):
        character = region.get("character") or "?"
        fix = (region.get("fix_suggestion") or region.get("problem") or "").strip()
        lines.append(f"- Region {i} ({character}): {fix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# gpt-image-2 region editor

class GptImage2RegionEditor:
    """gpt-image-2 `images.edit` wrapper for region-scoped recolors.

    Reuses gpt_colorizer.py's request plumbing: atlas upload as a
    `("atlas.jpg", buffer)` filename-carrying tuple (mandatory — a bare
    BytesIO uploads as application/octet-stream and gpt-image-2 rejects it),
    minimal aspect-preserving size, fixed `medium` quality, the same
    transient-error retry policy, and `_parse_usage` cost accounting.

    Thread safety mirrors GptImage2Colorizer: a shared instance is safe for
    concurrent `images.edit` calls (each call builds its own payloads).
    """

    def __init__(
        self,
        prompt_template: str,
        model: str = "gpt-image-2",
        size: tuple[int, int] | None = None,   # explicit --gpt-size override
        atlas_scale: float = 1.0,
        api_key_env: str = "OPENAI_API_KEY",
        output_format: str = "png",
        timeout: float = _TIMEOUT_SECONDS,
        retries: int = 3,
        retry_backoff_s: float = 5.0,
    ) -> None:
        self.prompt_template = prompt_template
        self.model = model
        self.size = size
        self.atlas_scale = atlas_scale
        self.api_key_env = api_key_env
        self.output_format = output_format
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"gpt-image-2 region editor requires {api_key_env} in the "
                "environment (load .env via run.py before building backends)"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout)

    # -- prompt rendering ---------------------------------------------------

    def _prompt(
        self,
        width: int,
        height: int,
        instruction: str,
        palette_instruction: str = "",
    ) -> str:
        return self.prompt_template.format(
            width=width,
            height=height,
            region_instruction=instruction,
            palette_instruction=palette_instruction,
        )

    # -- the editor protocol ------------------------------------------------

    def edit(
        self,
        boxed_image: Path,
        atlas: Path | None,
        output: Path,
        region_instruction_text: str,
        palette_instruction: str = "",
    ) -> ColorizeRecord:
        """Recolor only the boxed regions of `boxed_image` (Luna's boxes
        drawn on it) with the canonical colors; returns a ColorizeRecord so
        the verify loop records it exactly like a colorization attempt."""
        with Image.open(boxed_image) as image:
            original = (image.width, image.height)
        width, height = self.size or minimal_gpt_image_size(*original)
        prompt = self._prompt(width, height, region_instruction_text,
                              palette_instruction)

        started = time.monotonic()
        last_error: str | None = None
        result: dict | None = None
        for attempt in range(self.retries + 1):
            try:
                result = self._call(boxed_image, atlas, prompt, width, height)
                last_error = None
                break
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = f"{type(error).__name__}: {error}"
                if not self._retryable(error) or attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff_s * (2 ** attempt))

        latency = time.monotonic() - started
        usage, cost = None, None
        if result is not None:
            usage, cost = _parse_usage(result.get("usage"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(__import__("base64").b64decode(result["b64_json"]))
        return ColorizeRecord(
            status="ok" if last_error is None else "error",
            output=output if last_error is None else None,
            requested_size=(width, height),
            latency_s=latency,
            error=last_error,
            seed=None,
            original_size=original,
            scale=None,
            cap_applied=False,
            max_megapixels=None,
            upscaled=False,
            model=self.model,
            quality=GPT_IMAGE_QUALITY,
            usage=usage,
            est_cost_usd=cost,
        )

    # -- internals ----------------------------------------------------------

    def _call(
        self,
        boxed_image: Path,
        atlas: Path | None,
        prompt: str,
        width: int,
        height: int,
    ) -> dict:
        """One `images.edit` attempt. Raises on API errors (caller retries)."""
        images = [open(boxed_image, "rb")]
        if atlas is not None:
            images.append(self._scaled_atlas(atlas))
        try:
            response = self._client.images.edit(
                model=self.model,
                image=images,
                prompt=prompt,
                quality=GPT_IMAGE_QUALITY,
                size=f"{width}x{height}",
                output_format=self.output_format,
                n=1,
            )
        finally:
            for handle in images:
                if isinstance(handle, tuple):
                    handle[1].close()
                else:
                    handle.close()
        data = response.data[0]
        return {
            "b64_json": data.b64_json,
            "usage": getattr(response, "usage", None),
        }

    def _scaled_atlas(self, atlas: Path) -> tuple[str, io.BytesIO]:
        """Atlas upload as a `(filename, buffer)` tuple, downscaled by
        `self.atlas_scale` — same contract as gpt_colorizer.GptImage2Colorizer
        (the filename is what httpx uses to sniff image/jpeg)."""
        if self.atlas_scale == 1.0:
            buffer = io.BytesIO(atlas.read_bytes())
        else:
            with Image.open(atlas) as image:
                target = (
                    max(1, round(image.width * self.atlas_scale)),
                    max(1, round(image.height * self.atlas_scale)),
                )
                buffer = io.BytesIO()
                image.convert("RGB").resize(
                    target, Image.Resampling.LANCZOS
                ).save(buffer, format="JPEG", quality=94, subsampling=0)
        buffer.seek(0)
        return ("atlas.jpg", buffer)

    def _retryable(self, error: BaseException) -> bool:
        name = type(error).__name__
        if name.endswith(_RETRYABLE_EXCEPTION_SUFFIXES):
            return True
        status = getattr(error, "status_code", None)
        if status in _RETRYABLE_STATUS:
            return True
        # openai.APIStatusError for 5xx: no dedicated class name, so match on
        # the status code of any error that carries one.
        if isinstance(status, int) and 500 <= status < 600:
            return True
        return False
