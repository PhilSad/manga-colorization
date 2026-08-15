"""Full-page colorizer using OpenAI gpt-image-2 (`images/edit`).

The full-page mode backend: the whole B&W page plus a labelled reference
atlas are sent to the OpenAI Images API as input images (no mask -> extra
images act as references) and the API returns a colorized page.

Contract decisions (plan `docs/plans/fullpage-gpt-image2-atlas.md`,
user-confirmed 2026-08-15):

- Quality is **fixed at `medium`** (`config.GPT_IMAGE_QUALITY`): "low" is
  rejected by the edit endpoint for input-image requests and "high" only
  exists on gpt-image-1. There is deliberately no quality flag.
- Output size is the **minimal** aspect-preserving size that satisfies the
  API constraints (`config.minimal_gpt_image_size`), unless an explicit
  `--gpt-size WxH` override is configured.
- **Retry policy**: transient errors (connection errors, 429, 5xx) are
  retried with exponential backoff up to `retries=3` (4 attempts total);
  after the last failed attempt the call is recorded as
  `ColorizeRecord(status="error", error=<last error>)` — fail loudly, no
  silent partial state (mirrors the FLUX retry loop in `colorizer.py`).
- The atlas is downscaled by `--gpt-atlas-scale` before upload; gpt-image-2
  bills image-input tokens by image size, so a smaller atlas cuts the fixed
  input cost (research-v2 `--atlas-scale` knob).

Cost accounting (standard tier, OpenAI pricing page / image generation guide,
2026-06, same as research-v2): image input $8.00 / 1M image tokens, text
input $5.00 / 1M text tokens, image output $30.00 / 1M output tokens.
`est_cost_usd` is computed from the response's `usage` details when the API
returns them (None otherwise).
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

from PIL import Image

from colorizer import (
    ATLAS_INSTRUCTION,
    NO_ATLAS_INSTRUCTION,
    NO_PROFILE_INSTRUCTION,
    ColorizeRecord,
)
from config import GPT_IMAGE_QUALITY, minimal_gpt_image_size

_TIMEOUT_SECONDS = 600
# OpenAI error classes worth retrying: connection hiccups, rate limits (429),
# and server errors (5xx). Everything else is permanent and fails immediately.
_RETRYABLE_STATUS = {429}
_RETRYABLE_EXCEPTION_SUFFIXES = ("APIConnectionError", "InternalServerError")

# Standard-tier per-1M-token rates (USD) used for est_cost_usd.
_RATE_IMAGE_INPUT_USD = 8.0
_RATE_TEXT_INPUT_USD = 5.0
_RATE_IMAGE_OUTPUT_USD = 30.0
_RATE_TEXT_OUTPUT_USD = 30.0


class GptImage2Colorizer:
    """Implements the `Colorizer` protocol for the OpenAI Images edit API.

    Thread safety: the OpenAI SDK client is safe for concurrent `images.edit`
    calls, and every `colorize` call builds its own byte payloads — a single
    shared instance can be used from the parallel colorize step.
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
                f"gpt-image-2 backend requires {api_key_env} in the environment "
                "(load .env via run.py before building backends)"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout)

    # -- prompt rendering (mirrors FluxColorizer._prompt) --------------------

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

    # -- the Colorizer protocol ----------------------------------------------

    def colorize(
        self,
        panel: Path,
        atlas: Path | None,
        output: Path,
        palette_instruction: str = "",
    ) -> ColorizeRecord:
        """Colorize one full page. `panel` is the whole page; `atlas` is the
        labelled reference atlas (None -> panel-only colorization)."""
        with Image.open(panel) as image:
            original = (image.width, image.height)
        width, height = self.size or minimal_gpt_image_size(*original)
        prompt = self._prompt(width, height, atlas, palette_instruction)

        started = time.monotonic()
        last_error: str | None = None
        result: dict | None = None
        for attempt in range(self.retries + 1):
            try:
                result = self._call(panel, atlas, prompt, width, height)
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
            output.write_bytes(base64.b64decode(result["b64_json"]))
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

    # -- internals -----------------------------------------------------------

    def _call(
        self,
        panel: Path,
        atlas: Path | None,
        prompt: str,
        width: int,
        height: int,
    ) -> dict:
        """One `images.edit` attempt. Raises on API errors (caller retries)."""
        images = [open(panel, "rb")]
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
                handle.close()
        data = response.data[0]
        return {
            "b64_json": data.b64_json,
            "usage": getattr(response, "usage", None),
        }

    def _scaled_atlas(self, atlas: Path) -> io.BytesIO:
        """Atlas bytes for upload, downscaled by `self.atlas_scale` (1.0 ->
        the original file bytes; the resize keeps JPEG compression)."""
        if self.atlas_scale == 1.0:
            buffer = io.BytesIO(atlas.read_bytes())
            buffer.seek(0)
            return buffer
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
        return buffer

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


def _parse_usage(raw_usage: object) -> tuple[dict, float | None]:
    """Normalize the Images API `usage` object into the research-v2 token
    accounting and compute `est_cost_usd` (None when usage is missing)."""
    if raw_usage is None:
        return {}, None
    input_details = getattr(raw_usage, "input_tokens_details", None)
    output_details = getattr(raw_usage, "output_tokens_details", None)
    input_image_tokens = (
        getattr(input_details, "image_tokens", None) if input_details else None
    )
    input_text_tokens = (
        getattr(input_details, "text_tokens", None) if input_details else None
    )
    output_image_tokens = (
        getattr(output_details, "image_tokens", None) if output_details else None
    )
    output_text_tokens = (
        getattr(output_details, "text_tokens", None) if output_details else None
    )
    usage = {
        "input_tokens": getattr(raw_usage, "input_tokens", None),
        "output_tokens": getattr(raw_usage, "output_tokens", None),
        "total_tokens": getattr(raw_usage, "total_tokens", None),
        "input_tokens_details": {
            "image_tokens": input_image_tokens,
            "text_tokens": input_text_tokens,
        },
        "output_tokens_details": {
            "image_tokens": output_image_tokens,
            "text_tokens": output_text_tokens,
        },
    }
    cost = round(
        (input_image_tokens or 0) / 1e6 * _RATE_IMAGE_INPUT_USD
        + (input_text_tokens or 0) / 1e6 * _RATE_TEXT_INPUT_USD
        + (output_image_tokens or 0) / 1e6 * _RATE_IMAGE_OUTPUT_USD
        + (output_text_tokens or 0) / 1e6 * _RATE_TEXT_OUTPUT_USD,
        6,
    )
    return usage, cost
