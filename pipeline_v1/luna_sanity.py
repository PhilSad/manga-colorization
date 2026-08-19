"""Luna-based semantic line-art check (companion of the sanity step).

The pipeline's `sanity` stage scores every colorized panel against its B&W
original with *hand-crafted geometry metrics* (thin-stroke line maps, IoU /
chamfer / components / drift, see sanity.py). `luna_sanity.py` is the
complementary *semantic* check: one OpenRouter vision-language-model call —
`openai/gpt-5.6-luna` — per panel asking, in plain sight of the two images,
whether the line art of the colorized panel matches the B&W original. The
verdict is a **real structured output** — `response_format` type
`json_schema` (strict) with the fields `analyse: str` and
`line_art_matches: bool`, routed with `provider.require_parameters: true`
so the request only reaches endpoints that natively support structured
outputs and never silently degrades to loose JSON (the exact convention as
verify_color.py).

Both images are downscaled onto the same analysis grid
(`sanity.analysis_size`, `--max-edge`, default 1536 px) before being sent,
so the model sees the same geometry the local metrics score; the analysis
size is recorded per panel. Unlike the local metrics, the verdict is
subjective (the model's judgment of line-art fidelity) and costs one paid
OpenRouter call per panel — it is a review tool, not a pipeline stage.

Shared with the character detector and the color verifier: the
OpenAI-compatible call machinery with retry/backoff and `usage.cost`
accounting lives in `characters.call_vlm`; the prompt template is
`luna_sanity_prompt.txt`. The offline tool lives in
`scripts/check_luna_sanity.py`.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from characters import _extract_json_object, _iso_now, call_vlm
from sanity import analysis_size

API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_LUNA_SANITY_MODEL = "openai/gpt-5.6-luna"
DEFAULT_MAX_EDGE = 1536  # analysis-grid long-edge cap (px)

LUNA_SANITY_PROMPT_FILE = Path(__file__).resolve().parent / "luna_sanity_prompt.txt"

# Status values: ok | mismatch | unparseable | error
MATCHES = "ok"
MISMATCH = "mismatch"
UNPARSEABLE = "unparseable"
ERROR = "error"

# Strict json_schema structured output (OpenRouter structured-outputs mode),
# the same convention as verify_color.py: a boolean verdict plus a short
# textual analysis, with no extra fields.
LINE_ART_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analyse": {
            "type": "string",
            "description": (
                "Which line structures of the colorized panel match or "
                "diverge from the black & white original (strokes added, "
                "removed, moved, or redrawn)."
            ),
        },
        "line_art_matches": {
            "type": "boolean",
            "description": (
                "True if the colorized panel's line art matches the black & "
                "white original's line art well enough for an AI-assisted "
                "colorization (no strokes missing, added, or redrawn)."
            ),
        },
    },
    "required": ["analyse", "line_art_matches"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "line_art_verdict",
        "strict": True,
        "schema": LINE_ART_VERDICT_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# Answer parsing

def parse_line_art_verdict(text: str) -> dict | None:
    """Parse `{"analyse": str, "line_art_matches": bool}`.

    Returns None for unparseable/malformed answers. `line_art_matches`
    accepts a bool or the strings "true"/"false" (case-insensitive);
    `analyse` defaults to "" when missing. With the strict json_schema
    request the content is guaranteed-valid JSON, so this parser is a
    safety net only.
    """
    if not text:
        return None
    data = _extract_json_object(text.strip())
    if not isinstance(data, dict):
        return None
    raw = data.get("line_art_matches")
    if isinstance(raw, bool):
        line_art_matches = raw
    elif isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        line_art_matches = raw.strip().lower() == "true"
    else:
        return None
    analyse = data.get("analyse")
    return {
        "analyse": str(analyse or ""),
        "line_art_matches": line_art_matches,
    }


# ---------------------------------------------------------------------------
# Analysis-grid preparation

def analysis_pair(
    bw: Image.Image,
    color: Image.Image,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """(bw, color, (w, h)) on the shared analysis grid.

    The grid is defined by the *B&W* image's longest edge capped at
    `max_edge` (the reference geometry — colorized outputs may come back at
    a slightly different size, exactly like `sanity.score_pair`). Both
    images are LANCZOS-resampled onto it, so the model judges the same
    content the local metrics score.
    """
    size = analysis_size(bw.width, bw.height, max_edge)

    def _grid(image: Image.Image) -> Image.Image:
        if (image.width, image.height) == size:
            return image
        return image.resize(size, Image.Resampling.LANCZOS)

    return _grid(bw), _grid(color), size


# ---------------------------------------------------------------------------
# Checker client

@dataclass
class LineArtCheckRecord:
    """Result of one Luna line-art check for one panel."""

    status: str                       # ok | mismatch | unparseable | error
    line_art_matches: bool | None
    analyse: str
    response_text: str
    usage: dict[str, Any]
    cost_usd: float | None
    cost_source: str
    latency_s: float
    model_returned: str | None
    attempts: int
    analysis_size: tuple[int, int]    # (w, h) the model actually saw
    error: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "line_art_matches": self.line_art_matches,
            "analyse": self.analyse,
            "response_text": self.response_text,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "cost_source": self.cost_source,
            "latency_s": round(self.latency_s, 3),
            "model_returned": self.model_returned,
            "attempts": self.attempts,
            "analysis_size": list(self.analysis_size),
            "error": self.error,
            "finished_at": self.finished_at,
        }


class LineArtChecker:
    """OpenRouter VLM that checks a colorized panel's line art against its
    B&W original with one generic prompt and a strict structured-output
    verdict (`analyse`, `line_art_matches`)."""

    def __init__(
        self,
        model: str = DEFAULT_LUNA_SANITY_MODEL,
        api_key: str = "",
        api_base: str = API_BASE,
        client: Any = None,  # injected OpenAI-compatible client (tests)
        max_tokens: int = 1024,
        prompt_template: str | None = None,
        response_format: dict | None = None,  # default: LINE_ART_VERDICT_SCHEMA
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.prompt_template = prompt_template
        self.response_format = response_format or RESPONSE_FORMAT

    def check(
        self,
        colorized: Image.Image,
        input_crop: Image.Image,
        max_edge: int | None = DEFAULT_MAX_EDGE,
    ) -> LineArtCheckRecord:
        """Ask the VLM whether the colorized panel's line art matches the B&W
        original.

        Both images are downscaled onto the shared analysis grid (long edge
        capped at `max_edge`, default 1536 px — the same geometry the local
        sanity metrics use) unless the caller passes pre-prepared images with
        `max_edge=None` (then they are sent as-is and `analysis_size` is
        read from the input crop). One paid OpenRouter call with strict
        json_schema structured output (`analyse`/`line_art_matches`). The
        request omits `temperature` (gpt-5.6-luna does not support it) and
        is routed with `provider.require_parameters: true` by `call_vlm`
        for structured outputs; a rejection is recorded as an error, never
        downgraded.
        """
        if max_edge is None:
            bw_image, color_image = input_crop, colorized
            size = (input_crop.width, input_crop.height)
        else:
            bw_image, color_image, size = analysis_pair(
                input_crop, colorized, max_edge
            )
        template = self.prompt_template or LUNA_SANITY_PROMPT_FILE.read_text(
            encoding="utf-8"
        )
        content = _content_with_images(template, (color_image, bw_image))

        result = call_vlm(
            self.client, self.model, content,
            max_tokens=self.max_tokens,
            response_format=self.response_format,
        )
        parsed = None
        if result.error is None:
            parsed = parse_line_art_verdict(result.text)

        if result.error is not None:
            status = ERROR
            line_art_matches = None
            analyse = ""
        elif parsed is None:
            status = UNPARSEABLE
            line_art_matches = None
            analyse = ""
        else:
            line_art_matches = parsed["line_art_matches"]
            analyse = parsed["analyse"]
            status = MATCHES if line_art_matches else MISMATCH

        return LineArtCheckRecord(
            status=status,
            line_art_matches=line_art_matches,
            analyse=analyse,
            response_text=result.text,
            usage=result.usage,
            cost_usd=result.cost_usd,
            cost_source=result.cost_source,
            latency_s=result.latency_s,
            model_returned=result.model_returned,
            attempts=result.attempts,
            analysis_size=size,
            error=result.error,
            finished_at=_iso_now(),
        )


def _content_with_images(
    prompt: str,
    images: tuple[Image.Image | None, ...],
) -> list[dict[str, Any]]:
    """Text + base64 data-URL PNG images for an OpenAI-compatible request."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
    ]
    for image in images:
        if image is None:
            continue
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        content.append(
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
    return content
