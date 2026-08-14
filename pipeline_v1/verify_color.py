"""Generic canonical-palette verification for the COL-* evaluation cases.

One verifier for all color cases: asks an OpenRouter vision-language model —
`openai/gpt-5.6-luna` — whether every character in the colorized panel has
its canonical Frieren palette. The prompt is deliberately generic: no
fixture expectations (required/forbidden colors, left-to-right order) are
rendered into it; the model judges from its own knowledge of the manga. The
verdict is a **real structured output** — `response_format` type
`json_schema` (strict) with the fields `analyse: str` and
`good_color: bool`, routed with `provider.require_parameters: true` so the
request only reaches endpoints that natively support structured outputs and
never silently degrades to loose JSON.

Shared with the character detector: the OpenAI-compatible call machinery
with retry/backoff and `usage.cost` accounting lives in
`characters.call_vlm`; the prompt template is `verify_color_prompt.txt`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from characters import _extract_json_object, _iso_now, call_vlm

API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_VERIFY_MODEL = "openai/gpt-5.6-luna"

VERIFY_PROMPT_FILE = Path(__file__).resolve().parent / "verify_color_prompt.txt"

# Status values: verified | mismatch | unparseable | error
VERIFIED = "verified"
MISMATCH = "mismatch"
UNPARSEABLE = "unparseable"
ERROR = "error"

# Strict json_schema structured output (OpenRouter structured-outputs mode).
COLOR_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analyse": {
            "type": "string",
            "description": (
                "Explain which characters' palettes are correct or wrong."
            ),
        },
        "good_color": {
            "type": "boolean",
            "description": (
                "True if every character in the panel has its canonical "
                "Frieren color palette."
            ),
        },
    },
    "required": ["analyse", "good_color"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "color_verdict",
        "strict": True,
        "schema": COLOR_VERDICT_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# Answer parsing

def parse_color_verdict(text: str) -> dict | None:
    """Parse `{"analyse": str, "good_color": bool}`.

    Returns None for unparseable/malformed answers. `good_color` accepts a
    bool or the strings "true"/"false" (case-insensitive); `analyse` is
    defaulted to "" when missing. With the strict json_schema request the
    content is guaranteed-valid JSON, so this parser is a safety net only.
    """
    if not text:
        return None
    data = _extract_json_object(text.strip())
    if not isinstance(data, dict):
        return None
    raw = data.get("good_color")
    if isinstance(raw, bool):
        good_color = raw
    elif isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        good_color = raw.strip().lower() == "true"
    else:
        return None
    analyse = data.get("analyse")
    return {
        "analyse": str(analyse or ""),
        "good_color": good_color,
    }


# ---------------------------------------------------------------------------
# Verifier client

@dataclass
class ColorVerifyRecord:
    """Result of one color verification call for one COL-* case."""

    status: str                       # verified | mismatch | unparseable | error
    good_color: bool | None
    analyse: str
    response_text: str
    usage: dict[str, int]
    cost_usd: float | None
    cost_source: str
    latency_s: float
    model_returned: str | None
    attempts: int
    error: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "good_color": self.good_color,
            "analyse": self.analyse,
            "response_text": self.response_text,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "cost_source": self.cost_source,
            "latency_s": round(self.latency_s, 3),
            "model_returned": self.model_returned,
            "attempts": self.attempts,
            "error": self.error,
            "finished_at": self.finished_at,
        }


class ColorVerifier:
    """OpenRouter VLM that checks a colorized panel with one generic prompt
    and a strict structured-output verdict (`analyse`, `good_color`)."""

    def __init__(
        self,
        model: str = DEFAULT_VERIFY_MODEL,
        api_key: str = "",
        api_base: str = API_BASE,
        client: Any = None,  # injected OpenAI-compatible client (tests)
        max_tokens: int = 1024,
        temperature: float = 0.0,
        prompt_template: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.prompt_template = prompt_template

    def verify(
        self,
        colorized: Path,
        input_crop: Path | None,
        atlas: Path | None = None,
    ) -> ColorVerifyRecord:
        """Ask the VLM whether every character in the colorized panel has its
        canonical Frieren palette.

        Sends the colorized panel as the primary image plus the monochrome
        crop and the reference atlas of the detected characters (the same
        contact sheet the colorizer saw) as context when available. One paid
        OpenRouter call with strict json_schema structured output
        (`analyse`/`good_color`). The request omits `temperature`
        (gpt-5.6-luna does not support it; sending it would make
        `provider.require_parameters` reject every endpoint).
        """
        template = self.prompt_template or VERIFY_PROMPT_FILE.read_text(
            encoding="utf-8"
        )
        content = _content_with_images(template, (colorized, input_crop, atlas))

        result = call_vlm(
            self.client, self.model, content,
            max_tokens=self.max_tokens, temperature=self.temperature,
            response_format=RESPONSE_FORMAT,
        )
        parsed = parse_color_verdict(result.text) if result.error is None else None

        if result.error is not None:
            status = ERROR
            good_color = None
            analyse = ""
        elif parsed is None:
            status = UNPARSEABLE
            good_color = None
            analyse = ""
        else:
            good_color = parsed["good_color"]
            analyse = parsed["analyse"]
            status = VERIFIED if good_color else MISMATCH

        return ColorVerifyRecord(
            status=status,
            good_color=good_color,
            analyse=analyse,
            response_text=result.text,
            usage=result.usage,
            cost_usd=result.cost_usd,
            cost_source=result.cost_source,
            latency_s=result.latency_s,
            model_returned=result.model_returned,
            attempts=result.attempts,
            error=result.error,
            finished_at=_iso_now(),
        )


def _content_with_images(
    prompt: str,
    images: tuple[Path | None, ...],
) -> list[dict[str, Any]]:
    """Text + base64 data-URL images for an OpenAI-compatible chat request."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
    ]
    for image in images:
        if image is None or not Path(image).is_file():
            continue
        mime = _mime(Path(image))
        b64 = base64.b64encode(Path(image).read_bytes()).decode()
        content.append(
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )
    return content


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/png")
