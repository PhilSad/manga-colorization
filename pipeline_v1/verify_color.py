"""Generic canonical-palette verification for the COL-* evaluation cases.

One verifier for all color cases: asks an OpenRouter vision-language model —
`openai/gpt-5.6-luna` — whether every character in the colorized panel has
its canonical Frieren palette. The prompt is deliberately generic: no
fixture expectations (required/forbidden colors, left-to-right order) are
rendered into it; the model judges from its own knowledge of the manga. The
verdict is a **real structured output** — `response_format` type
`json_schema` (strict) with the fields `analyse: str`, `good_color: bool`
and `fix_prompt: str`, routed with `provider.require_parameters: true` so the
request only reaches endpoints that natively support structured outputs and
never silently degrades to loose JSON. The third field is the corrective
instruction consumed by the verification loop (verify_loop.py); the eval
suite ignores it.

Shared with the character detector: the OpenAI-compatible call machinery
with retry/backoff and `usage.cost` accounting lives in
`characters.call_vlm`; the prompt template is `verify_color_prompt.txt`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
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
# The schema is a superset of the COL-* evaluation verdict: `fix_prompt` is
# required by the verify loop (verify_loop.py) so one schema serves both the
# eval suite and the per-panel loop (the eval's parse only reads the fields it
# needs; the extra field is ignored there).
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
        "fix_prompt": {
            "type": "string",
            "description": (
                "If good_color is false: a concise corrective instruction "
                "for the colorizer naming each wrong character and the exact "
                "canonical colors to apply (e.g. 'Frieren: hair silver-white, "
                "eyes teal — the hair was colored lavender'). Empty string "
                "when good_color is true."
            ),
        },
    },
    "required": ["analyse", "good_color", "fix_prompt"],
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

# bbox mode (`--verify-mode bbox`, see verify_loop.py): the retry verdict
# carries the corrective regions too, so one Luna call per retry provides the
# fix_prompt (full-panel fallback) AND the edit-need bounding boxes. The
# probe schema (scripts/probe_luna_bboxes.py) plus `fix_prompt` — the probe's
# exact recipe and the empty-regions fallback coexist (user decision 1, plan
# docs/plans/verify-bbox-region-edit.md).
BBOX_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analyse": {
            "type": "string",
            "description": (
                "Which characters' palettes are correct or wrong, and where "
                "the wrong regions are."
            ),
        },
        "good_color": {
            "type": "boolean",
            "description": "True if every character has its canonical palette.",
        },
        "fix_prompt": {
            "type": "string",
            "description": (
                "If good_color is false: a concise corrective instruction "
                "for the colorizer naming each wrong character and the exact "
                "canonical colors to apply (e.g. 'Frieren: hair silver-white, "
                "eyes teal — the hair was colored lavender'). Empty string "
                "when good_color is true."
            ),
        },
        "regions": {
            "type": "array",
            "description": (
                "One entry per region of the colorized image that needs a "
                "color edit; empty when good_color is true or when no region "
                "can be localized (the fix_prompt then covers the whole panel)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "character": {
                        "type": "string",
                        "description": (
                            "Canonical character name from the atlas, or a "
                            "short description."
                        ),
                    },
                    "problem": {
                        "type": "string",
                        "description": (
                            "What is wrong and what the canonical color "
                            "should be."
                        ),
                    },
                    "fix_suggestion": {
                        "type": "string",
                        "description": (
                            "Exact corrective instruction for the colorizer."
                        ),
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "[x1, y1, x2, y2] in normalized 0-1000 integer "
                            "coordinates; (0,0) top-left, (1000,1000) "
                            "bottom-right of the colorized image."
                        ),
                    },
                },
                "required": [
                    "character", "problem", "fix_suggestion", "bbox",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analyse", "good_color", "fix_prompt", "regions"],
    "additionalProperties": False,
}

BBOX_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "bbox_verdict",
        "strict": True,
        "schema": BBOX_VERDICT_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# Answer parsing

def parse_color_verdict(text: str) -> dict | None:
    """Parse `{"analyse": str, "good_color": bool, "fix_prompt": str}`.

    Returns None for unparseable/malformed answers. `good_color` accepts a
    bool or the strings "true"/"false" (case-insensitive); `analyse` and
    `fix_prompt` default to "" when missing. With the strict json_schema
    request the content is guaranteed-valid JSON, so this parser is a safety
    net only (fix_prompt is present on real structured outputs).
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
    fix_prompt = data.get("fix_prompt")
    return {
        "analyse": str(analyse or ""),
        "good_color": good_color,
        "fix_prompt": str(fix_prompt or ""),
    }


def parse_bbox_verdict(text: str) -> dict | None:
    """Parse `{analyse, good_color, fix_prompt, regions[]}` for bbox mode.

    Mirrors the probe's parser (scripts/probe_luna_bboxes.py) plus the
    `fix_prompt` field; returns None for malformed answers. Regions whose
    bbox is missing/malformed are kept with `bbox: None` (draw_boxes skips
    them; the region text is still recorded)."""
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
    regions = data.get("regions")
    if regions is None:
        regions = []
    elif not isinstance(regions, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            bbox = [int(round(float(v))) for v in bbox]
        else:
            bbox = None
        cleaned.append(
            {
                "character": str(region.get("character") or ""),
                "problem": str(region.get("problem") or ""),
                "fix_suggestion": str(region.get("fix_suggestion") or ""),
                "bbox": bbox,
            }
        )
    return {
        "analyse": str(data.get("analyse") or ""),
        "good_color": good_color,
        "fix_prompt": str(data.get("fix_prompt") or ""),
        "regions": cleaned,
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
    usage: dict[str, Any]
    cost_usd: float | None
    cost_source: str
    latency_s: float
    model_returned: str | None
    attempts: int
    error: str | None = None
    fix_prompt: str = ""        # corrective instruction ("" when good_color)
    finished_at: str | None = None
    regions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        doc = {
            "status": self.status,
            "good_color": self.good_color,
            "analyse": self.analyse,
            "fix_prompt": self.fix_prompt,
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
        if self.regions:
            doc["regions"] = self.regions
        return doc


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
        response_format: dict | None = None,  # default: COLOR_VERDICT_SCHEMA
        reasoning_effort: str | None = None,  # bbox mode: "high"
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
        self.response_format = response_format or RESPONSE_FORMAT
        self.reasoning_effort = reasoning_effort

    def verify(
        self,
        colorized: Path,
        input_crop: Path | None,
        atlas: Path | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ColorVerifyRecord:
        """Ask the VLM whether every character in the colorized panel has its
        canonical Frieren palette.

        Sends the colorized panel as the primary image plus the monochrome
        crop and the reference atlas of the detected characters (the same
        contact sheet the colorizer saw) as context when available. One paid
        OpenRouter call with strict json_schema structured output
        (`analyse`/`good_color`/`fix_prompt`; the eval suite reads only the
        first two, the verify loop uses `fix_prompt` to re-colorize). The
        request omits `temperature` (gpt-5.6-luna does not support it;
        sending it would make `provider.require_parameters` reject every
        endpoint).
        """
        template = self.prompt_template or VERIFY_PROMPT_FILE.read_text(
            encoding="utf-8"
        )
        content = _content_with_images(template, (colorized, input_crop, atlas))

        merged_body: dict[str, Any] = dict(extra_body or {})
        if self.reasoning_effort:
            merged_body.setdefault(
                "reasoning", {"effort": self.reasoning_effort}
            )
        result = call_vlm(
            self.client, self.model, content,
            max_tokens=self.max_tokens, temperature=self.temperature,
            response_format=self.response_format,
            extra_body=merged_body,
        )
        parsed = None
        if result.error is None:
            schema = self.response_format.get("json_schema", {}).get("schema", {})
            if "regions" in schema.get("properties", {}):
                parsed = parse_bbox_verdict(result.text)
            else:
                parsed = parse_color_verdict(result.text)

        if result.error is not None:
            status = ERROR
            good_color = None
            analyse = ""
            fix_prompt = ""
            regions: list[dict[str, Any]] = []
        elif parsed is None:
            status = UNPARSEABLE
            good_color = None
            analyse = ""
            fix_prompt = ""
            regions = []
        else:
            good_color = parsed["good_color"]
            analyse = parsed["analyse"]
            fix_prompt = parsed["fix_prompt"]
            regions = parsed.get("regions", [])
            status = VERIFIED if good_color else MISMATCH

        return ColorVerifyRecord(
            status=status,
            good_color=good_color,
            analyse=analyse,
            fix_prompt=fix_prompt,
            regions=regions,
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
