"""Left-to-right palette verification for the COL-* evaluation cases.

The evaluator's color cases are human-review only; COL-004 (palette
geography, V1.2 problem 1) is instead resolved by asking an OpenRouter
vision-language model — `openai/gpt-5.6-luna` — whether the generated
colorized panel matches the fixture's expected left-to-right hair-color
order (green Heiter / blue Himmel / white-pink Frieren / yellow Eisen).
The verifier sends the colorized panel (image #1) plus the original
monochrome crop (image #2) and expects a strict JSON verdict.

Shared with the character detector: the OpenAI-compatible call machinery
with retry/backoff and `usage.cost` accounting lives in
`characters.call_vlm`; the prompt template is `verify_l2r_prompt.txt`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from characters import _extract_json_object, _iso_now, call_vlm

API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_VERIFY_MODEL = "openai/gpt-5.6-luna"

VERIFY_PROMPT_FILE = Path(__file__).resolve().parent / "verify_l2r_prompt.txt"

# Status values: verified | mismatch | unparseable | error
VERIFIED = "verified"
MISMATCH = "mismatch"
UNPARSEABLE = "unparseable"
ERROR = "error"


# ---------------------------------------------------------------------------
# Prompt rendering

def build_l2r_prompt(template: str, left_to_right: list[dict]) -> str:
    """Render `{left_to_right}` with the fixture's ordered expectation, e.g.
    `- 1. Heiter: light green hair`."""
    lines = [
        "- {}. {}: {} hair".format(
            index, entry.get("character", "?"), entry.get("hair", "?")
        )
        for index, entry in enumerate(left_to_right, start=1)
    ]
    return template.replace("{left_to_right}", "\n".join(lines))


def default_l2r_prompt(left_to_right: list[dict]) -> str:
    return build_l2r_prompt(VERIFY_PROMPT_FILE.read_text(encoding="utf-8"),
                            left_to_right)


# ---------------------------------------------------------------------------
# Answer parsing

def parse_l2r_verdict(text: str) -> dict | None:
    """Parse `{"left_to_right_matches": bool, "per_position": [...], "notes"}`.

    Returns None for unparseable/malformed answers. `left_to_right_matches`
    accepts a bool or the strings "true"/"false" (case-insensitive);
    `per_position` entries are kept as-is when valid.
    """
    if not text:
        return None
    data = _extract_json_object(text.strip())
    if not isinstance(data, dict) or "left_to_right_matches" not in data:
        return None
    raw = data["left_to_right_matches"]
    if isinstance(raw, bool):
        matches = raw
    elif isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        matches = raw.strip().lower() == "true"
    else:
        return None
    per_position = data.get("per_position")
    if not isinstance(per_position, list):
        per_position = []
    cleaned: list[dict] = []
    for entry in per_position:
        if isinstance(entry, dict):
            cleaned.append(entry)
    return {
        "left_to_right_matches": matches,
        "per_position": cleaned,
        "notes": str(data.get("notes", "") or ""),
    }


# ---------------------------------------------------------------------------
# Verifier client

@dataclass
class L2RVerifyRecord:
    """Result of one left-to-right verification call for one color case."""

    status: str                       # verified | mismatch | unparseable | error
    left_to_right_matches: bool | None
    per_position: list[dict]
    notes: str
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
            "left_to_right_matches": self.left_to_right_matches,
            "per_position": self.per_position,
            "notes": self.notes,
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


class LeftToRightVerifier:
    """OpenRouter VLM that checks the left-to-right hair-color assignment of
    a colorized panel against an ordered expectation list."""

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
        left_to_right: list[dict],
    ) -> L2RVerifyRecord:
        """Ask the VLM whether the left-to-right hair colors of the colorized
        panel match `left_to_right` (ordered `{character, hair}` entries).

        Sends the colorized panel as the primary image plus the monochrome
        crop as context when available. One paid OpenRouter call.
        """
        template = self.prompt_template or VERIFY_PROMPT_FILE.read_text(
            encoding="utf-8"
        )
        prompt = build_l2r_prompt(template, left_to_right)

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
        ]
        for image in (colorized, input_crop):
            if image is None or not Path(image).is_file():
                continue
            mime = _mime(Path(image))
            b64 = base64.b64encode(Path(image).read_bytes()).decode()
            content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

        result = call_vlm(
            self.client, self.model, content,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )
        parsed = parse_l2r_verdict(result.text) if result.error is None else None

        if result.error is not None:
            status = ERROR
            matches = None
            per_position: list[dict] = []
            notes = ""
        elif parsed is None:
            status = UNPARSEABLE
            matches = None
            per_position = []
            notes = ""
        else:
            matches = parsed["left_to_right_matches"]
            per_position = parsed["per_position"]
            notes = parsed["notes"]
            status = VERIFIED if matches else MISMATCH

        return L2RVerifyRecord(
            status=status,
            left_to_right_matches=matches,
            per_position=per_position,
            notes=notes,
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


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/png")
