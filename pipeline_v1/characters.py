"""Per-panel character detection via OpenRouter vision-language models.

Library port of
`research/character_detection_methods/character-detection-openrouter-vlm/run.py`
(the standalone experiment stays untouched): same prompt contract, JSON parsing
and validation, retry policy, and per-call cost accounting (`usage.cost` from
OpenRouter, computed from published pricing as a fallback).

One image per call: the panel is sent as the only image alongside a prompt that
lists the canonical reference characters (from `data/refs/`) with short hints and
asks for exactly `{"characters": ["Name1", ...]}`.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from util import file_record

API_BASE = "https://openrouter.ai/api/v1"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Short neutral hints from the anime, used to help the VLM tell characters
# apart in black-and-white panels. Only names present in data/refs are used.
CHARACTER_HINTS: dict[str, str] = {
    "frieren": "elf, long silver hair, elven ears",
    "fern": "Frieren's apprentice, pale hair, mage staff",
    "stark": "warrior, spiky red hair, large build",
    "himmel": "hero, blue hair, confident",
    "heiter": "priest, light gray hair, cross pendant",
    "eisen": "dwarf, tall, white beard, heavy armor",
    "aura": "demon general, long black hair, dark dress",
    "denken": "elderly mage, black hair and beard",
    "flamme": "Frieren's master, tall elf, long white hair",
    "serie": "great mage, elf, long black hair, dark dress",
    "sein": "priest, black hair, relaxed",
    "kanne": "timid mage girl, dark hair",
    "laufen": "energetic girl, short dark hair, dark outfit",
    "lawine": "mage girl, light hair",
    "richter": "older city guard, short dark hair, mustache",
    "wirbel": "mage, light hair",
    "uebel": "shadow warrior, black hair, dark clothes",
}

MAX_ATTEMPTS = 8
BASE_BACKOFF_S = 5.0


# ---------------------------------------------------------------------------
# Canonical character names (from the refs directory)

def reference_label(path: Path) -> str:
    name = path.stem
    for suffix in ("_reference", "_anime_profile"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip().title()


def canonical_characters(refs_dir: Path) -> list[str]:
    seen: dict[str, str] = {}
    for path in sorted(refs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        label = reference_label(path)
        seen.setdefault(label.lower(), label)
    return [seen[key] for key in sorted(seen)]


def build_prompt(template: str, characters: list[str]) -> str:
    lines = []
    for name in characters:
        hint = CHARACTER_HINTS.get(name.lower())
        lines.append(f"- {name}: {hint}" if hint else f"- {name}")
    return template.replace("{characters}", "\n".join(lines))


# ---------------------------------------------------------------------------
# Parsing and validation of the model's answer

def parse_characters(text: str) -> list[Any] | None:
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.S)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "characters" in data:
            if isinstance(data["characters"], list):
                return [str(item) for item in data["characters"]]
            return None
        if isinstance(data, list):
            return [str(item) for item in data]
    return None


def validate_characters(
    parsed: list[Any] | None, canonical: list[str]
) -> tuple[list[str], list[str]]:
    """Return (known characters in canonical casing, unknown entries)."""
    if parsed is None:
        return [], []
    lookup = {name.lower(): name for name in canonical}
    known: list[str] = []
    unknown: list[str] = []
    for item in parsed:
        key = str(item).strip()
        if not key:
            continue
        if key.lower() in lookup:
            name = lookup[key.lower()]
            if name not in known:
                known.append(name)
        elif key not in unknown:
            unknown.append(key)
    return known, unknown


# ---------------------------------------------------------------------------
# Detection client

@dataclass
class CharacterRecord:
    """Result of one character-detection call for one panel."""

    status: str                       # ok | ok-with-unknown | unparseable | error
    characters: list[str]             # canonical names, validated
    unknown_entries: list[str]
    response_text: str
    usage: dict[str, int]
    cost_usd: float | None
    cost_source: str
    latency_s: float
    model_returned: str | None
    attempts: int
    error: str | None = None
    finished_at: str | None = None

    def to_dict(self, panel: Path, page: str) -> dict[str, Any]:
        return {
            "status": self.status,
            "page": page,
            "panel": panel.name,
            "panel_sha256": file_record(panel)["sha256"],
            "characters": self.characters,
            "unknown_entries": self.unknown_entries,
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


class CharacterDetector(Protocol):
    """Interface for anything that lists which reference characters appear in
    a manga panel."""

    def detect(self, panel: Path, refs_dir: Path) -> CharacterRecord:
        ...


class OpenRouterCharacterDetector:
    """One OpenRouter VLM call per panel (vision-capable chat model)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = API_BASE,
        prompt_template: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        client: Any = None,  # injected OpenAI-compatible client (tests)
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.canonical: list[str] = []
        self.prompt: str = ""

    def prepare(self, refs_dir: Path, prompt_file: Path | None = None) -> None:
        """Load the canonical character list and build the prompt once per run."""
        self.canonical = canonical_characters(refs_dir)
        template = self.prompt_template
        if template is None:
            if prompt_file is None:
                raise ValueError("a prompt template or prompt_file is required")
            template = prompt_file.read_text(encoding="utf-8")
        self.prompt = build_prompt(template, self.canonical)

    def detect(self, panel: Path, refs_dir: Path) -> CharacterRecord:
        if not self.prompt or not self.canonical:
            self.prepare(refs_dir)
        info = file_record(panel)
        info["data_base64"] = base64.b64encode(panel.read_bytes()).decode()
        return self._classify(info)

    def _classify(self, panel: dict[str, Any]) -> CharacterRecord:
        from openai import APIError, APIConnectionError, BadRequestError, RateLimitError

        content: list[dict[str, Any]] = [
            {"type": "text", "text": self.prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{panel['mime_type']};base64,{panel['data_base64']}"
                },
            },
        ]
        messages = [{"role": "user", "content": content}]
        response_format: str | None = "json_object"
        attempts = 0
        while True:
            attempts += 1
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if response_format:
                kwargs["response_format"] = {"type": response_format}
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(**kwargs)
            except BadRequestError as error:
                message = str(error)
                unsupported_format = response_format == "json_object" and (
                    "response_format" in message
                    or "json_object" in message
                    or "json" in message.lower()
                )
                if unsupported_format and attempts <= 2:
                    print(
                        "    response_format=json_object unsupported, retrying without it",
                        flush=True,
                    )
                    response_format = None
                    attempts = 0
                    continue
                return self._error_record(
                    f"BadRequestError: {error}", started, attempts
                )
            except RateLimitError as error:
                if attempts >= MAX_ATTEMPTS:
                    return self._error_record(
                        f"RateLimitError: {error}", started, attempts
                    )
                headers = getattr(getattr(error, "response", None), "headers", {}) or {}
                retry_after = headers.get("retry-after")
                delay = float(retry_after) if retry_after else BASE_BACKOFF_S * attempts
                print(
                    f"    rate limited (attempt {attempts}), retrying in {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
                continue
            except (APIConnectionError, APIError) as error:
                if attempts >= MAX_ATTEMPTS:
                    return self._error_record(
                        f"{type(error).__name__}: {error}", started, attempts
                    )
                delay = BASE_BACKOFF_S * attempts
                print(
                    f"    transient error (attempt {attempts}): {type(error).__name__}, "
                    f"retrying in {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
                continue
            except Exception as error:  # noqa: BLE001 - record anything else
                return self._error_record(
                    f"{type(error).__name__}: {error}", started, attempts
                )

            latency = time.monotonic() - started
            usage = getattr(response, "usage", None)
            usage_record: dict[str, int] = {}
            cost_usd: float | None = None
            cost_source = "unavailable"
            if usage is not None:
                usage_record = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                cost_value = getattr(usage, "cost", None)
                if cost_value is None:
                    raw_usage = getattr(usage, "model_extra", None) or {}
                    cost_value = raw_usage.get("cost")
                if cost_value is not None:
                    try:
                        cost_usd = round(float(cost_value), 8)
                        cost_source = "usage.cost"
                    except (TypeError, ValueError):
                        cost_usd = None
            text = (
                response.choices[0].message.content or ""
                if response.choices
                else ""
            )
            parsed = parse_characters(text)
            known, unknown = validate_characters(parsed, self.canonical)
            status = "ok"
            if parsed is None:
                status = "unparseable"
            elif unknown:
                status = "ok-with-unknown"
            return CharacterRecord(
                status=status,
                characters=known,
                unknown_entries=unknown,
                response_text=text,
                usage=usage_record,
                cost_usd=cost_usd,
                cost_source=cost_source,
                latency_s=latency,
                model_returned=getattr(response, "model", None),
                attempts=attempts,
                error=None,
                finished_at=_iso_now(),
            )

    def _error_record(self, message: str, started: float, attempts: int) -> CharacterRecord:
        return CharacterRecord(
            status="error",
            characters=[],
            unknown_entries=[],
            response_text="",
            usage={},
            cost_usd=None,
            cost_source="unavailable",
            latency_s=time.monotonic() - started,
            model_returned=None,
            attempts=attempts,
            error=message,
            finished_at=_iso_now(),
        )


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
