"""Character detection via OpenRouter vision-language models.

The detection algorithm is selected by `--detection-mode` and implemented as
one strategy per mode (`DETECTION_STRATEGIES`, the strategy design pattern);
`OpenRouterCharacterDetector` is the context owning the shared state (client,
prompts, canonical roster, profiles, cast files) and dispatching through the
unified `detect(mode, ...)` entry point:

- `panel` (V1): one call per panel, the panel as the only image, prompt
  listing the canonical reference characters (hints come from the shared
  character profiles, task 0002) and asking for exactly
  `{"characters": ["Name1", ...]}`.
- `page` (V1.1, task 0003): one paid call per page. The full page is sent
  with the panels numbered in reading order (same numbers the extraction
  uses); the model returns a strict per-panel mapping. Missing, invalid, or
  explicitly `uncertain` panel entries trigger a cropped-panel fallback call
  (the V1 per-panel prompt).
- `panel-page` (V1.2): one paid call per panel that sends the full
  page (numbered, target panel highlighted) as global context *plus* the
  cropped panel, with the same cropped-panel fallback as page mode.
- `panel-page-prev2`: panel-page that also sends the two preceding pages in
  reading order as extra story context (fewer when they do not exist; blank
  pages are skipped), so the model can use recent story events to disambiguate.
- `panel-page-cast`: panel-page with an automatically derived per-chapter
  cast shortlist (explicit `cast_key` -> `--cast-key` -> `cast_key_for_page`).
- `panel-page-prev2-cast` (default): panel-page-prev2 with an automatically derived
  per-chapter cast shortlist (same resolution order as `panel-page-cast`).

Parsing, validation, retry policy, and per-call cost accounting (`usage.cost`
from OpenRouter) are shared with the standalone research method
`character-detection-openrouter-vlm`.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from profiles import hint_text, load_profiles
from tqdm import tqdm
from util import file_record

API_BASE = "https://openrouter.ai/api/v1"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

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


def cast_shortlist_for(chapter_casts_file: Path, cast_key: str | None) -> list[str] | None:
    """Deterministic cached chapter cast shortlist (never fetched remotely)."""
    if not cast_key:
        return None
    data = json.loads(Path(chapter_casts_file).read_text(encoding="utf-8"))
    casts = data.get("casts", {})
    if cast_key not in casts:
        raise ValueError(
            f"cast key {cast_key!r} not found in {chapter_casts_file} "
            f"(available: {sorted(casts)})"
        )
    return [str(name) for name in casts[cast_key]["characters"]]


# Chapter tag / page-number patterns for auto-cast derivation.
#   - "... - c005 (v01) - p130 ..."   (volume pages; spreads: "p004-p005")
#   - "0134-004.png"                  (data/chapter_134/ style)
CAST_TAG_RE = re.compile(r"- c(\d{1,3})(?:\s|-)")
P_NUMBER_RE = re.compile(r"- p(\d{1,4})(?:-p\d{1,4})?(?:\s|-)")
LEADING_CHAPTER_RE = re.compile(r"^(\d{3,4})-")


def _cast_key_for_number(number: int, casts: dict) -> str | None:
    """`c{number:03d}` if that shortlist exists, else None (full roster)."""
    key = f"c{number:03d}"
    return key if key in casts else None


def _cast_key_from_map(
    page_path: Path, chapter_page_map_file: Path, casts: dict
) -> str | None:
    """Authoritative lookup via chapter_page_map.json: find the chapter whose
    volume contains this page and whose p-number range covers the file's
    p-number. Corrects volumes with mislabeled filename tags (v09 is tagged
    c078 everywhere but spans chapters 78-87)."""
    match = P_NUMBER_RE.search(page_path.name)
    if not match:
        return None
    p_number = int(match.group(1))
    data = json.loads(
        Path(chapter_page_map_file).read_text(encoding="utf-8")
    )
    for chapter in data.get("chapters", []):
        volume_dir = chapter.get("volume_dir")
        if not volume_dir or volume_dir not in str(page_path):
            continue
        p_start = chapter.get("p_start")
        p_end = chapter.get("p_end", p_start)
        if p_start is None or not (p_start <= p_number <= p_end):
            continue
        return _cast_key_for_number(chapter["number"], casts)
    return None


def cast_key_for_page(
    page_path: Path,
    chapter_casts_file: Path,
    chapter_page_map_file: Path | None = None,
) -> str | None:
    """Auto-derive the chapter-cast shortlist key for a page, or None when no
    cast applies (full roster). Order:

    1. `chapter_page_map.json` (authoritative when the page lives in a mapped
       volume; fixes the mislabeled v09 where filename tags say c078);
    2. the filename chapter tag (`- c005 -`);
    3. a leading `NNN-` chapter prefix (`0134-004.png` -> c134).

    A derived key is only returned when it exists in `chapter_casts_file`.
    """
    casts = json.loads(
        Path(chapter_casts_file).read_text(encoding="utf-8")
    ).get("casts", {})
    if chapter_page_map_file is not None and chapter_page_map_file.is_file():
        key = _cast_key_from_map(page_path, chapter_page_map_file, casts)
        if key is not None:
            return key
    name = page_path.name
    match = CAST_TAG_RE.search(name)
    if match:
        key = _cast_key_for_number(int(match.group(1)), casts)
        if key is not None:
            return key
    match = LEADING_CHAPTER_RE.match(name)
    if match:
        key = _cast_key_for_number(int(match.group(1)), casts)
        if key is not None:
            return key
    return None


def build_prompt(
    template: str,
    characters: list[str],
    profiles: dict | None = None,
    cast_shortlist: list[str] | None = None,
) -> str:
    lines = []
    for name in characters:
        hint = hint_text(name, profiles) if profiles else None
        lines.append(f"- {name}: {hint}" if hint else f"- {name}")
    rendered = template.replace("{characters}", "\n".join(lines))
    if cast_shortlist:
        cast_text = (
            "\nThis page comes from a chapter whose likely cast is limited to: "
            + ", ".join(cast_shortlist)
            + ". Do not name characters outside this list; report them as "
            + "unknown (\"uncertain\": true)."
        )
    else:
        cast_text = ""
    return rendered.replace("{cast_shortlist}", cast_text)


# ---------------------------------------------------------------------------
# Parsing and validation of the model's answer

def _extract_json_object(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.S)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def parse_characters(text: str) -> list[Any] | None:
    if not text:
        return None
    data = _extract_json_object(text.strip())
    if data is None:
        return None
    if isinstance(data, dict) and "characters" in data:
        if isinstance(data["characters"], list):
            return [str(item) for item in data["characters"]]
        return None
    if isinstance(data, list):
        return [str(item) for item in data]
    return None


def parse_page_mapping(text: str) -> dict[str, dict[str, Any]] | None:
    """Parse the page-level response into `{panel_key: {characters, uncertain}}`.

    Returns None for unparseable/malformed answers. Panel keys are validated
    by the caller against the expected reading-order set.
    """
    if not text:
        return None
    data = _extract_json_object(text.strip())
    if not isinstance(data, dict) or not isinstance(data.get("panels"), dict):
        return None
    mapping: dict[str, dict[str, Any]] = {}
    for panel_key, entry in data["panels"].items():
        if not isinstance(entry, dict):
            return None
        characters = entry.get("characters", [])
        if not isinstance(characters, list):
            return None
        mapping[str(panel_key)] = {
            "characters": [str(item) for item in characters],
            "uncertain": bool(entry.get("uncertain", False)),
        }
    return mapping


def parse_panel_with_page(text: str) -> dict | None:
    """Parse the panel+page answer `{"characters": [...], "uncertain": bool}`.

    Same per-panel shape as `parse_characters` plus an optional `uncertain`
    flag; returns None for unparseable/malformed answers.
    """
    if not text:
        return None
    data = _extract_json_object(text.strip())
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list):
        return None
    return {
        "characters": [str(item) for item in data["characters"]],
        "uncertain": bool(data.get("uncertain", False)),
    }


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

    status: str                       # ok | ok-with-unknown | unparseable | error | forced
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
    source: str = "panel"             # "page" | "fallback" | "forced"
    uncertain: bool = False

    def to_dict(self, panel: Path, page: str) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "uncertain": self.uncertain,
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


@dataclass
class PageCharacterRecord:
    """Result of one page-level detection call plus its fallbacks."""

    status: str                       # "ok" | "partial" | "error"
    page: str
    panels: dict[str, CharacterRecord] = field(default_factory=dict)
    page_calls: int = 0
    fallback_calls: int = 0
    unpriced_calls: int = 0
    cost_usd: float = 0.0
    total_latency_s: float = 0.0
    error: str | None = None
    page_response_text: str = ""      # raw page-level answer (provenance)
    page_parse_ok: bool = False
    cast_key: str | None = None        # effective cast shortlist (panel-page-cast)


class OpenRouterCharacterDetector:
    """OpenRouter VLM client; the context for the detection strategies.

    Owns the shared state (client, prompts, canonical roster, profiles, cast
    files) and exposes `strategy_for(mode)` / the unified `detect(mode, ...)`
    entry point; the per-mode algorithms live in the `DetectionStrategy`
    implementations (see `DETECTION_STRATEGIES`).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = API_BASE,
        prompt_template: str | None = None,
        panel_prompt_template: str | None = None,
        panel_page_prompt_template: str | None = None,
        panel_page_prev2_prompt_template: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,  # injected OpenAI-compatible client (tests)
        profiles_file: Path | None = None,
        chapter_casts_file: Path | None = None,
        chapter_page_map_file: Path | None = None,
        cast_key: str | None = None,
        workers: int = 1,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.prompt_template = prompt_template
        self.panel_prompt_template = panel_prompt_template
        self.panel_page_prompt_template = panel_page_prompt_template
        self.panel_page_prev2_prompt_template = panel_page_prev2_prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.profiles_file = Path(profiles_file) if profiles_file else None
        self.chapter_casts_file = Path(chapter_casts_file) if chapter_casts_file else None
        self.chapter_page_map_file = (
            Path(chapter_page_map_file) if chapter_page_map_file else None
        )
        self.cast_key = cast_key
        self.workers = workers
        self._strategies: dict = {}  # memoized DetectionStrategy per mode
        self.canonical: list[str] = []
        self.profiles: dict = {}
        self.prompt: str = ""
        self.panel_prompt: str = ""
        self.panel_page_prompt: str = ""
        self.panel_page_prev2_prompt: str = ""
        # Guards prompt swaps (set_cast) against readers in worker threads.
        self._prompt_lock = threading.Lock()
        # Raw templates (kept so prompts can be rebuilt per chapter cast).
        self._page_template: str = ""
        self._panel_template: str = ""
        self._panel_page_template: str = ""
        self._panel_page_prev2_template: str = ""

    def prepare(
        self,
        refs_dir: Path,
        prompt_file: Path | None = None,
        panel_prompt_file: Path | None = None,
        panel_page_prompt_file: Path | None = None,
        panel_page_prev2_prompt_file: Path | None = None,
    ) -> None:
        """Load the canonical list, profiles, cast shortlist and build the
        prompts once per run."""
        self.canonical = canonical_characters(refs_dir)
        if self.profiles_file is not None and self.profiles_file.is_file():
            self.profiles = load_profiles(self.profiles_file)

        template = self.prompt_template
        if template is None:
            if prompt_file is None:
                raise ValueError("a page prompt template or prompt_file is required")
            template = Path(prompt_file).read_text(encoding="utf-8")
        self._page_template = template

        panel_template = self.panel_prompt_template
        if panel_template is None and panel_prompt_file is not None:
            panel_template = Path(panel_prompt_file).read_text(encoding="utf-8")
        self._panel_template = panel_template or ""

        panel_page_template = self.panel_page_prompt_template
        if panel_page_template is None and panel_page_prompt_file is not None:
            panel_page_template = Path(panel_page_prompt_file).read_text(encoding="utf-8")
        self._panel_page_template = panel_page_template or ""

        panel_page_prev2_template = self.panel_page_prev2_prompt_template
        if (
            panel_page_prev2_template is None
            and panel_page_prev2_prompt_file is not None
        ):
            panel_page_prev2_template = Path(
                panel_page_prev2_prompt_file
            ).read_text(encoding="utf-8")
        self._panel_page_prev2_template = panel_page_prev2_template or ""
        self._build_prompts()

    def _build_prompts(self) -> None:
        """(Re)build the four prompts from the stored templates + current
        cast shortlist. Called by `prepare` and by `set_cast`."""
        shortlist = cast_shortlist_for(self.chapter_casts_file, self.cast_key)
        self.prompt = build_prompt(
            self._page_template, self.canonical,
            profiles=self.profiles, cast_shortlist=shortlist,
        )
        self.panel_prompt = build_prompt(
            self._panel_template, self.canonical,
            profiles=self.profiles, cast_shortlist=shortlist,
        )
        self.panel_page_prompt = build_prompt(
            self._panel_page_template, self.canonical,
            profiles=self.profiles, cast_shortlist=shortlist,
        )
        self.panel_page_prev2_prompt = build_prompt(
            self._panel_page_prev2_template, self.canonical,
            profiles=self.profiles, cast_shortlist=shortlist,
        )

    def panel_page_prompt_for(self, cast_key: str | None) -> str:
        """The panel-page prompt rendered for a given cast shortlist. Always
        renders fresh from immutable inputs (template, roster, profiles), so
        per-page casts are thread-safe: it never reads or mutates shared
        prompt state. `None` restores the full roster."""
        shortlist = cast_shortlist_for(self.chapter_casts_file, cast_key)
        return build_prompt(
            self._panel_page_template, self.canonical,
            profiles=self.profiles, cast_shortlist=shortlist,
        )

    def panel_page_prev2_prompt_for(self, cast_key: str | None) -> str:
        """The panel-page-prev2 prompt rendered for a given cast shortlist.
        Always renders fresh from immutable inputs (template, roster,
        profiles), so per-page casts are thread-safe. `None` restores the
        full roster."""
        shortlist = cast_shortlist_for(self.chapter_casts_file, cast_key)
        return build_prompt(
            self._panel_page_prev2_template, self.canonical,
            profiles=self.profiles, cast_shortlist=shortlist,
        )

    def set_cast(self, cast_key: str | None) -> None:
        """Switch the chapter-cast shortlist for all four prompts (no-op when
        unchanged). Called per page in `panel-page-cast` mode so that
        cropped-panel fallbacks reuse the page's cast; lock-guarded for
        worker threads."""
        with self._prompt_lock:
            if cast_key == self.cast_key and self.prompt:
                return
            self.cast_key = cast_key
            self._build_prompts()

    # -- strategy selection ------------------------------------------------

    def strategy_for(self, mode: str) -> "DetectionStrategy":
        """The detection strategy for `mode` (one of the four
        `--detection-mode` values), built once and memoized per detector."""
        strategy_cls = DETECTION_STRATEGIES.get(mode)
        if strategy_cls is None:
            raise ValueError(
                f"unknown detection mode {mode!r} "
                f"(available: {sorted(DETECTION_STRATEGIES)})"
            )
        strategy = self._strategies.get(mode)
        if strategy is None:
            strategy = strategy_cls(self)
            self._strategies[mode] = strategy
        return strategy

    def detect(
        self,
        mode: str,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        """Unified character detection: dispatch `mode` ("panel", "page",
        "panel-page", "panel-page-cast", "panel-page-prev2") to its
        strategy. Every mode returns a per-page `PageCharacterRecord`; the
        per-panel results live in `record.panels[key]`."""
        return self.strategy_for(mode).detect(
            page, panels_dir, expected_panels, refs_dir, cast_key=cast_key
        )

    def _fallback_panel(
        self, panel_key: str, panels_dir: Path, refs_dir: Path
    ) -> CharacterRecord:
        """Cropped-panel fallback call (the V1 panel-only prompt), shared by
        the page-context strategies."""
        panel = _find_panel_file(panels_dir, panel_key)
        if panel is None:
            return CharacterRecord(
                status="error", characters=[], unknown_entries=[],
                response_text="", usage={}, cost_usd=None, cost_source="unavailable",
                latency_s=0.0, model_returned=None, attempts=0,
                error=f"fallback crop missing: {panels_dir / panel_key}",
                finished_at=_iso_now(), source="fallback",
            )
        record = self.strategy_for("panel").detect_panel(panel, refs_dir)
        record.source = "fallback"
        return record

    # -- shared OpenAI call machinery --------------------------------------

    def _call(self, content: list[dict[str, Any]]) -> "_CallResult":
        """One chat completion call (shared machinery in `call_vlm`)."""
        return call_vlm(
            self.client, self.model, content,
            max_tokens=self.max_tokens, temperature=self.temperature,
        )


def call_vlm(
    client: Any,
    model: str,
    content: list[dict[str, Any]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> "_CallResult":
    """One OpenAI-compatible chat completion with retry/backoff and
    `usage.cost` accounting (OpenRouter). Shared by character detection
    (characters.py) and color verification (verify_color.py).

    `response_format` is optional. When omitted (detection), the legacy
    `{"type": "json_object"}` mode is used, with a retry-without fallback
    for endpoints that reject it. When a full response_format object is
    passed (verify_color.py's strict json_schema structured output), it is
    sent verbatim plus `provider.require_parameters: true`, so the request
    only routes to endpoints that support it; a rejection is recorded as an
    error and never silently downgraded to loose JSON. Structured requests
    also omit `temperature` (gpt-5.6-luna does not list it as a supported
    parameter — `require_parameters` would otherwise reject every endpoint)
    and a routing 404 (NotFoundError) is recorded immediately, not
    retried."""
    from openai import (
        APIError,
        APIConnectionError,
        BadRequestError,
        NotFoundError,
        RateLimitError,
    )

    messages = [{"role": "user", "content": content}]
    structured = response_format is not None
    current_format: dict | None = (
        response_format if structured else {"type": "json_object"}
    )
    attempts = 0
    while True:
        attempts += 1
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not structured:
            kwargs["temperature"] = temperature
        if current_format is not None:
            kwargs["response_format"] = current_format
        if structured:
            kwargs["extra_body"] = {"provider": {"require_parameters": True}}
        started = time.monotonic()
        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as error:
            message = str(error)
            if not structured:
                unsupported_format = current_format == {"type": "json_object"} and (
                    "response_format" in message
                    or "json_object" in message
                    or "json" in message.lower()
                )
                if unsupported_format and attempts <= 2:
                    print(
                        "    response_format=json_object unsupported, retrying without it",
                        flush=True,
                    )
                    current_format = None
                    attempts = 0
                    continue
            return _CallResult(
                text="", usage={}, cost_usd=None, cost_source="unavailable",
                latency_s=time.monotonic() - started, model_returned=None,
                attempts=attempts, error=f"BadRequestError: {error}",
            )
        except RateLimitError as error:
            if attempts >= MAX_ATTEMPTS:
                return _CallResult(
                    text="", usage={}, cost_usd=None, cost_source="unavailable",
                    latency_s=time.monotonic() - started, model_returned=None,
                    attempts=attempts, error=f"RateLimitError: {error}",
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
        except NotFoundError as error:
            # Routing/config 404 (e.g. no endpoint supports the required
            # parameters under require_parameters): deterministic, no retry.
            return _CallResult(
                text="", usage={}, cost_usd=None, cost_source="unavailable",
                latency_s=time.monotonic() - started, model_returned=None,
                attempts=attempts, error=f"NotFoundError: {error}",
            )
        except (APIConnectionError, APIError) as error:
            if attempts >= MAX_ATTEMPTS:
                return _CallResult(
                    text="", usage={}, cost_usd=None, cost_source="unavailable",
                    latency_s=time.monotonic() - started, model_returned=None,
                    attempts=attempts,
                    error=f"{type(error).__name__}: {error}",
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
            return _CallResult(
                text="", usage={}, cost_usd=None, cost_source="unavailable",
                latency_s=time.monotonic() - started, model_returned=None,
                attempts=attempts, error=f"{type(error).__name__}: {error}",
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
        return _CallResult(
            text=text, usage=usage_record, cost_usd=cost_usd,
            cost_source=cost_source, latency_s=latency,
            model_returned=getattr(response, "model", None),
            attempts=attempts, error=None,
        )


@dataclass
class _CallResult:
    text: str
    usage: dict[str, int]
    cost_usd: float | None
    cost_source: str
    latency_s: float
    model_returned: str | None
    attempts: int
    error: str | None


def _page_geometry(panels_dir: Path) -> tuple[Path, list]:
    """(page image path, panel boxes in reading order) from panels.json."""
    geometry_path = panels_dir / "panels.json"
    if not geometry_path.is_file():
        raise ValueError(f"missing {geometry_path}; cannot annotate page for detection")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    from detection import PanelBox

    boxes = [
        PanelBox(*d["box"], d.get("confidence", 0.9))
        for d in geometry["detections"]
    ]
    return Path(geometry["page_path"]), boxes


def _annotated_page(page: Path, panels_dir: Path) -> Path:
    """A copy of the page with panel numbers overlaid in reading order (the
    same numbers the extraction uses). Written next to the page in a temp
    location inside the run's panels dir so it is never confused with inputs.

    The write is atomic (temp file + rename) so concurrent workers re-rendering
    the same page dir never expose a torn PNG to readers.
    """
    import os
    import tempfile

    from extraction import draw_overlay
    from PIL import Image

    _page, boxes = _page_geometry(panels_dir)
    annotated = panels_dir / "detection_annotated.png"
    fd, tmp_name = tempfile.mkstemp(dir=panels_dir, suffix=".png")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with Image.open(page) as image:
            draw_overlay(image.convert("RGB"), boxes, tmp)
        tmp.replace(annotated)
    finally:
        if tmp.exists():
            tmp.unlink()
    return annotated


def _panel_index(panel_key: str) -> int:
    """Zero-based index for a `panel_0001`-style key."""
    return int(panel_key.rsplit("_", 1)[1]) - 1


def _find_panel_file(panels_dir: Path, panel_key: str) -> Path | None:
    """The crop file for `panel_key` (any supported extension), or None."""
    panel = panels_dir / f"{panel_key}.png"
    if panel.is_file():
        return panel
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = panels_dir / f"{panel_key}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _highlighted_page(annotated: Path, boxes: list, panel_key: str) -> bytes:
    """PNG bytes of the annotated page with the target panel box outlined in
    blue, so the model knows which crop the second image is."""
    import io

    from PIL import Image, ImageDraw

    index = _panel_index(panel_key)
    if not (0 <= index < len(boxes)):
        raise ValueError(
            f"panel key {panel_key!r} out of range (have {len(boxes)} panels)"
        )
    with Image.open(annotated) as image:
        overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = boxes[index].as_int_tuple()
    draw.rectangle((x1, y1, x2, y2), outline=(40, 120, 255), width=10)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return buffer.getvalue()


def forced_record(
    panel: Path, names: list[str], profiles: dict | None = None
) -> CharacterRecord:
    """Ground-truth identity record: no API call, zero cost (task 0001)."""
    from profiles import unknown_names

    return CharacterRecord(
        status="forced",
        characters=list(names),
        unknown_entries=unknown_names(names, profiles or {}),
        response_text="",
        usage={},
        cost_usd=0.0,
        cost_source="forced ground-truth",
        latency_s=0.0,
        model_returned=None,
        attempts=0,
        error=None,
        finished_at=_iso_now(),
        source="forced",
    )


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Detection strategies (one per --detection-mode)

class DetectionStrategy(Protocol):
    """A detection mode: how characters are detected for `expected_panels`
    of one page. Every mode returns a per-page `PageCharacterRecord` so the
    pipeline step, the integration suite, and the sweep share one uniform
    interface; per-panel results live in `record.panels[key]`."""

    mode: str
    label: str                 # step progress label
    provenance: str | None     # provenance call-file name (None: no file)


    def detect(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        ...


class PanelStrategy:
    """mode="panel": one call per panel, the crop alone (V1 prompt). This is
    also the cropped-panel fallback prompt for the page-context modes."""

    mode = "panel"
    label = "panel"
    provenance = None

    def __init__(self, detector: OpenRouterCharacterDetector) -> None:
        self.detector = detector

    def detect_panel(self, panel: Path, refs_dir: Path) -> CharacterRecord:
        """One panel-only call (also used by the detector's fallback)."""
        detector = self.detector
        if not detector.panel_prompt or not detector.canonical:
            detector.prepare(refs_dir)
        with detector._prompt_lock:
            panel_prompt = detector.panel_prompt
        info = file_record(panel)
        info["data_base64"] = base64.b64encode(panel.read_bytes()).decode()
        content = [
            {"type": "text", "text": panel_prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{info['mime_type']};base64,{info['data_base64']}"}},
        ]
        result = detector._call(content)
        parsed = parse_characters(result.text)
        known, unknown = validate_characters(parsed, detector.canonical)
        if result.error is not None:
            status = "error"
        elif parsed is None:
            status = "unparseable"
        elif unknown:
            status = "ok-with-unknown"
        else:
            status = "ok"
        return CharacterRecord(
            status=status,
            characters=known,
            unknown_entries=unknown,
            response_text=result.text,
            usage=result.usage,
            cost_usd=result.cost_usd,
            cost_source=result.cost_source,
            latency_s=result.latency_s,
            model_returned=result.model_returned,
            attempts=result.attempts,
            error=result.error,
            finished_at=_iso_now(),
            source="panel",
        )

    def detect(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        """One panel-only call per expected panel, aggregated into a per-page
        record (`page_calls` counts the direct per-panel calls)."""
        record = PageCharacterRecord(
            status="ok", page=page.stem if page is not None else panels_dir.name
        )
        for panel_key in expected_panels:
            panel = _find_panel_file(panels_dir, panel_key)
            if panel is None:
                record.status = "partial" if record.status == "ok" else record.status
                record.panels[panel_key] = CharacterRecord(
                    status="error", characters=[], unknown_entries=[],
                    response_text="", usage={}, cost_usd=None,
                    cost_source="unavailable", latency_s=0.0,
                    model_returned=None, attempts=0,
                    error=f"crop missing: {panels_dir / panel_key}",
                    finished_at=_iso_now(), source="panel",
                )
                continue
            rec = self.detect_panel(panel, refs_dir)
            record.panels[panel_key] = rec
            record.page_calls += 1
            record.cost_usd += rec.cost_usd or 0.0
            record.total_latency_s += rec.latency_s
            if rec.cost_usd is None:
                record.unpriced_calls += 1
            if rec.status == "error":
                record.status = "partial" if record.status == "ok" else record.status
        return record


class PageStrategy:
    """mode="page": one page-level call mapping the numbered panels, with
    cropped-panel fallbacks for missing/invalid/uncertain entries."""

    mode = "page"
    label = "page-level"
    provenance = "page_call.json"

    def __init__(self, detector: OpenRouterCharacterDetector) -> None:
        self.detector = detector

    def detect(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        """One page-level call; per-panel fallbacks for missing/invalid/
        uncertain entries."""
        detector = self.detector
        if not detector.prompt or not detector.canonical:
            detector.prepare(refs_dir)
        record = PageCharacterRecord(status="ok", page=page.stem)

        annotated = _annotated_page(page, panels_dir)
        info = file_record(annotated)
        info["data_base64"] = base64.b64encode(annotated.read_bytes()).decode()
        content = [
            {"type": "text", "text": detector.prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{info['mime_type']};base64,{info['data_base64']}"}},
        ]
        result = detector._call(content)
        record.page_calls = 1
        record.cost_usd += result.cost_usd or 0.0
        record.total_latency_s += result.latency_s
        if result.cost_usd is None:
            record.unpriced_calls += 1

        mapping = parse_page_mapping(result.text) if result.error is None else None
        # An unparseable/empty page-level answer is a page-call failure, not a
        # per-panel uncertainty: retry the page call once before falling back
        # per panel (avoids an explosion of fallback calls).
        if mapping is None and result.error is None:
            print(
                f"    page detection: page-level answer for {page.stem} did not parse "
                f"(first 200 chars: {result.text[:200]!r}); retrying once",
                flush=True,
            )
            retry = detector._call(content)
            record.page_calls += 1
            record.cost_usd += retry.cost_usd or 0.0
            record.total_latency_s += retry.latency_s
            if retry.cost_usd is None:
                record.unpriced_calls += 1
            if retry.error is not None:
                record.error = retry.error
            else:
                result = retry
            mapping = parse_page_mapping(result.text) if result.error is None else None

        record.page_response_text = result.text
        record.page_parse_ok = mapping is not None
        mapping = mapping or {}
        expected_set = set(expected_panels)
        if result.error is not None:
            record.status = "error"
            record.error = result.error
        if not record.page_parse_ok and result.error is None:
            print(
                f"    page detection: response for {page.stem} still did not parse; "
                f"falling back per panel",
                flush=True,
            )

        for panel_key in expected_panels:
            entry = mapping.get(panel_key)
            if entry is None:
                # missing or invalid mapping entry -> cropped-panel fallback
                record.status = "partial" if record.status == "ok" else record.status
                fallback = detector._fallback_panel(panel_key, panels_dir, refs_dir)
                record.fallback_calls += 1
                record.cost_usd += fallback.cost_usd or 0.0
                record.total_latency_s += fallback.latency_s
                if fallback.cost_usd is None:
                    record.unpriced_calls += 1
                record.panels[panel_key] = fallback
                continue
            known, unknown = validate_characters(
                entry["characters"], detector.canonical
            )
            if entry["uncertain"] or unknown:
                record.status = "partial" if record.status == "ok" else record.status
                fallback = detector._fallback_panel(panel_key, panels_dir, refs_dir)
                record.fallback_calls += 1
                record.cost_usd += fallback.cost_usd or 0.0
                record.total_latency_s += fallback.latency_s
                if fallback.cost_usd is None:
                    record.unpriced_calls += 1
                record.panels[panel_key] = fallback
                continue
            # Keep only expected panel keys (extra keys are not accepted).
            record.panels[panel_key] = CharacterRecord(
                status="ok",
                characters=known,
                unknown_entries=[],
                response_text=result.text,
                usage=result.usage,
                cost_usd=None,  # cost attributed at page level
                cost_source="page-level",
                latency_s=0.0,
                model_returned=result.model_returned,
                attempts=result.attempts,
                error=None,
                finished_at=_iso_now(),
                source="page",
                uncertain=False,
            )
        # Panels the model invented that were not expected are rejected.
        for extra in set(mapping) - expected_set:
            print(f"    page detection: rejecting unexpected panel key {extra!r}",
                  flush=True)
        return record


def _detect_panels_with_page_context(
    detector: OpenRouterCharacterDetector,
    record: PageCharacterRecord,
    prompt: str,
    progress_desc: str,
    source: str,
    panels_dir: Path,
    expected_panels: list[str],
    refs_dir: Path,
    annotated: Path,
    boxes: list,
    context_images: list[tuple[str, bytes]] | None = None,
) -> None:
    """Shared per-panel loop for the page-context strategies (panel-page,
    panel-page-prev2): one call per panel sending `prompt`, optional extra
    context images (e.g. the preceding pages), the numbered annotated page
    with the target highlighted, and the cropped panel. Unparseable,
    explicit-`uncertain`, unknown-character, and error results fall back to
    the cropped-panel V1 call. Mutates `record` in place."""
    for panel_key in tqdm(
        expected_panels,
        desc=progress_desc,
        unit="panel", leave=False, disable=detector.workers > 1,
    ):
        panel = _find_panel_file(panels_dir, panel_key)
        if panel is None:
            fallback = detector._fallback_panel(panel_key, panels_dir, refs_dir)
            record.fallback_calls += 1
            record.cost_usd += fallback.cost_usd or 0.0
            record.total_latency_s += fallback.latency_s
            if fallback.cost_usd is None:
                record.unpriced_calls += 1
            record.panels[panel_key] = fallback
            record.status = "partial" if record.status == "ok" else record.status
            continue

        highlighted = _highlighted_page(annotated, boxes, panel_key)
        info = file_record(panel)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
        ]
        if context_images:
            for mime, data in context_images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{_b64(data)}"},
                })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_b64(highlighted)}"},
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{info['mime_type']};base64,{_b64(panel.read_bytes())}"
            },
        })
        result = detector._call(content)
        record.page_calls += 1
        record.cost_usd += result.cost_usd or 0.0
        record.total_latency_s += result.latency_s
        if result.cost_usd is None:
            record.unpriced_calls += 1

        known: list[str] = []
        unknown: list[str] = []
        uncertain = False
        parsed = parse_panel_with_page(result.text) if result.error is None else None
        if parsed is not None:
            known, unknown = validate_characters(
                parsed["characters"], detector.canonical
            )
            uncertain = parsed["uncertain"]

        if (
            result.error is None
            and parsed is not None
            and not uncertain
            and not unknown
        ):
            record.panels[panel_key] = CharacterRecord(
                status="ok",
                characters=known,
                unknown_entries=[],
                response_text=result.text,
                usage=result.usage,
                cost_usd=result.cost_usd,
                cost_source=result.cost_source,
                latency_s=result.latency_s,
                model_returned=result.model_returned,
                attempts=result.attempts,
                error=None,
                finished_at=_iso_now(),
                source=source,
                uncertain=False,
            )
            continue

        # Fallback: panel-only prompt (V1), mirrors page-mode behaviour.
        record.status = "partial" if record.status == "ok" else record.status
        fallback = detector._fallback_panel(panel_key, panels_dir, refs_dir)
        record.fallback_calls += 1
        record.cost_usd += fallback.cost_usd or 0.0
        record.total_latency_s += fallback.latency_s
        if fallback.cost_usd is None:
            record.unpriced_calls += 1
        record.panels[panel_key] = fallback


def _previous_page_images(
    panels_dir: Path, count: int = 2
) -> list[tuple[str, bytes]]:
    """(mime, bytes) of the up-to-`count` preceding pages in reading order,
    oldest first, for use as extra detection context.

    The preceding pages are the nearest page dirs before `panels_dir` in the
    run's `1_panels/` layout, read via each sibling's `panels.json`
    `page_path`. Blank pages (no detections) are skipped and the search
    continues further back. Returns fewer than `count` images at the start
    of a book, and `[]` when there are no preceding pages — the caller then
    degrades to plain panel-page behaviour."""
    panels_root = panels_dir.parent
    ordered = sorted(path for path in panels_root.iterdir() if path.is_dir())
    names = [path.name for path in ordered]
    if panels_dir.name not in names:
        return []
    index = names.index(panels_dir.name)
    found: list[Path] = []
    for candidate in reversed(ordered[:index]):
        if len(found) >= count:
            break
        geometry_path = candidate / "panels.json"
        if not geometry_path.is_file():
            continue
        try:
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if geometry.get("blank_page") or not geometry.get("detections"):
            continue
        page_path = Path(geometry.get("page_path") or "")
        if not page_path.is_file():
            continue
        found.append(page_path)
    found.reverse()  # oldest first
    images: list[tuple[str, bytes]] = []
    for path in found:
        info = file_record(path)
        images.append((info["mime_type"], path.read_bytes()))
    return images


class PanelPageStrategy:
    """mode="panel-page": one paid call per panel — the full page (numbered,
    target panel highlighted) as global context *plus* the cropped panel.

    Per-panel fallback (same as page mode): unparseable answers, explicit
    `uncertain`, unknown-character entries, and call errors fall back to the
    V1 panel-only prompt call.

    `cast_key` renders the panel-page prompt for that chapter shortlist
    without touching shared state (thread-safe for `panel-page-cast` mode);
    `None` uses the detector's current cast.
    """

    mode = "panel-page"
    label = "panel+page"
    provenance = "panel_page_calls.json"

    def __init__(self, detector: OpenRouterCharacterDetector) -> None:
        self.detector = detector

    def detect(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        detector = self.detector
        if not detector.canonical:
            detector.prepare(refs_dir)
        if not detector._panel_page_template:
            raise ValueError(
                "panel-page detection needs a panel-page prompt "
                "(--vlm-panel-page-prompt-file)"
            )
        record = PageCharacterRecord(status="ok", page=page.stem)
        # cast_key=None means "the detector's current cast" (panel-page with
        # --cast-key); an explicit key (panel-page-cast) renders that cast.
        effective_cast = detector.cast_key if cast_key is None else cast_key
        print(f'effective cast for {page.stem}: {effective_cast}', flush=True)
        record.cast_key = effective_cast
        prompt = detector.panel_page_prompt_for(effective_cast)

        page_image, boxes = _page_geometry(panels_dir)
        annotated = _annotated_page(page_image, panels_dir)
        _detect_panels_with_page_context(
            detector, record, prompt,
            progress_desc=f"characters: {page.stem} (panel-page)",
            source="panel-page",
            panels_dir=panels_dir, expected_panels=expected_panels,
            refs_dir=refs_dir, annotated=annotated, boxes=boxes,
        )
        return record


class PanelPagePrev2Strategy(PanelPageStrategy):
    """mode="panel-page-prev2": panel-page with the two preceding pages (in
    reading order) sent as extra story-context images, so the model can use
    recent story events, dialogue and outfits to disambiguate identity.
    Pages with fewer preceding pages degrade gracefully to plain panel-page
    behaviour (0 or 1 images instead of 2).

    The preceding pages are the nearest non-blank page dirs before the
    current one in the run's `1_panels/` layout (`_previous_page_images`);
    their raw page images are read via `panels.json` `page_path`. Per-panel
    fallback semantics are identical to panel-page (shared loop).
    """

    mode = "panel-page-prev2"
    label = "panel+page+prev2"
    provenance = "panel_page_prev2_calls.json"

    PREV_PAGE_COUNT = 2

    def detect(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        detector = self.detector
        if not detector.canonical:
            detector.prepare(refs_dir)
        if not detector._panel_page_prev2_template:
            raise ValueError(
                "panel-page-prev2 detection needs a panel-page-prev2 prompt "
                "(--vlm-panel-page-prev2-prompt-file)"
            )
        record = PageCharacterRecord(status="ok", page=page.stem)
        effective_cast = detector.cast_key if cast_key is None else cast_key
        print(f'effective cast for {page.stem}: {effective_cast}', flush=True)
        record.cast_key = effective_cast
        prompt = detector.panel_page_prev2_prompt_for(effective_cast)

        page_image, boxes = _page_geometry(panels_dir)
        annotated = _annotated_page(page_image, panels_dir)
        context_images = _previous_page_images(
            panels_dir, count=self.PREV_PAGE_COUNT
        )
        _detect_panels_with_page_context(
            detector, record, prompt,
            progress_desc=f"characters: {page.stem} (panel-page-prev2)",
            source="panel-page-prev2",
            panels_dir=panels_dir, expected_panels=expected_panels,
            refs_dir=refs_dir, annotated=annotated, boxes=boxes,
            context_images=context_images,
        )
        return record


class PanelPageCastStrategy(PanelPageStrategy):
    """mode="panel-page-cast": panel-page with the per-chapter cast
    shortlist. The effective cast key is the explicit `cast_key` argument,
    else the detector's fixed `--cast-key`, else derived per page via
    `cast_key_for_page` (chapter_page_map.json -> filename tag -> `NNN-`
    prefix). `set_cast` switches the detector's prompts so cropped-panel
    fallbacks stay in-cast; pages without a cast fall back to the full
    roster."""

    mode = "panel-page-cast"

    def detect(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        detector = self.detector
        key = cast_key
        if key is None:
            key = detector.cast_key
        if key is None and detector.chapter_casts_file is not None:
            key = cast_key_for_page(
                page,
                detector.chapter_casts_file,
                detector.chapter_page_map_file,
            )
        if key is not None:
            detector.set_cast(key)
        else:
            print(
                f"  characters: {page.stem}: no chapter cast derivable for "
                "panel-page-cast (full roster used); pass --cast-key or use a "
                "page name/volume the map can resolve",
                file=sys.stderr,
                flush=True,
            )
        return super().detect(
            page, panels_dir, expected_panels, refs_dir, cast_key=key
        )


class PanelPagePrev2CastStrategy(PanelPagePrev2Strategy):
    """mode="panel-page-prev2-cast": panel-page-prev2 with the per-chapter
    cast shortlist. The effective cast key is the explicit `cast_key`
    argument, else the detector's fixed `--cast-key`, else derived per page
    via `cast_key_for_page` (chapter_page_map.json -> filename tag -> `NNN-`
    prefix). `set_cast` switches the detector's prompts so cropped-panel
    fallbacks stay in-cast; pages without a cast fall back to the full
    roster."""

    mode = "panel-page-prev2-cast"
    label = "panel+page+prev2+cast"

    def detect(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        detector = self.detector
        key = cast_key
        if key is None:
            key = detector.cast_key
        if key is None and detector.chapter_casts_file is not None:
            key = cast_key_for_page(
                page,
                detector.chapter_casts_file,
                detector.chapter_page_map_file,
            )
        if key is not None:
            detector.set_cast(key)
        else:
            print(
                f"  characters: {page.stem}: no chapter cast derivable for "
                "panel-page-prev2-cast (full roster used); pass --cast-key or "
                "use a page name/volume the map can resolve",
                file=sys.stderr,
                flush=True,
            )
        return super().detect(
            page, panels_dir, expected_panels, refs_dir, cast_key=key
        )


DETECTION_STRATEGIES: dict[str, type[DetectionStrategy]] = {
    "panel": PanelStrategy,
    "page": PageStrategy,
    "panel-page": PanelPageStrategy,
    "panel-page-prev2": PanelPagePrev2Strategy,
    "panel-page-cast": PanelPageCastStrategy,
    "panel-page-prev2-cast": PanelPagePrev2CastStrategy,
}
