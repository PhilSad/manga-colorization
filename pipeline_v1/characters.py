"""Character detection via OpenRouter vision-language models.

V1 behaviour (`detection_mode="panel"`): one call per panel, the panel as the
only image, prompt listing the canonical reference characters (hints come from
the shared character profiles, task 0002) and asking for exactly
`{"characters": ["Name1", ...]}`.

V1.1 behaviour (`detection_mode="page"`, default, task 0003): one paid call
per page. The full page is sent with the panels numbered in reading order
(same numbers the extraction uses); the model returns a strict per-panel
mapping. Missing, invalid, or explicitly `uncertain` panel entries trigger a
cropped-panel fallback call (the V1 per-panel prompt). An optional cached
chapter cast shortlist (`--cast-key`) replaces the full roster in the prompt.

V1.2 behaviour (`detection_mode="panel-page"`): one paid call per panel that
sends the full page (numbered, target panel highlighted) as global context
*plus* the cropped panel. Each panel keeps the same cropped-panel fallback as
page mode: unparseable, `uncertain`, or unknown-character answers fall back to
the V1 per-panel prompt.

Parsing, validation, retry policy, and per-call cost accounting (`usage.cost`
from OpenRouter) are shared with the standalone research method
`character-detection-openrouter-vlm`.
"""

from __future__ import annotations

import base64
import json
import re
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


class CharacterDetector(Protocol):
    """Interface for anything that lists which reference characters appear in
    a manga panel."""

    def detect(self, panel: Path, refs_dir: Path) -> CharacterRecord:
        ...


class PageCharacterDetector(Protocol):
    """Interface for page-level detection (task 0003): one call per page
    mapping numbered panels to canonical characters."""

    def detect_page(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
    ) -> PageCharacterRecord:
        ...


class OpenRouterCharacterDetector:
    """OpenRouter VLM client supporting both per-panel and page-level calls."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = API_BASE,
        prompt_template: str | None = None,
        panel_prompt_template: str | None = None,
        panel_page_prompt_template: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        client: Any = None,  # injected OpenAI-compatible client (tests)
        profiles_file: Path | None = None,
        chapter_casts_file: Path | None = None,
        cast_key: str | None = None,
        workers: int = 1,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.prompt_template = prompt_template
        self.panel_prompt_template = panel_prompt_template
        self.panel_page_prompt_template = panel_page_prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.profiles_file = Path(profiles_file) if profiles_file else None
        self.chapter_casts_file = Path(chapter_casts_file) if chapter_casts_file else None
        self.cast_key = cast_key
        self.workers = workers
        self.canonical: list[str] = []
        self.profiles: dict = {}
        self.prompt: str = ""
        self.panel_prompt: str = ""
        self.panel_page_prompt: str = ""
        # Guards prompt swaps (set_cast) against readers in worker threads.
        self._prompt_lock = threading.Lock()
        # Raw templates (kept so prompts can be rebuilt per chapter cast).
        self._page_template: str = ""
        self._panel_template: str = ""
        self._panel_page_template: str = ""

    def prepare(
        self,
        refs_dir: Path,
        prompt_file: Path | None = None,
        panel_prompt_file: Path | None = None,
        panel_page_prompt_file: Path | None = None,
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
        self._build_prompts()

    def _build_prompts(self) -> None:
        """(Re)build the three prompts from the stored templates + current
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

    def set_cast(self, cast_key: str | None) -> None:
        """Switch the chapter-cast shortlist for all three prompts (no-op when
        unchanged). Called per page in `panel-page-cast` mode so that
        cropped-panel fallbacks reuse the page's cast; lock-guarded for
        worker threads."""
        with self._prompt_lock:
            if cast_key == self.cast_key and self.prompt:
                return
            self.cast_key = cast_key
            self._build_prompts()

    # -- per-panel ---------------------------------------------------------

    def detect(self, panel: Path, refs_dir: Path) -> CharacterRecord:
        if not self.panel_prompt or not self.canonical:
            self.prepare(refs_dir)
        with self._prompt_lock:
            panel_prompt = self.panel_prompt
        info = file_record(panel)
        info["data_base64"] = base64.b64encode(panel.read_bytes()).decode()
        content = [
            {"type": "text", "text": panel_prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{info['mime_type']};base64,{info['data_base64']}"}},
        ]
        result = self._call(content)
        parsed = parse_characters(result.text)
        known, unknown = validate_characters(parsed, self.canonical)
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

    # -- page-level (task 0003) --------------------------------------------

    def detect_page(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
    ) -> PageCharacterRecord:
        """One page-level call; per-panel fallbacks for missing/invalid/
        uncertain entries."""
        if not self.prompt or not self.canonical:
            self.prepare(refs_dir)
        record = PageCharacterRecord(status="ok", page=page.stem)

        annotated = _annotated_page(page, panels_dir)
        info = file_record(annotated)
        info["data_base64"] = base64.b64encode(annotated.read_bytes()).decode()
        content = [
            {"type": "text", "text": self.prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{info['mime_type']};base64,{info['data_base64']}"}},
        ]
        result = self._call(content)
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
            retry = self._call(content)
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
                fallback = self._fallback_panel(panel_key, panels_dir, refs_dir)
                record.fallback_calls += 1
                record.cost_usd += fallback.cost_usd or 0.0
                record.total_latency_s += fallback.latency_s
                if fallback.cost_usd is None:
                    record.unpriced_calls += 1
                record.panels[panel_key] = fallback
                continue
            known, unknown = validate_characters(
                entry["characters"], self.canonical
            )
            if entry["uncertain"] or unknown:
                record.status = "partial" if record.status == "ok" else record.status
                fallback = self._fallback_panel(panel_key, panels_dir, refs_dir)
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

    def _fallback_panel(
        self, panel_key: str, panels_dir: Path, refs_dir: Path
    ) -> CharacterRecord:
        panel = _find_panel_file(panels_dir, panel_key)
        if panel is None:
            return CharacterRecord(
                status="error", characters=[], unknown_entries=[],
                response_text="", usage={}, cost_usd=None, cost_source="unavailable",
                latency_s=0.0, model_returned=None, attempts=0,
                error=f"fallback crop missing: {panels_dir / panel_key}", 
                finished_at=_iso_now(), source="fallback",
            )
        record = self.detect(panel, refs_dir)
        record.source = "fallback"
        return record

    # -- panel+page (V1.2) -----------------------------------------------

    def detect_panels_with_page(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
    ) -> PageCharacterRecord:
        """One paid call per panel: the full page (numbered, target panel
        highlighted) as global context *plus* the cropped panel.

        Per-panel fallback (same as page mode): unparseable answers, explicit
        `uncertain`, unknown-character entries, and call errors fall back to
        the V1 panel-only prompt call.

        `cast_key` renders the panel-page prompt for that chapter shortlist
        without touching shared state (thread-safe for `panel-page-cast`
        mode); `None` uses the detector's current cast.
        """
        if not self.canonical:
            self.prepare(refs_dir)
        if not self._panel_page_template:
            raise ValueError(
                "panel-page detection needs a panel-page prompt "
                "(--vlm-panel-page-prompt-file)"
            )
        record = PageCharacterRecord(status="ok", page=page.stem)
        # cast_key=None means "the detector's current cast" (panel-page with
        # --cast-key); an explicit key (panel-page-cast) renders that cast.
        effective_cast = self.cast_key if cast_key is None else cast_key
        panel_page_prompt = self.panel_page_prompt_for(effective_cast)

        page_image, boxes = _page_geometry(panels_dir)
        annotated = _annotated_page(page_image, panels_dir)
        page_info = file_record(annotated)
        page_b64 = base64.b64encode(annotated.read_bytes()).decode()
        page_mime = page_info["mime_type"]

        for panel_key in tqdm(
            expected_panels,
            desc=f"characters: {page.stem} (panel-page)",
            unit="panel", leave=False, disable=self.workers > 1,
        ):
            panel = _find_panel_file(panels_dir, panel_key)
            if panel is None:
                fallback = self._fallback_panel(panel_key, panels_dir, refs_dir)
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
            content = [
                {"type": "text", "text": panel_page_prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{page_mime};base64,{page_b64}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:{info['mime_type']};base64,{_b64(panel.read_bytes())}"}},
            ]
            result = self._call(content)
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
                    parsed["characters"], self.canonical
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
                    source="panel-page",
                    uncertain=False,
                )
                continue

            # Fallback: panel-only prompt (V1), mirrors page-mode behaviour.
            record.status = "partial" if record.status == "ok" else record.status
            fallback = self._fallback_panel(panel_key, panels_dir, refs_dir)
            record.fallback_calls += 1
            record.cost_usd += fallback.cost_usd or 0.0
            record.total_latency_s += fallback.latency_s
            if fallback.cost_usd is None:
                record.unpriced_calls += 1
            record.panels[panel_key] = fallback
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
    temperature: float = 0.2,
) -> "_CallResult":
    """One OpenAI-compatible chat completion with retry/backoff and
    `usage.cost` accounting (OpenRouter). Shared by character detection
    (characters.py) and color verification (verify_color.py)."""
    from openai import APIError, APIConnectionError, BadRequestError, RateLimitError

    messages = [{"role": "user", "content": content}]
    response_format: str | None = "json_object"
    attempts = 0
    while True:
        attempts += 1
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = {"type": response_format}
        started = time.monotonic()
        try:
            response = client.chat.completions.create(**kwargs)
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
    """
    from extraction import draw_overlay
    from PIL import Image

    _page, boxes = _page_geometry(panels_dir)
    annotated = panels_dir / "detection_annotated.png"
    with Image.open(page) as image:
        draw_overlay(image.convert("RGB"), boxes, annotated)
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
