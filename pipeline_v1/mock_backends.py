"""Mock backends for offline runs (`--mock`) and the test suite.

These implement the same interfaces as the real backends (PanelDetector,
Colorizer, and the character-detection strategy interface via
`strategy_for(mode)`) with deterministic, dependency-free behaviour:
no YOLO weights, no OpenRouter calls, no FLUX server.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from characters import CharacterRecord
from colorizer import ColorizeRecord
from detection import PanelBox

# Default 2x2 mock layout for a 1200x1800 page; scaled to the actual page
# size by detect().
DEFAULT_MOCK_BOXES = [
    PanelBox(600, 50, 1150, 550, 0.99),
    PanelBox(50, 50, 580, 550, 0.99),
    PanelBox(600, 600, 1150, 1750, 0.99),
    PanelBox(50, 600, 580, 1750, 0.99),
]

_LAYOUT_PAGE_SIZE = (1200, 1800)


class MockPanelDetector:
    """Returns the same canned boxes for every page, scaled to the page
    size (so the default 2x2 layout works on pages of any dimensions)."""

    def __init__(self, boxes: list[PanelBox] | None = None) -> None:
        self.scale_to_page = boxes is None
        self.boxes = list(boxes) if boxes is not None else list(DEFAULT_MOCK_BOXES)

    def detect(self, page: Path) -> list[PanelBox]:
        if not self.scale_to_page:
            return list(self.boxes)
        with Image.open(page) as image:
            width, height = image.size
        scale_x = width / _LAYOUT_PAGE_SIZE[0]
        scale_y = height / _LAYOUT_PAGE_SIZE[1]
        scaled = []
        for box in self.boxes:
            scaled.append(
                PanelBox(
                    x1=box.x1 * scale_x,
                    y1=box.y1 * scale_y,
                    x2=box.x2 * scale_x,
                    y2=box.y2 * scale_y,
                    confidence=box.confidence,
                )
            )
        return scaled


class MockCharacterDetector:
    """Returns canned characters keyed by panel stem."""

    def __init__(self, by_panel: dict[str, list[str]] | None = None) -> None:
        self.by_panel = by_panel or {}

    def strategy_for(self, mode: str) -> "_MockPanelStrategy":
        """Per-panel mock: every mode falls back to per-panel mock calls
        (mirrors the pre-strategy step behaviour, where a detector without
        page capabilities ran the per-panel loop regardless of mode)."""
        return _MockPanelStrategy(self)

    def detect(self, panel: Path, refs_dir: Path) -> CharacterRecord:
        names = self.by_panel.get(panel.stem, [])
        return CharacterRecord(
            status="ok",
            characters=names,
            unknown_entries=[],
            response_text="{}",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost_usd=0.0001,
            cost_source="mock",
            latency_s=0.01,
            model_returned="mock",
            attempts=1,
            finished_at="mock",
        )


class MockColorizer:
    """Tints the panel with a fixed color; records every call.

    `backend` is a provenance marker ("flux" | "gpt-image-2") so tests and
    manifests can assert which backend was selected (run.py passes
    "gpt-image-2" for `--full-page`). Gpt mode uses a distinct tint so
    full-page mock outputs are visually distinguishable from panel-mode mocks.
    """

    _BACKEND_TINTS = {
        "flux": (205, 92, 92),        # indian red (default)
        "gpt-image-2": (92, 155, 205),  # steel blue
    }

    def __init__(
        self,
        color: tuple[int, int, int] | None = None,
        backend: str = "flux",
    ) -> None:
        if backend not in self._BACKEND_TINTS:
            raise ValueError(
                f"unknown mock backend {backend!r} (expected one of "
                f"{sorted(self._BACKEND_TINTS)})"
            )
        self.backend = backend
        self.color = color or self._BACKEND_TINTS[backend]
        self.calls: list[tuple[Path, Path | None, Path, str]] = []

    def colorize(
        self,
        panel: Path,
        atlas: Path | None,
        output: Path,
        palette_instruction: str = "",
    ) -> ColorizeRecord:
        self.calls.append((panel, atlas, output, palette_instruction))
        with Image.open(panel) as image:
            rgb = image.convert("RGB")
        tint = Image.new("RGB", rgb.size, self.color)
        blended = Image.blend(rgb, tint, 0.6)
        output.parent.mkdir(parents=True, exist_ok=True)
        blended.save(output)
        return ColorizeRecord(
            status="ok",
            output=output,
            requested_size=(rgb.width, rgb.height),
            latency_s=0.01,
            error=None,
            seed=None,
            original_size=(rgb.width, rgb.height),
            scale=1.0,
            cap_applied=False,
            max_megapixels=None,
            model=f"{self.backend} (mock)",
        )


class MockVerifier:
    """Deterministic offline stand-in for verify_color.ColorVerifier.

    Verdicts are canned per panel stem (the monochrome crop's stem — stable
    across retries even though the colorized file is renamed per attempt):
    `by_panel: {stem: ("good"|"bad"|"bad-once", fix_prompt, regions?)}`.
    Default: good (verified). "bad" yields a MISMATCH verdict carrying the
    given fix prompt (or a default one); "bad-once" is bad on the first
    verify call for that stem and good afterwards (the fix worked) — both
    exercise the verify loop's retry path without any network. An optional
    third tuple element provides canned `regions` (bbox verdicts) for
    --verify-mode bbox tests.
    """

    def __init__(
        self, by_panel: dict[str, tuple[str, str, list] | tuple[str, str]] | None = None
    ) -> None:
        self.by_panel = by_panel or {}
        self.calls: list[tuple[Path, Path | None, Path | None]] = []
        self._stem_counts: dict[str, int] = {}

    def verify(
        self,
        colorized: Path,
        input_crop: Path | None,
        atlas: Path | None = None,
    ) -> "ColorVerifyRecord":
        from verify_color import ColorVerifyRecord

        colorized = Path(colorized)
        self.calls.append((colorized, input_crop, atlas))
        stem = Path(input_crop).stem if input_crop is not None else colorized.stem
        verdict, fix, *rest = self.by_panel.get(stem, ("good", ""))
        regions = rest[0] if rest else []
        if verdict == "bad-once":
            n = self._stem_counts.get(stem, 0)
            self._stem_counts[stem] = n + 1
            if n == 0:
                verdict, fix = "bad", fix
            else:
                verdict, fix, regions = "good", "", []
        good = verdict == "good"
        return ColorVerifyRecord(
            status="verified" if good else "mismatch",
            good_color=good,
            analyse="mock analysis",
            fix_prompt="" if good else (fix or "mock: re-colorize with canonical palette"),
            regions=list(regions),
            response_text="",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost_usd=0.0001,
            cost_source="mock",
            latency_s=0.01,
            model_returned="mock",
            attempts=1,
            finished_at="mock",
            error=None,
        )


class MockRegionEditor:
    """Deterministic offline stand-in for region_edit.GptImage2RegionEditor
    (--verify-mode bbox): "edits" the boxed image with a distinct tint,
    records every call, and charges a canned est_cost_usd so totals and
    per-attempt cost accounting are exercised offline.
    """

    def __init__(
        self, color: tuple[int, int, int] | None = None, cost_usd: float = 0.02
    ) -> None:
        self.color = color or (205, 92, 155)  # distinct from the colorizer tints
        self.cost_usd = cost_usd
        self.calls: list[tuple[Path, Path | None, Path, str, str]] = []

    def target_size(self, image: Path) -> tuple[int, int]:
        with Image.open(image) as im:
            return (im.width, im.height)

    def render_prompt(
        self, width: int, height: int, instruction: str, palette_instruction: str = ""
    ) -> str:
        return (
            f"region edit {width}x{height}\n{instruction}\n{palette_instruction}"
        )

    def edit(
        self,
        boxed_image: Path,
        atlas: Path | None,
        output: Path,
        region_instruction_text: str,
        palette_instruction: str = "",
    ) -> ColorizeRecord:
        self.calls.append(
            (Path(boxed_image), atlas, Path(output),
             region_instruction_text, palette_instruction)
        )
        with Image.open(boxed_image) as image:
            rgb = image.convert("RGB")
        tint = Image.new("RGB", rgb.size, self.color)
        blended = Image.blend(rgb, tint, 0.6)
        output.parent.mkdir(parents=True, exist_ok=True)
        blended.save(output)
        return ColorizeRecord(
            status="ok",
            output=output,
            requested_size=(rgb.width, rgb.height),
            latency_s=0.01,
            error=None,
            seed=None,
            original_size=(rgb.width, rgb.height),
            scale=1.0,
            cap_applied=False,
            max_megapixels=None,
            model="gpt-image-2 (mock region editor)",
            quality="medium",
            usage={},
            est_cost_usd=self.cost_usd,
        )


class MockPageCharacterDetector:
    """Canned page-level detections keyed by page stem: `{page_stem: {panel_stem:
    (characters, uncertain)}}`. Panels not covered are reported as a fallback
    with no characters (deterministic, offline). Implements both the
    page-level (`detect_page`) and the panel+page (`detect_panels_with_page`)
    protocols."""

    def __init__(
        self,
        by_page: dict[str, dict[str, tuple[list[str], bool]]] | None = None,
        cast_key: str | None = None,
    ):
        self.by_page = by_page or {}
        self.calls: list[tuple[Path, list[str]]] = []
        self.cast_keys: list[str | None] = []
        self.current_cast: str | None = None
        self.cast_key: str | None = cast_key  # fixed --cast-key override

    def strategy_for(self, mode: str):
        """Page-context mock supports "page", "panel-page",
        "panel-page-prev2", "panel-page-cast", "panel-page-prev2-cast"
        (panel mode uses MockCharacterDetector)."""
        strategies = {
            "page": _MockPageStrategy,
            "panel-page": _MockPanelPageStrategy,
            "panel-page-prev2": _MockPanelPagePrev2Strategy,
            "panel-page-cast": _MockPanelPageCastStrategy,
            "panel-page-prev2-cast": _MockPanelPagePrev2CastStrategy,
        }
        strategy_cls = strategies.get(mode)
        if strategy_cls is None:
            raise ValueError(
                f"MockPageCharacterDetector supports modes "
                f"{sorted(strategies)}, got {mode!r}"
            )
        return strategy_cls(self)

    def set_cast(self, cast_key: str | None) -> None:
        """Mirror of OpenRouterCharacterDetector.set_cast (panel-page-cast)."""
        self.current_cast = cast_key
        self.cast_keys.append(cast_key)

    def detect_page(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
    ) -> "PageCharacterRecord":
        from characters import CharacterRecord, PageCharacterRecord

        self.calls.append((page, list(expected_panels)))
        page_map = self.by_page.get(page.stem, {})
        record = PageCharacterRecord(status="ok", page=page.stem, page_calls=1)
        record.cost_usd = 0.0002
        for panel_key in expected_panels:
            entry = page_map.get(panel_key)
            if entry is None:
                record.status = "partial"
                record.fallback_calls += 1
                record.cost_usd += 0.0001
                record.panels[panel_key] = CharacterRecord(
                    status="ok", characters=[], unknown_entries=[],
                    response_text="{}", usage={"total_tokens": 10},
                    cost_usd=0.0001, cost_source="mock", latency_s=0.01,
                    model_returned="mock", attempts=1, finished_at="mock",
                    source="fallback",
                )
                continue
            characters, uncertain = entry
            if uncertain:
                record.status = "partial"
                record.fallback_calls += 1
                record.cost_usd += 0.0001
                record.panels[panel_key] = CharacterRecord(
                    status="ok", characters=list(characters), unknown_entries=[],
                    response_text="{}", usage={"total_tokens": 10},
                    cost_usd=0.0001, cost_source="mock", latency_s=0.01,
                    model_returned="mock", attempts=1, finished_at="mock",
                    source="fallback", uncertain=False,
                )
                continue
            record.panels[panel_key] = CharacterRecord(
                status="ok", characters=list(characters), unknown_entries=[],
                response_text="{}", usage={"total_tokens": 10},
                cost_usd=None, cost_source="page-level", latency_s=0.0,
                model_returned="mock", attempts=1, finished_at="mock",
                source="page", uncertain=uncertain,
            )
        return record

    def detect_panels_with_page(
        self,
        page: Path,
        panels_dir: Path,
        expected_panels: list[str],
        refs_dir: Path,
        *,
        cast_key: str | None = None,
        source: str = "panel-page",
    ) -> "PageCharacterRecord":
        from characters import CharacterRecord, PageCharacterRecord

        self.calls.append((page, list(expected_panels)))
        self.cast_keys.append(cast_key)
        page_map = self.by_page.get(page.stem, {})
        record = PageCharacterRecord(status="ok", page=page.stem)
        record.page_calls = len(expected_panels)  # one panel+page call per panel
        record.cost_usd = 0.0002 * len(expected_panels)
        for panel_key in expected_panels:
            entry = page_map.get(panel_key)
            characters, uncertain = entry if entry is not None else ([], True)
            if entry is None or uncertain:
                record.status = "partial"
                record.fallback_calls += 1
                record.cost_usd += 0.0001
                record.panels[panel_key] = CharacterRecord(
                    status="ok", characters=list(characters), unknown_entries=[],
                    response_text="{}", usage={"total_tokens": 10},
                    cost_usd=0.0001, cost_source="mock", latency_s=0.01,
                    model_returned="mock", attempts=1, finished_at="mock",
                    source="fallback", uncertain=False,
                )
                continue
            record.panels[panel_key] = CharacterRecord(
                status="ok", characters=list(characters), unknown_entries=[],
                response_text="{}", usage={"total_tokens": 10},
                cost_usd=0.0002, cost_source="mock", latency_s=0.01,
                model_returned="mock", attempts=1, finished_at="mock",
                source=source, uncertain=False,
            )
        return record


# ---------------------------------------------------------------------------
# Detection strategies (mock adapters)

# The step selects a strategy per --detection-mode; the mocks mirror the real
# strategies' uniform interface (mode/label/provenance + detect()).
PIPELINE_DIR = Path(__file__).resolve().parent
CHAPTER_CASTS_FILE = PIPELINE_DIR / "chapter_casts.json"
CHAPTER_PAGE_MAP_FILE = (
    PIPELINE_DIR.parent / "frieren_wiki_dataset" / "chapter_page_map.json"
)


class _MockPanelStrategy:
    """mode="panel": one mock call per panel, aggregated per page."""

    mode = "panel"
    label = "panel"
    provenance = None

    def __init__(self, detector: MockCharacterDetector) -> None:
        self.detector = detector

    def detect(self, page, panels_dir, expected_panels, refs_dir, *, cast_key=None):
        from characters import CharacterRecord, PageCharacterRecord

        record = PageCharacterRecord(
            status="ok",
            page=Path(page).stem if page is not None else panels_dir.name,
        )
        for panel_key in expected_panels:
            panel = panels_dir / f"{panel_key}.png"
            if not panel.is_file():
                record.status = "partial" if record.status == "ok" else record.status
                record.panels[panel_key] = CharacterRecord(
                    status="error", characters=[], unknown_entries=[],
                    response_text="", usage={}, cost_usd=None,
                    cost_source="unavailable", latency_s=0.0,
                    model_returned=None, attempts=0,
                    error=f"crop missing: {panel}", finished_at="mock",
                    source="panel",
                )
                continue
            rec = self.detector.detect(panel, refs_dir)
            record.panels[panel_key] = rec
            record.page_calls += 1
            record.cost_usd += rec.cost_usd or 0.0
            record.total_latency_s += rec.latency_s
            if rec.cost_usd is None:
                record.unpriced_calls += 1
        return record


class _MockPageStrategy:
    """mode="page": delegates to MockPageCharacterDetector.detect_page."""

    mode = "page"
    label = "page-level"
    provenance = "page_call.json"

    def __init__(self, detector: MockPageCharacterDetector) -> None:
        self.detector = detector

    def detect(self, page, panels_dir, expected_panels, refs_dir, *, cast_key=None):
        return self.detector.detect_page(
            page, panels_dir, expected_panels, refs_dir
        )


class _MockPanelPageStrategy:
    """mode="panel-page": delegates to detect_panels_with_page."""

    mode = "panel-page"
    label = "panel+page"
    provenance = "panel_page_calls.json"

    def __init__(self, detector: MockPageCharacterDetector) -> None:
        self.detector = detector

    def detect(self, page, panels_dir, expected_panels, refs_dir, *, cast_key=None):
        return self.detector.detect_panels_with_page(
            page, panels_dir, expected_panels, refs_dir, cast_key=cast_key
        )


class _MockPanelPagePrev2Strategy(_MockPanelPageStrategy):
    """mode="panel-page-prev2": delegates exactly like panel-page (the mock
    has no page images to send, so the preceding pages are not represented),
    recording the prev2 source on panel records."""

    mode = "panel-page-prev2"
    label = "panel+page+prev2"
    provenance = "panel_page_prev2_calls.json"

    def detect(self, page, panels_dir, expected_panels, refs_dir, *, cast_key=None):
        return self.detector.detect_panels_with_page(
            page, panels_dir, expected_panels, refs_dir,
            cast_key=cast_key, source="panel-page-prev2",
        )


class _MockPanelPageCastStrategy(_MockPanelPageStrategy):
    """mode="panel-page-cast": derive the chapter cast like the real
    strategy (explicit key -> detector's fixed cast -> cast_key_for_page),
    switch the mock's prompts, and delegate."""

    mode = "panel-page-cast"

    def detect(self, page, panels_dir, expected_panels, refs_dir, *, cast_key=None):
        from characters import cast_key_for_page

        key = cast_key or self.detector.cast_key
        if key is None:
            key = cast_key_for_page(
                page, CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE
            )
        if key is not None:
            self.detector.set_cast(key)
        else:
            print(
                f"  characters: {page.name}: no chapter cast derivable for "
                "panel-page-cast (full roster used)",
                file=sys.stderr,
                flush=True,
            )
        record = self.detector.detect_panels_with_page(
            page, panels_dir, expected_panels, refs_dir, cast_key=key
        )
        record.cast_key = key
        return record


class _MockPanelPagePrev2CastStrategy(_MockPanelPagePrev2Strategy):
    """mode="panel-page-prev2-cast": derive the chapter cast like the real
    strategy (explicit key -> detector's fixed cast -> cast_key_for_page),
    switch the mock's prompts, and delegate with the prev2 source."""

    mode = "panel-page-prev2-cast"
    label = "panel+page+prev2+cast"

    def detect(self, page, panels_dir, expected_panels, refs_dir, *, cast_key=None):
        from characters import cast_key_for_page

        key = cast_key or self.detector.cast_key
        if key is None:
            key = cast_key_for_page(
                page, CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE
            )
        if key is not None:
            self.detector.set_cast(key)
        else:
            print(
                f"  characters: {page.name}: no chapter cast derivable for "
                "panel-page-prev2-cast (full roster used)",
                file=sys.stderr,
                flush=True,
            )
        record = self.detector.detect_panels_with_page(
            page, panels_dir, expected_panels, refs_dir,
            cast_key=key, source="panel-page-prev2",
        )
        record.cast_key = key
        return record
