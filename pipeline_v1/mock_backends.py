"""Mock backends for offline runs (`--mock`) and the test suite.

These implement the same protocols as the real backends (PanelDetector,
CharacterDetector, Colorizer) with deterministic, dependency-free behaviour:
no YOLO weights, no OpenRouter calls, no FLUX server.
"""

from __future__ import annotations

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
    """Tints the panel with a fixed color; records every call."""

    def __init__(self, color: tuple[int, int, int] = (205, 92, 92)) -> None:
        self.color = color
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
        )


class MockPageCharacterDetector:
    """Canned page-level detections keyed by page stem: `{page_stem: {panel_stem:
    (characters, uncertain)}}`. Panels not covered are reported as a fallback
    with no characters (deterministic, offline). Implements both the
    page-level (`detect_page`) and the panel+page (`detect_panels_with_page`)
    protocols."""

    def __init__(self, by_page: dict[str, dict[str, tuple[list[str], bool]]] | None = None):
        self.by_page = by_page or {}
        self.calls: list[tuple[Path, list[str]]] = []

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
    ) -> "PageCharacterRecord":
        from characters import CharacterRecord, PageCharacterRecord

        self.calls.append((page, list(expected_panels)))
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
                source="panel-page", uncertain=False,
            )
        return record
