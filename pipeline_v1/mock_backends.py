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

# Default 2x2 mock layout (used by `run.py --mock` on real pages).
DEFAULT_MOCK_BOXES = [
    PanelBox(600, 50, 1150, 550, 0.99),
    PanelBox(50, 50, 580, 550, 0.99),
    PanelBox(600, 600, 1150, 1750, 0.99),
    PanelBox(50, 600, 580, 1750, 0.99),
]


class MockPanelDetector:
    """Returns the same canned boxes for every page."""

    def __init__(self, boxes: list[PanelBox] | None = None) -> None:
        self.boxes = list(boxes) if boxes is not None else list(DEFAULT_MOCK_BOXES)

    def detect(self, page: Path) -> list[PanelBox]:
        return list(self.boxes)


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
        self.calls: list[tuple[Path, Path | None, Path]] = []

    def colorize(self, panel: Path, atlas: Path | None, output: Path) -> ColorizeRecord:
        self.calls.append((panel, atlas, output))
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
        )
