"""Panel detection for manga pages.

`YoloPanelDetector` wraps the HuggingFace model
`leoxs22/manga-panel-detector-yolo26n` (YOLO26-nano, Apache-2.0, classes
`0: panel`, `1: text`) via the `ultralytics` package. `ultralytics` is
imported lazily so the rest of the pipeline and the offline test suite never
require torch. The weights file is downloaded once into
`pipeline_v1/models/` (gitignored) and reused afterwards.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from config import PIPELINE_DIR, SUPPORTED_IMAGE_SUFFIXES

MODEL_REPO = "leoxs22/manga-panel-detector-yolo26n"
MODEL_FILE = "manga_panel_detector_fp32.pt"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
DEFAULT_MODEL_PATH = PIPELINE_DIR / "models" / MODEL_FILE

# Model card recommendation.
DEFAULT_CONFIDENCE = 0.25
PANEL_CLASS = 0  # 0=panel, 1=text — text bubbles are ignored.


@dataclass
class PanelBox:
    """One detected panel, in page pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
        }


class PanelDetector(Protocol):
    """Interface for anything that finds panels in a page image."""

    def detect(self, page: Path) -> list[PanelBox]:
        """Return detected panel boxes in page pixel coordinates (not yet
        sorted in reading order — see panel_ordering.reading_order)."""
        ...


def _ensure_model_file(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading panel detector weights -> {model_path}", flush=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "manga-colorization/1"})
    with urllib.request.urlopen(request, timeout=300) as response:
        temporary = model_path.with_suffix(model_path.suffix + ".part")
        temporary.write_bytes(response.read())
    temporary.replace(model_path)
    return model_path


class YoloPanelDetector:
    """YOLO26n panel detector (leoxs22/manga-panel-detector-yolo26n).

    `ultralytics` and `torch` are imported on first `detect()` call only.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence = confidence
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: torch is a heavy dependency

            weights = _ensure_model_file(self.model_path)
            self._model = YOLO(str(weights))
        return self._model

    def detect(self, page: Path) -> list[PanelBox]:
        model = self._load_model()
        results = model.predict(
            str(page),
            conf=self.confidence,
            imgsz=640,
            device=self.device,
            verbose=False,
        )
        boxes: list[PanelBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls)
                if cls != PANEL_CLASS:
                    continue
                confidence = float(box.conf)
                if confidence < self.confidence:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                boxes.append(
                    PanelBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                    )
                )
        return boxes


def list_page_images(input_dir: Path) -> list[Path]:
    """Page images of an input directory in natural (filename) order."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
