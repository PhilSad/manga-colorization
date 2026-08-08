"""Tests for detection.py with a monkeypatched ultralytics (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

import detection
from detection import PanelBox, YoloPanelDetector


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self._xyxy = xyxy
        self._cls = cls
        self._conf = conf

    @property
    def cls(self):
        return self._cls

    @property
    def conf(self):
        return self._conf

    @property
    def xyxy(self):
        class _Arr:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return self._values

        return [_Arr(self._xyxy)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """Records the predict() call and returns canned boxes."""

    last_predict_kwargs = None

    def __init__(self, weights_path):
        FakeYOLO.weights_path = weights_path

    def predict(self, source, **kwargs):
        FakeYOLO.last_predict_kwargs = {"source": source, **kwargs}
        return [
            FakeResult(
                [
                    FakeBox([10, 20, 110, 120], cls=0, conf=0.93),   # panel
                    FakeBox([10, 20, 110, 120], cls=0, conf=0.10),   # panel, low conf
                    FakeBox([130, 20, 200, 60], cls=1, conf=0.90),   # text bubble -> drop
                    FakeBox([210, 20, 310, 220], cls=0, conf=0.88),  # panel
                ]
            )
        ]


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    return tmp_path / "models" / "manga_panel_detector_fp32.pt"


@pytest.fixture
def fake_ultralytics(monkeypatch, fake_model_path):
    """Install a fake YOLO class and pre-create the weights file so the
    download path is not exercised."""
    fake_model_path.parent.mkdir(parents=True)
    fake_model_path.write_bytes(b"fake weights")
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    monkeypatch.setattr(detection, "DEFAULT_MODEL_PATH", fake_model_path)
    return FakeYOLO


def test_detect_filters_text_and_low_confidence(fake_ultralytics, fake_model_path, tmp_path):
    page = tmp_path / "page.png"
    page.write_bytes(b"not a real image")
    detector = YoloPanelDetector(model_path=fake_model_path)
    boxes = detector.detect(page)
    assert len(boxes) == 2  # text box and low-conf box dropped
    assert boxes[0] == PanelBox(10, 20, 110, 120, 0.93)
    assert boxes[1] == PanelBox(210, 20, 310, 220, 0.88)


def test_detect_calls_predict_with_sane_kwargs(fake_ultralytics, fake_model_path, tmp_path):
    page = tmp_path / "page.png"
    page.write_bytes(b"x")
    detector = YoloPanelDetector(model_path=fake_model_path, confidence=0.5)
    detector.detect(page)
    kwargs = FakeYOLO.last_predict_kwargs
    assert kwargs["conf"] == 0.5
    assert kwargs["imgsz"] == 640
    assert kwargs["verbose"] is False


def test_model_file_downloaded_on_first_use(monkeypatch, tmp_path):
    """Weights are fetched from the HF repo when missing."""
    model_path = tmp_path / "cache" / "model.pt"
    fetched: list[str] = []

    class EmptyFakeYOLO:
        def __init__(self, weights_path):
            self.weights_path = weights_path

        def predict(self, source, **kwargs):
            return [FakeResult([])]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"downloaded weights"

    def fake_urlopen(request, timeout):
        fetched.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr("ultralytics.YOLO", EmptyFakeYOLO)
    monkeypatch.setattr(detection.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(detection, "MODEL_URL", "https://hf.example/model.pt")

    detector = YoloPanelDetector(model_path=model_path)
    assert detector.detect(tmp_path / "p.png") == []
    assert model_path.read_bytes() == b"downloaded weights"
    assert fetched == ["https://hf.example/model.pt"]
    # Second use does not re-download.
    detector.detect(tmp_path / "p.png")
    assert len(fetched) == 1


def test_ensure_model_file_existing_skips_download(tmp_path):
    model_path = tmp_path / "m.pt"
    model_path.write_bytes(b"weights")
    assert detection._ensure_model_file(model_path) == model_path


def test_list_page_images(tmp_path):
    (tmp_path / "0134-001.png").write_bytes(b"")
    (tmp_path / "0134-002.webp").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    files = detection.list_page_images(tmp_path)
    assert [p.name for p in files] == ["0134-001.png", "0134-002.webp"]
