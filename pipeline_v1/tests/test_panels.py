"""Tests for the panels step (task 0004): blank-page detection, synthetic
full-page fallback, and `--only-panel` page filtering. Fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from detection import PanelBox
from steps.panels import ink_ratio, run_panels_step


class FakeDetector:
    def __init__(self, boxes: list[PanelBox] | None = None):
        self.boxes = boxes or []

    def detect(self, page: Path) -> list[PanelBox]:
        return list(self.boxes)


def make_page(tmp_path: Path, name: str, size=(500, 700), ink=False) -> Path:
    page = Image.new("RGB", size, "white")
    if ink:
        draw = ImageDraw.Draw(page)
        draw.rectangle((100, 100, 400, 600), outline="black", width=4)
        draw.line((150, 150, 350, 350), fill="black", width=4)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)
    return path


def make_config(tmp_path, **overrides):
    from config import PipelineConfig

    base = dict(
        input_dir=tmp_path / "pages",
        refs_dir=tmp_path / "refs",
        output_root=tmp_path / "output",
        mock=True,
        sleep_s=0.0,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ink_ratio

def test_ink_ratio_blank_is_zero():
    blank = Image.new("RGB", (100, 100), "white")
    assert ink_ratio(blank) == 0.0


def test_ink_ratio_counts_dark_ink():
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 9, 9), fill="black")  # 100 dark px / 10000
    assert ink_ratio(image) == pytest.approx(0.01, abs=1e-6)


def test_ink_ratio_sparse_line_art_is_not_blank():
    """A full-page illustration with sparse line art (~2% ink) must not be
    classified as blank with the default threshold (0.005)."""
    image = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    for i in range(0, 1000, 40):
        draw.line((i, 0, i, 1000), fill="black", width=1)  # 25 lines = 2.5%
    ratio = ink_ratio(image)
    assert ratio > 0.005
    assert ratio < 0.1


# ---------------------------------------------------------------------------
# full-page fallback + blank skip

def test_zero_panels_sparse_page_gets_full_page_fallback(tmp_path):
    page = make_page(tmp_path / "pages", "p006.png", ink=True)
    config = make_config(tmp_path, full_page_fallback=True)
    ctx = make_ctx(tmp_path)
    result = run_panels_step(ctx, config, FakeDetector([]))

    geometry = read_json(ctx.run_dir / "1_panels" / "p006" / "panels.json")
    assert geometry["blank_page"] is False
    assert geometry["full_page_fallback"] is True
    assert len(geometry["detections"]) == 1
    detection = geometry["detections"][0]
    assert detection["box"] == [0, 0, 500, 700]
    assert detection["crop"] == "panel_0001.png"
    assert detection["provenance"] == "full-page-fallback"
    # The full-page crop exists and equals the canvas.
    crop = ctx.run_dir / "1_panels" / "p006" / "panel_0001.png"
    with Image.open(crop) as image:
        assert image.size == (500, 700)
    assert result["pages"][0]["full_page_fallback"] is True


def test_zero_panels_blank_page_skipped(tmp_path):
    page = make_page(tmp_path / "pages", "blank.png")  # all white
    config = make_config(tmp_path, full_page_fallback=True)
    ctx = make_ctx(tmp_path)
    run_panels_step(ctx, config, FakeDetector([]))

    geometry = read_json(ctx.run_dir / "1_panels" / "blank" / "panels.json")
    assert geometry["blank_page"] is True
    assert geometry["skip_reason"] == "blank-page"
    assert geometry["detections"] == []
    # No crops, no overlay.
    page_dir = ctx.run_dir / "1_panels" / "blank"
    assert not (page_dir / "panel_0001.png").exists()
    assert not (page_dir / "overlay.png").exists()


def test_blank_page_respects_threshold(tmp_path):
    """A barely-inked page with ink above the threshold is not blank."""
    page = make_page(tmp_path / "pages", "p.png")
    with Image.open(page) as image:
        image = image.copy()
    draw = ImageDraw.Draw(image)
    # 46x46 = 2116 px on 500x700 (350000 px) -> 0.6% ink > 0.005 threshold
    draw.rectangle((10, 10, 55, 55), fill="black")
    page.write_bytes(_png(image))
    config = make_config(tmp_path, full_page_fallback=True)
    ctx = make_ctx(tmp_path)
    run_panels_step(ctx, config, FakeDetector([]))
    geometry = read_json(ctx.run_dir / "1_panels" / "p" / "panels.json")
    assert geometry["blank_page"] is False
    assert geometry["full_page_fallback"] is True


def test_normal_detections_never_get_synthetic_box(tmp_path):
    page = make_page(tmp_path / "pages", "p001.png", ink=True)
    boxes = [PanelBox(20, 20, 240, 340, 0.9), PanelBox(260, 20, 480, 340, 0.9)]
    config = make_config(tmp_path, full_page_fallback=True)
    ctx = make_ctx(tmp_path)
    run_panels_step(ctx, config, FakeDetector(boxes))

    geometry = read_json(ctx.run_dir / "1_panels" / "p001" / "panels.json")
    assert len(geometry["detections"]) == 2
    assert geometry["full_page_fallback"] is False
    assert all(d["provenance"] == "yolo" for d in geometry["detections"])


def test_full_page_fallback_disabled(tmp_path):
    page = make_page(tmp_path / "pages", "p006.png", ink=True)
    config = make_config(tmp_path, full_page_fallback=False)
    ctx = make_ctx(tmp_path)
    run_panels_step(ctx, config, FakeDetector([]))
    geometry = read_json(ctx.run_dir / "1_panels" / "p006" / "panels.json")
    assert geometry["detections"] == []
    assert geometry["blank_page"] is False  # not blank, just unhandled


# ---------------------------------------------------------------------------
# --only-panel page filtering

def test_only_panels_filters_pages(tmp_path):
    make_page(tmp_path / "pages", "p001.png", ink=True)
    make_page(tmp_path / "pages", "p002.png", ink=True)
    config = make_config(tmp_path, only_panels=("p002:panel_0001",))
    ctx = make_ctx(tmp_path)
    result = run_panels_step(ctx, config, FakeDetector([PanelBox(10, 10, 100, 100, 0.9)]))

    pages = result["pages"]
    assert [p["page"] for p in pages] == ["p002.png"]
    assert (ctx.run_dir / "1_panels" / "p002").is_dir()
    assert not (ctx.run_dir / "1_panels" / "p001").exists()


def _png(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_ctx(tmp_path):
    from run_context import RunContext

    return RunContext.create(tmp_path / "output", {"status": "running"})
