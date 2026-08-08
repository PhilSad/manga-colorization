"""Tests for extraction.py and steps/panels.py (offline, fake detector)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from detection import PanelBox
from extraction import crop_panel, draw_overlay, panel_filename, save_panels
from run_context import RunContext


def make_synthetic_page(width=400, height=400) -> Image.Image:
    """White page with two distinct colored rectangles (stand-in panels)."""
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((20, 20, 180, 180), fill=(255, 0, 0))    # red square
    draw.rectangle((200, 20, 380, 180), fill=(0, 0, 255))   # blue square
    return page


def test_crop_panel_exact_region():
    page = make_synthetic_page()
    crop = crop_panel(page, PanelBox(20, 20, 180, 180, 0.9))
    assert crop.size == (160, 160)
    assert crop.getpixel((0, 0)) == (255, 0, 0)
    assert crop.getpixel((159, 159)) == (255, 0, 0)


def test_crop_panel_inset():
    page = make_synthetic_page()
    crop = crop_panel(page, PanelBox(20, 20, 180, 180, 0.9), inset=10)
    assert crop.size == (140, 140)
    assert crop.getpixel((0, 0)) == (255, 0, 0)


def test_crop_panel_clipped_to_page():
    page = make_synthetic_page()
    crop = crop_panel(page, PanelBox(-10, -10, 40, 40, 0.9))
    assert crop.size == (40, 40)


def test_crop_panel_collapses_with_big_inset():
    page = make_synthetic_page()
    with pytest.raises(ValueError):
        crop_panel(page, PanelBox(10, 10, 20, 20, 0.9), inset=10)


def test_panel_filename():
    assert panel_filename(1) == "panel_0001.png"
    assert panel_filename(12, ".webp") == "panel_0012.webp"
    assert panel_filename(3, ".jpg", prefix="x") == "x_0003.jpg"


def test_save_panels_names_in_given_order(tmp_path):
    page = make_synthetic_page()
    detections = [
        PanelBox(200, 20, 380, 180, 0.9),  # blue, right
        PanelBox(20, 20, 180, 180, 0.9),   # red, left
    ]
    records = save_panels(page, detections, tmp_path)
    assert [r["filename"] for r in records] == ["panel_0001.png", "panel_0002.png"]
    assert (tmp_path / "panel_0001.png").is_file()
    with Image.open(tmp_path / "panel_0001.png") as crop:
        assert crop.getpixel((0, 0)) == (0, 0, 255)  # right panel is #1


def test_draw_overlay(tmp_path):
    page = make_synthetic_page()
    out = tmp_path / "overlay.png"
    draw_overlay(page, [PanelBox(20, 20, 180, 180, 0.9)], out)
    assert out.is_file()
    with Image.open(out) as image:
        assert image.size == page.size


# ---- step-level test with a fake detector -------------------------------

class FakeDetector:
    """Returns a fixed set of boxes regardless of the page."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, page: Path):
        return list(self.boxes)


def make_step_inputs(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    # Two pages; the second has one more panel.
    make_synthetic_page().save(pages / "p001.png")
    make_synthetic_page().save(pages / "p002.png")
    refs = tmp_path / "refs"
    refs.mkdir()
    from config import PipelineConfig

    config = PipelineConfig(
        input_dir=pages, refs_dir=refs, output_root=tmp_path / "output", mock=True
    )
    return config


def test_panels_step_full_flow(tmp_path):
    from steps.panels import run_panels_step

    config = make_step_inputs(tmp_path)
    detector = FakeDetector(
        [
            PanelBox(200, 20, 380, 180, 0.9),  # right -> #1
            PanelBox(20, 20, 180, 180, 0.9),   # left  -> #2
        ]
    )
    ctx = RunContext.create(tmp_path / "output", {"status": "running"})
    result = run_panels_step(ctx, config, detector)

    assert len(result["pages"]) == 2
    first = result["pages"][0]
    assert first["page"] == "p001.png"
    assert first["reading_order"] == [1, 2]

    page_dir = ctx.run_dir / "1_panels" / "p001"
    assert (page_dir / "panel_0001.png").is_file()
    assert (page_dir / "panel_0002.png").is_file()
    assert (page_dir / "overlay.png").is_file()
    assert (page_dir / "panels.json").is_file()

    import json

    geometry = json.loads((page_dir / "panels.json").read_text())
    assert geometry["detections"][0]["box"] == [200, 20, 380, 180]
    assert geometry["detections"][0]["crop"] == "panel_0001.png"
    assert geometry["detections"][1]["box"] == [20, 20, 180, 180]

    # Blue (right) panel must be numbered 1 in reading order.
    with Image.open(page_dir / "panel_0001.png") as crop:
        assert crop.getpixel((0, 0)) == (0, 0, 255)


def test_panels_step_empty_input(tmp_path):
    from steps.panels import run_panels_step

    pages = tmp_path / "pages"
    pages.mkdir()  # no images inside
    refs = tmp_path / "refs"
    refs.mkdir()
    from config import PipelineConfig

    config = PipelineConfig(
        input_dir=pages, refs_dir=refs, output_root=tmp_path / "output2", mock=True
    )
    ctx = RunContext.create(tmp_path / "output2", {"status": "running"})
    with pytest.raises(ValueError):
        run_panels_step(ctx, config, FakeDetector([]))


def test_panels_step_skip_and_limit(tmp_path):
    from steps.panels import run_panels_step

    config = make_step_inputs(tmp_path)
    config.skip_first = 1
    config.limit = 1
    ctx = RunContext.create(tmp_path / "output3", {"status": "running"})
    result = run_panels_step(ctx, config, FakeDetector([]))
    assert [p["page"] for p in result["pages"]] == ["p002.png"]
