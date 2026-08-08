"""Tests for stitching.py and steps/stitch.py (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from detection import PanelBox
from stitching import stitch_page


def make_page(width=400, height=400) -> Image.Image:
    """B&W-ish page: white background with gray panel rectangles."""
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((20, 20, 180, 180), fill=(128, 128, 128))   # panel A
    draw.rectangle((200, 20, 380, 180), fill=(64, 64, 64))     # panel B
    return page


def make_colorized(path: Path, color=(255, 0, 0), size=(160, 160)) -> Image.Image:
    Image.new("RGB", size, color).save(path)
    return Image.open(path)


def test_stitch_page_pastes_at_box_and_keeps_rest():
    page = make_page()
    red = Image.new("RGB", (160, 160), (255, 0, 0))
    blue = Image.new("RGB", (180, 160), (0, 0, 255))
    stitched = stitch_page(
        page,
        [
            (PanelBox(20, 20, 180, 180, 0.9), red),     # panel A
            (PanelBox(200, 20, 380, 180, 0.9), blue),   # panel B
        ],
    )
    assert stitched.size == page.size
    assert stitched.getpixel((100, 100)) == (255, 0, 0)
    assert stitched.getpixel((300, 100)) == (0, 0, 255)
    # Outside the boxes: untouched white.
    assert stitched.getpixel((200, 300)) == (255, 255, 255)
    # Source page is not modified.
    assert page.getpixel((100, 100)) == (128, 128, 128)


def test_stitch_page_resizes_back_to_box_dimensions():
    """A colorized panel at 336x496 pasted into a 160x160 box is resized."""
    page = make_page()
    big = Image.new("RGB", (336, 496), (0, 255, 0))
    stitched = stitch_page(page, [(PanelBox(20, 20, 180, 180, 0.9), big)])
    assert stitched.getpixel((100, 100)) == (0, 255, 0)


def test_stitch_page_box_out_of_bounds_clipped():
    page = make_page()
    panel = Image.new("RGB", (400, 400), (255, 0, 0))
    stitched = stitch_page(page, [(PanelBox(-100, -100, 100, 100, 0.9), panel)])
    assert stitched.size == page.size  # no crash, clipped paste


def test_stitch_page_empty_colorized_returns_copy():
    page = make_page()
    stitched = stitch_page(page, [])
    assert stitched.size == page.size
    assert stitched.getpixel((100, 100)) == (128, 128, 128)


def test_stitch_page_inset():
    page = make_page()
    panel = Image.new("RGB", (160, 160), (255, 0, 0))
    stitched = stitch_page(page, [(PanelBox(20, 20, 180, 180, 0.9), panel)], inset=5)
    # The panel border pixels (x=20..24) keep the original gray.
    assert stitched.getpixel((22, 100)) == (128, 128, 128)
    assert stitched.getpixel((30, 100)) == (255, 0, 0)


# ---------------------------------------------------------------------------
# Step-level test

def _build_step_layout(tmp_path: Path):
    """1_panels/<page>/ + 3_colorized/<page>/ + a source page."""
    page_path = tmp_path / "pages" / "p001.png"
    page_path.parent.mkdir()
    make_page().save(page_path)

    panels_dir = tmp_path / "1_panels" / "p001"
    panels_dir.mkdir(parents=True)
    Image.new("RGB", (160, 160), "white").save(panels_dir / "panel_0001.png")
    Image.new("RGB", (180, 160), "white").save(panels_dir / "panel_0002.png")
    geometry = {
        "page": "p001.png",
        "page_path": str(page_path.resolve()),
        "detections": [
            {"panel_index": 1, "box": [20, 20, 180, 180], "confidence": 0.9,
             "crop": "panel_0001.png"},
            {"panel_index": 2, "box": [200, 20, 380, 180], "confidence": 0.9,
             "crop": "panel_0002.png"},
        ],
        "reading_order": [1, 2],
    }
    (panels_dir / "panels.json").write_text(json.dumps(geometry))

    colorized_dir = tmp_path / "3_colorized" / "p001"
    colorized_dir.mkdir(parents=True)
    Image.new("RGB", (160, 160), (255, 0, 0)).save(colorized_dir / "panel_0001.png")
    Image.new("RGB", (180, 160), (0, 0, 255)).save(colorized_dir / "panel_0002.png")
    return page_path


def test_stitch_step(tmp_path):
    from config import PipelineConfig
    from run_context import RunContext
    from steps.stitch import run_stitch_step

    _build_step_layout(tmp_path)
    refs = tmp_path / "refs"
    refs.mkdir()
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output",
        mock=True,
    )
    ctx = RunContext.create(tmp_path / "output", {"status": "running"})
    # Move fixtures into the run dir layout.
    import shutil

    shutil.copytree(tmp_path / "1_panels", ctx.step_dir("panels"), dirs_exist_ok=True)
    shutil.copytree(tmp_path / "3_colorized", ctx.step_dir("colorize"), dirs_exist_ok=True)

    result = run_stitch_step(ctx, config)
    assert len(result["outputs"]) == 1
    output_path = ctx.step_dir("stitch") / "p001.png"
    assert output_path.is_file()
    with Image.open(output_path) as image:
        assert image.getpixel((100, 100)) == (255, 0, 0)
        assert image.getpixel((300, 100)) == (0, 0, 255)
        assert image.getpixel((200, 300)) == (255, 255, 255)  # outside B&W


def test_stitch_step_missing_colorized_raises(tmp_path):
    from config import PipelineConfig
    from run_context import RunContext
    from steps.stitch import run_stitch_step

    _build_step_layout(tmp_path)
    (tmp_path / "3_colorized").rename(tmp_path / "3_colorized_gone")
    refs = tmp_path / "refs"
    refs.mkdir()
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output2",
        mock=True,
    )
    ctx = RunContext.create(tmp_path / "output2", {"status": "running"})
    import shutil

    shutil.copytree(tmp_path / "1_panels", ctx.step_dir("panels"), dirs_exist_ok=True)
    with pytest.raises(ValueError):
        run_stitch_step(ctx, config)
