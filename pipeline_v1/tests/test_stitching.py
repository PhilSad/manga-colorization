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
    # Original crops mirror the page's panel rectangles (gray shades).
    Image.new("RGB", (160, 160), (128, 128, 128)).save(panels_dir / "panel_0001.png")
    Image.new("RGB", (180, 160), (64, 64, 64)).save(panels_dir / "panel_0002.png")
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


def test_stitch_step_missing_colorized_raises_without_flag(tmp_path):
    """A single missing colorized panel still fails loudly unless
    --stitch-bw-fallback is enabled."""
    from config import PipelineConfig
    from run_context import RunContext
    from steps.stitch import run_stitch_step

    _build_step_layout(tmp_path)
    (tmp_path / "3_colorized" / "p001" / "panel_0002.png").unlink()
    refs = tmp_path / "refs"
    refs.mkdir()
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output4",
        mock=True,
    )
    ctx = RunContext.create(tmp_path / "output4", {"status": "running"})
    import shutil

    shutil.copytree(tmp_path / "1_panels", ctx.step_dir("panels"), dirs_exist_ok=True)
    shutil.copytree(tmp_path / "3_colorized", ctx.step_dir("colorize"),
                    dirs_exist_ok=True)
    with pytest.raises(ValueError, match="panel_0002"):
        run_stitch_step(ctx, config)


def test_stitch_step_bw_fallback_missing_panel(tmp_path, capsys):
    """--stitch-bw-fallback: a missing colorized panel is stitched from the
    original black & white crop, logged, and recorded in the step record."""
    from config import PipelineConfig
    from run_context import RunContext
    from steps.stitch import run_stitch_step

    _build_step_layout(tmp_path)
    (tmp_path / "3_colorized" / "p001" / "panel_0002.png").unlink()
    refs = tmp_path / "refs"
    refs.mkdir()
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output5",
        mock=True,
        stitch_bw_fallback=True,
    )
    ctx = RunContext.create(tmp_path / "output5", {"status": "running"})
    import shutil

    shutil.copytree(tmp_path / "1_panels", ctx.step_dir("panels"), dirs_exist_ok=True)
    shutil.copytree(tmp_path / "3_colorized", ctx.step_dir("colorize"),
                    dirs_exist_ok=True)

    result = run_stitch_step(ctx, config)
    assert result["panels_bw_fallback"] == 1
    assert len(result["outputs"]) == 1
    page_record = result["outputs"][0]
    assert page_record["panels_bw_fallback"] == ["panel_0002.png"]
    assert page_record["panels_skipped_black_white"] == []
    output_path = ctx.step_dir("stitch") / "p001.png"
    assert output_path.is_file()
    with Image.open(output_path) as image:
        # panel A colorized red; panel B keeps the original gray (B&W crop).
        assert image.getpixel((100, 100)) == (255, 0, 0)
        assert image.getpixel((300, 100)) == (64, 64, 64)
        assert image.getpixel((200, 300)) == (255, 255, 255)  # outside B&W
    warning = capsys.readouterr().err
    assert "panel_0002.png" in warning and "black & white" in warning


def test_stitch_step_bw_fallback_missing_whole_page(tmp_path, capsys):
    """--stitch-bw-fallback with no colorized dir at all for a page stitches
    every panel from its original crop and logs a warning."""
    from config import PipelineConfig
    from run_context import RunContext
    from steps.stitch import run_stitch_step

    _build_step_layout(tmp_path)
    (tmp_path / "3_colorized" / "p001").rename(tmp_path / "3_colorized_gone")
    refs = tmp_path / "refs"
    refs.mkdir()
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output6",
        mock=True,
        stitch_bw_fallback=True,
    )
    ctx = RunContext.create(tmp_path / "output6", {"status": "running"})
    import shutil

    shutil.copytree(tmp_path / "1_panels", ctx.step_dir("panels"), dirs_exist_ok=True)

    result = run_stitch_step(ctx, config)
    assert result["panels_bw_fallback"] == 2
    page_record = result["outputs"][0]
    assert sorted(page_record["panels_bw_fallback"]) == [
        "panel_0001.png", "panel_0002.png"
    ]
    with Image.open(ctx.step_dir("stitch") / "p001.png") as image:
        assert image.getpixel((100, 100)) == (128, 128, 128)  # original gray
        assert image.getpixel((300, 100)) == (64, 64, 64)
    warning = capsys.readouterr().err
    assert "no colorized panels for p001" in warning


# ---------------------------------------------------------------------------
# V1.1 (task 0004): full-page fallback placement

def test_stitch_full_page_fallback_returns_canvas_dimensions(tmp_path):
    """A capped full-page output (requested size != canvas) must be resized
    back to exactly the source canvas dimensions when stitched."""
    from config import PipelineConfig
    from run_context import RunContext
    from steps.stitch import run_stitch_step

    canvas = (1500, 2250)
    page_path = tmp_path / "pages" / "p006.png"
    page_path.parent.mkdir(parents=True)
    Image.new("RGB", canvas, "white").save(page_path)

    panels_dir = tmp_path / "1_panels" / "p006"
    panels_dir.mkdir(parents=True)
    geometry = {
        "page": "p006.png",
        "page_path": str(page_path.resolve()),
        "detections": [{
            "panel_index": 1, "box": [0, 0, 1500, 2250], "confidence": 1.0,
            "crop": "panel_0001.png", "provenance": "full-page-fallback",
        }],
        "reading_order": [1],
        "full_page_fallback": True,
    }
    (panels_dir / "panels.json").write_text(json.dumps(geometry))

    colorized_dir = tmp_path / "3_colorized" / "p006"
    colorized_dir.mkdir(parents=True)
    # The FLUX server returned a capped 1152x1728 image.
    Image.new("RGB", (1152, 1728), (220, 60, 60)).save(
        colorized_dir / "panel_0001.png"
    )

    refs = tmp_path / "refs"
    refs.mkdir()
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output",
        mock=True,
    )
    ctx = RunContext.create(tmp_path / "output", {"status": "running"})
    import shutil

    shutil.copytree(tmp_path / "1_panels", ctx.step_dir("panels"), dirs_exist_ok=True)
    shutil.copytree(tmp_path / "3_colorized", ctx.step_dir("colorize"),
                    dirs_exist_ok=True)

    result = run_stitch_step(ctx, config)
    assert len(result["outputs"]) == 1
    with Image.open(ctx.step_dir("stitch") / "p006.png") as image:
        assert image.size == canvas  # exactly the source canvas
        assert image.getpixel((750, 1000)) == (220, 60, 60)  # colorized


def test_stitch_step_only_panels_skips_unselected_pages(tmp_path):
    from config import PipelineConfig
    from run_context import RunContext
    from steps.stitch import run_stitch_step

    _build_step_layout(tmp_path)
    # A second page without colorized outputs (like a resumed run's extra page).
    second = tmp_path / "1_panels" / "p002"
    second.mkdir(parents=True)
    (second / "panels.json").write_text(json.dumps({
        "page": "p002.png",
        "page_path": str((tmp_path / "pages" / "p002.png").resolve()),
        "detections": [{"panel_index": 1, "box": [20, 20, 180, 180],
                         "crop": "panel_0001.png"}],
        "reading_order": [1],
    }))
    Image.new("RGB", (160, 160), "white").save(second / "panel_0001.png")

    refs = tmp_path / "refs"
    refs.mkdir()
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output3",
        mock=True,
        only_panels=("p001:panel_0001",),
    )
    ctx = RunContext.create(tmp_path / "output3", {"status": "running"})
    import shutil

    shutil.copytree(tmp_path / "1_panels", ctx.step_dir("panels"), dirs_exist_ok=True)
    shutil.copytree(tmp_path / "3_colorized", ctx.step_dir("colorize"),
                    dirs_exist_ok=True)

    result = run_stitch_step(ctx, config)
    assert [output["page"] for output in result["outputs"]] == ["p001"]
    assert not (ctx.step_dir("stitch") / "p002.png").exists()
