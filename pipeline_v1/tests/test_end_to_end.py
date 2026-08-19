"""End-to-end tests: the full pipeline (all five stages) with mock backends on
a synthetic manga page, fully offline.

The real YOLO detector / OpenRouter API / Spark FLUX server are exercised only
by the manual smoke script (scripts/smoke_real.sh), never by pytest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from config import PIPELINE_DIR
from detection import PanelBox
from mock_backends import (
    MockCharacterDetector,
    MockColorizer,
    MockPanelDetector,
)
from orchestrator import Backends, PipelineRunner
from tests.synthetic_page import READING_ORDER, build_page, panel_box

# Characters per reading-order number (panel_000N), exercising the empty
# detection path on panel 3.
CHARACTERS_BY_PANEL = {
    "panel_0001": ["Frieren", "Fern"],   # banner
    "panel_0002": ["Frieren"],           # right-top
    "panel_0003": [],                    # left-top (empty -> panel-only colorize)
    "panel_0004": ["Fern"],              # right-bottom
    "panel_0005": ["Frieren", "Fern"],   # left-bottom
}


# ---------------------------------------------------------------------------
# Fixtures

@pytest.fixture
def pipeline_inputs(tmp_path):
    page_path = tmp_path / "pages" / "0134-999.png"
    build_page(page_path)

    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ("frieren_reference.webp", "fern_reference.webp"):
        Image.new("RGB", (8, 8), "gray").save(refs / name)
    return page_path, refs, tmp_path


@pytest.fixture
def mock_backends():
    boxes = [PanelBox(*panel_box(panel_id), 0.95) for panel_id in READING_ORDER]
    return Backends(
        detector=MockPanelDetector(boxes),
        character_detector=MockCharacterDetector(CHARACTERS_BY_PANEL),
        colorizer=MockColorizer(),
    )


def make_config(tmp_path, refs, **overrides):
    from config import PipelineConfig

    base = dict(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output",
        mock=True,
        sleep_s=0.0,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The end-to-end test

def test_pipeline_end_to_end(pipeline_inputs, mock_backends):
    page_path, refs, tmp_path = pipeline_inputs
    config = make_config(tmp_path, refs)
    ctx = PipelineRunner(config, mock_backends).run()

    # 1. Manifest completed with the seven stages.
    assert ctx.manifest["status"] == "completed"
    assert list(ctx.manifest["steps"].keys()) == [
        "panels", "characters", "colorize", "stitch", "debug", "pdf",
        "sanity",
    ]
    assert ctx.manifest["configuration"]["mock"] is True

    # 2. Numbered intermediate directories.
    for name in ("1_panels", "2_characters", "3_colorized", "4_stitched",
                 "5_debug", "6_pdf", "7_sanity"):
        assert (ctx.run_dir / name).is_dir()

    # 3. Panels extracted in Japanese reading order with geometry saved.
    page_name = page_path.stem
    panels_dir = ctx.run_dir / "1_panels" / page_name
    crops = sorted(p.name for p in panels_dir.iterdir() if p.name.startswith("panel_"))
    assert crops == [f"panel_000{i}.png" for i in range(1, 6)]
    geometry = read_json(panels_dir / "panels.json")
    assert geometry["reading_order"] == [1, 2, 3, 4, 5]
    # Reading order verified by pixels: crop 1 = banner (lightest), crop 3 =
    # left-top (110), crop 5 = left-bottom (darkest).
    with Image.open(panels_dir / "panel_0001.png") as crop:
        assert crop.getpixel((5, 5)) == (200, 200, 200)
    with Image.open(panels_dir / "panel_0003.png") as crop:
        assert crop.getpixel((5, 5)) == (110, 110, 110)
    with Image.open(panels_dir / "panel_0005.png") as crop:
        assert crop.getpixel((5, 5)) == (60, 60, 60)

    # 4. Per-panel character records.
    chars_dir = ctx.run_dir / "2_characters" / page_name
    for i in range(1, 6):
        doc = read_json(chars_dir / f"panel_000{i}.json")
        assert doc["characters"] == CHARACTERS_BY_PANEL[f"panel_000{i}"]
        assert doc["status"] == "ok"

    # 5. Colorized panels + filtered atlases.
    colorized_dir = ctx.run_dir / "3_colorized" / page_name
    for i in range(1, 6):
        assert (colorized_dir / f"panel_000{i}.png").is_file()
    # Atlas sent only when characters were detected.
    for panel, atlas, _output, _palette in mock_backends.colorizer.calls:
        expected_atlas = bool(CHARACTERS_BY_PANEL[panel.stem])
        assert (atlas is not None) == expected_atlas, panel.name
        if atlas is not None:
            assert atlas.name == f"{panel.stem}_atlas.jpg"

    # 6. Stitched page: tinted inside every box, white outside.
    stitched = ctx.run_dir / "4_stitched" / f"{page_name}.png"
    with Image.open(stitched) as image:
        assert image.size == (500, 700)
        for panel_id in READING_ORDER:
            x1, y1, x2, y2 = panel_box(panel_id)
            r, g, b = image.getpixel(((x1 + x2) // 2, (y1 + y2) // 2))
            assert r > g == b, f"{panel_id}: got {(r, g, b)}"
        assert image.getpixel((250, 130)) == (255, 255, 255)   # gutter
        assert image.getpixel((250, 410)) == (255, 255, 255)   # gutter

    # 7. Debug annotation: same page size as the stitched one; one record per
    #    panel with the detected characters, from the run's 5_debug/.
    debug_dir = ctx.run_dir / "5_debug"
    debug_page = debug_dir / f"{page_name}.png"
    with Image.open(debug_page) as image:
        assert image.size == (500, 700)
    debug_summary = read_json(debug_dir / "summary.json")
    assert debug_summary["pages_annotated"] == 1
    by_panel = {
        p["panel"]: p for p in debug_summary["records"][0]["panels"]
    }
    assert by_panel["panel_0001.png"]["characters"] == ["Frieren", "Fern"]
    assert by_panel["panel_0003.png"]["characters"] == []
    assert all(not p["bw_fallback"] for p in by_panel.values())

    # 8. Totals.
    totals = ctx.manifest["totals"]
    assert totals["character_calls"] == 5
    assert totals["flux_calls"] == 5
    assert totals["successful_flux_calls"] == 5
    assert totals["panels_colorized"] == 5
    assert totals["pages_stitched"] == 1
    assert totals["pages_annotated"] == 1
    assert totals["openrouter_cost_usd"] == pytest.approx(0.0005, abs=1e-9)


def test_pipeline_end_to_end_panel_page_prev2(pipeline_inputs, tmp_path):
    """The full pipeline with detection_mode='panel-page-prev2' and the
    page-context mock: per-panel records sourced 'panel-page-prev2' and the
    panel+page+prev2 provenance file written."""
    from mock_backends import MockPageCharacterDetector

    page_path, refs, _ = pipeline_inputs
    page_name = page_path.stem
    by_page = {
        page_name: {
            f"panel_000{i}": (list(CHARACTERS_BY_PANEL[f"panel_000{i}"]), False)
            for i in range(1, 6)
        }
    }
    backends = Backends(
        detector=MockPanelDetector(
            [PanelBox(*panel_box(panel_id), 0.95) for panel_id in READING_ORDER]
        ),
        character_detector=MockPageCharacterDetector(by_page),
        colorizer=MockColorizer(),
    )
    config = make_config(tmp_path, refs, detection_mode="panel-page-prev2")
    ctx = PipelineRunner(config, backends).run()

    assert ctx.manifest["status"] == "completed"
    assert ctx.manifest["configuration"]["detection_mode"] == "panel-page-prev2"
    chars_dir = ctx.run_dir / "2_characters" / page_name
    for i in range(1, 6):
        doc = read_json(chars_dir / f"panel_000{i}.json")
        assert doc["source"] == "panel-page-prev2"
        assert doc["characters"] == CHARACTERS_BY_PANEL[f"panel_000{i}"]
    assert (chars_dir / "panel_page_prev2_calls.json").is_file()
    totals = ctx.manifest["totals"]
    assert totals["character_calls"] == 5
    assert totals["page_character_calls"] == 5
    assert totals["fallback_character_calls"] == 0
    assert totals["openrouter_cost_usd"] == pytest.approx(0.001, abs=1e-9)
    assert totals["pages_stitched"] == 1


def test_pipeline_end_to_end_via_cli(pipeline_inputs, tmp_path):
    """The CLI wiring itself: `run.py --mock` on the synthetic page."""
    page_path, refs, _ = pipeline_inputs
    output = tmp_path / "cli-output"
    result = subprocess.run(
        [
            sys.executable, str(PIPELINE_DIR / "run.py"),
            "--mock",
            "--input-dir", str(page_path.parent),
            "--refs-dir", str(refs),
            "--output-root", str(output),
            "--sleep", "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    run_dirs = sorted(output.iterdir())
    assert run_dirs, "no run directory created"
    manifest = read_json(run_dirs[-1] / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["totals"]["pages_stitched"] == 1


def test_reading_order_non_trivial():
    """The synthetic layout must actually exercise the ordering algorithm
    (banner first, rows right-to-left)."""
    boxes = [PanelBox(*panel_box(panel_id), 0.95) for panel_id in READING_ORDER]
    from panel_ordering import reading_order

    order = reading_order(boxes)
    # reading_order returns indices into the input list, which here is already
    # in reading order, so the output must be [0, 1, 2, 3, 4].
    assert order == [0, 1, 2, 3, 4]
