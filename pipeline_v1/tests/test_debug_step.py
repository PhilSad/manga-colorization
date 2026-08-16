"""Tests for the pipeline's debug stage (steps/debug.py -> 5_debug/).

The final pipeline stage annotates each stitched page with the detected panel
bounding boxes (from 1_panels/<page>/panels.json) and the characters detected
per panel (from 2_characters/<page>/<panel>.json). Panels stitched from their
original B&W crop (the stitch step's always-on fallback) get an orange box. The standalone
offline tool scripts/annotate_stitch.py delegates to the same
`run_debug_step`, so these tests also cover its rendering.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from config import PipelineConfig
from mock_backends import MockCharacterDetector, MockColorizer, MockPanelDetector
from orchestrator import Backends, PipelineRunner
from run_context import RunContext
from steps.debug import run_debug_step
from test_orchestrator import make_config, make_refs, make_synthetic_page


# ---------------------------------------------------------------------------
# Fixtures

def make_backends(by_panel: dict[str, list[str]] | None = None):
    from detection import PanelBox

    def box(x1, y1, x2, y2):
        return PanelBox(x1, y1, x2, y2, 0.9)

    return Backends(
        detector=MockPanelDetector([box(200, 20, 380, 180), box(20, 20, 180, 180)]),
        character_detector=MockCharacterDetector(
            by_panel or {"panel_0001": ["Frieren", "Fern"], "panel_0002": []}
        ),
        colorizer=MockColorizer(),
    )


def _geometry(tmp_path: Path) -> dict:
    return {
        "page": "p001.png",
        "page_path": str((tmp_path / "pages" / "p001.png").resolve()),
        "detections": [
            {"panel_index": 1, "box": [20, 20, 180, 180], "confidence": 0.9,
             "crop": "panel_0001.png"},
            {"panel_index": 2, "box": [200, 20, 380, 180], "confidence": 0.9,
             "crop": "panel_0002.png"},
        ],
        "reading_order": [1, 2],
    }


def _write_character_record(run_dir: Path, stem: str, names: list[str]) -> None:
    path = run_dir / "2_characters" / "p001" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "ok", "source": "panel-page", "uncertain": False,
        "panel": f"{stem}.png", "characters": names,
    }))


def _build_minimal_run(tmp_path: Path, *, with_manifest_fallback: bool = False,
                       with_characters: bool = True) -> Path:
    """Run dir with a stitched page + geometry (+ characters records), usable
    directly by run_debug_step. Optional stitch-record B&W fallback."""
    run_dir = tmp_path / "run"
    (run_dir / "4_stitched").mkdir(parents=True)
    (run_dir / "1_panels" / "p001").mkdir(parents=True)
    make_synthetic_page(tmp_path / "pages" / "p001.png", size=(400, 400))
    (run_dir / "4_stitched" / "p001.png").write_bytes(
        (tmp_path / "pages" / "p001.png").read_bytes()
    )
    (run_dir / "1_panels" / "p001" / "panels.json").write_text(
        json.dumps(_geometry(tmp_path))
    )
    if with_characters:
        _write_character_record(run_dir, "panel_0001", ["Frieren", "Himmel"])
        _write_character_record(run_dir, "panel_0002", [])
    manifest: dict = {"steps": {}}
    if with_manifest_fallback:
        manifest["steps"]["stitch"] = {"outputs": [{
            "page": "p001",
            "panels_bw_fallback": ["panel_0002.png"],
        }]}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


# ---------------------------------------------------------------------------
# Tests

def test_full_run_creates_debug_step_outputs(tmp_path):
    """The final pipeline stage writes 5_debug/ with an annotated copy of
    each stitched page + summary.json."""
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    make_synthetic_page(pages / "p001.png")
    make_synthetic_page(pages / "p002.png")
    config = PipelineConfig(
        input_dir=pages, refs_dir=make_refs(tmp_path),
        output_root=tmp_path / "output", mock=True, sleep_s=0.0,
    )
    ctx = PipelineRunner(config, make_backends()).run()

    assert ctx.manifest["status"] == "completed"
    assert ctx.manifest["totals"]["pages_annotated"] == 2
    debug_dir = ctx.run_dir / "5_debug"
    assert (debug_dir / "p001.png").is_file()
    assert (debug_dir / "p002.png").is_file()

    summary = json.loads((debug_dir / "summary.json").read_text())
    assert summary["pages_annotated"] == 2
    page_record = summary["records"][0]
    assert page_record["page"] == "p001"
    by_panel = {p["panel"]: p for p in page_record["panels"]}
    assert by_panel["panel_0001.png"]["characters"] == ["Frieren", "Fern"]
    assert by_panel["panel_0002.png"]["characters"] == []

    with Image.open(debug_dir / "p001.png") as image:
        # Red bboxes around both panels; badge (white) near top-left of A.
        assert image.getpixel((100, 24)) == (220, 30, 30)
        assert image.getpixel((300, 24)) == (220, 30, 30)
        assert image.getpixel((30, 30)) == (255, 255, 255)


def test_debug_marks_bw_fallback_panels_orange(tmp_path):
    """Panels recorded as B&W fallbacks in the run manifest (stitch step's
    always-on fallback) get an orange bbox and a [B&W fallback] tag."""
    run_dir = _build_minimal_run(tmp_path, with_manifest_fallback=True)
    ctx = RunContext.load(run_dir)
    record = run_debug_step(ctx, PipelineConfig(), output_dir=run_dir / "5_debug")

    assert record["pages_annotated"] == 1
    by_panel = {p["panel"]: p for p in record["outputs"][0]["panels"]}
    assert by_panel["panel_0002.png"]["bw_fallback"] is True
    assert by_panel["panel_0001.png"]["bw_fallback"] is False
    with Image.open(run_dir / "5_debug" / "p001.png") as image:
        assert image.getpixel((100, 24)) == (220, 30, 30)    # normal, red
        assert image.getpixel((300, 24)) == (230, 110, 0)    # fallback, orange


def test_debug_reads_resume_manifest_fallbacks(tmp_path):
    """`--from-step debug --resume RUN`: the fresh run has no stitch record,
    so the fallback map comes from the resume run's manifest."""
    source = _build_minimal_run(tmp_path / "s", with_manifest_fallback=True)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "manifest.json").write_text(json.dumps({"steps": {}}))
    for name in ("4_stitched", "1_panels", "2_characters"):
        shutil.copytree(source / name, fresh / name)

    ctx = RunContext(fresh, {"steps": {}})
    config = PipelineConfig(resume=source)
    run_debug_step(ctx, config, output_dir=fresh / "5_debug")

    with Image.open(fresh / "5_debug" / "p001.png") as image:
        assert image.getpixel((300, 24)) == (230, 110, 0)  # orange fallback


def test_debug_tolerates_missing_character_records(tmp_path):
    """A run without 2_characters/ (e.g. --steps panels,colorize,stitch)
    still annotates: panels get a '(no record)' label."""
    run_dir = _build_minimal_run(tmp_path, with_characters=False)
    ctx = RunContext.load(run_dir)
    record = run_debug_step(ctx, PipelineConfig(), output_dir=run_dir / "5_debug")

    assert record["pages_annotated"] == 1
    assert (run_dir / "5_debug" / "p001.png").is_file()
    by_panel = {p["panel"]: p for p in record["outputs"][0]["panels"]}
    assert by_panel["panel_0001.png"]["characters"] == []


def test_debug_without_stitch_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({}))
    ctx = RunContext.load(run_dir)
    with pytest.raises(ValueError, match="stitch"):
        run_debug_step(ctx, PipelineConfig())


def test_debug_only_panels_selection(tmp_path):
    """A targeted rerun annotates only the selected pages."""
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    make_synthetic_page(pages / "p001.png")
    make_synthetic_page(pages / "p002.png")
    config = PipelineConfig(
        input_dir=pages, refs_dir=make_refs(tmp_path),
        output_root=tmp_path / "output", mock=True, sleep_s=0.0,
        only_panels=("p001:panel_0001", "p001:panel_0002"),
    )
    ctx = PipelineRunner(config, make_backends()).run()

    assert ctx.manifest["totals"]["pages_annotated"] == 1
    debug_dir = ctx.run_dir / "5_debug"
    assert (debug_dir / "p001.png").is_file()
    assert not (debug_dir / "p002.png").exists()


def test_debug_step_respects_rendering_knobs(tmp_path):
    """--debug-font-size / --debug-bbox-width / overrides change the output."""
    run_dir = _build_minimal_run(tmp_path)
    ctx = RunContext.load(run_dir)
    config = PipelineConfig(debug_font_size=60, debug_bbox_width=9)
    run_debug_step(ctx, config, output_dir=run_dir / "5_debug")

    with Image.open(run_dir / "5_debug" / "p001.png") as image:
        # 9px stroke: rows 20..28 are red; with the default 5px they are not.
        assert image.getpixel((100, 27)) == (220, 30, 30)
        assert image.getpixel((100, 29)) != (220, 30, 30)
