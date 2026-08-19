"""Tests for the line-art fidelity sanity check (sanity.py + steps/sanity.py).

The sanity stage (step 7, 7_sanity/) compares each colorized panel with its
black & white original through structural line maps and flags panels whose
line art drifted below the threshold for review. These are fully offline:
synthetic line art drawn with Pillow, no backends, no network.

Also covers full-page runs (one synthetic box covering the whole page, e.g.
--full-page mode): the real panels are re-extracted with the panel detector
on the B&W page and each detected panel is checked individually against the
stitched page (YOLO box scaled to the stitched page's resolution).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from config import PipelineConfig
from detection import PanelBox
from run_context import RunContext
from sanity import score_pair
from steps.sanity import panel_check_tasks, run_sanity_step


# ---------------------------------------------------------------------------
# Synthetic fixtures

def _line_art(size: tuple[int, int] = (300, 400)) -> Image.Image:
    """Simple manga-like line art: thin strokes + an outline on white."""
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    draw.line([(40, 40), (260, 40)], fill=0, width=3)
    draw.line([(40, 60), (40, 340)], fill=0, width=2)
    draw.ellipse([(100, 100), (200, 200)], outline=0, width=3)
    draw.line([(220, 120), (280, 300)], fill=0, width=4)
    draw.rectangle([(50, 250), (90, 320)], outline=0, width=2)
    return image


def _tinted(image: Image.Image, ink=(90, 55, 40), paper=(250, 245, 235)):
    """'Perfect' colorization: same line art, brown ink on warm paper."""
    return ImageOps.colorize(image.convert("L"), black=ink, white=paper)


def _corrupted(image: Image.Image) -> Image.Image:
    """Unrelated content: a big filled shape where the line art used to be."""
    size = image.size
    corrupted = Image.new("L", size, 255)
    draw = ImageDraw.Draw(corrupted)
    draw.rectangle([(40, 40), (size[0] - 40, size[1] - 40)], fill=60)
    draw.ellipse([(90, 90), (size[0] - 90, size[1] - 90)], fill=180)
    return corrupted


# ---------------------------------------------------------------------------
# Scorer

def test_tinted_identical_pair_scores_high():
    bw = _line_art()
    metrics = score_pair(bw, _tinted(bw))
    assert metrics["flagged"] is False
    assert metrics["line_fidelity"] > 0.9, metrics
    assert metrics["line_iou"] > 0.9, metrics


def test_corrupted_pair_flagged():
    bw = _line_art()
    metrics = score_pair(bw, _corrupted(bw))
    assert metrics["flagged"] is True
    assert metrics["line_fidelity"] < 0.4, metrics
    assert metrics["line_iou"] < 0.2, metrics
    assert metrics["reasons"], metrics


def test_size_mismatch_still_scores_high():
    """FLUX/gpt-image-2 round sizes to multiples of 16; the scorer must be
    insensitive to that (both are resampled onto the B&W analysis grid)."""
    bw = _line_art((302, 401))  # awkward size
    color = _tinted(bw).resize((288, 384), Image.Resampling.LANCZOS)
    metrics = score_pair(bw, color)
    assert metrics["flagged"] is False
    assert metrics["line_fidelity"] > 0.85, metrics


def test_hard_iou_rule_trips_with_high_threshold_relaxed():
    """Even with a very low threshold, catastrophic content loss trips the
    hard IoU rule."""
    bw = _line_art()
    metrics = score_pair(bw, _corrupted(bw), threshold=0.0)
    assert metrics["flagged"] is True
    assert any("line_iou" in reason for reason in metrics["reasons"])


def test_empty_both_sides_scores_one():
    blank = Image.new("L", (100, 100), 255)
    metrics = score_pair(blank, blank)
    assert metrics["flagged"] is False
    assert metrics["line_fidelity"] == 1.0


# ---------------------------------------------------------------------------
# Step

def _minimal_run(tmp_path: Path, *, flagged_colorized: bool = False) -> Path:
    """Run dir with one page: a B&W panel crop + a stitched page."""
    run_dir = tmp_path / "run"
    page_dir = run_dir / "1_panels" / "p001"
    page_dir.mkdir(parents=True)
    bw = _line_art()
    bw.save(page_dir / "panel_0001.png")
    panels = {
        "page": "source/p001.png",
        "page_path": str((tmp_path / "p001.png").resolve()),
        "page_sha256": "x",
        "detections": [{
            "panel_index": 1,
            "box": [0, 0, 300, 400],
            "confidence": 1.0,
            "crop": "panel_0001.png",
            "provenance": "test",
        }],
        "blank_page": False,
        "skip_reason": None,
        "full_page_fallback": False,
    }
    (page_dir / "panels.json").write_text(json.dumps(panels), encoding="utf-8")
    stitched = run_dir / "4_stitched"
    stitched.mkdir(parents=True)
    color = _corrupted(bw) if flagged_colorized else _tinted(bw)
    color.save(stitched / "p001.png")
    manifest = {
        "kind": "pipeline-run",
        "steps": {"stitch": {"outputs": [
            {"page": "p001", "panels_bw_fallback": []},
        ]}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_run_sanity_step_clean(tmp_path):
    run_dir = _minimal_run(tmp_path, flagged_colorized=False)
    ctx = RunContext.load(run_dir)
    record = run_sanity_step(ctx, PipelineConfig())
    assert record["pages_checked"] == 1
    assert record["panels_checked"] == 1
    assert record["panels_flagged"] == 0
    summary = json.loads((run_dir / "7_sanity" / "summary.json").read_text())
    assert summary["panels_checked"] == 1
    assert summary["panels_flagged"] == 0
    page = json.loads((run_dir / "7_sanity" / "p001.json").read_text())
    assert page["flagged_panels"] == []
    assert not (run_dir / "7_sanity" / "p001_flagged.png").exists()


def test_run_sanity_step_flags_corrupted(tmp_path):
    run_dir = _minimal_run(tmp_path, flagged_colorized=True)
    ctx = RunContext.load(run_dir)
    record = run_sanity_step(ctx, PipelineConfig())
    assert record["panels_flagged"] == 1
    page = json.loads((run_dir / "7_sanity" / "p001.json").read_text())
    assert page["flagged_panels"] == ["panel_0001.png"]
    sheet = run_dir / "7_sanity" / "p001_flagged.png"
    assert sheet.is_file()
    with Image.open(sheet) as image:
        assert image.width > 0 and image.height > 0
    summary = json.loads((run_dir / "7_sanity" / "summary.json").read_text())
    assert summary["flagged"][0]["panels"] == ["panel_0001.png"]


def test_run_sanity_step_skips_bw_fallback(tmp_path):
    run_dir = _minimal_run(tmp_path, flagged_colorized=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["steps"]["stitch"]["outputs"][0]["panels_bw_fallback"] = [
        "panel_0001.png"
    ]
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ctx = RunContext.load(run_dir)
    record = run_sanity_step(ctx, PipelineConfig())
    assert record["panels_checked"] == 0
    assert record["panels_flagged"] == 0
    page = json.loads((run_dir / "7_sanity" / "p001.json").read_text())
    assert page["panels"][0]["bw_fallback"] is True


def test_run_sanity_step_missing_colorized_flags(tmp_path):
    run_dir = _minimal_run(tmp_path, flagged_colorized=True)
    (run_dir / "4_stitched" / "p001.png").unlink()
    ctx = RunContext.load(run_dir)
    record = run_sanity_step(ctx, PipelineConfig())
    assert record["panels_flagged"] == 1
    page = json.loads((run_dir / "7_sanity" / "p001.json").read_text())
    assert "no colorized output" in page["panels"][0]["note"]


def test_config_fields_roundtrip():
    config = PipelineConfig()
    data = config.to_dict()
    assert data["sanity_threshold"] == 0.45
    assert data["sanity_max_edge"] == 1024
    config.sanity_threshold = 0.6
    config.sanity_max_edge = 512
    assert config.to_dict()["sanity_threshold"] == 0.6
    assert config.to_dict()["sanity_max_edge"] == 512


# ---------------------------------------------------------------------------
# Full-page runs (synthetic whole-page box -> re-extract real panels)

class _FakeDetector:
    """Deterministic panel detector for the synthetic full-page branch."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, page):
        return [PanelBox(x1, y1, x2, y2, confidence=1.0)
                for (x1, y1, x2, y2) in self.boxes]


def _full_page_run(tmp_path: Path, *, corrupted_box: int | None = None,
                   stitched_scale: float = 1.0) -> Path:
    """Run dir with one full-page-mode page: a 600x800 B&W source page with
    two line-art panels, and a stitched page with the panels at their scaled
    positions (tinted = perfect, or corrupted for one box).

    `stitched_scale` shrinks the stitched page (full-page passthrough pages
    can be smaller than the source); the scoring tests use 1.0 so thin
    strokes survive, the task-extraction test uses 0.5 to verify the box
    scaling math."""
    run_dir = tmp_path / "run"
    page_dir = run_dir / "1_panels" / "p001"
    page_dir.mkdir(parents=True)

    bw_page = Image.new("RGB", (600, 800), "white")
    panel_a = _line_art((300, 400))          # top-left panel
    panel_b = _line_art((300, 400))          # bottom-right panel
    bw_page.paste(panel_a, (0, 0))
    bw_page.paste(panel_b, (300, 400))
    bw_path = tmp_path / "p001.png"
    bw_page.save(bw_path)

    panels = {
        "page": str(bw_path),
        "page_path": str(bw_path),
        "page_sha256": "x",
        "detections": [{
            "panel_index": 1,
            "box": [0, 0, 600, 800],
            "confidence": 1.0,
            "crop": "panel_0001.png",
            "provenance": "full-page-mode",
        }],
        "blank_page": False,
        "skip_reason": None,
        "full_page_fallback": False,
    }
    (page_dir / "panels.json").write_text(json.dumps(panels), encoding="utf-8")

    stitched = run_dir / "4_stitched"
    stitched.mkdir(parents=True)
    scale = stitched_scale
    page = Image.new("RGB", (round(600 * scale), round(800 * scale)), "white")
    a_pos = (0, 0)
    b_pos = (round(300 * scale), round(400 * scale))
    a_size = (round(300 * scale), round(400 * scale))
    b_size = (round(300 * scale), round(400 * scale))
    if scale == 1.0:
        page.paste(_tinted(panel_a), a_pos)
        page.paste(
            _corrupted(panel_b) if corrupted_box == 2 else _tinted(panel_b),
            b_pos,
        )
    else:
        page.paste(_tinted(panel_a).resize(a_size, Image.Resampling.LANCZOS), a_pos)
        content = (_corrupted(panel_b) if corrupted_box == 2 else _tinted(panel_b))
        page.paste(content.resize(b_size, Image.Resampling.LANCZOS), b_pos)
    page.save(stitched / "p001.png")

    manifest = {"kind": "pipeline-run", "steps": {}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_panel_check_tasks_full_page_uses_yolo_and_scales_boxes(tmp_path):
    """A synthetic whole-page box becomes one task per YOLO-detected panel,
    with the box scaled to the stitched page's resolution and bw_box kept in
    B&W source coordinates."""
    run_dir = _full_page_run(tmp_path, stitched_scale=0.5)
    page_dir = run_dir / "1_panels" / "p001"
    geometry = json.loads((page_dir / "panels.json").read_text())
    from PIL import Image as _Image
    stitched = _Image.open(run_dir / "4_stitched" / "p001.png").convert("RGB")
    assert stitched.size == (300, 400)

    detector = _FakeDetector([(0, 0, 300, 400), (300, 400, 600, 800)])
    tasks = panel_check_tasks(
        geometry, page_dir, stitched, run_dir / "3_colorized",
        set(), detector=detector,
    )

    assert [t["provenance"] for t in tasks] == ["yolo", "yolo"]
    assert [t["crop"] for t in tasks] == ["yolo_0001.png", "yolo_0002.png"]
    assert all(t["synthetic"] for t in tasks)
    # Stitched page is 300x400, source is 600x800 -> scale factor 0.5.
    assert tasks[0]["box"] == [0, 0, 150, 200]
    assert tasks[1]["box"] == [150, 200, 300, 400]
    assert tasks[0]["bw_box"] == [0, 0, 300, 400]
    assert tasks[1]["bw_box"] == [300, 400, 600, 800]


def test_run_sanity_step_full_page_clean(tmp_path):
    run_dir = _full_page_run(tmp_path)
    ctx = RunContext.load(run_dir)
    record = run_sanity_step(
        ctx, PipelineConfig(),
        detector=_FakeDetector([(0, 0, 300, 400), (300, 400, 600, 800)]),
    )
    assert record["panels_checked"] == 2
    assert record["panels_flagged"] == 0
    page = json.loads((run_dir / "7_sanity" / "p001.json").read_text())
    assert page["flagged_panels"] == []
    assert [p["panel"] for p in page["panels"]] == ["yolo_0001.png", "yolo_0002.png"]
    # Both the scaled box and the B&W-source box are recorded.
    assert page["panels"][0]["box"] == [0, 0, 300, 400]
    assert page["panels"][0]["bw_box"] == [0, 0, 300, 400]


def test_run_sanity_step_full_page_flags_corrupted_panel(tmp_path):
    run_dir = _full_page_run(tmp_path, corrupted_box=2)
    ctx = RunContext.load(run_dir)
    record = run_sanity_step(
        ctx, PipelineConfig(),
        detector=_FakeDetector([(0, 0, 300, 400), (300, 400, 600, 800)]),
    )
    assert record["panels_checked"] == 2
    assert record["panels_flagged"] == 1
    page = json.loads((run_dir / "7_sanity" / "p001.json").read_text())
    assert page["flagged_panels"] == ["yolo_0002.png"]
    assert (run_dir / "7_sanity" / "p001_flagged.png").is_file()
    summary = json.loads((run_dir / "7_sanity" / "summary.json").read_text())
    assert summary["flagged"][0]["panels"] == ["yolo_0002.png"]


def test_panel_check_tasks_full_page_empty_detections(tmp_path):
    """No YOLO detections on the B&W page -> no tasks, no crash."""
    run_dir = _full_page_run(tmp_path)
    page_dir = run_dir / "1_panels" / "p001"
    geometry = json.loads((page_dir / "panels.json").read_text())
    from PIL import Image as _Image
    stitched = _Image.open(run_dir / "4_stitched" / "p001.png").convert("RGB")

    tasks = panel_check_tasks(
        geometry, page_dir, stitched, run_dir / "3_colorized",
        set(), detector=_FakeDetector([]),
    )
    assert tasks == []


def test_panel_check_tasks_full_page_missing_source(tmp_path):
    """No B&W source page -> no tasks, no crash."""
    run_dir = tmp_path / "run"
    page_dir = run_dir / "1_panels" / "p001"
    page_dir.mkdir(parents=True)
    panels = {
        "page": "gone.png",
        "page_path": str(tmp_path / "gone.png"),  # does not exist
        "detections": [{
            "panel_index": 1,
            "box": [0, 0, 600, 800],
            "confidence": 1.0,
            "crop": "panel_0001.png",
            "provenance": "full-page-mode",
        }],
    }
    (page_dir / "panels.json").write_text(json.dumps(panels), encoding="utf-8")
    stitched = Image.new("RGB", (300, 400), "white")
    tasks = panel_check_tasks(
        panels, page_dir, stitched, run_dir / "3_colorized",
        set(), detector=_FakeDetector([(0, 0, 300, 400)]),
    )
    assert tasks == []
