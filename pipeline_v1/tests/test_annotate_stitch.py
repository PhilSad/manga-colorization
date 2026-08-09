"""Tests for scripts/annotate_stitch.py (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from test_stitching import make_page


def _build_run_dir(tmp_path: Path) -> Path:
    """Minimal completed run dir: 4_stitched/, 1_panels/, 2_characters/,
    manifest.json with one B&W fallback panel."""
    run_dir = tmp_path / "run"
    (run_dir / "4_stitched").mkdir(parents=True)
    (run_dir / "1_panels" / "p001").mkdir(parents=True)
    (run_dir / "2_characters" / "p001").mkdir(parents=True)

    make_page().save(run_dir / "4_stitched" / "p001.png")
    (run_dir / "1_panels" / "p001" / "panels.json").write_text(json.dumps({
        "page": "p001.png",
        "page_path": str((tmp_path / "pages" / "p001.png").resolve()),
        "detections": [
            {"panel_index": 1, "box": [20, 20, 180, 180], "confidence": 0.9,
             "crop": "panel_0001.png"},
            {"panel_index": 2, "box": [200, 20, 380, 180], "confidence": 0.9,
             "crop": "panel_0002.png"},
        ],
        "reading_order": [1, 2],
    }))

    (run_dir / "2_characters" / "p001" / "panel_0001.json").write_text(json.dumps({
        "status": "ok", "source": "panel-page", "uncertain": False,
        "panel": "panel_0001.png", "characters": ["Frieren", "Himmel"],
    }))
    (run_dir / "2_characters" / "p001" / "panel_0002.json").write_text(json.dumps({
        "status": "ok", "source": "panel-page", "uncertain": False,
        "panel": "panel_0002.png", "characters": [],
    }))
    (run_dir / "manifest.json").write_text(json.dumps({
        "steps": {"stitch": {"outputs": [{
            "page": "p001",
            "panels_bw_fallback": ["panel_0002.png"],
        }]}},
    }))
    return run_dir


def _run_script(tmp_path: Path, *extra_args: str) -> Path:
    import scripts.annotate_stitch as annotate

    out_dir = tmp_path / "debug"
    sys.argv = [
        "annotate_stitch",
        "--run-dir", str(_build_run_dir(tmp_path)),
        "--output-dir", str(out_dir),
        *extra_args,
    ]
    annotate.main()
    return out_dir


def test_annotate_writes_bboxes_and_characters(tmp_path):
    out_dir = _run_script(tmp_path)
    page_path = out_dir / "p001.png"
    assert page_path.is_file()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["pages_annotated"] == 1
    page_record = summary["records"][0]
    assert page_record["page"] == "p001"
    by_panel = {p["panel"]: p for p in page_record["panels"]}
    assert by_panel["panel_0001.png"]["characters"] == ["Frieren", "Himmel"]
    assert by_panel["panel_0002.png"]["characters"] == []
    assert by_panel["panel_0002.png"]["bw_fallback"] is True
    assert by_panel["panel_0001.png"]["bw_fallback"] is False

    with Image.open(page_path) as image:
        # Red bbox around panel A, orange (B&W fallback) around panel B.
        assert image.getpixel((100, 24)) == (220, 30, 30)
        assert image.getpixel((300, 24)) == (230, 110, 0)
        # Badge is drawn inside the box (white fill near the top-left corner).
        assert image.getpixel((30, 30)) == (255, 255, 255)


def test_annotate_page_filter(tmp_path):
    run_dir = _build_run_dir(tmp_path)
    out_dir = tmp_path / "debug"
    sys.argv = [
        "annotate_stitch",
        "--run-dir", str(run_dir),
        "--output-dir", str(out_dir),
        "--page", "does-not-exist",
    ]
    import scripts.annotate_stitch as annotate

    with pytest.raises(SystemExit):
        annotate.main()
    # The output dir is created, but no pages were annotated: no summary.
    assert not (out_dir / "summary.json").exists()
