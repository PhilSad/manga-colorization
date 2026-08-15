"""Tests for the pipeline's PDF export stage (steps/pdf.py -> 6_pdf/).

The final pipeline stage packs every stitched page from `4_stitched/` into a
single multi-page PDF (one page per manga page, filename order = reading
order) using Pillow's native PDF writer — no extra dependency. The standalone
offline tool scripts/make_pdf.py delegates to the same `run_pdf_step`, so
these tests also cover its export.

Pillow 12.x can only *write* PDFs (reading was removed upstream), so the
generated file is verified structurally: `/Count N` for the page count and
`/MediaBox [0 0 W H]` per page (points; pt = px * 72 / dpi).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from PIL import Image

from config import PipelineConfig
from mock_backends import MockCharacterDetector, MockColorizer, MockPanelDetector
from orchestrator import Backends, PipelineRunner
from run_context import RunContext
from steps.pdf import run_pdf_step
from test_orchestrator import make_config, make_refs, make_synthetic_page


# ---------------------------------------------------------------------------
# PDF structure helpers (Pillow writes PDFs but cannot read them back)

def _pdf_page_count(pdf: bytes) -> int:
    match = re.search(rb"/Count\s+(\d+)", pdf)
    return int(match.group(1)) if match else 0


def _pdf_page_sizes(pdf: bytes) -> list[tuple[float, float]]:
    return [
        (float(w), float(h))
        for w, h in re.findall(
            rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf
        )
    ]


def _make_backends():
    from detection import PanelBox

    return Backends(
        detector=MockPanelDetector([
            PanelBox(200, 20, 380, 180, 0.9),
            PanelBox(20, 20, 180, 180, 0.9),
        ]),
        character_detector=MockCharacterDetector(
            {"panel_0001": ["Frieren", "Fern"], "panel_0002": []}
        ),
        colorizer=MockColorizer(),
    )


def _build_minimal_run(tmp_path: Path) -> Path:
    """Run dir with two stitched pages of different sizes, usable directly
    by run_pdf_step (no pipeline run needed)."""
    run_dir = tmp_path / "run"
    stitched = run_dir / "4_stitched"
    stitched.mkdir(parents=True)
    Image.new("RGB", (300, 400), "red").save(stitched / "p001.png")
    Image.new("RGB", (400, 300), "blue").save(stitched / "p002.png")
    (run_dir / "manifest.json").write_text(json.dumps({"steps": {}}))
    return run_dir


# ---------------------------------------------------------------------------
# Tests

def test_full_run_creates_pdf(tmp_path):
    """The final pipeline stage writes 6_pdf/ with a multi-page PDF of every
    stitched page + summary.json; the manifest records pdf_pages."""
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    make_synthetic_page(pages / "p001.png")
    make_synthetic_page(pages / "p002.png")
    config = PipelineConfig(
        input_dir=pages, refs_dir=make_refs(tmp_path),
        output_root=tmp_path / "output", mock=True, sleep_s=0.0,
    )
    ctx = PipelineRunner(config, _make_backends()).run()

    assert ctx.manifest["status"] == "completed"
    assert ctx.manifest["totals"]["pdf_pages"] == 2
    pdf_dir = ctx.run_dir / "6_pdf"
    pdf_path = pdf_dir / "colorized.pdf"
    assert pdf_path.is_file()
    assert _pdf_page_count(pdf_path.read_bytes()) == 2

    record = ctx.manifest["steps"]["pdf"]
    assert record["pages_in_pdf"] == 2
    assert [r["page"] for r in record["records"]] == ["p001", "p002"]
    assert [r["pdf_page"] for r in record["records"]] == [1, 2]

    summary = json.loads((pdf_dir / "summary.json").read_text())
    assert summary["pages_in_pdf"] == 2
    assert summary["pdf"]["path"].endswith("colorized.pdf")


def test_pdf_pages_follow_dpi(tmp_path):
    """Page size in points = pixel size * 72 / dpi: at 72 dpi 1 px = 1 pt, at
    144 dpi the page shrinks by half."""
    run_dir = _build_minimal_run(tmp_path)
    ctx = RunContext.load(run_dir)
    record = run_pdf_step(ctx, PipelineConfig(pdf_dpi=72))
    pdf = (run_dir / "6_pdf" / "colorized.pdf").read_bytes()
    assert _pdf_page_sizes(pdf) == [(300.0, 400.0), (400.0, 300.0)]

    record = run_pdf_step(ctx, PipelineConfig(pdf_dpi=144))
    pdf = (run_dir / "6_pdf" / "colorized.pdf").read_bytes()
    assert _pdf_page_sizes(pdf) == [(150.0, 200.0), (200.0, 150.0)]
    assert record["pages_in_pdf"] == 2


def test_pdf_custom_name_and_override_output_dir(tmp_path):
    run_dir = _build_minimal_run(tmp_path)
    ctx = RunContext.load(run_dir)
    out_dir = tmp_path / "custom"
    record = run_pdf_step(
        ctx, PipelineConfig(pdf_name="volume-1.pdf"),
        output_dir=out_dir,
    )
    assert (out_dir / "volume-1.pdf").is_file()
    assert record["output"]["path"].endswith("volume-1.pdf")
    assert (out_dir / "summary.json").is_file()


def test_pdf_page_substrings_filter(tmp_path):
    run_dir = _build_minimal_run(tmp_path)
    ctx = RunContext.load(run_dir)
    record = run_pdf_step(ctx, PipelineConfig(), page_substrings=("p002",))
    assert record["pages_in_pdf"] == 1
    assert record["records"][0]["page"] == "p002"
    pdf = (run_dir / "6_pdf" / "colorized.pdf").read_bytes()
    assert _pdf_page_count(pdf) == 1


def test_pdf_without_stitch_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({}))
    ctx = RunContext.load(run_dir)
    with pytest.raises(ValueError, match="stitch"):
        run_pdf_step(ctx, PipelineConfig())


def test_pdf_only_panels_selection(tmp_path):
    """A targeted rerun exports only the selected pages."""
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    make_synthetic_page(pages / "p001.png")
    make_synthetic_page(pages / "p002.png")
    config = PipelineConfig(
        input_dir=pages, refs_dir=make_refs(tmp_path),
        output_root=tmp_path / "output", mock=True, sleep_s=0.0,
        only_panels=("p001:panel_0001", "p001:panel_0002"),
    )
    ctx = PipelineRunner(config, _make_backends()).run()

    assert ctx.manifest["totals"]["pdf_pages"] == 1
    pdf = (ctx.run_dir / "6_pdf" / "colorized.pdf").read_bytes()
    assert _pdf_page_count(pdf) == 1


def test_make_pdf_script_cli(tmp_path, monkeypatch):
    """scripts/make_pdf.py exports a completed run with custom options."""
    run_dir = _build_minimal_run(tmp_path)
    out_dir = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", [
        "make_pdf",
        "--run-dir", str(run_dir),
        "--output-dir", str(out_dir),
        "--name", "volume.pdf",
        "--dpi", "144",
        "--page", "p002",
    ])
    import scripts.make_pdf as make_pdf

    make_pdf.main()
    assert (out_dir / "volume.pdf").is_file()
    pdf = (out_dir / "volume.pdf").read_bytes()
    assert _pdf_page_count(pdf) == 1
    assert _pdf_page_sizes(pdf) == [(200.0, 150.0)]
    assert (out_dir / "summary.json").is_file()


def test_make_pdf_script_empty_filter_exits(tmp_path, monkeypatch):
    run_dir = _build_minimal_run(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "make_pdf",
        "--run-dir", str(run_dir),
        "--page", "does-not-exist",
    ])
    import scripts.make_pdf as make_pdf

    with pytest.raises(SystemExit):
        make_pdf.main()
    assert not (run_dir / "6_pdf" / "summary.json").exists()
