"""Real-network layout integration tests (LAY-001, LAY-002).

Stage-isolated: the layout stage's unit of input IS the full page, so these
tests run the **real** YOLO26n panel detection (reusing the pipeline's
`steps.panels.run_panels_step`, including the blank-ink check and the
full-page fallback) on the committed page / a generated white page:

- LAY-001: p006 (full-page illustration, no panel frames) must fall back to
  one synthetic full-page panel (provenance `full-page-fallback`), not be
  classified as blank.
- LAY-002: a generated all-white page must yield zero panels with an
  explicit `blank-page` skip reason.

No API key needed (local YOLO); artifacts land in the timestamped run dir.
"""

from __future__ import annotations

import json
import shutil

import pytest
from PIL import Image

from config import PipelineConfig
from integration_support import (
    REFS_DIR,
    case_by_id,
    load_fixture,
    page_path,
    write_json,
)
from run_context import RunContext

pytestmark = pytest.mark.integration


def _run_real_panels(page: "object", run_dir, name: str) -> dict:
    """Run the real panels step (YOLO + reading order + blank/fallback
    policy) on `page` inside `run_dir`. Returns the written panels.json."""
    from detection import YoloPanelDetector
    from steps.panels import run_panels_step

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(page, "save"):  # a PIL image: save it as the input page
        target = input_dir / f"{name}.png"
        page.save(target)
    else:
        target = input_dir / name
        shutil.copy(page, target)

    ctx = RunContext(run_dir / "pipeline_run")
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        input_dir=input_dir,
        refs_dir=REFS_DIR,
        output_root=run_dir / "output",
        endpoint=None,
        mock=True,
    )
    run_panels_step(ctx, config, YoloPanelDetector())

    page_dir = ctx.step_dir("panels") / target.stem
    return json.loads((page_dir / "panels.json").read_text(encoding="utf-8"))


def test_lay_001_full_page_fallback(integration_run):
    fixture = load_fixture()
    case = case_by_id(fixture, "LAY-001")
    expected = case["expected"]
    page = page_path("lay_001_page.png")

    case_dir = integration_run.run_dir / "layout" / "LAY-001"
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(page, case_dir / "input.png")
    geometry = _run_real_panels(page, case_dir, page.name)
    write_json(case_dir / "panels.json", geometry)

    detections = geometry.get("detections", [])
    integration_run.record(
        "LAY-001",
        stage="layout",
        failure=case.get("failure"),
        panels=len(detections),
        blank_page=geometry.get("blank_page"),
        skip_reason=geometry.get("skip_reason"),
        provenance=detections[0].get("provenance") if detections else None,
        box=detections[0].get("box") if detections else None,
        crop=detections[0].get("crop") if detections else None,
    )

    assert len(detections) == 1, f"LAY-001: expected 1 panel, got {len(detections)}"
    assert detections[0]["box"] == expected["box"], detections[0]["box"]
    assert detections[0]["provenance"] == "full-page-fallback"
    assert detections[0].get("crop") == f"{expected['crop']}.png"
    assert geometry.get("blank_page") is False


def test_lay_002_blank_page_skip(integration_run):
    fixture = load_fixture()
    case = case_by_id(fixture, "LAY-002")
    expected = case["expected"]
    name = case["input"]["generate"]["name"]
    width, height = case["input"]["generate"]["size"]

    case_dir = integration_run.run_dir / "layout" / "LAY-002"
    case_dir.mkdir(parents=True, exist_ok=True)
    white = Image.new("RGB", (width, height), "white")
    white.save(case_dir / name)
    geometry = _run_real_panels(white, case_dir, name)
    write_json(case_dir / "panels.json", geometry)

    detections = geometry.get("detections", [])
    integration_run.record(
        "LAY-002",
        stage="layout",
        failure=case.get("failure"),
        panels=len(detections),
        blank_page=geometry.get("blank_page"),
        skip_reason=geometry.get("skip_reason"),
    )

    assert len(detections) == 0, f"LAY-002: expected 0 panels, got {len(detections)}"
    assert geometry.get("blank_page") is True
    assert geometry.get("skip_reason") == "blank-page"
