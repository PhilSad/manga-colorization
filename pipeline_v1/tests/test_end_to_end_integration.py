"""Real-network end-to-end pipeline test on volume-1 p130 (ch. 5 "Killing
Magic") — the six-panel Flamme/Frieren look-alike page the DET-005..010
evaluation cases are built on.

Deliberately NOT stage-isolated: this runs the FULL four-stage pipeline on one
real page with the same real backends `run.py` builds (`run.build_backends`)
— real YOLO26n panel detection + reading-order extraction, real OpenRouter
`google/gemma-4-31b-it` character detection in the `panel-page-prev2-cast`
variant (`DETECTION_MODE` = `panel-page-prev2` with the chapter-cast
shortlist rendered into the prompt via an explicit `CAST_KEY`, mirroring the
stage-isolated `test_detection_panel_page_prev2_cast`), real FLUX.2 Klein 9B
+ LoRA colorization on the Spark server, and
stitching — and asserts the wiring end to end:

- the panel extraction reproduces the committed fixture set byte-for-byte
  (`tests/data/panels/P130/`, the same crop-stability guarantee the layout
  stage enforces);
- every panel gets a character record with no call errors (identities are
  recorded for provenance but NOT asserted — DET-006..008 are known
  Flamme/Frieren failures owned by the stage-isolated detection suite);
- every panel gets colorized and the output differs from its B&W crop;
- the stitched page preserves the black & white gutters pixel-exactly and
  changes only the panel interiors.

Input: the committed `tests/data/pages/P130.png` — a byte-identical copy of
the gitignored volume page (see `prepare_integration_data.py` provenance) —
copied into the run under its real volume filename (fixture alias basename,
"... - c005 (v01) - p130 ...png") so the run looks like a real volume run;
the chapter-cast shortlist comes from the fixture's `cast_keys` map (passed
explicitly, like the stage-isolated prev2-cast test), so the test needs no
extracted-volume data.

Prerequisites (skipped with a printed reason when missing, like the rest of
the integration suite): Spark FLUX server up (`curl http://spark:3000/healthz`)
and `OPENROUTER_API_KEY` in the repo `.env`.

Cost: 6 OpenRouter detection calls (measured `usage.cost`, ~$0.0005 total) +
6 FLUX calls on the self-hosted server ($0/call, electricity only; the first
call pays the ~1-3 min model load).

Run with: `.venv/bin/pytest pipeline_v1/tests -m integration -k end_to_end`
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config import PipelineConfig
from integration_support import (
    DETECTION_MODEL,
    FLUX_GUIDANCE,
    FLUX_LORA_SCALE,
    FLUX_SEED,
    FLUX_STEPS,
    PANELS_ROOT,
    REFS_DIR,
    committed_page,
    load_fixture,
)
from orchestrator import PipelineRunner
from run import build_backends

pytestmark = pytest.mark.integration

FIXTURE = load_fixture()

# Volume-1 p130 (ch. 5), the DET-005..010 fixture page: 6 panels in reading
# order; the committed per-page set lives at tests/data/panels/P130/.
PAGE_ALIAS = "P130"
EXPECTED_PANELS = 6
CAST_KEY = FIXTURE["cast_keys"][PAGE_ALIAS]  # "c005" — single source of truth
# panel-page-prev2-cast: the prev2 mode with the chapter-cast shortlist
# rendered into the prompt (explicit CAST_KEY, like the stage-isolated
# test_detection_panel_page_prev2_cast). The prev2 strategy's provenance file
# records the effective key, asserted below.
DETECTION_MODE = "panel-page-prev2"   # the detection mode under test
PROVENANCE_FILE = "panel_page_prev2_calls.json"
# The run input is copied under its real volume filename (the fixture alias's
# basename) so the run looks like a real volume run ('- c005 -' tag and all).
INPUT_FILENAME = Path(FIXTURE["aliases"][PAGE_ALIAS]).name


def _panel_stem(index: int) -> str:
    return f"panel_{index:04d}"


def _read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_end_to_end_on_p130(integration_run, openrouter_key,
                                     spark_endpoint):
    """Full pipeline, real backends, one page (volume-1 p130)."""
    page = committed_page(PAGE_ALIAS)          # tests/data/pages/P130.png

    # Input dir containing only this page, copied under its real volume
    # filename (INPUT_FILENAME): the run stays self-contained (page_path
    # inside the run dir) and looks like a real volume run; the cast key is
    # explicit (CAST_KEY), not derived from the filename.
    e2e_root = integration_run.run_dir / "e2e"
    input_dir = e2e_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_image = input_dir / INPUT_FILENAME
    shutil.copy(page, input_image)

    config = PipelineConfig(
        input_dir=input_dir,
        refs_dir=REFS_DIR,
        output_root=e2e_root,
        endpoint=spark_endpoint,
        vlm_model=DETECTION_MODEL,
        sleep_s=0.0,
        detection_mode=DETECTION_MODE,
        cast_key=CAST_KEY,
        flux_steps=FLUX_STEPS,
        guidance_scale=FLUX_GUIDANCE,
        lora_scale=FLUX_LORA_SCALE,
        seed=FLUX_SEED,
        output_format="png",
    )
    ctx = PipelineRunner(config, build_backends(config)).run()

    assert ctx.manifest["status"] == "completed"

    # 1. The four numbered intermediate directories.
    for name in ("1_panels", "2_characters", "3_colorized", "4_stitched"):
        assert (ctx.run_dir / name).is_dir(), f"missing {name}/"

    # Run page stem is the real volume filename (the input copy), not the
    # committed alias stem.
    page_stem = input_image.stem

    # 2. Panels: reading order, geometry and byte-identical crops vs the
    #    committed fixture set (YOLO/ordering drift guard).
    panels_dir = ctx.run_dir / "1_panels" / page_stem
    geometry = _read_json(panels_dir / "panels.json")
    committed_dir = PANELS_ROOT / PAGE_ALIAS
    committed_geometry = _read_json(committed_dir / "panels.json")
    assert geometry["reading_order"] == list(range(1, EXPECTED_PANELS + 1))
    assert len(geometry["detections"]) == EXPECTED_PANELS
    for live, committed in zip(geometry["detections"],
                               committed_geometry["detections"]):
        assert live["box"] == committed["box"], live
        assert live["crop"] == committed["crop"], live
        live_bytes = (panels_dir / live["crop"]).read_bytes()
        committed_bytes = (committed_dir / committed["crop"]).read_bytes()
        assert live_bytes == committed_bytes, (
            f"{live['crop']}: live crop differs from the committed fixture one"
        )

    # 3. Characters: one record per panel, no call errors; identities are
    #    recorded for provenance only (DET-006..008 known failures).
    assert (ctx.run_dir / "2_characters" / "summary.json").is_file()
    chars_dir = ctx.run_dir / "2_characters" / page_stem
    detected: dict[str, list[str]] = {}
    for index in range(1, EXPECTED_PANELS + 1):
        stem = _panel_stem(index)
        doc = _read_json(chars_dir / f"{stem}.json")
        detected[stem] = doc["characters"]
        assert doc["status"] != "error", f"{stem}: {doc.get('error')}"

    # The cast shortlist must actually be applied in the prev2-cast variant:
    # the provenance records the effective cast key (a silent full-roster
    # fallback would defeat the page's purpose — DET-006..010 must not guess
    # Flamme, who is excluded from ch. 5's cast).
    provenance = _read_json(chars_dir / PROVENANCE_FILE)
    assert provenance["cast_key"] == CAST_KEY, (
        f"panel-page-prev2-cast run recorded cast "
        f"{provenance['cast_key']!r}, expected {CAST_KEY!r}"
    )

    # 4. Colorize: one output per panel, each differing from its B&W crop; a
    #    filtered atlas is sent exactly for the panels with characters.
    colorized_dir = ctx.run_dir / "3_colorized" / page_stem
    for index in range(1, EXPECTED_PANELS + 1):
        stem = _panel_stem(index)
        output = colorized_dir / f"{stem}.png"
        crop = panels_dir / f"{stem}.png"
        assert output.is_file(), f"missing colorized {stem}"
        assert output.read_bytes() != crop.read_bytes(), (
            f"{stem}: colorized output identical to the B&W crop"
        )
        atlas = colorized_dir / f"{stem}_atlas.jpg"
        assert atlas.is_file() == bool(detected[stem]), (
            f"{stem}: atlas presence mismatch (characters={detected[stem]})"
        )

    # 5. Stitch: same size as the source; panel interiors changed
    #    (colorized), everything outside the boxes untouched (B&W == source,
    #    pixel-exact).
    stitched_path = ctx.run_dir / "4_stitched" / f"{page_stem}.png"
    assert stitched_path.is_file()
    with Image.open(stitched_path) as stitched_img, Image.open(page) as source_img:
        assert stitched_img.size == source_img.size
        source = np.asarray(source_img.convert("RGB"))
        stitched = np.asarray(stitched_img.convert("RGB"))
        height, width = source.shape[:2]
        inside = np.zeros((height, width), dtype=bool)
        for detection in geometry["detections"]:
            x1, y1, x2, y2 = detection["box"]
            inside[y1:y2, x1:x2] = True
            differing = int(np.count_nonzero(
                np.any(stitched[y1:y2, x1:x2] != source[y1:y2, x1:x2], axis=2)
            ))
            assert differing > 100, (
                f"{detection['crop']}: only {differing} differing pixels "
                "inside its box — the panel was not colorized"
            )
        assert int(np.count_nonzero(inside)) > 0
        assert np.array_equal(stitched[~inside], source[~inside]), (
            "stitched page changed pixels outside the panel boxes"
        )

    # 6. Totals + session-manifest record (cost feeds the suite totals).
    totals = ctx.manifest["totals"]
    assert totals["flux_calls"] == EXPECTED_PANELS
    assert totals["successful_flux_calls"] == EXPECTED_PANELS
    assert totals["panels_colorized"] == EXPECTED_PANELS
    assert totals["pages_stitched"] == 1
    assert totals["character_calls"] >= EXPECTED_PANELS
    assert totals["openrouter_cost_usd"] > 0

    integration_run.record(
        "E2E-P130",
        stage="end-to-end",
        run_dir=str(ctx.run_dir),
        status=ctx.manifest["status"],
        panels=EXPECTED_PANELS,
        detected=detected,
        cost_usd=totals["openrouter_cost_usd"],
        flux_calls=totals["flux_calls"],
        successful_flux_calls=totals["successful_flux_calls"],
        pages_stitched=totals["pages_stitched"],
        wall_time_s=totals["wall_time_s"],
    )
