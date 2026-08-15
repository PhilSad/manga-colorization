"""Offline tests for the verification loop (verify_loop.py + the colorize
step's verification wiring) with mock colorizer + mock verifier, no network.

Covers the loop outcomes (verified / mismatch / verifier_error /
colorize_error), retry semantics (fix prompt appended to the palette
instruction, attempt_<n> images kept, canonical output = final successful
attempt), per-panel provenance (verify.json / fix_prompt.txt), the step
totals, and the full pipeline end-to-end with --verify-attempts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from colorizer import ColorizeRecord
from detection import PanelBox
from mock_backends import MockColorizer, MockVerifier
from orchestrator import Backends, PipelineRunner
from tests.synthetic_page import READING_ORDER, build_page, panel_box
from verify_color import ColorVerifyRecord, VERIFIED, MISMATCH, ERROR
from verify_loop import (
    FIX_HEADER,
    OUTCOME_COLORIZE_ERROR,
    OUTCOME_MISMATCH,
    OUTCOME_VERIFIED,
    OUTCOME_VERIFIER_ERROR,
    _apply_fix,
    run_verify_loop,
)

CHARACTERS_BY_PANEL = {
    "panel_0001": ["Frieren", "Fern"],
    "panel_0002": ["Frieren"],
    "panel_0003": [],
    "panel_0004": ["Fern"],
    "panel_0005": ["Frieren", "Fern"],
}


# ---------------------------------------------------------------------------
# Small helpers / fixtures

def _panel(path, size=(32, 32), color="white"):
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def pipeline_inputs(tmp_path):
    page_path = tmp_path / "pages" / "0134-999.png"
    build_page(page_path)
    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ("frieren_reference.webp", "fern_reference.webp"):
        Image.new("RGB", (8, 8), "gray").save(refs / name)
    return page_path, refs, tmp_path


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
# run_verify_loop unit tests

def _loop_output(tmp_path):
    out = tmp_path / "colorized"
    out.mkdir(exist_ok=True)
    return out / "panel_0001.png"


def test_loop_verified_first_attempt(tmp_path):
    panel = _panel(tmp_path / "panel_0001.png")
    output = _loop_output(tmp_path)
    colorizer = MockColorizer()
    verifier = MockVerifier()  # everything good

    result = run_verify_loop(
        colorizer, verifier, panel, None, output,
        palette_instruction="palette: canonical",
        max_attempts=2,
    )

    assert result.outcome == OUTCOME_VERIFIED
    assert len(result.attempts) == 1
    assert result.colorization_retries == 0
    assert result.verify_calls == 1
    assert result.successful_verify_calls == 1
    assert result.verify_cost_usd == pytest.approx(0.0001, abs=1e-9)
    assert result.fix_prompt == ""
    assert output.is_file()
    assert not (tmp_path / "panel_0001.attempt_2.png").exists()
    # the canonical output is the attempt-1 file itself (no copy needed)
    assert result.colorize.output == output


def test_loop_mismatch_then_verified(tmp_path):
    panel = _panel(tmp_path / "panel_0001.png")
    output = _loop_output(tmp_path)
    colorizer = MockColorizer()
    verifier = MockVerifier(
        {"panel_0001": ("bad-once", "Frieren: hair silver-white, eyes teal")}
    )

    result = run_verify_loop(
        colorizer, verifier, panel, None, output,
        palette_instruction="palette: canonical",
        max_attempts=2,
    )

    assert result.outcome == OUTCOME_VERIFIED
    assert len(result.attempts) == 2
    assert result.colorization_retries == 1
    assert result.verify_calls == 2
    assert result.successful_verify_calls == 1  # only the VERIFIED verdict
    assert result.fix_prompt == "Frieren: hair silver-white, eyes teal"

    # EVERY superseded attempt keeps an image: attempt_1 (the original, which
    # wrote directly to the canonical name) is preserved as attempt_1.png,
    # attempt_2 kept, canonical copied from the final attempt.
    attempt1 = output.with_name("panel_0001.attempt_1.png")
    attempt2 = output.with_name("panel_0001.attempt_2.png")
    assert attempt1.is_file()
    assert attempt2.is_file()
    assert output.read_bytes() == attempt2.read_bytes()
    assert result.colorize.output == output
    # provenance: the attempt-1 record points at the preserved attempt_1 file
    assert result.attempts[0]["colorize"]["output"]["filename"] == "panel_0001.attempt_1.png"

    # the retry prompt carried the fix (authoritative, appended)
    prompts = [c[3] for c in colorizer.calls]
    assert prompts[0] == "palette: canonical"
    assert FIX_HEADER in prompts[1]
    assert "hair silver-white" in prompts[1]

    # attempt records recorded both colorize + verify
    assert result.attempts[0]["verify"]["status"] == MISMATCH
    assert result.attempts[1]["verify"]["status"] == VERIFIED


def test_loop_mismatch_exhausted(tmp_path):
    panel = _panel(tmp_path / "panel_0001.png")
    output = _loop_output(tmp_path)
    colorizer = MockColorizer()
    verifier = MockVerifier({"panel_0001": ("bad", "still wrong")})

    result = run_verify_loop(
        colorizer, verifier, panel, None, output,
        palette_instruction="palette: canonical",
        max_attempts=2,
    )

    assert result.outcome == OUTCOME_MISMATCH
    assert len(result.attempts) == 2
    assert result.colorization_retries == 1
    assert result.verify_calls == 2
    # panel keeps its last colorization as the canonical output, and BOTH
    # superseded attempts keep their images (attempt_1 preserved too).
    assert output.is_file()
    assert output.with_name("panel_0001.attempt_1.png").is_file()
    assert output.with_name("panel_0001.attempt_2.png").is_file()
    assert output.read_bytes() == output.with_name("panel_0001.attempt_2.png").read_bytes()


def test_loop_verify_attempts_one_never_retries(tmp_path):
    """--verify-attempts 1: verify + output the fix prompt, no re-colorize."""
    panel = _panel(tmp_path / "panel_0001.png")
    output = _loop_output(tmp_path)
    colorizer = MockColorizer()
    verifier = MockVerifier({"panel_0001": ("bad", "fix it")})

    result = run_verify_loop(
        colorizer, verifier, panel, None, output, max_attempts=1,
    )

    assert result.outcome == OUTCOME_MISMATCH
    assert len(result.attempts) == 1
    assert result.colorization_retries == 0
    assert result.verify_calls == 1
    assert result.fix_prompt == "fix it"
    assert len(colorizer.calls) == 1
    assert output.is_file()


def test_loop_verifier_error_stops_without_retry(tmp_path):
    """A broken verifier must not burn retries: error verdict stops the loop
    and the panel keeps its latest colorization."""
    panel = _panel(tmp_path / "panel_0001.png")
    output = _loop_output(tmp_path)
    colorizer = MockColorizer()

    class BrokenVerifier:
        def verify(self, colorized, input_crop, atlas=None):
            return ColorVerifyRecord(
                status=ERROR, good_color=None, analyse="", fix_prompt="",
                response_text="boom", usage={}, cost_usd=None,
                cost_source="", latency_s=0.01, model_returned="mock",
                attempts=1, finished_at="now", error="http 500",
            )

    result = run_verify_loop(
        colorizer, BrokenVerifier(), panel, None, output, max_attempts=2,
    )

    assert result.outcome == OUTCOME_VERIFIER_ERROR
    assert len(result.attempts) == 1
    assert result.colorization_retries == 0
    assert result.verify_calls == 1
    assert len(colorizer.calls) == 1
    assert output.is_file()


def test_loop_colorize_error_stops_before_verify(tmp_path):
    panel = _panel(tmp_path / "panel_0001.png")
    output = _loop_output(tmp_path)
    verifier = MockVerifier()

    class FailingColorizer:
        def colorize(self, panel, atlas, output, palette_instruction=""):
            return ColorizeRecord(
                status="error", output=None, requested_size=(32, 32),
                latency_s=0.01, error="connection refused", seed=None,
                original_size=(32, 32), scale=1.0, cap_applied=False,
                max_megapixels=None, model="mock",
            )

    result = run_verify_loop(
        FailingColorizer(), verifier, panel, None, output, max_attempts=2,
    )

    assert result.outcome == OUTCOME_COLORIZE_ERROR
    assert result.verify_calls == 0
    assert result.colorization_retries == 0
    assert not output.exists()


def test_apply_fix():
    assert _apply_fix("palette: canonical", "use teal") == (
        f"palette: canonical\n\n{FIX_HEADER}\nuse teal"
    )
    # empty palette -> the fix block alone (still authoritative)
    assert _apply_fix("", "use teal") == f"{FIX_HEADER}\nuse teal"


# ---------------------------------------------------------------------------
# Full pipeline end-to-end with the verify loop

def test_pipeline_end_to_end_verify_loop(pipeline_inputs, tmp_path):
    from mock_backends import MockCharacterDetector, MockPanelDetector

    page_path, refs, _ = pipeline_inputs
    page_name = page_path.stem
    boxes = [PanelBox(*panel_box(panel_id), 0.95) for panel_id in READING_ORDER]
    verifier = MockVerifier(
        {"panel_0001": ("bad-once", "Frieren: hair silver-white, eyes teal")}
    )
    backends = Backends(
        detector=MockPanelDetector(boxes),
        character_detector=MockCharacterDetector(CHARACTERS_BY_PANEL),
        colorizer=MockColorizer(),
        verifier=verifier,
    )
    config = make_config(tmp_path, refs, verify_attempts=2)
    ctx = PipelineRunner(config, backends).run()

    assert ctx.manifest["status"] == "completed"
    assert ctx.manifest["configuration"]["verify_attempts"] == 2

    # 1. Totals: one retry for panel_0001, everything else verified once.
    totals = ctx.manifest["totals"]
    assert totals["flux_calls"] == 6          # 5 panels + 1 fix retry
    assert totals["successful_flux_calls"] == 6
    assert totals["panels_colorized"] == 5
    assert totals["verify_calls"] == 6
    assert totals["successful_verify_calls"] == 5  # panel_0001's first verdict was a mismatch
    assert totals["verified_panels"] == 5
    assert totals["mismatch_panels"] == 0
    assert totals["verifier_error_panels"] == 0
    assert totals["colorization_retries"] == 1
    assert totals["verify_cost_usd"] == pytest.approx(0.0006, abs=1e-9)

    # 2. Colorized outputs: canonical name for every panel; the retried panel
    #    keeps BOTH superseded attempt images (attempt_1 original + retry).
    colorized_dir = ctx.run_dir / "3_colorized" / page_name
    for i in range(1, 6):
        assert (colorized_dir / f"panel_000{i}.png").is_file()
    assert (colorized_dir / "panel_0001.attempt_1.png").is_file()
    assert (colorized_dir / "panel_0001.attempt_2.png").is_file()
    assert not (colorized_dir / "panel_0002.attempt_2.png").exists()
    assert not (colorized_dir / "panel_0002.attempt_1.png").exists()

    # 3. Fix prompt recorded only for the mismatched panel.
    fix_text = (colorized_dir / "panel_0001.fix_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "silver-white" in fix_text
    assert not (colorized_dir / "panel_0002.fix_prompt.txt").exists()

    # 4. Per-panel verify.json: all attempts, every verify verdict recorded.
    for i in range(1, 6):
        verify_doc = read_json(colorized_dir / f"panel_000{i}.verify.json")
        assert verify_doc["outcome"] == "verified"
        expected_attempts = 2 if i == 1 else 1
        assert len(verify_doc["attempts"]) == expected_attempts
        for attempt in verify_doc["attempts"]:
            assert attempt["colorize"]["status"] == "ok"
            assert attempt["verify"]["status"] in (VERIFIED, MISMATCH)
    first = read_json(colorized_dir / "panel_0001.verify.json")
    assert first["fix_prompt"] == "Frieren: hair silver-white, eyes teal"
    assert first["verify_calls"] == 2

    # 5. The retry colorize call carried the fix prompt (authoritative block).
    prompts = {
        call[0].stem: [c[3] for c in backends.colorizer.calls
                       if c[0].stem == call[0].stem]
        for call in backends.colorizer.calls
    }
    assert len(prompts["panel_0001"]) == 2
    assert FIX_HEADER in prompts["panel_0001"][1]
    assert "silver-white" in prompts["panel_0001"][1]

    # 6. Stitch still produces a complete page.
    assert totals["pages_stitched"] == 1
    assert (ctx.run_dir / "4_stitched" / f"{page_name}.png").is_file()


def test_pipeline_verify_attempts_one_check_only(pipeline_inputs, tmp_path):
    """--verify-attempts 1: verify each panel, output fix prompts, never
    re-colorize. Totals must show no retries and no extra flux calls."""
    from mock_backends import MockCharacterDetector, MockPanelDetector

    page_path, refs, _ = pipeline_inputs
    boxes = [PanelBox(*panel_box(panel_id), 0.95) for panel_id in READING_ORDER]
    verifier = MockVerifier(
        {"panel_0002": ("bad", "Fern: green hair")}
    )
    backends = Backends(
        detector=MockPanelDetector(boxes),
        character_detector=MockCharacterDetector(CHARACTERS_BY_PANEL),
        colorizer=MockColorizer(),
        verifier=verifier,
    )
    config = make_config(tmp_path, refs, verify_attempts=1)
    ctx = PipelineRunner(config, backends).run()

    totals = ctx.manifest["totals"]
    assert totals["flux_calls"] == 5
    assert totals["colorization_retries"] == 0
    assert totals["verify_calls"] == 5
    assert totals["mismatch_panels"] == 1
    assert totals["verified_panels"] == 4
    assert totals["verify_cost_usd"] == pytest.approx(0.0005, abs=1e-9)

    colorized_dir = ctx.run_dir / "3_colorized" / page_path.stem
    assert (colorized_dir / "panel_0002.fix_prompt.txt").is_file()
    assert (colorized_dir / "panel_0002.png").is_file()
    assert not (colorized_dir / "panel_0002.attempt_2.png").exists()
