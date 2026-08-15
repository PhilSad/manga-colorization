"""Full-page mode (`--full-page`, gpt-image-2 atlas pipeline) end-to-end
tests with mock backends: the orchestrator runs all five stages against a
tiny fake page set and we assert on the run directory layout, manifest
totals (gpt_image_* vs flux_*), the characters-step no-op for
`--atlas-source cast`, and the parallel colorize path. Fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from config import parse_args
from mock_backends import MockVerifier
from orchestrator import PipelineRunner
from run import build_backends


def make_page(tmp_path: Path, name: str, size=(500, 700)) -> Path:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 400, 600), outline="black", width=4)
    draw.line((150, 150, 350, 350), fill="black", width=4)
    path = tmp_path / "pages" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def make_ref(tmp_path: Path, name: str, size=(100, 150)) -> Path:
    image = Image.new("RGB", size, (40, 120, 200))
    path = tmp_path / "refs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_full_page(tmp_path: Path, pages: int, extra_args: list[str] | None = None):
    """Build pages + run the pipeline with `--full-page --mock` and any
    extra args. Returns (ctx, config, backends)."""
    make_page(tmp_path, "p001.png")
    if pages >= 2:
        make_page(tmp_path, "p002.png")
    if pages >= 3:
        make_page(tmp_path, "p003.png")
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)  # refs_dir must exist
    argv = [
        "--input-dir", str(tmp_path / "pages"),
        "--refs-dir", str(tmp_path / "refs"),
        "--output-root", str(tmp_path / "output"),
        "--mock",
        "--full-page",
        *(extra_args or []),
    ]
    config = parse_args(argv)
    backends = build_backends(config)
    runner = PipelineRunner(config, backends)
    ctx = runner.run()
    return ctx, config, backends


# ---------------------------------------------------------------------------
# --atlas-source detected (default): one VLM page call, full pipeline

def test_full_page_detected_full_pipeline(tmp_path):
    ctx, config, _ = run_full_page(tmp_path, pages=2)

    assert ctx.manifest["status"] == "completed"
    assert ctx.manifest["configuration"]["full_page"] is True
    assert ctx.manifest["configuration"]["atlas_source"] == "detected"
    # --full-page --atlas-source detected forces the page-level mode.
    assert ctx.manifest["configuration"]["detection_mode"] == "page"

    # Panels: one synthetic full-page panel per page.
    for page in ("p001", "p002"):
        geometry = read_json(ctx.run_dir / "1_panels" / page / "panels.json")
        assert len(geometry["detections"]) == 1
        assert geometry["detections"][0]["provenance"] == "full-page-mode"
        assert geometry["detections"][0]["crop"] == "panel_0001.png"
        assert geometry["blank_page"] is False

    # Characters: page-level detection ran (not the cast no-op).
    characters_summary = read_json(
        ctx.run_dir / "2_characters" / "summary.json"
    )
    assert "skipped" not in characters_summary
    assert "records" in characters_summary

    # Colorize: one gpt-image-2 mock call per page.
    colorize_summary = read_json(ctx.run_dir / "3_colorized" / "summary.json")
    assert len(colorize_summary["records"]) == 2
    assert all(r["status"] == "ok" for r in colorize_summary["records"])
    assert all(r["model"] == "gpt-image-2 (mock)" for r in colorize_summary["records"])
    for page in ("p001", "p002"):
        out = ctx.run_dir / "3_colorized" / page / "panel_0001.png"
        assert out.is_file()
        with Image.open(out) as image:
            assert image.size == (500, 700)

    # Stitch + debug produced outputs the same size as the page.
    for page in ("p001", "p002"):
        stitched = ctx.run_dir / "4_stitched" / f"{page}.png"
        assert stitched.is_file()
        with Image.open(stitched) as image:
            assert image.size == (500, 700)
    debug_summary = read_json(ctx.run_dir / "5_debug" / "summary.json")
    assert debug_summary["pages_annotated"] == 2
    assert (ctx.run_dir / "5_debug" / "p001.png").is_file()

    # Totals: gpt-image-2 accounting, not FLUX.
    totals = ctx.manifest["totals"]
    assert totals["gpt_image_calls"] == 2
    assert totals["successful_gpt_image_calls"] == 2
    assert totals["gpt_image_cost_usd"] == 0.0     # mock records carry no usage
    assert totals["flux_calls"] == 0
    # distinct panels, not calls: 2 pages colorized in 2 calls, no retries
    assert totals["panels_colorized"] == 2
    # Mock mode records the mock pricing note (no external calls, no cost).
    pricing = ctx.manifest["pricing_assumptions"]
    assert pricing["note"].startswith("mock backends")


# ---------------------------------------------------------------------------
# --atlas-source cast: zero VLM calls, chapter-cast atlas

def test_full_page_cast_zero_vlm_calls(tmp_path):
    make_ref(tmp_path, "Frieren_reference.png")
    make_ref(tmp_path, "Stark_reference.png")
    ctx, config, _ = run_full_page(
        tmp_path, pages=1, extra_args=["--atlas-source", "cast"]
    )

    # Characters step is a documented no-op; no OpenRouter calls or cost.
    characters_summary = read_json(
        ctx.run_dir / "2_characters" / "summary.json"
    )
    assert characters_summary["skipped"] is True
    assert characters_summary["reason"].startswith("atlas-source cast")
    totals = ctx.manifest["totals"]
    assert totals["character_calls"] == 0
    assert totals["openrouter_cost_usd"] == 0.0

    # The colorize step derives the cast: no chapter derivable for the tmp
    # page name -> full canonical roster from the refs dir.
    colorize_summary = read_json(ctx.run_dir / "3_colorized" / "summary.json")
    assert len(colorize_summary["records"]) == 1
    record = colorize_summary["records"][0]
    assert record["status"] == "ok"
    assert record["characters"] == ["Frieren", "Stark"]
    # The labelled atlas for the cast was built and saved next to the output.
    assert (ctx.run_dir / "3_colorized" / "p001" / "panel_0001_atlas.jpg").is_file()
    assert totals["gpt_image_calls"] == 1


# ---------------------------------------------------------------------------
# parallel colorization (--worker-colorization)

def test_full_page_parallel_colorize(tmp_path):
    ctx, _, _ = run_full_page(tmp_path, pages=3, extra_args=["--worker-colorization", "3"])

    colorize_summary = read_json(ctx.run_dir / "3_colorized" / "summary.json")
    assert len(colorize_summary["records"]) == 3
    assert all(r["status"] == "ok" for r in colorize_summary["records"])
    totals = ctx.manifest["totals"]
    assert totals["gpt_image_calls"] == 3
    assert totals["successful_gpt_image_calls"] == 3
    for page in ("p001", "p002", "p003"):
        assert (ctx.run_dir / "3_colorized" / page / "panel_0001.png").is_file()
        assert (ctx.run_dir / "4_stitched" / f"{page}.png").is_file()


# ---------------------------------------------------------------------------
# verify loop in full-page mode: retry accounting (calls vs distinct panels,
# per-attempt costs) must hold with the gpt-image-2 backend too.

def test_full_page_verify_retry_totals(tmp_path):
    make_page(tmp_path, "p001.png")
    make_page(tmp_path, "p002.png")
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    argv = [
        "--input-dir", str(tmp_path / "pages"),
        "--refs-dir", str(tmp_path / "refs"),
        "--output-root", str(tmp_path / "output"),
        "--mock",
        "--full-page",
        "--verify-attempts", "3",
    ]
    config = parse_args(argv)
    backends = build_backends(config)
    # Both full-page panels share the stem panel_0001: "bad-once" makes the
    # first verify call mismatch (-> one retry) and later calls good.
    backends.verifier = MockVerifier(by_panel={"panel_0001": ("bad-once", "fix: canonical palette")})
    runner = PipelineRunner(config, backends)
    ctx = runner.run()

    assert ctx.manifest["status"] == "completed"
    totals = ctx.manifest["totals"]
    # p001 colorizes twice (retry), p002 once -> 3 gpt-image-2 calls,
    # but only 2 distinct panels colorized.
    assert totals["gpt_image_calls"] == 3
    assert totals["successful_gpt_image_calls"] == 3
    assert totals["panels_colorized"] == 2
    assert totals["colorization_retries"] == 1
    assert totals["verify_calls"] == 3
    assert totals["successful_verify_calls"] == 2
    assert totals["verified_panels"] == 2
    assert totals["mismatch_panels"] == 0
    assert totals["verifier_error_panels"] == 0
    # Retry images are kept: every superseded attempt keeps an image, so the
    # original (attempt_1) is preserved too — the canonical name is a copy of
    # the final attempt, which is what the stitch step consumes.
    colorized_p001 = ctx.run_dir / "3_colorized" / "p001"
    assert (colorized_p001 / "panel_0001.attempt_1.png").is_file()
    assert (colorized_p001 / "panel_0001.attempt_2.png").is_file()
    assert (colorized_p001 / "panel_0001.png").is_file()
    # The retry's verify loop is recorded on the p001 page record.
    p001_rec = next(
        r for r in ctx.manifest["steps"]["colorize"]["records"]
        if r["page"] == "p001"
    )
    assert p001_rec["verify_loop"]["outcome"] == "verified"
    assert p001_rec["verify_loop"]["verify_calls"] == 2
    assert len(p001_rec["verify_loop"]["attempts"]) == 2


# ---------------------------------------------------------------------------
# sum_gpt_image_cost: retry attempts must be counted even though the top-level
# record only carries the final attempt's cost.

def test_sum_gpt_image_cost_includes_retry_attempts():
    from orchestrator import sum_gpt_image_cost

    records = [
        # No verify loop: cost lives on the record itself.
        {"est_cost_usd": 0.05},
        # Verify loop: final top-level cost (0.06) plus attempt 1 (0.04).
        {
            "est_cost_usd": 0.06,
            "verify_loop": {
                "attempts": [
                    {"colorize": {"est_cost_usd": 0.04}},
                    {"colorize": {"est_cost_usd": 0.06}},
                ]
            },
        },
        # Missing usage (mock/failed) contributes nothing.
        {"est_cost_usd": None},
        {"verify_loop": {"attempts": [{"colorize": {}}]}},
    ]
    assert sum_gpt_image_cost(records) == pytest.approx(0.05 + 0.04 + 0.06)


# ---------------------------------------------------------------------------
# CLI validation already tested in test_config.py; one belt-and-braces check
# that `--atlas-source cast` without `--full-page` is rejected.

def test_cast_without_full_page_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--atlas-source", "cast"])
