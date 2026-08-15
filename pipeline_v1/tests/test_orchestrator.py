"""Tests for orchestrator.py: step sequencing, manifest totals, resume, and
failure handling (all mock backends, offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from config import PipelineConfig
from mock_backends import (
    MockCharacterDetector,
    MockColorizer,
    MockPanelDetector,
)
from orchestrator import Backends, PipelineRunner
from run_context import RunContext


# ---------------------------------------------------------------------------
# Fixtures

def make_synthetic_page(path: Path, size=(400, 400)) -> Path:
    """Page with two gray rectangles (stand-in panels)."""
    from PIL import ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 180, 180), fill=(128, 128, 128))
    draw.rectangle((200, 20, 380, 180), fill=(64, 64, 64))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def make_refs(tmp_path: Path) -> Path:
    refs = tmp_path / "refs"
    refs.mkdir(exist_ok=True)
    for name in ("frieren_reference.webp", "fern_reference.webp"):
        Image.new("RGB", (8, 8), "gray").save(refs / name)
    return refs


def make_config(tmp_path: Path, **overrides) -> PipelineConfig:
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)
    make_synthetic_page(pages / "p001.png")
    make_synthetic_page(pages / "p002.png")
    refs = make_refs(tmp_path)
    base = dict(
        input_dir=pages,
        refs_dir=refs,
        output_root=tmp_path / "output",
        mock=True,
        sleep_s=0.0,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def make_backends(by_panel: dict[str, list[str]] | None = None):
    return Backends(
        detector=MockPanelDetector(
            [
                # right panel -> #1, left panel -> #2 (reading order)
                _box(200, 20, 380, 180),
                _box(20, 20, 180, 180),
            ]
        ),
        character_detector=MockCharacterDetector(
            by_panel or {"panel_0001": ["Frieren", "Fern"], "panel_0002": []}
        ),
        colorizer=MockColorizer(),
    )


def _box(x1, y1, x2, y2):
    from detection import PanelBox

    return PanelBox(x1, y1, x2, y2, 0.9)


def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests

def test_full_run_success(tmp_path):
    config = make_config(tmp_path)
    backends = make_backends()
    ctx = PipelineRunner(config, backends).run()

    assert ctx.manifest["status"] == "completed"
    assert (ctx.run_dir / "1_panels").is_dir()
    assert (ctx.run_dir / "2_characters").is_dir()
    assert (ctx.run_dir / "3_colorized").is_dir()
    assert (ctx.run_dir / "4_stitched").is_dir()
    assert (ctx.run_dir / "5_debug").is_dir()
    assert (ctx.run_dir / "6_pdf").is_dir()

    # Two pages x two panels.
    totals = ctx.manifest["totals"]
    assert totals["character_calls"] == 4
    assert totals["successful_character_calls"] == 4
    assert totals["openrouter_cost_usd"] == pytest.approx(0.0004, abs=1e-9)
    assert totals["flux_calls"] == 4
    assert totals["successful_flux_calls"] == 4
    assert totals["panels_colorized"] == 4
    assert totals["pages_stitched"] == 2
    assert totals["pages_annotated"] == 2
    assert totals["pdf_pages"] == 2
    assert "wall_time_s" in totals

    # Step records are present in the manifest.
    assert set(ctx.manifest["steps"].keys()) == {
        "panels", "characters", "colorize", "stitch", "debug", "pdf",
    }


def test_full_run_stitched_pixels(tmp_path):
    config = make_config(tmp_path)
    backends = make_backends()
    ctx = PipelineRunner(config, backends).run()

    stitched = ctx.run_dir / "4_stitched" / "p001.png"
    with Image.open(stitched) as image:
        # The mock blends its tint over the gray panels: warm pixels inside
        # both boxes, untouched white outside.
        for position in ((300, 100), (100, 100)):
            r, g, b = image.getpixel(position)
            assert r > g == b, f"expected warm tint at {position}, got {(r, g, b)}"
        assert image.getpixel((200, 300)) == (255, 255, 255)  # untouched


def test_filtered_atlas_used_only_when_characters_detected(tmp_path):
    config = make_config(tmp_path)
    backends = make_backends()
    ctx = PipelineRunner(config, backends).run()

    colorizer: MockColorizer = backends.colorizer
    # panel_0001 has characters -> atlas sent; panel_0002 none -> no atlas.
    for panel, atlas, _output, _palette in colorizer.calls:
        assert (panel.stem == "panel_0001") == (atlas is not None)
        if atlas is not None:
            assert atlas.is_file()
            assert "panel_0001_atlas" in atlas.name


def test_steps_subset(tmp_path):
    config = make_config(tmp_path, steps=("panels", "characters"))
    ctx = PipelineRunner(config, make_backends()).run()
    assert ctx.manifest["status"] == "completed"
    assert set(ctx.manifest["steps"].keys()) == {"panels", "characters"}
    assert ctx.manifest["totals"]["character_calls"] == 4
    assert ctx.manifest["totals"]["flux_calls"] == 0


def test_stitch_without_colorize_fails_and_keeps_artifacts(tmp_path):
    config = make_config(tmp_path, steps=("panels", "characters", "stitch"))
    runner = PipelineRunner(config, make_backends())
    with pytest.raises(ValueError):
        runner.run()
    ctx = RunContext.load(_latest_run(tmp_path / "output"))
    assert ctx.manifest["status"] == "failed"
    assert "colorize" in (ctx.manifest.get("error") or "")
    # Earlier step artifacts are preserved.
    assert (ctx.run_dir / "1_panels").is_dir()
    assert (ctx.run_dir / "2_characters").is_dir()


def test_failure_records_error(tmp_path):
    class BoomColorizer:
        def colorize(self, panel, atlas, output, palette_instruction=""):
            raise RuntimeError("boom")

    backends = make_backends()
    backends.colorizer = BoomColorizer()
    config = make_config(tmp_path)
    with pytest.raises(RuntimeError):
        PipelineRunner(config, backends).run()
    ctx = RunContext.load(_latest_run(tmp_path / "output"))
    assert ctx.manifest["status"] == "failed"
    assert "RuntimeError: boom" in ctx.manifest["error"]
    # panels + characters artifacts kept even though colorize blew up.
    assert (ctx.run_dir / "1_panels").is_dir()
    assert (ctx.run_dir / "2_characters").is_dir()
    assert not (ctx.run_dir / "4_stitched" / "p001.png").exists()


def test_from_step_skips_earlier(tmp_path):
    config = make_config(tmp_path, from_step="colorize")
    # from_step skips panels/characters, but colorize needs panels -> fails.
    with pytest.raises(ValueError):
        PipelineRunner(config, make_backends()).run()
    ctx = RunContext.load(_latest_run(tmp_path / "output"))
    assert ctx.manifest["status"] == "failed"


def test_resume_copies_previous_outputs(tmp_path):
    config = make_config(tmp_path)
    first = PipelineRunner(config, make_backends()).run()
    # Re-run with resume: everything is copied, so no step re-runs.
    config2 = make_config(tmp_path, resume=first.run_dir)
    ctx2 = PipelineRunner(config2, make_backends()).run()
    assert ctx2.manifest["status"] == "completed"
    assert ctx2.run_dir != first.run_dir
    # All step dirs were copied from the first run.
    for name in ("1_panels", "2_characters", "3_colorized", "4_stitched",
                 "5_debug", "6_pdf"):
        assert (ctx2.run_dir / name).is_dir()
    # The new run has no fresh calls (all steps skipped).
    totals = ctx2.manifest["totals"]
    assert totals["character_calls"] == 0
    assert totals["flux_calls"] == 0


def test_run_dirs_never_overwritten(tmp_path):
    config = make_config(tmp_path)
    first = PipelineRunner(config, make_backends()).run()
    second = PipelineRunner(config, make_backends()).run()
    assert first.run_dir != second.run_dir
    assert first.run_dir.is_dir()
    assert second.run_dir.is_dir()


# ---------------------------------------------------------------------------
# V1.1 targeted reruns (task 0001)

def test_resume_from_step_copies_only_earlier_steps(tmp_path):
    """`--resume RUN --from-step colorize` copies panels+characters but NOT
    colorize/stitch outputs; colorize/stitch are regenerated in the fresh
    run (task 0001)."""
    config = make_config(tmp_path)
    first = PipelineRunner(config, make_backends()).run()

    config2 = make_config(tmp_path, resume=first.run_dir, from_step="colorize")
    backends2 = make_backends()
    ctx2 = PipelineRunner(config2, backends2).run()

    assert ctx2.manifest["status"] == "completed"
    assert ctx2.run_dir != first.run_dir
    # Earlier steps reused from the resume dir.
    assert (ctx2.run_dir / "1_panels").is_dir()
    assert (ctx2.run_dir / "2_characters").is_dir()
    # No fresh character calls (reused), all panels regenerated.
    totals = ctx2.manifest["totals"]
    assert totals["character_calls"] == 0
    assert totals["flux_calls"] == 4
    assert totals["pages_stitched"] == 2
    assert len(backends2.colorizer.calls) == 4


def test_resume_from_step_colorize_does_not_copy_later_outputs(tmp_path):
    """Later-stage outputs (3_colorized, 4_stitched) must NOT be copied into
    the fresh run by --resume --from-step."""
    config = make_config(tmp_path)
    first = PipelineRunner(config, make_backends()).run()
    # Marker files inside the first run's colorize/stitch dirs.
    (first.run_dir / "3_colorized" / "marker.txt").write_text("v1")
    (first.run_dir / "4_stitched" / "marker.txt").write_text("v1")

    config2 = make_config(tmp_path, resume=first.run_dir, from_step="colorize")
    ctx2 = PipelineRunner(config2, make_backends()).run()

    assert not (ctx2.run_dir / "3_colorized" / "marker.txt").exists()
    assert not (ctx2.run_dir / "4_stitched" / "marker.txt").exists()
    # Reused earlier outputs ARE copied.
    assert (ctx2.run_dir / "1_panels" / "p001").is_dir()


def test_only_panel_targeted_rerun(tmp_path):
    """A targeted rerun processes only the selected panels while stitching the
    correct page (task 0001)."""
    config = make_config(tmp_path)
    first = PipelineRunner(config, make_backends()).run()

    # Re-run: resume panels/characters, colorize only p001:panel_0001.
    config2 = make_config(
        tmp_path,
        resume=first.run_dir,
        from_step="colorize",
        only_panels=("p001:panel_0001",),
    )
    backends2 = make_backends()
    ctx2 = PipelineRunner(config2, backends2).run()

    assert ctx2.manifest["status"] == "completed"
    # Only one panel was re-colorized.
    assert len(backends2.colorizer.calls) == 1
    assert backends2.colorizer.calls[0][0].name == "panel_0001.png"
    assert ctx2.manifest["totals"]["flux_calls"] == 1
    # Non-selected panels were reused from the resume dir so the page stitches.
    assert ctx2.manifest["totals"]["pages_stitched"] == 1
    stitched = ctx2.run_dir / "4_stitched" / "p001.png"
    assert stitched.is_file()
    # The other page was not touched.
    assert not (ctx2.run_dir / "4_stitched" / "p002.png").exists()


def test_forced_characters_skip_paid_calls(tmp_path):
    """Ground-truth identities never make a paid detection call (task 0001)."""
    config = make_config(
        tmp_path,
        only_panels=("p001:panel_0001",),
        forced_characters={"p001:panel_0001": ["Frieren"]},
        steps=("panels", "characters"),
    )
    backends = make_backends()
    ctx = PipelineRunner(config, backends).run()

    totals = ctx.manifest["totals"]
    assert totals["character_calls"] == 0
    assert totals["successful_character_calls"] == 0
    assert totals["openrouter_cost_usd"] == 0.0
    assert totals["forced_character_panels"] == 1
    # The forced identity is available to later steps.
    chars_dir = ctx.run_dir / "2_characters" / "p001"
    doc = json.loads((chars_dir / "panel_0001.json").read_text(encoding="utf-8"))
    assert doc["status"] == "forced"
    assert doc["characters"] == ["Frieren"]


def test_only_panel_without_resume_still_stitches_selected_page(tmp_path):
    """A targeted run from scratch processes only selected pages/panels."""
    config = make_config(
        tmp_path,
        only_panels=("p001:panel_0001", "p001:panel_0002"),
    )
    backends = make_backends()
    ctx = PipelineRunner(config, backends).run()

    assert ctx.manifest["status"] == "completed"
    totals = ctx.manifest["totals"]
    # Only p001 processed: 2 panels -> 2 character + 2 flux calls.
    assert totals["character_calls"] == 2
    assert totals["flux_calls"] == 2
    assert totals["pages_stitched"] == 1
    assert (ctx.run_dir / "4_stitched" / "p001.png").is_file()
    assert not (ctx.run_dir / "4_stitched" / "p002.png").exists()


def test_only_panel_partial_page_stitches_selected_only(tmp_path):
    """A from-scratch targeted run selecting one panel of a two-panel page
    stitches that panel and leaves the other black & white (task 0001)."""
    config = make_config(
        tmp_path,
        only_panels=("p001:panel_0001",),
    )
    backends = make_backends()
    ctx = PipelineRunner(config, backends).run()

    assert ctx.manifest["status"] == "completed"
    totals = ctx.manifest["totals"]
    assert totals["character_calls"] == 1
    assert totals["flux_calls"] == 1
    assert totals["pages_stitched"] == 1
    stitch_record = ctx.manifest["steps"]["stitch"]["outputs"][0]
    assert stitch_record["panels_stitched"] == 1
    assert stitch_record["panels_skipped_black_white"] == ["panel_0002.png"]
    with Image.open(ctx.run_dir / "4_stitched" / "p001.png") as image:
        # Selected panel (right, x~300) colorized; un-selected (left) B&W.
        r, g, b = image.getpixel((300, 100))
        assert r > g == b
        assert image.getpixel((100, 100)) == (128, 128, 128)  # untouched gray


# ---------------------------------------------------------------------------
# V1.1 page-level detection (task 0003) through the orchestrator

def test_page_mode_full_run_totals(tmp_path):
    """One page-level call per page; manifest totals split page/fallback."""
    from mock_backends import MockPageCharacterDetector

    config = make_config(tmp_path, detection_mode="page")
    page_detector = MockPageCharacterDetector({
        "p001": {
            "panel_0001": (["Frieren", "Fern"], False),
            "panel_0002": ([], False),
        },
        "p002": {
            "panel_0001": (["Frieren"], False),
            "panel_0002": (["Fern"], True),  # uncertain -> fallback
        },
    })
    backends = Backends(
        detector=MockPanelDetector(
            [_box(200, 20, 380, 180), _box(20, 20, 180, 180)]
        ),
        character_detector=page_detector,
        colorizer=MockColorizer(),
    )
    ctx = PipelineRunner(config, backends).run()

    assert ctx.manifest["status"] == "completed"
    totals = ctx.manifest["totals"]
    assert totals["character_calls"] == 3      # 2 page + 1 fallback
    assert totals["page_character_calls"] == 2
    assert totals["fallback_character_calls"] == 1
    assert totals["openrouter_cost_usd"] == pytest.approx(0.0005, abs=1e-9)
    assert totals["flux_calls"] == 4
    assert totals["pages_stitched"] == 2
    # The fallback panel carries the fallback identity into colorization.
    colorizer: MockColorizer = backends.colorizer
    panel_names = {panel.stem for panel, _a, _o, _p in colorizer.calls}
    assert panel_names == {"panel_0001", "panel_0002"}


def test_colorize_records_palette_and_profiles_hash(tmp_path):
    """Task 0002 provenance: palette instruction + profiles hash recorded."""
    config = make_config(tmp_path)
    backends = make_backends()
    ctx = PipelineRunner(config, backends).run()

    records = ctx.manifest["steps"]["colorize"]["records"]
    with_profile = [r for r in records if r["characters"]]
    assert with_profile, "expected at least one record with characters"
    doc = with_profile[0]
    assert "Frieren" in doc["palette_instruction"]
    assert len(doc["profiles_sha256"]) == 64
    assert doc["unknown_characters"] == []
    # prompt/profile hashes in the manifest header
    hashes = ctx.manifest["prompt_hashes"]
    assert len(hashes["colorizer_prompt_sha256"]) == 64
    assert len(hashes["profiles_sha256"]) == 64


def test_forced_characters_via_cli_parse():
    from config import parse_args

    config = parse_args([
        "--mock", "--limit", "1",
        "--only-panel", "P003:panel_0006",
        "--force-characters", "P003:panel_0006=Frieren",
    ])
    assert config.only_panels == ("P003:panel_0006",)
    assert config.forced_characters == {"P003:panel_0006": ["Frieren"]}


def _latest_run(output_root: Path) -> Path:
    runs = sorted(output_root.iterdir())
    assert runs
    return runs[-1]
