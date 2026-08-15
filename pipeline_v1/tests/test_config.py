"""Tests for config.py: CLI parsing, validation, step selection, size policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    GPT_IMAGE_MAX_EDGE,
    GPT_IMAGE_MAX_PIXELS,
    GPT_IMAGE_MIN_PIXELS,
    STEP_ORDER,
    PipelineConfig,
    minimal_gpt_image_size,
    nearest_multiple_of,
    parse_args,
    parse_gpt_size,
    parse_steps,
    requested_panel_size,
)


def test_parse_args_defaults():
    config = parse_args([])
    assert config.steps == STEP_ORDER
    assert config.skip_first == 0
    assert config.limit is None
    assert config.mock is False
    assert config.output_format == "png"
    assert config.atlas_columns is None
    assert config.to_dict()["input_dir"]
    # V1.2 defaults
    assert config.detection_mode == "panel-page-prev2-cast"
    assert config.cast_key is None
    assert config.full_page_fallback is True
    assert config.blank_ink_threshold == 0.005
    assert config.max_megapixels == 2.0
    assert config.max_tokens == 1024
    assert config.sleep_s == 1.0
    assert config.only_panels == ()
    assert config.forced_characters == {}
    assert config.debug_font_size == 42
    assert config.debug_bbox_width == 5


def test_parse_args_overrides():
    config = parse_args(
        ["--skip-first", "3", "--limit", "2", "--steps", "panels,stitch",
         "--mock", "--output-format", "webp"]
    )
    assert config.skip_first == 3
    assert config.limit == 2
    assert config.steps == ("panels", "stitch")
    assert config.mock is True
    assert config.output_format == "webp"


@pytest.mark.parametrize(
    "argv",
    [
        ["--limit", "0"],
        ["--limit", "-1"],
        ["--skip-first", "-2"],
        ["--num-inference-steps", "0"],
        ["--guidance-scale", "0"],
        ["--lora-scale", "-0.5"],
        ["--panel-inset", "-1"],
    ],
)
def test_invalid_values_rejected(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_v1_1_flags_parse():
    config = parse_args([
        "--detection-mode", "panel",
        "--cast-key", "c001",
        "--no-full-page-fallback",
        "--max-megapixels", "3.5",
        "--blank-ink-threshold", "0.01",
        "--only-panel", "P003:panel_0006",
        "--only-panel", "p007:panel_0003",
        "--force-characters", "P003:panel_0006=Frieren",
        "--force-characters", "p007:panel_0003=Heiter,Fern",
    ])
    assert config.detection_mode == "panel"
    assert config.cast_key == "c001"
    assert config.full_page_fallback is False
    assert config.max_megapixels == 3.5
    assert config.blank_ink_threshold == 0.01
    assert config.only_panels == ("P003:panel_0006", "p007:panel_0003")
    assert config.forced_characters == {
        "P003:panel_0006": ["Frieren"],
        "p007:panel_0003": ["Heiter", "Fern"],
    }


def test_panel_page_cast_flags_parse():
    config = parse_args([
        "--detection-mode", "panel-page-cast",
        "--chapter-page-map", "custom/map.json",
    ])
    assert config.detection_mode == "panel-page-cast"
    assert config.chapter_page_map_file == Path("custom/map.json")


def test_panel_page_prev2_flags_parse():
    config = parse_args([
        "--detection-mode", "panel-page-prev2",
        "--vlm-panel-page-prev2-prompt-file", "custom/prev2.txt",
    ])
    assert config.detection_mode == "panel-page-prev2"
    assert config.vlm_panel_page_prev2_prompt_file == Path("custom/prev2.txt")
    assert config.to_dict()["vlm_panel_page_prev2_prompt_file"] == str(
        Path("custom/prev2.txt")
    )


def test_invalid_detection_mode_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--detection-mode", "panel-page-prev3"])


def test_invalid_v1_1_values_rejected():
    for argv in (
        ["--detection-mode", "pagex"],
        ["--max-megapixels", "0"],
        ["--max-megapixels", "-1"],
        ["--blank-ink-threshold", "1.5"],
        ["--blank-ink-threshold", "-0.1"],
        ["--only-panel", "missing-colon"],
        ["--force-characters", "no-equals"],
        ["--force-characters", "P003:panel_0006="],
    ):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_unknown_step_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--steps", "panels,nope"])


def test_parse_steps():
    assert parse_steps("") == STEP_ORDER
    assert parse_steps(None) == STEP_ORDER
    assert parse_steps(" panels , stitch ") == ("panels", "stitch")
    assert parse_steps("stitch,panels") == ("stitch", "panels")
    with pytest.raises(ValueError):
        parse_steps("bogus")


def test_step_dir_names():
    config = PipelineConfig()
    assert config.step_dir("panels") == "1_panels"
    assert config.step_dir("characters") == "2_characters"
    assert config.step_dir("colorize") == "3_colorized"
    assert config.step_dir("stitch") == "4_stitched"
    assert config.step_dir("debug") == "5_debug"
    with pytest.raises(ValueError):
        config.step_dir("bogus")


@pytest.mark.parametrize(
    "value,multiple,expected",
    [
        (340, 16, 336),   # 21.25 -> 21
        (345, 16, 352),   # 21.5625 -> 22
        (500, 16, 496),   # 31.25 -> 31
        (505, 16, 512),   # 31.5625 -> 32
        (600, 16, 608),   # 37.5 -> 38 (half-even)
        (900, 16, 896),   # 56.25 -> 56
        (24, 16, 32),     # 1.5 -> 2 (half-even)
        (8, 16, 16),      # clamps to minimum, never 0
        (1, 16, 16),
    ],
)
def test_nearest_multiple_of(value, multiple, expected):
    assert nearest_multiple_of(value, multiple) == expected


def test_requested_panel_size():
    assert requested_panel_size(340, 500) == (336, 496)
    assert requested_panel_size(345, 505) == (352, 512)
    assert requested_panel_size(10, 10) == (16, 16)
    # Both axes independent; aspect ratio may shift by <8px per axis.
    w, h = requested_panel_size(1200, 1800)
    assert w % 16 == 0 and h % 16 == 0
    assert abs(w - 1200) <= 8 and abs(h - 1800) <= 8


# ---------------------------------------------------------------------------
# V1.2 full-page gpt-image-2: new flags and validation

def test_full_page_flags_parse():
    config = parse_args([
        "--full-page",
        "--atlas-source", "cast",
        "--worker-detection", "4",
        "--worker-colorization", "2",
        "--gpt-model", "gpt-image-2-preview",
        "--gpt-size", "1024x1536",
        "--gpt-atlas-scale", "0.5",
        "--gpt-image-prompt-file", "custom/gpt.txt",
        "--openai-api-key-env", "MY_OPENAI_KEY",
    ])
    assert config.full_page is True
    assert config.atlas_source == "cast"
    assert config.worker_detection == 4
    assert config.worker_colorization == 2
    assert config.gpt_model == "gpt-image-2-preview"
    assert config.gpt_size == "1024x1536"
    assert config.gpt_atlas_scale == 0.5
    assert config.gpt_image_prompt_file == Path("custom/gpt.txt")
    assert config.openai_api_key_env == "MY_OPENAI_KEY"
    doc = config.to_dict()
    assert doc["worker_detection"] == 4
    assert doc["worker_colorization"] == 2
    assert doc["full_page"] is True


def test_full_page_detected_forces_page_detection_mode(capsys):
    config = parse_args(["--full-page", "--detection-mode", "panel"])
    # _validate coerces page mode for --atlas-source detected (the default).
    assert config.detection_mode == "page"
    assert "forces --detection-mode 'page'" in capsys.readouterr().err


def test_atlas_source_cast_requires_full_page():
    with pytest.raises(SystemExit):
        parse_args(["--atlas-source", "cast"])


def test_atlas_source_value_validated():
    with pytest.raises(SystemExit):
        parse_args(["--full-page", "--atlas-source", "everything"])


def test_workers_flag_removed():
    # The old umbrella --workers flag is gone: --worker-detection /
    # --worker-colorization replaced it.
    with pytest.raises(SystemExit):
        parse_args(["--workers", "4"])


def test_worker_flags_parse():
    config = parse_args(["--worker-detection", "3"])
    assert config.worker_detection == 3
    assert config.worker_colorization == 1  # default stays 1
    config = parse_args(["--worker-colorization", "6"])
    assert config.worker_detection == 1
    assert config.worker_colorization == 6


@pytest.mark.parametrize(
    "argv",
    [
        ["--worker-detection", "0"],
        ["--worker-detection", "-1"],
        ["--worker-colorization", "0"],
        ["--worker-colorization", "-1"],
    ],
)
def test_worker_flags_reject_less_than_one(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--gpt-atlas-scale", "0"],
        ["--gpt-atlas-scale", "-0.5"],
        ["--gpt-atlas-scale", "1.5"],
        ["--gpt-atlas-scale", "2"],
    ],
)
def test_gpt_atlas_scale_validated(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--gpt-size", "abc"],
        ["--gpt-size", "672"],          # not WxH
        ["--gpt-size", "672x"],         # not WxH
        ["--gpt-size", "671x1008"],     # not a multiple of 16
        ["--gpt-size", "672x672"],      # below the pixel floor
        ["--gpt-size", "4096x2048"],    # over max edge / max pixels
        ["--gpt-size", "4000x1000"],    # 4:1 ratio > 3:1
    ],
)
def test_gpt_size_validated(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_parse_gpt_size():
    assert parse_gpt_size("672x1008") == (672, 1008)
    assert parse_gpt_size("1024x1536") == (1024, 1536)
    assert parse_gpt_size("3840x2160") == (3840, 2160)
    with pytest.raises(ValueError):
        parse_gpt_size("abc")
    with pytest.raises(ValueError):
        parse_gpt_size("672")


# ---------------------------------------------------------------------------
# minimal_gpt_image_size (full-page output size policy)

def test_minimal_gpt_image_size_plan_examples():
    # Research-v2 measured sizes must be reproduced exactly.
    assert minimal_gpt_image_size(1500, 2250) == (672, 1008)   # 2:3
    assert minimal_gpt_image_size(1200, 1800) == (672, 1008)
    assert minimal_gpt_image_size(3000, 2250) == (960, 720)    # 4:3 spread
    # Already minimal: never upscales.
    assert minimal_gpt_image_size(672, 1008) == (672, 1008)


@pytest.mark.parametrize(
    "width,height",
    [
        (1500, 2250),   # 2:3 manga page (research-v2)
        (3000, 2250),   # 4:3 spread
        (2000, 2000),   # square
        (3840, 2160),   # 16:9
        (700, 1000),    # odd prime-ish ratio
        # NOTE: 1240x1754 / 2480x3508 (B5 scans) are deliberately absent: their
        # exact ratios are unsolvable within the API caps and must raise
        # (see test_minimal_gpt_image_size_rejects_unsolvable_ratios).
    ],
)
def test_minimal_gpt_image_size_constraints(width, height):
    w, h = minimal_gpt_image_size(width, height)
    assert w % 16 == 0 and h % 16 == 0
    assert GPT_IMAGE_MIN_PIXELS <= w * h <= GPT_IMAGE_MAX_PIXELS
    assert max(w, h) <= GPT_IMAGE_MAX_EDGE
    # Exact aspect ratio is preserved (the whole point of the policy).
    assert w / h == pytest.approx(width / height)
    assert w * h >= GPT_IMAGE_MIN_PIXELS


def test_minimal_gpt_image_size_is_minimal():
    # For the 2:3 headline case: one 16px step smaller falls below the
    # pixel floor, so (672, 1008) is the smallest valid exact-ratio size.
    assert 672 * 1008 >= GPT_IMAGE_MIN_PIXELS
    assert 656 * 984 < GPT_IMAGE_MIN_PIXELS


def test_minimal_gpt_image_size_rejects_unsolvable_ratios():
    # 4:1 — every size at that ratio is rejected by the API: fail loudly
    # rather than distort.
    with pytest.raises(ValueError, match="outside"):
        minimal_gpt_image_size(4000, 1000)
    with pytest.raises(ValueError, match="outside"):
        minimal_gpt_image_size(500, 2000)  # 1:4
    # 2480x3508 (300 dpi B5): reduced ratio 620:877 needs k=16 ->
    # 9920x14032, beyond the max edge/pixels; no exact-ratio size exists.
    with pytest.raises(ValueError, match="no minimal size"):
        minimal_gpt_image_size(2480, 3508)
    with pytest.raises(ValueError, match="no minimal size"):
        minimal_gpt_image_size(1240, 1754)
    with pytest.raises(ValueError, match="positive"):
        minimal_gpt_image_size(0, 100)
    with pytest.raises(ValueError, match="positive"):
        minimal_gpt_image_size(100, -5)
