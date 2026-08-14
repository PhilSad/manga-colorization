"""Tests for config.py: CLI parsing, validation, step selection, size policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    STEP_ORDER,
    PipelineConfig,
    nearest_multiple_of,
    parse_args,
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
    assert config.detection_mode == "panel-page"
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
