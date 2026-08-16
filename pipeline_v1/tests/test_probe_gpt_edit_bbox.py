"""Offline unit tests for scripts/probe_gpt_edit_bbox.py (no network).

The probe itself is a paid behavior probe (gpt-image-2 + optional Luna
re-probe); these tests pin the pure helpers: the numbered region
instruction that matches the drawn boxes, the edit-prompt template
formatting, and the never-overwrite output-dir convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from probe_gpt_edit_bbox import (  # noqa: E402
    EDIT_PROMPT_FILE,
    fresh_output_dir,
    region_instruction,
)


# ---------------------------------------------------------------------------
# Region instruction

def test_region_instruction_numbers_regions_in_order():
    regions = [
        {"character": "Eisen", "fix_suggestion": "recolor beard to golden-brown"},
        {"character": "Eisen", "fix_suggestion": "recolor mustache to blond"},
    ]
    text = region_instruction(regions)
    assert "Region 0 (Eisen): recolor beard to golden-brown" in text
    assert "Region 1 (Eisen): recolor mustache to blond" in text


def test_region_instruction_falls_back_to_problem():
    """A region without fix_suggestion uses its problem text instead."""
    regions = [{"character": "Frieren", "problem": "hair lavender, should be silver"}]
    text = region_instruction(regions)
    assert "Region 0 (Frieren): hair lavender, should be silver" in text


def test_region_instruction_missing_fix_and_problem():
    text = region_instruction([{"character": "?"}])
    assert "Region 0 (?):" in text


# ---------------------------------------------------------------------------
# Edit prompt template

def test_edit_prompt_template_formats():
    """The template's placeholders are exactly the ones GptImage2Colorizer
    passes to format() — width, height, character_profiles (the region
    instruction), plus the extra atlas_instruction kwarg must be ignored."""
    template = EDIT_PROMPT_FILE.read_text(encoding="utf-8")
    instruction = region_instruction(
        [{"character": "Eisen", "fix_suggestion": "beard golden-brown"}]
    )
    prompt = template.format(
        width=672,
        height=1008,
        atlas_instruction="unused in this template",
        character_profiles=instruction,
    )
    assert "672x1008" in prompt
    assert "Region 0 (Eisen): beard golden-brown" in prompt
    assert "{character_profiles}" not in prompt
    assert "{width}" not in prompt and "{height}" not in prompt
    assert "red rectangles" in prompt
    assert "Remove the red rectangle outlines" in prompt


# ---------------------------------------------------------------------------
# Output dir convention

def test_fresh_output_dir_never_overwrites(tmp_path):
    first = fresh_output_dir(tmp_path, "gpt-edit-bbox")
    second = fresh_output_dir(tmp_path, "gpt-edit-bbox")
    assert first != second
    assert first.is_dir() and second.is_dir()
