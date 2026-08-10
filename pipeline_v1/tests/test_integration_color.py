"""Real-network color integration tests (COL-001..004) + request-size policy
(SIZE-001) — one parametrized test over all five cases.

Stage-isolated by design: the input is the committed **pre-cropped panel**
(`tests/data/panels/<case_id>.png`) plus the fixture's `forced_characters`
— there is no detection step and no page. The panel is colorized with the
**real** FLUX.2 Klein 9B (step-distilled + LoRA) on the Spark server, then
validated per case kind:

- COL-001..003 (palette adherence): a real `openai/gpt-5.6-luna` palette
  verdict — required colors present / forbidden absent.
- COL-004 (palette geography): a real left-to-right verifier — the
  hair-color assignment (green Heiter / blue Himmel / white-pink Frieren /
  yellow Eisen) must be spatially true. Known V1.2 failure: the test asserts
  it and currently FAILS until the geographic-atlas fix lands.
- SIZE-001 (size policy): no VLM verification — the live colorize request on
  the oversized spread crop must be capped to 1600x1248 (<= 2.0 MP,
  multiples of 16).

Skipped when the Spark server is unreachable (`spark_endpoint` fixture);
requires `OPENROUTER_API_KEY`. First FLUX call pays the model-load cost.
"""

from __future__ import annotations

import pytest

from integration_support import (
    REFS_DIR,
    VERIFY_MODEL,
    build_colorizer,
    build_verify_verifier,
    case_by_id,
    crop_path,
    load_fixture,
    palette_instruction_for,
    record_color,
)
from verify_color import PaletteVerifier

pytestmark = pytest.mark.integration

FIXTURE = load_fixture()

COLOR_CASES = ["COL-001", "COL-002", "COL-003", "COL-004", "SIZE-001"]


def _colorize(integration_run, case_id, spark_endpoint, case):
    """Real FLUX colorization of the committed crop with the forced
    characters' atlas (crop-only when no characters are forced, i.e.
    SIZE-001). Returns (colorize_record, colorized_path)."""
    from atlas import build_filtered_atlas

    forced = case["input"].get("forced_characters", [])
    crop = crop_path(case_id)
    case_dir = integration_run.run_dir / "color" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = None
    if forced:
        atlas_path = case_dir / "atlas.jpg"
        build_filtered_atlas(forced, REFS_DIR, atlas_path)

    colorized = case_dir / "colorized.png"
    colorizer = build_colorizer(spark_endpoint)
    record = colorizer.colorize(
        crop,
        atlas_path,
        colorized,
        palette_instruction=palette_instruction_for(forced) if forced else "",
    )
    return record, colorized


@pytest.mark.parametrize("case_id", COLOR_CASES)
def test_color_case(integration_run, openrouter_key, spark_endpoint, case_id):
    """Colorize the committed crop and verify it per case kind (palette
    adherence, palette geography, or size-cap policy)."""
    case = case_by_id(FIXTURE, case_id)
    expected = case["expected"]
    crop = crop_path(case_id)

    colorize_record, colorized = _colorize(
        integration_run, case_id, spark_endpoint, case
    )
    assert colorize_record.status == "ok", (
        f"{case_id}: colorize failed: {colorize_record.error}"
    )

    verdict = None
    if "left_to_right" in expected:
        # COL-004: the left-to-right hair-color assignment must be true.
        verifier = build_verify_verifier(openrouter_key)
        verdict = verifier.verify(colorized, crop, expected["left_to_right"])
        assert verdict.status == "verified", (
            "COL-004: left-to-right hair colors are NOT true — "
            f"{verdict.notes}; per-position: {verdict.per_position}. "
            "Known V1.2 failure (ideas.md problem 1): the atlas is not yet "
            "built in left-to-right reading order with geographic identity info."
        )
    elif "required_colors" in expected:
        # COL-001..003: required colors present, forbidden absent.
        verifier = PaletteVerifier(model=VERIFY_MODEL, api_key=openrouter_key)
        verdict = verifier.verify(
            colorized, crop,
            required_colors=expected["required_colors"],
            forbidden_colors=expected["forbidden_colors"],
        )
        assert verdict.status == "verified", (
            f"{case_id}: palette not adhered to — missing required: "
            f"{verdict.missing_required}, forbidden present: "
            f"{verdict.present_forbidden}; notes: {verdict.notes}"
        )
    else:
        # SIZE-001: the oversized spread crop must be requested at the cap.
        expected_size = (expected["requested_size"]["width"],
                         expected["requested_size"]["height"])
        assert colorize_record.requested_size == expected_size, (
            f"SIZE-001: requested {colorize_record.requested_size} "
            f"!= expected {expected_size}"
        )
        assert colorize_record.cap_applied is True
        assert colorize_record.original_size is not None
        original_pixels = (
            colorize_record.original_size[0] * colorize_record.original_size[1]
        )
        requested_pixels = (
            colorize_record.requested_size[0] * colorize_record.requested_size[1]
        )
        assert requested_pixels <= 2_000_000
        assert original_pixels > 2_000_000

    record_color(
        integration_run, case_id, case, colorize_record, verdict,
        original_size=colorize_record.original_size,
        cap_applied=colorize_record.cap_applied,
        scale=colorize_record.scale,
        max_megapixels=colorize_record.max_megapixels,
    )
