"""Real-network color integration tests (COL-001..004) + request-size policy
(SIZE-001) — one parametrized test over all five cases.

Stage-isolated by design: the input is the committed **pre-cropped panel**
(`tests/data/panels/<case_id>.png`) plus the fixture's `forced_characters`
— there is no detection step and no page. The panel is colorized with the
**real** FLUX.2 Klein 9B (step-distilled + LoRA) on the Spark server, then
validated with one generic structured-output verdict: a real
`openai/gpt-5.6-luna` call (strict `json_schema` response_format with
`provider.require_parameters: true`) judges whether every character in the
colorized panel has its canonical Frieren palette, answering
`analyse: str` + `good_color: bool`. No fixture expectations
(required/forbidden colors, left-to-right order) are rendered into the
prompt — the fixture keeps them as human-readable documentation only.
This replaces the previous two verifiers (palette adherence for
COL-001..003, left-to-right geography for COL-004); the
palette-adherence vs palette-geography distinction now lives only in the
fixture's per-case `failure` tag.

- COL-001..004: one generic palette verdict; a `good_color: false` verdict
  fails the test loudly. COL-004 is the known V1.2 geography failure
  (uniform blue wash, run 20260809-091129): the test asserts it and
  currently FAILS until the geographic-atlas fix lands.
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
    build_colorizer,
    build_color_verifier,
    case_by_id,
    crop_path,
    load_fixture,
    palette_instruction_for,
    record_color,
)

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
    """Colorize the committed crop and verify it per case kind (generic
    canonical-palette verdict, or size-cap policy)."""
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
    if "left_to_right" in expected or "required_colors" in expected:
        # COL-001..004: one generic canonical-palette verdict (strict
        # structured output). COL-004 is the known V1.2 geography failure
        # (uniform blue wash) and is expected to fail loudly until the
        # geographic-atlas fix lands.
        verifier = build_color_verifier(openrouter_key)
        verdict = verifier.verify(colorized, crop)
        known = (
            " Known V1.2 failure (ideas.md problem 1): the atlas is not yet "
            "built in left-to-right reading order with geographic identity info."
            if case_id == "COL-004"
            else ""
        )
        assert verdict.status == "verified", (
            f"{case_id}: color palette judged NOT canonical — "
            f"{verdict.analyse}; status={verdict.status}"
            f"{('; error: ' + verdict.error) if verdict.error else ''}"
            f"{known}"
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
