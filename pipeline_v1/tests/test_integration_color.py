"""Real-network color integration tests (COL-001..004) + request-size policy
(SIZE-001).

Stage-isolated by design: the input is the committed **pre-cropped panel**
(`tests/data/panels/<case_id>.png`) plus the fixture's `forced_characters`
— there is no detection step and no page. The panel is colorized with the
**real** FLUX.2 Klein 9B (step-distilled + LoRA) on the Spark server, then
validated with the **real** `openai/gpt-5.6-luna`:

- COL-001..003 (palette adherence): required colors present / forbidden
  absent, prompt rendered from the fixture's `required_colors` /
  `forbidden_colors`.
- COL-004 (palette geography): the left-to-right hair colors must be true
  (green Heiter / blue Himmel / white-pink Frieren / yellow Eisen). This is
  the V1.2 known failure — the test asserts it and currently FAILS until the
  geographic-atlas fix (ideas.md, V1.2 problem 1) lands.
- SIZE-001: the live colorize request on the oversized spread crop must be
  capped to 1600x1248 (<= 2.0 MP, multiples of 16).

Skipped when the Spark server is unreachable (`spark_endpoint` fixture);
requires `OPENROUTER_API_KEY`. First FLUX call pays the model-load cost.
"""

from __future__ import annotations

import shutil

import pytest

from integration_support import (
    REFS_DIR,
    build_colorizer,
    build_verify_verifier,
    case_by_id,
    crop_path,
    load_fixture,
    palette_instruction_for,
    write_json,
)
from verify_color import PaletteVerifier

pytestmark = pytest.mark.integration

PALETTE_CASES = ["COL-001", "COL-002", "COL-003"]


def _colorize_panel(integration_run, case_id, spark_endpoint, crop):
    """Real FLUX colorization of the crop with the forced characters' atlas.
    Returns (colorize_record, case_dir, colorized_path, atlas_path)."""
    from atlas import build_filtered_atlas

    fixture = load_fixture()
    case = case_by_id(fixture, case_id)
    forced = case["input"]["forced_characters"]

    case_dir = integration_run.run_dir / "color" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(crop, case_dir / "input.png")

    atlas_path = case_dir / "atlas.jpg"
    build_filtered_atlas(forced, REFS_DIR, atlas_path)
    colorized = case_dir / "colorized.png"

    colorizer = build_colorizer(spark_endpoint)
    record = colorizer.colorize(
        crop, atlas_path, colorized,
        palette_instruction=palette_instruction_for(forced),
    )
    return record, case_dir, colorized, atlas_path


@pytest.mark.parametrize("case_id", PALETTE_CASES)
def test_col_palette_adherence(integration_run, openrouter_key, spark_endpoint,
                               case_id):
    fixture = load_fixture()
    case = case_by_id(fixture, case_id)
    crop = crop_path(case_id)

    record, case_dir, colorized, atlas_path = _colorize_panel(
        integration_run, case_id, spark_endpoint, crop
    )
    assert record.status == "ok", f"{case_id}: colorize failed: {record.error}"

    # Real VLM validation against the fixture's palette.
    verifier = PaletteVerifier(model="openai/gpt-5.6-luna", api_key=openrouter_key)
    verdict = verifier.verify(
        colorized, crop,
        required_colors=case["expected"]["required_colors"],
        forbidden_colors=case["expected"]["forbidden_colors"],
    )

    write_json(case_dir / "record.json", {
        "colorize": record.to_dict(crop, atlas_path),
        "verdict": verdict.to_dict(),
    })
    integration_run.record(
        case_id,
        stage="color",
        failure=case.get("failure"),
        forced_characters=case["input"]["forced_characters"],
        colorize_status=record.status,
        requested_size=record.requested_size,
        verdict_status=verdict.status,
        adheres=verdict.adheres,
        missing_required=verdict.missing_required,
        present_forbidden=verdict.present_forbidden,
        verify_notes=verdict.notes,
        cost_usd=verdict.cost_usd,
        cost_source=verdict.cost_source,
        verify_model=verifier.model,
        error=record.error,
    )

    assert verdict.status == "verified", (
        f"{case_id}: palette not adhered to — missing required: "
        f"{verdict.missing_required}, forbidden present: "
        f"{verdict.present_forbidden}; notes: {verdict.notes}"
    )


def test_col_004_palette_geography(integration_run, openrouter_key,
                                   spark_endpoint):
    fixture = load_fixture()
    case = case_by_id(fixture, "COL-004")
    crop = crop_path("COL-004")

    record, case_dir, colorized, atlas_path = _colorize_panel(
        integration_run, "COL-004", spark_endpoint, crop
    )
    assert record.status == "ok", f"COL-004: colorize failed: {record.error}"

    # Real VLM validation: is the left-to-right hair-color assignment true?
    verifier = build_verify_verifier(openrouter_key)
    verdict = verifier.verify(colorized, crop, case["expected"]["left_to_right"])

    write_json(case_dir / "record.json", {
        "colorize": record.to_dict(crop, atlas_path),
        "verdict": verdict.to_dict(),
    })
    integration_run.record(
        "COL-004",
        stage="color",
        failure=case.get("failure"),
        forced_characters=case["input"]["forced_characters"],
        colorize_status=record.status,
        requested_size=record.requested_size,
        verdict_status=verdict.status,
        left_to_right_matches=verdict.left_to_right_matches,
        per_position=verdict.per_position,
        verify_notes=verdict.notes,
        cost_usd=verdict.cost_usd,
        cost_source=verdict.cost_source,
        verify_model=verifier.model,
        error=record.error,
    )

    assert verdict.status == "verified", (
        "COL-004: left-to-right hair colors are NOT true — "
        f"{verdict.notes}; per-position: {verdict.per_position}. "
        "Known V1.2 failure (ideas.md problem 1): the atlas is not yet built "
        "in left-to-right reading order with geographic identity info."
    )


def test_size_001_request_cap(integration_run, spark_endpoint):
    """The oversized spread crop must be requested at 1600x1248 (2.0 MP cap,
    multiples of 16), not at its native ~2900x2250."""
    crop = crop_path("SIZE-001")
    case_dir = integration_run.run_dir / "size" / "SIZE-001"
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(crop, case_dir / "input.png")
    colorized = case_dir / "colorized.png"

    colorizer = build_colorizer(spark_endpoint)
    record = colorizer.colorize(crop, None, colorized, palette_instruction="")
    write_json(case_dir / "record.json", record.to_dict(crop, None))

    assert record.status == "ok", f"SIZE-001: colorize failed: {record.error}"
    assert record.requested_size == (1600, 1248), (
        f"SIZE-001: requested {record.requested_size} != (1600, 1248)"
    )
    assert record.cap_applied is True
    assert record.original_size is not None
    original_pixels = record.original_size[0] * record.original_size[1]
    requested_pixels = record.requested_size[0] * record.requested_size[1]
    assert requested_pixels <= 2_000_000
    assert original_pixels > 2_000_000

    integration_run.record(
        "SIZE-001",
        stage="size",
        failure="oversized-input-capping",
        original_size=record.original_size,
        requested_size=record.requested_size,
        scale=record.scale,
        cap_applied=record.cap_applied,
        max_megapixels=record.max_megapixels,
        error=record.error,
    )
