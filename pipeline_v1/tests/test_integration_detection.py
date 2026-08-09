"""Real-network character-detection integration tests (DET-001..010, OOV-001).

The tests use the same **panel-page** path as the real pipeline: the durable
source page is extracted in reading order, then each target is sent with the
numbered full page as context plus its panel crop. One real OpenRouter
`google/gemma-4-31b-it` call per case; the fixture's expected character set /
`unknown_present` are asserted. The
detection record (status, characters, unknowns, `usage.cost`) and the input
crop land in the timestamped integration run dir.
"""

from __future__ import annotations

import shutil

import pytest

from integration_support import (
    DETECTION_MODEL,
    REFS_DIR,
    build_page_dir,
    build_panel_detector,
    case_by_id,
    crop_path,
    load_fixture,
    write_json,
)

pytestmark = pytest.mark.integration

DETECTION_CASES = ["DET-001", "DET-002", "DET-003", "DET-004", "OOV-001",
                  "DET-005", "DET-006", "DET-007", "DET-008", "DET-009", "DET-010"]


@pytest.fixture(scope="module")
def panel_page_records(integration_run, openrouter_key):
    """Run each source page once and cache its per-panel records for cases."""
    fixture = load_fixture()
    detector = build_panel_detector(openrouter_key)
    cases_by_alias = {}
    for case_id in DETECTION_CASES:
        case = case_by_id(fixture, case_id)
        cases_by_alias.setdefault(case["input"]["source_page"], []).append(case)

    records = {}
    for alias, cases in cases_by_alias.items():
        source_page, panels_dir = build_page_dir(
            fixture, alias, integration_run.run_dir / "detection_pages"
        )
        panel_keys = [case["input"]["panel"] for case in cases]
        page_record = detector.detect_panels_with_page(
            source_page, panels_dir, panel_keys, REFS_DIR
        )
        for case in cases:
            records[case["id"]] = (page_record.panels[case["input"]["panel"]], panels_dir)
    return records


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_case(integration_run, panel_page_records, case_id):
    fixture = load_fixture()
    case = case_by_id(fixture, case_id)
    committed_crop = crop_path(case_id)
    expected = set(case["expected"]["characters"])
    expected_unknown_present = case["expected"]["unknown_present"]
    expected_unknown = set(case["expected"].get("expected_unknown_characters", []))

    record, panels_dir = panel_page_records[case_id]
    crop = panels_dir / f"{case['input']['panel']}.png"
    assert crop.read_bytes() == committed_crop.read_bytes(), (
        f"{case_id}: live extraction differs from committed fixture crop"
    )

    case_dir = integration_run.run_dir / "detection" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(crop, case_dir / "input.png")
    write_json(case_dir / "record.json", record.to_dict(crop, page="integration"))
    integration_run.record(
        case_id,
        stage="detection",
        failure=case.get("failure"),
        model=DETECTION_MODEL,
        status=record.status,
        detected=record.characters,
        unknown_entries=record.unknown_entries,
        expected=sorted(expected),
        expected_unknown_present=expected_unknown_present,
        expected_unknown=sorted(expected_unknown),
        matches=record.status not in ("error", "unparseable")
        and set(record.characters) == expected
        and bool(record.unknown_entries) == expected_unknown_present
        and expected_unknown <= set(record.unknown_entries),
        cost_usd=record.cost_usd,
        cost_source=record.cost_source,
        latency_s=record.latency_s,
        model_returned=record.model_returned,
        error=record.error,
    )

    assert record.error is None, f"{case_id}: detection call failed: {record.error}"
    assert set(record.characters) == expected, (
        f"{case_id}: detected {sorted(record.characters)} != expected {sorted(expected)}"
    )
    assert bool(record.unknown_entries) == expected_unknown_present, (
        f"{case_id}: unknown_present {bool(record.unknown_entries)} "
        f"!= expected {expected_unknown_present}"
    )
    assert expected_unknown <= set(record.unknown_entries), (
        f"{case_id}: expected unknowns {sorted(expected_unknown)} not reported; "
        f"got {record.unknown_entries}"
    )
