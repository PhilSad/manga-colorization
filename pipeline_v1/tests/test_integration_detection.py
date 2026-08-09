"""Real-network character-detection integration tests (DET-001..010, OOV-001).

Stage-isolated by design: the input is the committed **pre-cropped panel**
(`tests/data/panels/<case_id>.png`, produced by the real reading-order
extraction) and nothing else — no page, no YOLO, no pipeline. One real
OpenRouter `google/gemma-4-31b-it` panel-detection call per case; the
fixture's expected character set / `unknown_present` are asserted. The
detection record (status, characters, unknowns, `usage.cost`) and the input
crop land in the timestamped integration run dir.

Known-failing cases stay failing (no xfail): with panel-only mode the V1
baselines that the fixture documents (e.g. OOV-001's Clematis) may still not
be met — the failure is the point, it tracks the known issue.
"""

from __future__ import annotations

import shutil

import pytest

from integration_support import (
    REFS_DIR,
    build_panel_detector,
    case_by_id,
    crop_path,
    load_fixture,
    write_json,
)

pytestmark = pytest.mark.integration

DETECTION_CASES = ["DET-001", "DET-002", "DET-003", "DET-004", "OOV-001",
                  "DET-005", "DET-006", "DET-007", "DET-008", "DET-009", "DET-010"]


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_case(integration_run, openrouter_key, case_id):
    fixture = load_fixture()
    case = case_by_id(fixture, case_id)
    crop = crop_path(case_id)
    expected = set(case["expected"]["characters"])
    expected_unknown_present = case["expected"]["unknown_present"]
    expected_unknown = set(case["expected"].get("expected_unknown_characters", []))

    # Real OpenRouter panel detection on the crop (V1 panel prompt, no mocks).
    detector = build_panel_detector(openrouter_key)
    record = detector.detect(crop, REFS_DIR)

    case_dir = integration_run.run_dir / "detection" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(crop, case_dir / "input.png")
    write_json(case_dir / "record.json", record.to_dict(crop, page="integration"))
    integration_run.record(
        case_id,
        stage="detection",
        failure=case.get("failure"),
        model=detector.model,
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
