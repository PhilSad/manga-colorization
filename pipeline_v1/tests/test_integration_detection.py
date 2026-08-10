"""Real-network character-detection integration tests (DET-001..010, OOV-001).

One parametrized test function per detection mode, each run over the full
case set with **committed inputs** from `tests/data/` — no panel detection in
the tests themselves (the layout-stage crop-stability tripwire guards that):

- `test_detection_panel` — `OpenRouterCharacterDetector.detect` on the
  committed crop (V1 prompt; also the fallback path).
- `test_detection_panel_page` — `detect_panels_with_page`: the numbered
  committed page + the committed crop (the pipeline's default mode).
- `test_detection_panel_page_cast` — same, with the chapter-cast shortlist
  (`fixture["cast_keys"][alias]`).
- `test_detection_page` — `detect_page`: one page-level mapping call per
  case; the case's panel entry is kept (missing/uncertain/unknown entries
  trigger the built-in per-panel fallback).

Every mode asserts the same expected character set / unknown flags from the
fixture (`assert_matches`); per-(mode, case) records with `usage.cost` land
in the timestamped integration run dir. Known-failing cases (DET-006..010,
OOV-001) assert the *correct* outcome and fail loudly on a live run — they
stay tracked in the fixture, never xfailed.
"""

from __future__ import annotations

import pytest

from integration_support import (
    REFS_DIR,
    assert_matches,
    build_panel_detector,
    case_by_id,
    committed_page,
    crop_path,
    load_fixture,
    materialize_panels_dir,
    record_detection,
)

pytestmark = pytest.mark.integration

FIXTURE = load_fixture()

DETECTION_CASES = ["DET-001", "DET-002", "DET-003", "DET-004", "OOV-001",
                   "DET-005", "DET-006", "DET-007", "DET-008", "DET-009", "DET-010"]


@pytest.fixture(scope="module")
def detector(openrouter_key):
    """One prepared real detector shared by the mode tests. Preparation is
    local-only (refs, profiles, prompt files); no cross-test state."""
    return build_panel_detector(openrouter_key)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_panel(integration_run, detector, case_id):
    """panel mode: the committed crop alone (V1 prompt, the fallback path)."""
    case = case_by_id(FIXTURE, case_id)
    record = detector.detect(crop_path(case_id), REFS_DIR)
    record_detection(integration_run, "panel", case_id, record, case)
    assert_matches(record, case)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_panel_page(integration_run, detector, case_id, tmp_path):
    """panel-page mode: numbered committed page + committed crop (the
    pipeline's default detection mode)."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect_panels_with_page(
        committed_page(alias),
        materialize_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(integration_run, "panel-page", case_id, record, case)
    assert_matches(record, case)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_panel_page_cast(integration_run, detector, case_id, tmp_path):
    """panel-page-cast mode: panel-page with the chapter-cast shortlist
    (Flamme is excluded from ch. 5's cast, so DET-006..010 must not guess
    her)."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    cast_key = FIXTURE["cast_keys"][alias]
    page_record = detector.detect_panels_with_page(
        committed_page(alias),
        materialize_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
        cast_key=cast_key,
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(integration_run, "panel-page-cast", case_id, record, case)
    assert_matches(record, case)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_page(integration_run, detector, case_id, tmp_path):
    """page mode: one page-level mapping call; the case's panel entry is
    kept (missing/uncertain/unknown entries fall back per panel)."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect_page(
        committed_page(alias),
        materialize_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(integration_run, "page", case_id, record, case)
    assert_matches(record, case)
