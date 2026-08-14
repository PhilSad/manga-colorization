"""Real-network character-detection integration tests (DET-001..010, OOV-001).

One parametrized test function per detection mode (plus the
`panel-page-prev2-cast` variant), each run over the full
case set with **committed inputs** from `tests/data/` — no panel detection in
the tests themselves (the layout-stage crop-stability tripwire guards that).
All five modes go through the unified `OpenRouterCharacterDetector.detect(
mode, ...)` entry point, which dispatches to the mode's strategy
(`characters.DETECTION_STRATEGIES`); the prev2-cast variant is the prev2
mode with the cast shortlist passed explicitly, mirroring `panel-page-cast`:

- `panel` — one call per committed crop (V1 prompt; also the fallback path).
- `panel-page` — the numbered committed page + the committed crop (the
  pipeline's default detection mode).
- `panel-page-cast` — panel-page with the chapter-cast shortlist
  (`fixture["cast_keys"][alias]`, passed explicitly so the fixture stays the
  single source of truth).
- `panel-page-prev2` — panel-page plus the two preceding pages as extra
  story-context images (`materialize_prev2_panels_dir` lays out two preceding
  page dirs; their `page_path` reuses the case's own committed page, so the
  two-image code path is exercised with committed inputs only and no
  wrong-story characters leak into the context). `panel-page-prev2-cast`
  (same mode) additionally renders the chapter-cast shortlist in the prompt,
  exactly like `panel-page-cast`.
- `page` — one page-level mapping call per case; the case's panel entry is
  kept (missing/uncertain/unknown entries trigger the built-in per-panel
  fallback).

Every mode returns a per-page `PageCharacterRecord`; the per-panel record is
asserted against the same fixture expectations (`assert_matches`), and a slim
record with `usage.cost` lands in the timestamped integration run dir.
Known-failing cases (DET-006..010, OOV-001) assert the *correct* outcome and
fail loudly on a live run — they stay tracked in the fixture, never xfailed.
"""

from __future__ import annotations

import pytest

from integration_support import (
    REFS_DIR,
    assert_matches,
    build_panel_detector,
    case_by_id,
    committed_page,
    load_fixture,
    materialize_panels_dir,
    materialize_prev2_panels_dir,
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
def test_detection_panel(integration_run, detector, case_id, tmp_path):
    """panel mode: the committed crop alone (V1 prompt, the fallback path)."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect(
        "panel",
        committed_page(alias),
        materialize_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(integration_run, "panel", case_id, record, case)
    assert_matches(record, case)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_panel_page(integration_run, detector, case_id, tmp_path):
    """panel-page mode: numbered committed page + committed crop (the
    pipeline's default detection mode)."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect(
        "panel-page",
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
    her). The cast key comes from the fixture, not from pipeline data."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect(
        "panel-page-cast",
        committed_page(alias),
        materialize_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
        cast_key=FIXTURE["cast_keys"][alias],
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(integration_run, "panel-page-cast", case_id, record, case)
    assert_matches(record, case)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_panel_page_prev2(integration_run, detector, case_id, tmp_path):
    """panel-page-prev2 mode: like panel-page, plus the two preceding pages
    as story-context images (same committed page, see the helper's docstring).
    Same fixture expectations as the other modes."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect(
        "panel-page-prev2",
        committed_page(alias),
        materialize_prev2_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(integration_run, "panel-page-prev2", case_id, record, case)
    assert_matches(record, case)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_panel_page_prev2_cast(integration_run, detector, case_id,
                                         tmp_path):
    """panel-page-prev2 with the chapter-cast shortlist: same mode as
    `test_detection_panel_page_prev2` plus the cast rendered in the prompt
    (Flamme is excluded from ch. 5's cast, so DET-006..010 must not guess
    her). The cast key comes from the fixture, not from pipeline data."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect(
        "panel-page-prev2",
        committed_page(alias),
        materialize_prev2_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
        cast_key=FIXTURE["cast_keys"][alias],
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(
        integration_run, "panel-page-prev2-cast", case_id, record, case
    )
    assert_matches(record, case)


@pytest.mark.parametrize("case_id", DETECTION_CASES)
def test_detection_page(integration_run, detector, case_id, tmp_path):
    """page mode: one page-level mapping call; the case's panel entry is
    kept (missing/uncertain/unknown entries fall back per panel)."""
    case = case_by_id(FIXTURE, case_id)
    alias = case["input"]["source_page"]
    page_record = detector.detect(
        "page",
        committed_page(alias),
        materialize_panels_dir(alias, tmp_path),
        [case["input"]["panel"]],
        REFS_DIR,
    )
    record = page_record.panels[case["input"]["panel"]]
    record_detection(integration_run, "page", case_id, record, case)
    assert_matches(record, case)
