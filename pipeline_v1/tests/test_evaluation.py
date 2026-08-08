"""Tests for evaluation/v1_1_cases.json + evaluate.py (offline).

The V1.1 fixed evaluation set covers every supported failure category:
character confusion (DET-001..004), out-of-vocabulary identity (OOV-001),
palette adherence (COL-001..003), zero-panel fallback (LAY-001), blank-page
skip (LAY-002), and oversized-input capping (SIZE-001).

Detection cases are scored automatically (set comparison with exact
TP/FP/FN). Color cases only produce a human-review Markdown report: no code
path may assign a pass/fail verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from config import PIPELINE_DIR

FIXTURE_PATH = PIPELINE_DIR / "evaluation" / "v1_1_cases.json"
REPO_ROOT = PIPELINE_DIR.parent


# ---------------------------------------------------------------------------
# Fixture helpers

def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def case_by_id(fixture: dict, case_id: str) -> dict:
    for case in fixture["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def resolve_alias(fixture: dict, alias: str) -> Path:
    relative = fixture["aliases"][alias]
    return (REPO_ROOT / relative).resolve()


# ---------------------------------------------------------------------------
# Fixture schema and resolvability

def test_fixture_loads_and_has_all_categories():
    fixture = load_fixture()
    stages = {case["stage"] for case in fixture["cases"]}
    assert stages == {"characters", "color", "layout", "size"}
    ids = [case["id"] for case in fixture["cases"]]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    expected_ids = {
        "DET-001", "DET-002", "DET-003", "DET-004", "OOV-001",
        "COL-001", "COL-002", "COL-003", "LAY-001", "LAY-002", "SIZE-001",
    }
    assert set(ids) == expected_ids
    assert fixture["schema_version"] == 1


def test_fixture_inputs_resolve_to_repo_paths():
    fixture = load_fixture()
    for alias, relative in fixture["aliases"].items():
        path = (REPO_ROOT / relative).resolve()
        assert path.is_relative_to(REPO_ROOT), f"{alias} escapes the repo"
        if path.exists():  # gitignored data dirs may be absent on other machines
            assert path.is_file(), f"{alias} -> {path} is not a file"


def test_fixture_reference_pages_exist():
    """The real volume-1 / chapter-134 pages must exist on this machine."""
    fixture = load_fixture()
    for alias in ("P003", "P004_005", "P006", "P007", "P008", "CH134_004"):
        path = resolve_alias(fixture, alias)
        assert path.is_file(), f"{alias} missing: {path}"


def test_fixture_expected_sets_match_task_spec():
    fixture = load_fixture()
    spec = {
        "DET-001": (["Frieren", "Himmel"], ["Fern", "Stark"]),
        "DET-002": (["Frieren", "Himmel", "Heiter", "Eisen"], ["Sein"]),
        "DET-003": (["Heiter"], ["Sein"]),
        "DET-004": (["Frieren", "Heiter"], ["Frieren", "Sein"]),
        "OOV-001": ([], ["Sein"]),
    }
    for case_id, (expected, baseline) in spec.items():
        case = case_by_id(fixture, case_id)
        assert set(case["expected"]["characters"]) == set(expected), case_id
        assert set(case["baseline"]["characters"]) == set(baseline), case_id


def test_color_cases_have_machine_readable_expectations():
    fixture = load_fixture()
    for case_id in ("COL-001", "COL-002", "COL-003"):
        case = case_by_id(fixture, case_id)
        expected = case["expected"]
        for key in ("characters", "required_colors", "forbidden_colors", "preserve"):
            assert isinstance(expected[key], list) and expected[key], f"{case_id}.{key}"
        assert "forced_characters" in case["input"], case_id


def test_layout_and_size_cases_have_exact_expectations():
    fixture = load_fixture()
    lay1 = case_by_id(fixture, "LAY-001")
    assert lay1["expected"]["box"] == [0, 0, 1500, 2250]
    assert lay1["expected"]["provenance"] == "full-page-fallback"
    lay2 = case_by_id(fixture, "LAY-002")
    assert lay2["expected"]["panels"] == 0
    assert lay2["expected"]["skip_reason"] == "blank-page"
    assert lay2["input"]["generate"]["kind"] == "white"
    size = case_by_id(fixture, "SIZE-001")
    assert size["expected"]["requested_size"] == {"width": 1600, "height": 1248}
    assert size["expected"]["requested_pixels"] == 1600 * 1248 <= 2_000_000


# ---------------------------------------------------------------------------
# evaluate.py: detection scoring

def _write_detection_record(page_dir: Path, panel: str, characters, unknown=()):
    page_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "status": "ok",
        "characters": list(characters),
        "unknown_entries": list(unknown),
        "cost_usd": 0.0001,
    }
    (page_dir / f"{panel}.json").write_text(json.dumps(doc), encoding="utf-8")


def _build_run_dir(tmp_path: Path) -> Path:
    """A run directory whose page stems match the fixture aliases, with
    detection records reproducing the V1 baselines for the DET-* cases."""
    import evaluate

    run = tmp_path / "run"
    fixture = load_fixture()
    p003 = resolve_alias(fixture, "P003").stem
    p008 = resolve_alias(fixture, "P008").stem
    ch134 = resolve_alias(fixture, "CH134_004").stem
    chars = run / "2_characters"
    _write_detection_record(chars / p003, "panel_0002", ["Fern", "Stark"])
    _write_detection_record(chars / p003, "panel_0004", ["Sein"])
    _write_detection_record(chars / p008, "panel_0003", ["Sein"])
    _write_detection_record(chars / p008, "panel_0004", ["Frieren", "Sein"])
    _write_detection_record(chars / ch134, "panel_0003", [], unknown=["Clematis"])
    (run / "manifest.json").write_text(json.dumps({
        "run_directory": str(run),
        "configuration": {"seed": 1, "vlm_model": "mock"},
        "steps": {},
        "totals": {},
    }), encoding="utf-8")
    return run


def test_detection_scoring_exact_sets_and_failures(tmp_path):
    import evaluate

    run = _build_run_dir(tmp_path)
    report = evaluate.run_evaluation(run, FIXTURE_PATH)

    detection = report["detection"]["cases"]
    by_id = {case["id"]: case for case in detection}

    det1 = by_id["DET-001"]
    assert det1["detected"] == ["Fern", "Stark"]
    assert det1["expected"] == ["Frieren", "Himmel"]
    assert det1["tp"] == 0 and det1["fp"] == 2 and det1["fn"] == 2
    assert det1["matches"] is False

    det3 = by_id["DET-003"]
    assert det3["tp"] == 0 and det3["fp"] == 1 and det3["fn"] == 1

    det4 = by_id["DET-004"]
    # Frieren correct, Sein spurious, Heiter missed.
    assert det4["tp"] == 1 and det4["fp"] == 1 and det4["fn"] == 1

    oov = by_id["OOV-001"]
    assert oov["detected"] == []
    assert oov["unknown_present"] is True
    assert oov["expected_unknown"] == ["Clematis"]
    assert oov["unknown_handled"] is True
    assert oov["tp"] == 0 and oov["fp"] == 0 and oov["fn"] == 0

    totals = report["detection"]["totals"]
    assert totals["tp"] == 1
    assert totals["fp"] == 5   # 2 (DET-001) + 1 (DET-002) + 1 (DET-003) + 1 (DET-004)
    assert totals["fn"] == 8   # 2 (DET-001) + 4 (DET-002) + 1 (DET-003) + 1 (DET-004)
    # order-independence: sets, not lists
    assert set(by_id["DET-001"]["detected"]) == {"Stark", "Fern"}


def test_detection_scoring_is_order_independent(tmp_path):
    import evaluate

    run = _build_run_dir(tmp_path)
    # Rewrite DET-001 in a different response order; sets must match.
    fixture = load_fixture()
    p003 = resolve_alias(fixture, "P003").stem
    _write_detection_record(run / "2_characters" / p003, "panel_0002",
                            ["Himmel", "Frieren"])
    report = evaluate.run_evaluation(run, FIXTURE_PATH)
    det1 = {c["id"]: c for c in report["detection"]["cases"]}["DET-001"]
    assert det1["tp"] == 2 and det1["fp"] == 0 and det1["fn"] == 0
    assert det1["matches"] is True


# ---------------------------------------------------------------------------
# evaluate.py: color review report

def test_color_review_report_never_auto_verdicts(tmp_path):
    import evaluate

    run = _build_run_dir(tmp_path)
    fixture = load_fixture()
    # Create the generated outputs for the COL-* cases.
    for alias, panel in (("P003", "panel_0006"), ("P007", "panel_0006"),
                         ("P008", "panel_0003")):
        page = resolve_alias(fixture, alias).stem
        out_dir = run / "3_colorized" / page
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (220, 60, 60)).save(out_dir / f"{panel}.png")
        Image.new("RGB", (8, 8), "gray").save(out_dir / f"{panel}_atlas.jpg")

    report = evaluate.run_evaluation(run, FIXTURE_PATH)
    color_cases = report["color"]["cases"]
    assert {c["id"] for c in color_cases} == {"COL-001", "COL-002", "COL-003"}
    for case in color_cases:
        assert case["review_status"] == "Pending user review"
        assert case["generated_image"] is not None
        # The report image link must resolve relative to the report.
        image = (run / "evaluation" / case["generated_image"]).resolve()
        assert image.is_file(), f"broken relative link: {case['generated_image']}"

    report_path = run / "evaluation" / "color_review.md"
    markdown = report_path.read_text(encoding="utf-8")
    assert markdown.count("**Review status:** Pending user review") == 3
    assert "[ ] Pass" in markdown and "[ ] Fail" in markdown
    assert "silver-white hair" in markdown
    assert "magenta" in markdown or "purple" in markdown
    assert "light green hair" in markdown
    assert "Reviewed by a human" not in markdown  # no automated verdict wording
    # Page names contain parens; image links must be angle-bracketed and
    # resolve relative to the report.
    for line in markdown.splitlines():
        if "![" in line:
            assert "](<" in line, f"link not angle-bracketed: {line}"


def test_color_review_expected_text_comes_from_fixture(tmp_path):
    import evaluate

    run = _build_run_dir(tmp_path)
    fixture = load_fixture()
    page = resolve_alias(fixture, "P008").stem
    out_dir = run / "3_colorized" / page
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "red").save(out_dir / "panel_0003.png")
    # Drift check: change the fixture expectation and the report must change.
    evaluate.run_evaluation(run, FIXTURE_PATH)
    markdown = (run / "evaluation" / "color_review.md").read_text(encoding="utf-8")
    assert "black clerical robe with gold trim" in markdown
    assert "Sein's brown-hair/purple-blue outfit palette" in markdown
    assert "text, linework" in markdown


def test_evaluation_distinguishes_detection_and_color_errors(tmp_path):
    import evaluate

    run = _build_run_dir(tmp_path)
    report = evaluate.run_evaluation(run, FIXTURE_PATH)
    # Detection errors are scored; color cases stay pending — a wrong label
    # in a detection record must never be reported as a color failure.
    assert "totals" in report["detection"]
    assert report["color"]["verdict_mode"] == "human review only"
    # A color case with no generated output is flagged, not failed.
    assert report["color"]["cases"]  # COL-001/2/3 present (no outputs here)
    for case in report["color"]["cases"]:
        assert case["review_status"] in ("Pending user review", "missing output")


# ---------------------------------------------------------------------------
# evaluate.py: layout + size assertions

def test_layout_and_size_evaluation(tmp_path):
    import evaluate

    run = _build_run_dir(tmp_path)
    fixture = load_fixture()

    # LAY-001: full-page fallback record in the run's panels.json.
    p006 = resolve_alias(fixture, "P006").stem
    panels_p006 = run / "1_panels" / p006
    panels_p006.mkdir(parents=True)
    (panels_p006 / "panels.json").write_text(json.dumps({
        "page": f"{p006}.png",
        "detections": [{
            "panel_index": 1,
            "box": [0, 0, 1500, 2250],
            "crop": "panel_0001.png",
            "provenance": "full-page-fallback",
        }],
        "blank_page": False,
    }), encoding="utf-8")

    # LAY-002: blank-page skip record for the generated white page.
    lay2 = case_by_id(fixture, "LAY-002")
    blank_stem = Path(lay2["input"]["generate"]["name"]).stem
    panels_blank = run / "1_panels" / blank_stem
    panels_blank.mkdir(parents=True)
    (panels_blank / "panels.json").write_text(json.dumps({
        "page": f"{blank_stem}.png",
        "detections": [],
        "blank_page": True,
        "skip_reason": "blank-page",
    }), encoding="utf-8")

    # SIZE-001: colorize record in the manifest.
    p004 = resolve_alias(fixture, "P004_005").stem
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["steps"]["colorize"] = {"records": [{
        "page": p004,
        "panel": "panel_0001.png",
        "original_size": {"width": 2895, "height": 2250},
        "requested_size": {"width": 1600, "height": 1248},
        "scale": 0.5541,
        "cap_applied": True,
        "max_megapixels": 2.0,
    }]}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate.run_evaluation(run, FIXTURE_PATH)

    layout = {c["id"]: c for c in report["layout"]["cases"]}
    assert layout["LAY-001"]["matches"] is True
    assert layout["LAY-001"]["provenance"] == "full-page-fallback"
    assert layout["LAY-002"]["matches"] is True
    assert layout["LAY-002"]["skip_reason"] == "blank-page"

    size = {c["id"]: c for c in report["size"]["cases"]}["SIZE-001"]
    assert size["requested_size"] == {"width": 1600, "height": 1248}
    assert size["matches"] is True
    assert size["cap_applied"] is True
    assert size["original_size"] == {"width": 2895, "height": 2250}
