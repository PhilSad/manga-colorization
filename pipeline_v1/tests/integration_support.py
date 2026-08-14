"""Shared helpers for the real-network integration suite (tests marked
`integration`, run with `pytest -m integration`).

The suite is stage-isolated: every case takes a fixed committed input from
`tests/data/` (a pre-cropped panel, or a committed page + its full panel set
for the page-context detection modes) and exercises ONE real backend — never
a mocked one and never the whole pipeline:

- detection (DET-*, OOV-*): the committed inputs -> real OpenRouter
  `google/gemma-4-31b-it`, one parametrized test per detection mode —
  `panel` (crop only), `panel-page` (full page context + crop), `panel-page-cast`
  (same, with the chapter cast shortlist), `panel-page-prev2` (panel-page plus
  the two preceding pages as story-context images), `panel-page-prev2-cast`
  (prev2 with the chapter cast shortlist), and `page` (one page-level mapping
  call per case). Assertions are identical across modes.
- color (COL-*, SIZE-*): the committed crop + `forced_characters` -> real
  FLUX.2 Klein 9B colorization on the Spark server -> real
  `openai/gpt-5.6-luna` validation of the output via one generic strict
  structured-output palette verdict (`analyse`/`good_color`; the size-policy
  case has no VLM verification).
- layout (LAY-*, crop-stability): real YOLO26n panel detection. The
  crop-stability tripwire re-extracts the committed pages and asserts the
  crops match the committed per-page sets byte-for-byte.

All artifacts and per-case records (with `usage.cost`) go into the
timestamped run dir created by the `integration_run` session fixture
(`tests/output/YYYYMMDD-HHMMSS/`).
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_DIR.parent

FIXTURE_PATH = PIPELINE_DIR / "evaluation" / "v1_1_cases.json"
DATA_ROOT = PIPELINE_DIR / "tests" / "data"
PANELS_ROOT = DATA_ROOT / "panels"
PAGES_ROOT = DATA_ROOT / "pages"

REFS_DIR = REPO_ROOT / "data" / "refs"
PROFILES_FILE = PIPELINE_DIR / "character_profiles.json"
CHAPTER_CASTS_FILE = PIPELINE_DIR / "chapter_casts.json"

PAGE_PROMPT_FILE = PIPELINE_DIR / "prompt.txt"
PANEL_PROMPT_FILE = PIPELINE_DIR / "prompt_panel.txt"
PANEL_PAGE_PROMPT_FILE = PIPELINE_DIR / "prompt_panel_page.txt"
PANEL_PAGE_PREV2_PROMPT_FILE = PIPELINE_DIR / "prompt_panel_page_prev2.txt"

# Real models used by the suite (override via env).
DETECTION_MODEL = os.environ.get(
    "INTEGRATION_DETECTION_MODEL", "openai/gpt-5.6-luna"
)
VERIFY_MODEL = "openai/gpt-5.6-luna"

# FLUX settings for the step-distilled Spark server (matches the pipeline).
FLUX_STEPS = 4
FLUX_GUIDANCE = 4.0
FLUX_LORA_SCALE = 1.0
FLUX_SEED = 1337

COLORIZER_PROMPT_FILE = PIPELINE_DIR / "colorizer_prompt.txt"


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def case_by_id(fixture: dict, case_id: str) -> dict:
    for case in fixture["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def crop_path(case_id: str) -> Path:
    """The committed pre-cropped panel for a DET/OOV/COL/SIZE case."""
    path = PANELS_ROOT / f"{case_id}.png"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing; regenerate with tests/prepare_integration_data.py"
        )
    return path


def page_path(filename: str) -> Path:
    path = PAGES_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing; regenerate with tests/prepare_integration_data.py"
        )
    return path


def committed_page(alias: str) -> Path:
    """The committed full page for a detection alias (P003, P008, ...)."""
    return page_path(f"{alias}.png")


def materialize_panels_dir(alias: str, work_dir: Path) -> Path:
    """Copy a committed per-page panel set (`panels/<alias>/`) into
    `work_dir` and point its `panels.json` at the committed page (absolute,
    resolved from the repo-relative committed value). Returns the panels dir,
    writable — the page-context detection code writes its annotation overlay
    there."""
    src = PANELS_ROOT / alias
    if not src.is_dir():
        raise FileNotFoundError(
            f"{src} missing; regenerate with tests/prepare_integration_data.py"
        )
    dst = work_dir / alias
    shutil.copytree(src, dst)
    geometry = json.loads((dst / "panels.json").read_text(encoding="utf-8"))
    geometry["page_path"] = str(committed_page(alias))
    write_json(dst / "panels.json", geometry)
    return dst


def materialize_prev2_panels_dir(alias: str, work_dir: Path) -> Path:
    """The committed panel set for `alias`, laid out so `panel-page-prev2`
    detection finds two preceding page dirs (`_previous_page_images` reads
    `panels_dir.parent` siblings, oldest first, each with a `panels.json`
    whose `page_path` exists and is non-blank).

    Both preceding dirs point their `page_path` at the case's own committed
    page image: the test stays fully committed-input (no fabricated pages),
    the two extra story-context images are sent deterministically on every
    call, and no wrong-story characters leak into the context. The degraded
    0/1-context-image shapes are covered by the offline unit tests in
    `test_characters.py`.

    Returns the current page's panels dir (same as `materialize_panels_dir`)."""
    panels_dir = materialize_panels_dir(alias, work_dir)
    page = committed_page(alias)
    # Dir names must sort strictly before the alias dir; the `00-`/`01-`
    # prefix guarantees that for every alias and keeps prev2 (older) first.
    for index, label in enumerate(("prev2", "prev1")):
        sibling = work_dir / f"0{index}-{alias}-{label}"
        sibling.mkdir(parents=True, exist_ok=True)
        write_json(sibling / "panels.json", {
            "page_path": str(page),
            "blank_page": False,
            "detections": [{"box": [0, 0, 10, 10], "crop": "panel_0001.png"}],
        })
    return panels_dir


def extract_page_crops(page: Path, work_dir: Path) -> list[dict]:
    """Real reading-order extraction of a committed page: YOLO26n detection
    + `panel_ordering.reading_order` + `extraction.save_panels`, the same
    code the pipeline runs. Writes the crops into `work_dir` and returns the
    per-panel records (panel_index, filename). Used by the crop-stability
    tripwire in the layout stage — the only place the suite re-runs panel
    detection."""
    from detection import YoloPanelDetector
    from extraction import save_panels
    from panel_ordering import reading_order
    from PIL import Image

    boxes = YoloPanelDetector().detect(page)
    order = reading_order(boxes)
    ordered = [boxes[index] for index in order]
    if not ordered:
        raise RuntimeError(f"{page.name}: panel detector returned no panels")
    work_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(page) as image:
        return save_panels(image.convert("RGB"), ordered, work_dir, inset=0)


def build_panel_detector(api_key: str, model: str = DETECTION_MODEL):
    """Real OpenRouter character detector prepared for all detection modes
    (page, panel, panel-page, panel-page-prev2 prompts all built).

    `chapter_casts_file` is wired so the cast modes (`panel-page-cast`,
    `panel-page-prev2` with a cast key) can render their chapter shortlist
    (`cast_shortlist_for` would otherwise crash on `Path(None)`)."""
    from characters import OpenRouterCharacterDetector

    detector = OpenRouterCharacterDetector(
        model=model,
        api_key=api_key,
        chapter_casts_file=CHAPTER_CASTS_FILE,
    )
    detector.prepare(
        REFS_DIR,
        prompt_file=PAGE_PROMPT_FILE,
        panel_prompt_file=PANEL_PROMPT_FILE,
        panel_page_prompt_file=PANEL_PAGE_PROMPT_FILE,
        panel_page_prev2_prompt_file=PANEL_PAGE_PREV2_PROMPT_FILE,
    )
    return detector


def build_colorizer(endpoint: str):
    """Real FLUX colorizer pointed at the Spark server."""
    from colorizer import FluxColorizer

    return FluxColorizer(
        endpoint=endpoint,
        prompt_template=COLORIZER_PROMPT_FILE.read_text(encoding="utf-8"),
        num_inference_steps=FLUX_STEPS,
        guidance_scale=FLUX_GUIDANCE,
        lora_scale=FLUX_LORA_SCALE,
        seed=FLUX_SEED,
        output_format="png",
    )


def build_color_verifier(api_key: str):
    """Real generic color verifier for COL-001..004 (openai/gpt-5.6-luna,
    strict json_schema structured output: analyse/good_color)."""
    from verify_color import ColorVerifier

    return ColorVerifier(model=VERIFY_MODEL, api_key=api_key)


def palette_instruction_for(names: list[str]) -> str:
    """Explicit canonical-palette instruction from the shared profiles."""
    from profiles import load_profiles, palette_instruction

    profiles = load_profiles(PROFILES_FILE)
    return palette_instruction(names, profiles)


# -- detection assertions + recording ---------------------------------------


def assert_matches(record: Any, case: dict) -> None:
    """The shared detection assertions, identical across the mode tests:
    character set equality, unknown-presence flag, and expected
    unknown characters (OOV). `record` is the per-panel CharacterRecord."""
    expected = set(case["expected"]["characters"])
    expected_unknown_present = case["expected"]["unknown_present"]
    expected_unknown = set(case["expected"].get("expected_unknown_characters", []))
    case_id = case["id"]
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


def record_detection(
    integration_run, mode: str, case_id: str, record: Any, case: dict
) -> None:
    """Slim manifest record for one (mode, case) detection call."""
    expected = set(case["expected"]["characters"])
    expected_unknown_present = case["expected"]["unknown_present"]
    expected_unknown = set(case["expected"].get("expected_unknown_characters", []))
    integration_run.record(
        case_id,
        stage="detection",
        mode=mode,
        failure=case.get("failure"),
        status=record.status,
        detected=record.characters,
        matches=(
            record.status not in ("error", "unparseable")
            and set(record.characters) == expected
            and bool(record.unknown_entries) == expected_unknown_present
            and expected_unknown <= set(record.unknown_entries)
        ),
        cost_usd=record.cost_usd,
        cost_source=record.cost_source,
        error=record.error,
    )


def record_color(
    integration_run, case_id: str, case: dict, colorize_record: Any,
    verdict: Any = None, **extra: Any,
) -> None:
    """Slim manifest record for one color case. `verdict` is the VLM
    verifier result (COL-*); size-policy cases pass no verdict and extra
    fields instead."""
    fields: dict[str, Any] = {
        "stage": "color",
        "failure": case.get("failure"),
        "colorize_status": colorize_record.status,
        "requested_size": colorize_record.requested_size,
        "error": colorize_record.error,
    }
    if verdict is not None:
        fields.update({
            "verdict_status": verdict.status,
            "good_color": verdict.good_color,
            "analyse": verdict.analyse,
            "cost_usd": verdict.cost_usd,
            "cost_source": verdict.cost_source,
        })
    fields.update(extra)
    integration_run.record(case_id, **fields)
