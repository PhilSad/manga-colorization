"""Shared helpers for the real-network integration suite (tests marked
`integration`, run with `pytest -m integration`).

The suite is stage-isolated: every case takes a fixed committed input from
`tests/data/` (a pre-cropped panel, or a full page for the layout stage) and
exercises ONE real backend — never a mocked one and never the whole pipeline:

- detection (DET-*, OOV-*): the durable source page + real reading-order
  extraction -> real OpenRouter `google/gemma-4-31b-it` panel-page detection
  (full page context plus committed target crop) -> assert the character set.
- color (COL-*, SIZE-*): the committed crop + `forced_characters` -> real
  FLUX.2 Klein 9B colorization on the Spark server -> real
  `openai/gpt-5.6-luna` validation of the output.
- layout (LAY-*): the committed full page -> real YOLO26n panel detection.

All artifacts and per-case records (with `usage.cost`) go into the
timestamped run dir created by the `integration_run` session fixture
(`tests/output/YYYYMMDD-HHMMSS/`).
"""

from __future__ import annotations

import json
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

# Real models used by the suite (override via env).
DETECTION_MODEL = "google/gemma-4-31b-it"
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


def build_panel_detector(api_key: str, model: str = DETECTION_MODEL):
    """Real OpenRouter character detector prepared for panel-page detection."""
    from characters import OpenRouterCharacterDetector

    detector = OpenRouterCharacterDetector(model=model, api_key=api_key)
    detector.prepare(
        REFS_DIR,
        prompt_file=PAGE_PROMPT_FILE,
        panel_prompt_file=PANEL_PROMPT_FILE,
        panel_page_prompt_file=PANEL_PAGE_PROMPT_FILE,
    )
    return detector


def build_page_dir(fixture: dict, alias: str, work_dir: Path) -> tuple[Path, Path]:
    """Extract one fixture page exactly as the real pipeline's panels step does.

    Returns ``(source_page, panels_dir)`` suitable for
    ``OpenRouterCharacterDetector.detect_panels_with_page``.
    """
    from detection import YoloPanelDetector
    from extraction import save_panels
    from panel_ordering import reading_order
    from PIL import Image
    from util import sha256

    source_page = (REPO_ROOT / fixture["aliases"][alias]).resolve()
    boxes = YoloPanelDetector().detect(source_page)
    order = reading_order(boxes)
    ordered = [boxes[index] for index in order]
    if not ordered:
        raise RuntimeError(f"{alias}: panel detector returned no panels")

    panels_dir = work_dir / alias
    panels_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_page) as image:
        records = save_panels(image.convert("RGB"), ordered, panels_dir, inset=0)
    detections = [
        {
            "panel_index": index,
            "box": [round(box.x1), round(box.y1), round(box.x2), round(box.y2)],
            "confidence": round(box.confidence, 4),
            "crop": record["filename"],
            "provenance": "yolo",
        }
        for index, (box, record) in enumerate(zip(ordered, records), start=1)
    ]
    write_json(panels_dir / "panels.json", {
        "page": source_page.name,
        "page_path": str(source_page),
        "page_sha256": sha256(source_page),
        "detection_order_into_reading_order": order,
        "detections": detections,
        "reading_order": [item["panel_index"] for item in detections],
        "blank_page": False,
        "skip_reason": None,
        "full_page_fallback": False,
    })
    return source_page, panels_dir


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


def build_verify_verifier(api_key: str):
    """Real L2R verifier for COL-004 (openai/gpt-5.6-luna)."""
    from verify_color import LeftToRightVerifier

    return LeftToRightVerifier(model=VERIFY_MODEL, api_key=api_key)


def palette_instruction_for(names: list[str]) -> str:
    """Explicit canonical-palette instruction from the shared profiles."""
    from profiles import load_profiles, palette_instruction

    profiles = load_profiles(PROFILES_FILE)
    return palette_instruction(names, profiles)
