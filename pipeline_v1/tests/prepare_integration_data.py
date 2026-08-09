#!/usr/bin/env python3
"""Regenerate the committed integration-test inputs under `tests/data/`.

The integration tests (test_integration_*.py) are stage-isolated: they take a
**pre-cropped panel** as input (never a full page), so `tests/data/panels/`
must hold the exact crops the fixture's panel IDs refer to — the crops
produced by the real reading-order extraction (`YoloPanelDetector` +
`panel_ordering.reading_order` + `extraction.save_panels`, same code the
pipeline uses). This script produces them from the durable source pages under
`data/` (gitignored) and commits the crops + the one full page the layout
tests need (P006) into `tests/data/`.

Usage:
    .venv/bin/python pipeline_v1/tests/prepare_integration_data.py

Rerun it whenever `evaluation/v1_1_cases.json` gains panel-based cases. Real
YOLO weights are auto-downloaded on first use (pipeline_v1/models/).
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))

from config import PIPELINE_DIR as _  # noqa: F401  (sys.path bootstrap)
from detection import YoloPanelDetector  # noqa: E402
from extraction import save_panels  # noqa: E402
from panel_ordering import reading_order  # noqa: E402
from PIL import Image  # noqa: E402

FIXTURE = PIPELINE_DIR / "evaluation" / "v1_1_cases.json"
DATA_ROOT = PIPELINE_DIR / "tests" / "data"
PANELS_ROOT = DATA_ROOT / "panels"
PAGES_ROOT = DATA_ROOT / "pages"

# Cases whose input is a panel crop (stage != layout, has a panel selector).
CROP_STAGES = ("characters", "color", "size")


def main() -> int:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    aliases = fixture["aliases"]

    # Collect (case_id, alias, panel) from the fixture.
    wanted: list[tuple[str, str, str]] = []
    for case in fixture["cases"]:
        entry = case.get("input", {})
        if case["stage"] in CROP_STAGES and entry.get("panel"):
            wanted.append((case["id"], entry["source_page"], entry["panel"]))

    by_page: dict[str, list[tuple[str, str]]] = {}
    for case_id, alias, panel in wanted:
        by_page.setdefault(alias, []).append((case_id, panel))

    PANELS_ROOT.mkdir(parents=True, exist_ok=True)
    PAGES_ROOT.mkdir(parents=True, exist_ok=True)
    detector = YoloPanelDetector()
    provenance: dict = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "detector": "leoxs22/manga-panel-detector-yolo26n",
        "confidence": detector.confidence,
        "panel_inset": 0,
        "ordering": "panel_ordering.reading_order (right-to-left, top-to-bottom)",
        "fixture": str(FIXTURE),
        "pages": {},
    }

    for alias, cases in sorted(by_page.items()):
        page_path = (REPO_ROOT / aliases[alias]).resolve()
        if not page_path.is_file():
            print(f"ERROR: source page missing: {page_path}", file=sys.stderr)
            return 1
        boxes = detector.detect(page_path)
        order = reading_order(boxes)
        ordered = [boxes[i] for i in order]
        if not ordered:
            print(f"ERROR: {alias} ({page_path.name[:50]}...) detected 0 panels "
                  f"-> cannot crop the fixture's panel cases", file=sys.stderr)
            return 1
        with Image.open(page_path) as image:
            page = image.convert("RGB")
        tmp = PIPELINE_DIR / "tests" / ".crops_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        records = save_panels(page, ordered, tmp, inset=0)
        boxes_by_index = {r["panel_index"]: b for r, b in zip(records, ordered)}
        provenance["pages"][alias] = {
            "source": str(page_path),
            "panels": len(ordered),
            "boxes": [b.as_int_tuple() for b in ordered],
        }
        for case_id, panel_key in cases:
            index = int(panel_key.rsplit("_", 1)[1])
            if index not in boxes_by_index:
                print(f"ERROR: {alias} {panel_key} (index {index}) out of range "
                      f"(have {len(ordered)} panels)", file=sys.stderr)
                return 1
            src = tmp / records[index - 1]["filename"]
            dst = PANELS_ROOT / f"{case_id}.png"
            shutil.copy(src, dst)
            print(f"  {case_id}: {alias} {panel_key} -> {dst.name} "
                  f"({src.stat().st_size // 1024} KiB, box {boxes_by_index[index].as_int_tuple()})")
        shutil.rmtree(tmp)

    # LAY-001 needs the full p006 page (no crop).
    p006 = (REPO_ROOT / aliases["P006"]).resolve()
    shutil.copy(p006, PAGES_ROOT / "lay_001_page.png")
    print(f"  LAY-001: {p006.name[:50]}... -> pages/lay_001_page.png")

    (DATA_ROOT / "README.md").write_text(
        _readme(provenance), encoding="utf-8"
    )
    print(f"\nprovenance written to {DATA_ROOT / 'README.md'}")
    return 0


def _readme(provenance: dict) -> str:
    lines = [
        "# Integration-test data",
        "",
        "Fixed inputs for the real-network integration suite "
        "(`pytest -m integration`). Regenerate with "
        "`pipeline_v1/tests/prepare_integration_data.py`.",
        "",
        "- `panels/<case_id>.png` — the pre-cropped panel for each "
        "DET/OOV/COL/SIZE case, produced by the real reading-order "
        "extraction (YOLO26n + `panel_ordering.reading_order`), so the "
        "fixture's panel IDs stay meaningful. The integration tests take "
        "these crops as input; they never run panel detection themselves.",
        "- `pages/lay_001_page.png` — p006, the full-page illustration the "
        "LAY-001 layout test runs real YOLO on.",
        "",
        "## Provenance of the last regeneration",
        "",
        "```json",
        json.dumps(provenance, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
