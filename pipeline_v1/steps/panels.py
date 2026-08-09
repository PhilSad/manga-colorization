"""Pipeline stage 1+2: detect panels and extract them in reading order.

Writes, per page, into the run directory's `1_panels/<page_stem>/`:
  panel_0001.png ...          crops numbered in Japanese reading order
  panels.json                 geometry + order (consumed by the stitch stage)
  overlay.png                 debug image with numbered boxes

V1.1 (task 0004): when YOLO returns zero boxes the page is measured for ink.
Sparse full-page art (e.g. p006) gets one synthetic full-page box marked
`provenance: full-page-fallback`; effectively blank pages get an explicit
`blank-page` skip record and no crops at all. `--only-panel` (task 0001)
restricts which pages are processed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from config import PipelineConfig
from detection import PanelBox, PanelDetector, list_page_images
from extraction import draw_overlay, save_panels
from panel_ordering import reading_order
from run_context import RunContext, write_json
from selection import page_selected
from tqdm import tqdm
from util import sha256


def ink_ratio(image: Image.Image, threshold: int = 128) -> float:
    """Fraction of pixels darker than `threshold` (deterministic, conservative:
    sparse line art like a full-page illustration keeps a small but non-zero
    ratio; a blank page measures ~0)."""
    gray = image.convert("L")
    lut = [255 if value < threshold else 0 for value in range(256)]
    dark = gray.point(lut).histogram()[255]
    total = gray.width * gray.height
    return dark / total if total else 0.0


def run_panels_step(
    ctx: RunContext,
    config: PipelineConfig,
    detector: PanelDetector,
) -> dict:
    """Run stage 1+2 for all selected pages. Returns per-page records for the
    manifest."""
    pages = list_page_images(config.input_dir)
    if config.skip_first:
        pages = pages[config.skip_first:]
    if config.limit:
        pages = pages[: config.limit]
    if config.only_panels:
        pages = [p for p in pages if page_selected(p.stem, config.only_panels)]
    if not pages:
        raise ValueError(f"No supported page images found in {config.input_dir}")

    pages_dir = ctx.step_dir("panels")
    page_records: list[dict] = []
    for page_path in tqdm(
        pages, desc="panels: detect+extract", unit="page", leave=False
    ):
        with Image.open(page_path) as page:
            page = page.convert("RGB")
            detections = detector.detect(page_path)
            order = reading_order(detections)
            ordered = [detections[i] for i in order]
            page_dir = pages_dir / page_path.stem
            page_dir.mkdir(parents=True, exist_ok=True)
            provenance = "yolo"
            fallback = False
            blank_page = False
            skip_reason = None

            if not ordered:
                ink = ink_ratio(page)
                blank_page = ink < config.blank_ink_threshold
                if blank_page:
                    skip_reason = "blank-page"
                elif config.full_page_fallback:
                    ordered = [
                        PanelBox(0, 0, page.width, page.height, 1.0)
                    ]
                    provenance = "full-page-fallback"
                    fallback = True

            records = []
            if ordered:
                records = save_panels(
                    page, ordered, page_dir,
                    inset=config.panel_inset, extension=_extension(config),
                )
                draw_overlay(page, ordered, page_dir / "overlay.png",
                             inset=config.panel_inset)
            geometry = _panels_json(
                page_path, ordered, records, order,
                provenance=provenance, fallback=fallback,
                blank_page=blank_page, skip_reason=skip_reason,
            )
            write_json(page_dir / "panels.json", geometry)
            page_records.append(geometry)
    return {"pages": page_records}


def _extension(config: PipelineConfig) -> str:
    return {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[config.output_format]


def _panels_json(
    page_path: Path,
    ordered: list[PanelBox],
    records: list[dict],
    order: list[int],
    *,
    provenance: str,
    fallback: bool,
    blank_page: bool,
    skip_reason: str | None,
) -> dict:
    detections = []
    for index, (box, record) in enumerate(zip(ordered, records), start=1):
        detections.append({
            "panel_index": index,
            "box": [round(box.x1), round(box.y1), round(box.x2), round(box.y2)],
            "confidence": round(box.confidence, 4),
            "crop": record["filename"],
            "provenance": provenance,
        })
    return {
        "page": page_path.name,
        "page_path": str(page_path.resolve()),
        "page_sha256": sha256(page_path),
        "detection_order_into_reading_order": order,
        "detections": detections,
        "reading_order": [d["panel_index"] for d in detections],
        "blank_page": blank_page,
        "skip_reason": skip_reason,
        "full_page_fallback": fallback,
    }
