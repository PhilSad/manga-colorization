#!/usr/bin/env python3
"""research-v2: split manga pages into panels, pipeline_v1-style.

Reuses the exact pipeline_v1 panel-extraction implementation (YOLO26n
detector `leoxs22/manga-panel-detector-yolo26n`, Japanese reading order,
blank-ink check + full-page fallback) by importing its modules; only the
output layout is research-v2's own.

Per page, writes into the timestamped run dir:
  <run>/<page_stem>/panel_0001.png ...   crops numbered in reading order
  <run>/<page_stem>/panels.json          geometry + order
  <run>/<page_stem>/overlay.png          debug image with numbered boxes

Usage:
    .venv/bin/python research-v2/split_panels.py                          # default: data/pages
    .venv/bin/python research-v2/split_panels.py --input-dir PATH --confidence 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Reuse pipeline_v1 modules verbatim (same weights, ordering, fallbacks).
PIPELINE_V1 = Path(__file__).resolve().parents[1] / "pipeline_v1"
sys.path.insert(0, str(PIPELINE_V1))

from PIL import Image  # noqa: E402

from detection import PanelBox, YoloPanelDetector, list_page_images  # noqa: E402
from extraction import draw_overlay, save_panels  # noqa: E402
from panel_ordering import reading_order  # noqa: E402
from steps.panels import ink_ratio  # noqa: E402
from util import sha256  # noqa: E402

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "data" / "pages"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_INSET = 0
DEFAULT_BLANK_INK_THRESHOLD = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split manga pages into panels (pipeline_v1 method)."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="directory with page images (default: research-v2/data/pages)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="parent of the timestamped run dir (default: research-v2/output)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help="YOLO panel confidence threshold (default: 0.25)")
    parser.add_argument("--panel-inset", type=int, default=DEFAULT_INSET,
                        help="px trimmed from each side of every panel crop (default: 0)")
    parser.add_argument("--blank-ink-threshold", type=float, default=DEFAULT_BLANK_INK_THRESHOLD,
                        help="ink ratio below which a page is blank (default: 0.005)")
    parser.add_argument("--no-full-page-fallback", dest="full_page_fallback",
                        action="store_false", default=True,
                        help="disable the synthetic full-page box for sparse art")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="skip the first N pages in natural order")
    parser.add_argument("--limit", type=int, default=0,
                        help="process at most N pages (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = list_page_images(args.input_dir)
    if args.skip_first:
        pages = pages[args.skip_first:]
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        raise SystemExit(f"No supported page images found in {args.input_dir}")

    run_dir = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    detector = YoloPanelDetector(confidence=args.confidence)
    page_records: list[dict] = []
    total_panels = 0
    n_blank = 0
    n_fallback = 0

    for page_path in pages:
        with Image.open(page_path) as image:
            image = image.convert("RGB")
            detections = detector.detect(page_path)
            order = reading_order(detections)
            ordered = [detections[i] for i in order]

            page_dir = run_dir / page_path.stem
            page_dir.mkdir(parents=True, exist_ok=True)
            provenance = "yolo"
            fallback = False
            blank_page = False
            skip_reason = None

            if not ordered:
                ink = ink_ratio(image)
                blank_page = ink < args.blank_ink_threshold
                if blank_page:
                    skip_reason = "blank-page"
                elif args.full_page_fallback:
                    ordered = [PanelBox(0, 0, image.width, image.height, 1.0)]
                    provenance = "full-page-fallback"
                    fallback = True

            records = []
            if ordered:
                records = save_panels(image, ordered, page_dir,
                                      inset=args.panel_inset)
                draw_overlay(image, ordered, page_dir / "overlay.png",
                             inset=args.panel_inset)

            geometry = _panels_json(
                page_path, ordered, records, order,
                provenance=provenance, fallback=fallback,
                blank_page=blank_page, skip_reason=skip_reason,
            )
            (page_dir / "panels.json").write_text(
                json.dumps(geometry, indent=2, ensure_ascii=False) + "\n"
            )
            page_records.append(geometry)
            total_panels += len(ordered)
            n_blank += blank_page
            n_fallback += fallback
            status = f"detected {len(ordered)} panels"
            if blank_page:
                status = "blank page (skipped)"
            elif fallback:
                status = "full-page fallback (1 panel)"
            print(f"{page_path.name}: {status}", flush=True)

    manifest = {
        "command": "research-v2/split_panels.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "input_dir": str(args.input_dir),
            "confidence": args.confidence,
            "panel_inset": args.panel_inset,
            "blank_ink_threshold": args.blank_ink_threshold,
            "full_page_fallback": args.full_page_fallback,
            "skip_first": args.skip_first,
            "limit": args.limit,
        },
        "backend": {
            "detector": "YOLO26n leoxs22/manga-panel-detector-yolo26n "
                        "(manga_panel_detector_fp32.pt, imgsz=640)",
            "cost": "self-hosted / local, $0 per call",
        },
        "totals": {
            "pages": len(page_records),
            "panels": total_panels,
            "blank_pages_skipped": n_blank,
            "full_page_fallbacks": n_fallback,
        },
        "pages": page_records,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print(
        f"\n{len(page_records)} pages, {total_panels} panels "
        f"({n_blank} blank skipped, {n_fallback} full-page fallbacks) -> {run_dir}",
        flush=True,
    )


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
    """Same schema as pipeline_v1 steps/panels.py (consumed by future v2
    stages and by the stitch step)."""
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


if __name__ == "__main__":
    main()
