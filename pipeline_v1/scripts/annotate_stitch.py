#!/usr/bin/env python3
"""Debug annotation of a completed pipeline run's stitched pages.

Reads a pipeline run directory and writes a debug copy of `4_stitched/` with,
per panel: a bounding box and a label with the panel name + the characters
detected for it (from `2_characters/<page>/<panel>.json`).

Input layout (a completed run, e.g. pipeline_v1/output/YYYYMMDD-HHMMSS/):
  4_stitched/<page>.png            final pages
  1_panels/<page>/panels.json      panel geometry (boxes + crop names)
  2_characters/<page>/<panel>.json detected characters per panel
  manifest.json                    optional: B&W fallback panels (steps.stitch)

Output (default <run-dir>/5_debug/):
  <page>.png                       same page + bbox + character label per panel
  summary.json                     per-page provenance records

Panels that were stitched from their original black & white crop
(--stitch-bw-fallback) get an orange bbox and a "[B&W fallback]" tag.

Usage:
  .venv/bin/python pipeline_v1/scripts/annotate_stitch.py \
      --run-dir pipeline_v1/output/20260809-125148
  .venv/bin/python pipeline_v1/scripts/annotate_stitch.py \
      --run-dir pipeline_v1/output/20260809-125148 --page p077 --font-size 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Make the pipeline package importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util import SUPPORTED_IMAGE_SUFFIXES, load_font  # noqa: E402

NORMAL_COLOR = (220, 30, 30)      # red bbox + badge outline
FALLBACK_COLOR = (230, 110, 0)    # orange: stitched from the B&W crop
TEXT_COLOR = (150, 0, 0)
FALLBACK_TEXT_COLOR = (180, 80, 0)
BADGE_PAD = 8
BADGE_OFFSET = 6


def load_character_record(run_dir: Path, page: str, crop_stem: str) -> dict:
    """Character record for one panel ({} when missing/unreadable)."""
    path = run_dir / "2_characters" / page / f"{crop_stem}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def label_text(crop_name: str, record: dict) -> str:
    stem = Path(crop_name).stem
    names = record.get("characters", [])
    if names:
        return f"{stem}: {', '.join(names)}"
    if record.get("uncertain"):
        return f"{stem}: (uncertain)"
    if record:
        return f"{stem}: (no characters)"
    return f"{stem}: (no record)"


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> list[str]:
    """Word-wrap `text` to `max_width` px."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    text: str,
    font,
    outline_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    """White badge with the panel label, anchored at the box's top-left."""
    x1, y1, x2, y2 = (round(value) for value in box)
    box_w = max(120, x2 - x1 - 2 * BADGE_OFFSET)
    lines = wrap_lines(draw, text, font, box_w - 2 * BADGE_PAD)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 4
    widths = [draw.textlength(line, font=font) for line in lines]
    label_w = max(widths) + 2 * BADGE_PAD
    label_h = line_h * len(lines) + 2 * BADGE_PAD
    bx1 = x1 + BADGE_OFFSET
    by1 = y1 + BADGE_OFFSET
    draw.rectangle(
        (bx1, by1, bx1 + label_w, by1 + label_h),
        fill="white", outline=outline_color, width=3,
    )
    for index, line in enumerate(lines):
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        draw.text(
            (bx1 + BADGE_PAD, by1 + BADGE_PAD + index * line_h - top),
            line, fill=text_color, font=font,
        )


def load_fallback_map(run_dir: Path) -> dict[str, set[str]]:
    """page stem -> set of panel stems stitched from the B&W crop
    (from manifest.json steps.stitch.outputs[].panels_bw_fallback)."""
    fallback_map: dict[str, set[str]] = {}
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return fallback_map
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback_map
    for record in manifest.get("steps", {}).get("stitch", {}).get("outputs", []):
        fallback = record.get("panels_bw_fallback") or []
        fallback_map[record["page"]] = {Path(crop).stem for crop in fallback}
    return fallback_map


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="annotate_stitch",
        description=(
            "Debug copy of a run's stitched pages: panel bboxes + detected "
            "characters per panel."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed pipeline run directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: <run-dir>/5_debug).")
    parser.add_argument("--page", action="append", default=[], metavar="SUBSTR",
                        help="Only annotate pages whose name contains SUBSTR "
                             "(repeatable).")
    parser.add_argument("--font-size", type=int, default=42,
                        help="Label font size in px.")
    parser.add_argument("--bbox-width", type=int, default=5,
                        help="Bounding-box stroke width in px.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    stitched_dir = run_dir / "4_stitched"
    if not stitched_dir.is_dir():
        raise SystemExit(f"no 4_stitched/ dir in run: {run_dir}")

    out_dir = (args.output_dir or run_dir / "5_debug").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(
        path for path in stitched_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if args.page:
        pages = [p for p in pages if any(s in p.stem for s in args.page)]
    if not pages:
        raise SystemExit(f"no stitched pages to annotate in {stitched_dir}")

    font = load_font(args.font_size)
    fallback_map = load_fallback_map(run_dir)
    records: list[dict] = []
    for page_path in pages:
        page = page_path.stem
        geometry_path = run_dir / "1_panels" / page / "panels.json"
        detections: list[dict] = []
        if geometry_path.is_file():
            try:
                detections = json.loads(
                    geometry_path.read_text(encoding="utf-8")
                )["detections"]
            except (OSError, ValueError, KeyError):
                detections = []

        with Image.open(page_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            panels: list[dict] = []
            for detection in detections:
                crop = detection["crop"]
                stem = Path(crop).stem
                record = load_character_record(run_dir, page, stem)
                is_fallback = stem in fallback_map.get(page, set())
                color = FALLBACK_COLOR if is_fallback else NORMAL_COLOR
                text_color = FALLBACK_TEXT_COLOR if is_fallback else TEXT_COLOR
                draw.rectangle(
                    tuple(round(value) for value in detection["box"]),
                    outline=color, width=args.bbox_width,
                )
                text = label_text(crop, record)
                if is_fallback:
                    text += " [B&W fallback]"
                draw_label(draw, detection["box"], text, font, color, text_color)
                panels.append({
                    "panel": crop,
                    "characters": record.get("characters", []),
                    "bw_fallback": is_fallback,
                })
            output_path = out_dir / page_path.name
            image.save(output_path)

        records.append({
            "page": page,
            "output": str(output_path.resolve()),
            "panels": panels,
        })

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "pages_annotated": len(records),
        "records": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(records)} annotated pages to {out_dir}", flush=True)
    print(f"summary: {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - friendly failure output
        print(f"annotate_stitch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
