"""Pipeline stage 7: line-art fidelity sanity check.

Compares each colorized panel with its black & white original through the
structural line-metric scorer in `sanity.py` (thin-stroke line maps + IoU /
chamfer / large components / phase-correlation drift) and flags panels whose
line art drifted below the threshold for review. Pure local compute (numpy +
OpenCV, no backends).

Reads:
  1_panels/<page>/panel_000N.png     the black & white panel crops
  1_panels/<page>/panels.json        panel boxes (for cropping the stitched page)
  4_stitched/<page>.<ext>            the final page (colorized panels at box positions)
  3_colorized/<page>/panel_000N.png  fallback when 4_stitched is unavailable
  manifest.json                      B&W-fallback panels are skipped (trivially identical)

Writes:
  7_sanity/<page>.json               per-panel metrics + verdicts
  7_sanity/<page>_flagged.png        contact sheet of the flagged panels (side-by-side)
  7_sanity/summary.json              run totals + flagged list

The standalone offline tool scripts/check_sanity.py re-runs this step on any
completed run directory (custom threshold / page filter / output dir) without
re-running the pipeline; both share this implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from config import STEP_DIRS, PipelineConfig
from run_context import RunContext, write_json
from sanity import analysis_size, score_pair
from selection import page_selected
from steps.debug import load_fallback_map
from tqdm import tqdm
from util import SUPPORTED_IMAGE_SUFFIXES, load_font


def _find_colorized(page_dir: Path, crop_name: str) -> Path | None:
    """Colorized file for a crop (any supported extension), or None."""
    if not page_dir.is_dir():
        return None
    stem = Path(crop_name).stem
    for path in page_dir.iterdir():
        if path.stem == stem and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            return path
    return None


def _load_stitched(stitched_dir: Path, page: str) -> Image.Image | None:
    """The final stitched page as RGB, or None when not produced yet."""
    if not stitched_dir.is_dir():
        return None
    for path in stitched_dir.iterdir():
        if path.stem == page and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            with Image.open(path) as image:
                return image.convert("RGB")
    return None


def _crop_box(image: Image.Image, box: list[float]) -> Image.Image | None:
    """Crop the stitched page at a panel box, clipped to the image bounds
    (full-page passthrough pages can be smaller than the source page)."""
    width, height = image.size
    x1, y1, x2, y2 = (round(v) for v in box)
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def run_sanity_step(
    ctx: RunContext,
    config: PipelineConfig,
    *,
    threshold: float | None = None,
    max_edge: int | None = None,
    page_substrings: tuple[str, ...] = (),
    output_dir: Path | None = None,
) -> dict:
    """Run stage 7. `threshold` / `max_edge` / `page_substrings` /
    `output_dir` override the config / run layout (used by the standalone
    scripts/check_sanity.py tool)."""
    threshold = config.sanity_threshold if threshold is None else threshold
    max_edge = config.sanity_max_edge if max_edge is None else max_edge

    panels_root = ctx.run_dir / STEP_DIRS["panels"]
    stitched_dir = ctx.run_dir / STEP_DIRS["stitch"]
    colorized_root = ctx.run_dir / STEP_DIRS["colorize"]
    if not panels_root.is_dir():
        raise ValueError("no 1_panels/ dir in the run; run the 'panels' step first")

    page_dirs = sorted(path for path in panels_root.iterdir() if path.is_dir())
    if config.only_panels:
        page_dirs = [
            path for path in page_dirs
            if page_selected(path.name, config.only_panels)
        ]
    if page_substrings:
        page_dirs = [
            path for path in page_dirs
            if any(sub in path.name for sub in page_substrings)
        ]
    if not page_dirs:
        raise ValueError(f"no pages selected in {panels_root}")

    out_dir = output_dir or ctx.step_dir("sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    fallback_map = load_fallback_map(ctx, config)

    page_records: list[dict] = []
    totals_checked = 0
    totals_flagged = 0
    for page_dir in tqdm(page_dirs, desc="sanity: pages", unit="page", leave=False):
        page = page_dir.name
        geometry_path = page_dir / "panels.json"
        if not geometry_path.is_file():
            continue
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        if geometry.get("blank_page"):
            continue

        stitched = _load_stitched(stitched_dir, page)
        panel_records: list[dict] = []
        sheet_items: list[tuple[dict, Image.Image, Image.Image]] = []
        for detection in geometry["detections"]:
            crop_name = detection["crop"]
            stem = Path(crop_name).stem
            bw_path = page_dir / crop_name
            if not bw_path.is_file():
                continue

            record: dict = {
                "panel": crop_name,
                "box": detection["box"],
                "provenance": detection.get("provenance"),
            }
            if stem in fallback_map.get(page, set()):
                # Stitched from the original B&W crop: trivially identical.
                record.update({
                    "bw_fallback": True,
                    "flagged": False,
                    "note": "stitched from B&W crop; skipped (trivially identical)",
                })
                panel_records.append(record)
                continue

            with Image.open(bw_path) as bw_image:
                bw_image = bw_image.convert("RGB")
            color_image = None
            if stitched is not None:
                color_image = _crop_box(stitched, detection["box"])
            if color_image is None:
                colorized_path = _find_colorized(colorized_root / page, crop_name)
                if colorized_path is not None:
                    with Image.open(colorized_path) as colorized_image:
                        color_image = colorized_image.convert("RGB")

            if color_image is None:
                record.update({
                    "bw_fallback": False,
                    "flagged": True,
                    "note": "no colorized output available for comparison",
                })
                panel_records.append(record)
                totals_checked += 1
                totals_flagged += 1
                continue

            metrics = score_pair(bw_image, color_image,
                                 max_edge=max_edge, threshold=threshold)
            record.update(metrics)
            record["bw_fallback"] = False
            panel_records.append(record)
            totals_checked += 1
            if record["flagged"]:
                totals_flagged += 1
                sheet_items.append((record, bw_image, color_image))

        sheet_path = None
        if sheet_items:
            sheet_path = _contact_sheet(out_dir, page, sheet_items, max_edge)
        page_records.append({
            "page": page,
            "input": geometry.get("page", geometry.get("page_path")),
            "panels": panel_records,
            "flagged_panels": [r["panel"] for r in panel_records if r.get("flagged")],
            "contact_sheet": str(sheet_path) if sheet_path else None,
        })
        write_json(out_dir / f"{page}.json", page_records[-1])

    flagged_pages = [r for r in page_records if r["flagged_panels"]]
    summary = {
        "run_dir": str(ctx.run_dir),
        "output_dir": str(out_dir),
        "threshold": threshold,
        "max_edge": max_edge,
        "pages_checked": len(page_records),
        "panels_checked": totals_checked,
        "panels_flagged": totals_flagged,
        "flagged": [
            {"page": r["page"], "panels": r["flagged_panels"],
             "contact_sheet": r["contact_sheet"]}
            for r in flagged_pages
        ],
        "pages": page_records,
    }
    write_json(out_dir / "summary.json", summary)
    return {
        "outputs": page_records,
        "pages_checked": len(page_records),
        "panels_checked": totals_checked,
        "panels_flagged": totals_flagged,
        "threshold": threshold,
        "max_edge": max_edge,
        "output_dir": str(out_dir),
    }


def _contact_sheet(out_dir: Path, page: str,
                   items: list[tuple[dict, Image.Image, Image.Image]],
                   max_edge: int) -> Path:
    """Side-by-side B&W | colorized tiles for every flagged panel of a page,
    stacked vertically, red border + verdict label; one PNG per page."""
    tiles: list[Image.Image] = []
    for record, bw_image, color_image in items:
        size = analysis_size(bw_image.width, bw_image.height, max_edge)
        bw_tile = bw_image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        color_tile = color_image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        gap = 12
        label_height = 30
        pair = Image.new("RGB", (size[0] * 2 + gap * 3, size[1] + label_height),
                         "white")
        pair.paste(bw_tile, (gap, label_height))
        pair.paste(color_tile, (size[0] + gap * 2, label_height))
        draw = ImageDraw.Draw(pair)
        draw.rectangle([0, 0, pair.width - 1, pair.height - 1],
                       outline=(200, 20, 20), width=4)
        label = (
            f"{Path(record['panel']).stem}  fidelity="
            f"{record.get('line_fidelity', 'n/a')}  "
            f"{', '.join(record.get('reasons') or [record.get('note', '')])}"
        )
        font = load_font(24)
        max_width = pair.width - 2 * gap
        while font.getlength(label) > max_width and len(label) > 8:
            label = label[:-1]
        draw.text((gap, 4), label, fill=(150, 0, 0), font=font)
        tiles.append(pair)

    height = sum(tile.height for tile in tiles)
    width = max(tile.width for tile in tiles)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    path = out_dir / f"{page}_flagged.png"
    sheet.save(path)
    return path
