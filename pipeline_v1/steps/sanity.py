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
  YOLO panel detector                full-page runs: the recorded geometry is one
                                     synthetic box covering the whole page, so the
                                     real panels are re-extracted from the B&W page
                                     (YoloPanelDetector + reading order) and the
                                     stitched page is cropped at those boxes (scaled
                                     to the stitched page's resolution)

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

# Provenance values that mean "one synthetic box covering the whole page"
# instead of real per-panel YOLO detections.
FULL_PAGE_PROVENANCES = ("full-page-mode", "full-page-fallback")


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


def _is_synthetic_full_page(geometry: dict) -> bool:
    """True when the recorded geometry is one synthetic box covering the
    whole page (full-page mode or the sparse-art full-page fallback) rather
    than real per-panel YOLO detections."""
    detections = geometry.get("detections") or []
    return (
        len(detections) == 1
        and detections[0].get("provenance") in FULL_PAGE_PROVENANCES
    )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def panel_check_tasks(
    geometry: dict,
    page_dir: Path,
    stitched: Image.Image | None,
    colorized_root: Path,
    fallback_stems: set[str],
    *,
    detector: object | None = None,
) -> list[dict]:
    """Per-panel check tasks for one page (shared by the sanity step and the
    offline tools scripts/check_sanity.py + scripts/check_luna_sanity.py).

    Normal runs: one task per recorded detection — the B&W crop file plus the
    recorded box (page coordinates == stitched-page coordinates).

    Full-page runs (one synthetic box covering the whole page): the YOLO panel
    detector is run on the B&W page to extract the real panels (Japanese
    reading order), and one task is returned per detected panel. `box` is the
    detection scaled to the stitched page's resolution (full-page passthrough
    pages can be smaller than the source), `bw_box` is the detection in B&W
    page coordinates, and `bw_path` is the full B&W page to crop `bw_box`
    from. When the whole full-page panel was stitched from B&W, every
    extracted panel is marked `bw_fallback=True` (trivially identical).

    `detector` is any object with `detect(page_path) -> list[PanelBox]`;
    defaults to a lazily-constructed `YoloPanelDetector` (torch loaded only
    when the synthetic branch actually runs).
    """
    if not _is_synthetic_full_page(geometry):
        tasks: list[dict] = []
        for detection in geometry.get("detections") or []:
            crop_name = detection["crop"]
            stem = Path(crop_name).stem
            bw_path = page_dir / crop_name
            if not bw_path.is_file():
                continue
            tasks.append({
                "page": page_dir.name,
                "crop": crop_name,
                "stem": stem,
                "colorized_crop": crop_name,
                "box": detection["box"],
                "bw_box": None,
                "bw_path": bw_path,
                "provenance": detection.get("provenance"),
                "bw_fallback": stem in fallback_stems,
                "stitched": stitched,
                "colorized_root": colorized_root,
            })
        return tasks

    # Synthetic full-page geometry: re-extract the real panels with YOLO.
    recorded_crop = (geometry.get("detections") or [{}])[0].get(
        "crop", "panel_0001.png"
    )
    bw_source = Path(geometry.get("page_path") or "")
    if not bw_source.is_file():
        candidate = page_dir / recorded_crop
        if candidate.is_file():
            bw_source = candidate
    if not bw_source.is_file():
        return []  # nothing to detect against

    if detector is None:
        from detection import DEFAULT_CONFIDENCE, YoloPanelDetector

        detector = YoloPanelDetector(confidence=DEFAULT_CONFIDENCE)
    from panel_ordering import reading_order

    detections = detector.detect(bw_source)
    order = reading_order(detections)
    ordered = [detections[i] for i in order]
    if not ordered:
        return []

    bw_width, bw_height = _image_size(bw_source)
    scale_x = scale_y = 1.0
    if stitched is not None:
        stitched_width, stitched_height = stitched.size
        scale_x = stitched_width / bw_width
        scale_y = stitched_height / bw_height

    fallback_recorded = any(
        Path(d.get("crop", "")).stem in fallback_stems
        for d in geometry.get("detections") or []
    )
    tasks = []
    for index, box in enumerate(ordered, start=1):
        crop_name = f"yolo_{index:04d}.png"
        tasks.append({
            "page": page_dir.name,
            "crop": crop_name,
            "stem": Path(crop_name).stem,
            "colorized_crop": recorded_crop,
            "box": [
                round(box.x1 * scale_x),
                round(box.y1 * scale_y),
                round(box.x2 * scale_x),
                round(box.y2 * scale_y),
            ],
            "bw_box": [
                round(box.x1), round(box.y1),
                round(box.x2), round(box.y2),
            ],
            "bw_path": bw_source,
            "provenance": "yolo",
            "bw_fallback": fallback_recorded,
            "stitched": stitched,
            "colorized_root": colorized_root,
            "synthetic": True,
        })
    return tasks


def run_sanity_step(
    ctx: RunContext,
    config: PipelineConfig,
    *,
    threshold: float | None = None,
    max_edge: int | None = None,
    page_substrings: tuple[str, ...] = (),
    output_dir: Path | None = None,
    detector: object | None = None,
) -> dict:
    """Run stage 7. `threshold` / `max_edge` / `page_substrings` /
    `output_dir` override the config / run layout (used by the standalone
    scripts/check_sanity.py tool); `detector` injects a panel detector for
    full-page runs (default: YoloPanelDetector)."""
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
        for task in panel_check_tasks(
            geometry,
            page_dir,
            stitched,
            colorized_root,
            fallback_map.get(page, set()),
            detector=detector,
        ):
            crop_name = task["crop"]
            stem = task["stem"]
            record: dict = {
                "panel": crop_name,
                "box": task["box"],
                "provenance": task["provenance"],
            }
            if task.get("bw_box"):
                record["bw_box"] = task["bw_box"]
            if task["bw_fallback"]:
                # Stitched from the original B&W crop: trivially identical.
                record.update({
                    "bw_fallback": True,
                    "flagged": False,
                    "note": "stitched from B&W crop; skipped (trivially identical)",
                })
                panel_records.append(record)
                continue

            with Image.open(task["bw_path"]) as bw_image:
                bw_image = bw_image.convert("RGB")
            if task.get("bw_box"):
                # Full-page runs: the B&W panel is the YOLO crop of the page.
                bw_cropped = _crop_box(bw_image, task["bw_box"])
                if bw_cropped is not None:
                    bw_image = bw_cropped
            color_image = None
            if stitched is not None:
                color_image = _crop_box(stitched, task["box"])
            if color_image is None:
                colorized_path = _find_colorized(
                    colorized_root / page, task["colorized_crop"]
                )
                if colorized_path is not None:
                    with Image.open(colorized_path) as colorized_image:
                        color_image = colorized_image.convert("RGB")
                    if task.get("synthetic"):
                        # Full-page colorized output: crop the real panel box.
                        color_cropped = _crop_box(color_image, task["box"])
                        if color_cropped is not None:
                            color_image = color_cropped

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
