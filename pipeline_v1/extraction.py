"""Panel extraction: crop detected panels and name them in reading order."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from detection import PanelBox


def crop_panel(page: Image.Image, box: PanelBox, inset: int = 0) -> Image.Image:
    """Crop the panel region from the page.

    `inset` trims that many pixels from each side (used to drop the panel
    border if the detector box hugs the frame). The crop is clipped to the
    page bounds.
    """
    x1, y1, x2, y2 = box.as_int_tuple()
    x1 = max(0, x1 + inset)
    y1 = max(0, y1 + inset)
    x2 = min(page.width, x2 - inset)
    y2 = min(page.height, y2 - inset)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"panel box {box} collapses to nothing after inset={inset}")
    return page.crop((x1, y1, x2, y2))


def panel_filename(index: int, extension: str = ".png", prefix: str = "panel") -> str:
    """Zero-padded panel filename, e.g. panel_0001.png (matches data/panels/)."""
    return f"{prefix}_{index:04d}{extension}"


def save_panels(
    page: Image.Image,
    detections: list[PanelBox],
    out_dir: Path,
    *,
    inset: int = 0,
    prefix: str = "panel",
    extension: str = ".png",
) -> list[dict]:
    """Crop each panel and save it as `out_dir/<prefix>_NNNN<ext>`.

    `detections` must already be in reading order (see
    panel_ordering.reading_order). Returns the per-panel records:
    {panel_index, crop, filename}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index, box in enumerate(detections, start=1):
        crop = crop_panel(page, box, inset=inset)
        filename = panel_filename(index, extension, prefix)
        crop.save(out_dir / filename)
        records.append({"panel_index": index, "filename": filename})
    return records


def draw_overlay(
    page: Image.Image,
    detections: list[PanelBox],
    out_path: Path,
    *,
    inset: int = 0,
) -> None:
    """Debug image: the page with each panel box drawn and numbered in
    reading order (indices correspond to `detections` order, 1-based).
    Numbers are drawn large on a white badge so they are readable even on
    dense linework."""
    from util import load_font

    overlay = page.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = load_font(56)
    for index, box in enumerate(detections, start=1):
        x1, y1, x2, y2 = box.as_int_tuple()
        x1, y1 = x1 + inset, y1 + inset
        x2, y2 = x2 - inset, y2 - inset
        draw.rectangle((x1, y1, x2, y2), outline=(220, 30, 30), width=4)
        text = str(index)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        label_w = right - left
        label_h = bottom - top
        pad = 6
        badge = (x1 + 6, y1 + 6, x1 + 6 + label_w + 2 * pad, y1 + 6 + label_h + 2 * pad)
        draw.rectangle(badge, fill="white", outline=(220, 30, 30), width=3)
        draw.text(
            (x1 + 6 + pad, y1 + 6 + pad - top), text,
            fill=(200, 0, 0), font=font,
        )
    overlay.save(out_path)
