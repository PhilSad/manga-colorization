"""Stitch colorized panels back onto the original page.

The colorized panel from the FLUX server is at the nearest-multiple-of-16
resolution; it is resized back to the exact panel box (the panel's original
dimension, from panels.json) before being pasted. Everything outside the panel
boxes stays untouched (black & white).
"""

from __future__ import annotations

from typing import Sequence

from PIL import Image

from detection import PanelBox


def stitch_page(
    page: Image.Image,
    colorized: Sequence[tuple[PanelBox, Image.Image]],
    *,
    inset: int = 0,
    resize_filter: object = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Return a new page image with each colorized panel pasted at its box.

    `colorized` is a sequence of (box, panel_image) in any order (panels do
    not overlap); pasting order follows the given order.
    """
    result = page.convert("RGB").copy()
    for box, panel_image in colorized:
        x1 = max(0, round(box.x1) + inset)
        y1 = max(0, round(box.y1) + inset)
        x2 = min(result.width, round(box.x2) - inset)
        y2 = min(result.height, round(box.y2) - inset)
        if x2 <= x1 or y2 <= y1:
            continue
        target_size = (x2 - x1, y2 - y1)
        resized = panel_image.convert("RGB").resize(target_size, resize_filter)
        result.paste(resized, (x1, y1))
    return result
