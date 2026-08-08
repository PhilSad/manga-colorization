"""Builder for a manga-like synthetic page used by the end-to-end tests.

Layout (unambiguous Japanese reading order): a full-width banner panel on top,
then a 2x2 grid below. Reading order: banner (1), top-right (2), top-left (3),
bottom-right (4), bottom-left (5).

```text
+----------------------------------------+
|  banner  (full width)                  |  1
+------------------+---------------------+
|  left-top    (3) |  right-top    (2)   |
|                  |                     |
+------------------+---------------------+
|  left-bottom (5) |  right-bottom (4)   |
|                  |                     |
+------------------+---------------------+
```

Each panel is filled with a distinct color so crops/stitching can be verified
by pixels.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

PAGE_SIZE = (500, 700)

# panel_id -> (box, rgb)
PANELS: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int]]] = {
    "banner": ((20, 20, 480, 120), (200, 200, 200)),
    "right_top": ((300, 140, 480, 400), (150, 150, 150)),
    "left_top": ((20, 140, 280, 400), (110, 110, 110)),
    "right_bottom": ((300, 420, 480, 680), (90, 90, 90)),
    "left_bottom": ((20, 420, 280, 680), (60, 60, 60)),
}

# Expected reading order (reading-order number -> panel_id).
READING_ORDER: list[str] = [
    "banner", "right_top", "left_top", "right_bottom", "left_bottom",
]


def build_page(path: Path | None = None) -> Image.Image:
    page = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    for (x1, y1, x2, y2), color in PANELS.values():
        draw.rectangle((x1, y1, x2, y2), fill=color)
    # Page number in the corner (outside any panel).
    draw.text((460, 684), "134", fill=(0, 0, 0))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        page.save(path)
    return page


def panel_box(panel_id: str) -> tuple[int, int, int, int]:
    box, _color = PANELS[panel_id]
    return box


def panel_color(panel_id: str) -> tuple[int, int, int]:
    _box, color = PANELS[panel_id]
    return color
