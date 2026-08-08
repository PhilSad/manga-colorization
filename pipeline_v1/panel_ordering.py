"""Japanese reading order for manga panels.

Manga pages read right-to-left, top-to-bottom. Panels that share a horizontal
band are ordered rightmost first; bands are ordered top to bottom. A panel is
assigned to a band when its vertical interval overlaps the band's vertical
interval by more than a threshold fraction of the panel's height.
"""

from __future__ import annotations

from typing import Sequence

from detection import PanelBox

# A panel joins a band when overlap / panel_height exceeds this ratio.
OVERLAP_RATIO = 0.3


def _vertical_overlap(a: PanelBox, b: PanelBox) -> float:
    return max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))


def _bands(panels: Sequence[PanelBox]) -> list[list[int]]:
    """Cluster panel indices into horizontal bands (top to bottom)."""
    ordered = sorted(range(len(panels)), key=lambda i: (panels[i].y1, panels[i].x1))
    bands: list[list[int]] = []
    band_bounds: list[tuple[float, float]] = []  # (min_y1, max_y2) per band
    for index in ordered:
        box = panels[index]
        for position, (band_min_y1, band_max_y2) in enumerate(band_bounds):
            overlap = max(0.0, min(box.y2, band_max_y2) - max(box.y1, band_min_y1))
            if overlap > OVERLAP_RATIO * box.height:
                bands[position].append(index)
                band_bounds[position] = (
                    min(band_min_y1, box.y1),
                    max(band_max_y2, box.y2),
                )
                break
        else:
            bands.append([index])
            band_bounds.append((box.y1, box.y2))
    return bands


def reading_order(panels: Sequence[PanelBox]) -> list[int]:
    """Return panel indices (into `panels`) in Japanese reading order.

    Bands are ordered top to bottom; within a band, panels are ordered
    rightmost first. The returned list has one entry per panel; its position
    in the list is the reading-order number minus one.
    """
    if not panels:
        return []
    result: list[int] = []
    for band in _bands(panels):
        result.extend(sorted(band, key=lambda i: panels[i].x1, reverse=True))
    return result
