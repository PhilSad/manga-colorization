"""Tests for panel_ordering.py: Japanese reading order on synthetic layouts."""

from __future__ import annotations

import pytest

from detection import PanelBox
from panel_ordering import reading_order


def box(x1, y1, x2, y2, conf=0.9):
    return PanelBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


def indices(panels, order):
    """Map reading-order positions back to the expected panel indices."""
    return order


def test_empty():
    assert reading_order([]) == []


def test_single_panel():
    assert reading_order([box(0, 0, 100, 100)]) == [0]


def test_two_by_two_grid():
    """Two rows x two columns; each row reads right-to-left, rows top-to-bottom."""
    panels = [
        box(0, 0, 100, 100),     # 0: top-left
        box(110, 0, 210, 100),   # 1: top-right
        box(0, 110, 100, 210),   # 2: bottom-left
        box(110, 110, 210, 210), # 3: bottom-right
    ]
    assert reading_order(panels) == [1, 0, 3, 2]


def test_right_column_two_small_plus_left_tall():
    """Right column split in two stacked small panels, left panel tall."""
    panels = [
        box(0, 0, 100, 300),     # 0: left tall
        box(110, 0, 210, 140),   # 1: right-top
        box(110, 150, 210, 300), # 2: right-bottom
    ]
    assert reading_order(panels) == [1, 2, 0]


def test_title_banner_across_top():
    """Wide banner panel across the top reads first, then the row below."""
    panels = [
        box(0, 0, 300, 60),      # 0: banner
        box(0, 70, 140, 200),    # 1: bottom-left
        box(150, 70, 300, 200),  # 2: bottom-right
    ]
    assert reading_order(panels) == [0, 2, 1]


def test_three_columns_single_row():
    panels = [
        box(0, 0, 80, 200),      # left
        box(90, 0, 170, 200),    # middle
        box(180, 0, 260, 200),   # right
    ]
    assert reading_order(panels) == [2, 1, 0]


def test_bottom_wide_panel_reads_last():
    """A wide bottom panel (below both top panels) reads after the top row."""
    panels = [
        box(0, 0, 100, 60),      # 0: top-left small
        box(110, 0, 210, 60),    # 1: top-right small
        box(0, 70, 210, 250),    # 2: wide bottom panel spanning below
    ]
    order = reading_order(panels)
    assert order == [1, 0, 2]


def test_overlapping_boxes_still_total_order():
    """Overlapping detections (not expected from the model) never crash."""
    panels = [
        box(10, 10, 110, 110),
        box(50, 50, 150, 150),
        box(200, 200, 300, 300),
    ]
    order = reading_order(panels)
    assert sorted(order) == [0, 1, 2]
    assert len(order) == 3


def test_reading_order_positions_are_1_based_contiguous():
    panels = [box(i * 10, 0, i * 10 + 5, 5) for i in range(5)]
    order = reading_order(panels)
    assert sorted(order) == list(range(5))
