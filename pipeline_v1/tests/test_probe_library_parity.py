"""Parity tests: the library helpers duplicated from the committed probe
scripts must not drift.

Per user decision 3 (docs/plans/verify-bbox-region-edit.md) the probes
(scripts/probe_luna_bboxes.py, scripts/probe_gpt_edit_bbox.py) stay
standalone and the library modules (verify_color.py, region_edit.py) carry
their own copies of `parse_bbox_verdict` / `draw_boxes` /
`region_instruction` — accepted duplication. These tests pin behavioral
parity: same input -> same output, so a future edit to either copy that
changes behavior fails loudly here instead of silently diverging.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

# Make the scripts dir importable (same pattern as test_probe_gpt_edit_bbox.py).
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from probe_gpt_edit_bbox import region_instruction as probe_region_instruction  # noqa: E402
from probe_luna_bboxes import draw_boxes as probe_draw_boxes  # noqa: E402
from probe_luna_bboxes import parse_bbox_verdict as probe_parse_bbox_verdict  # noqa: E402
from region_edit import draw_boxes, region_instruction  # noqa: E402
from verify_color import parse_bbox_verdict  # noqa: E402

REGIONS = [
    {"character": "Eisen", "problem": "beard was colored white",
     "fix_suggestion": "Eisen: beard golden-brown/blond",
     "bbox": [93, 197, 304, 375]},
    {"character": "Frieren", "problem": "hair lavender, should be silver-white",
     "fix_suggestion": "Frieren: hair silver-white",
     "bbox": [781, 154, 878, 240]},
    {"character": "Himmel", "problem": "cloak wrong shade",
     "fix_suggestion": "Himmel: cloak navy", "bbox": None},
]

VERDICT_TEXT = (
    '{"analyse": "Eisen beard white, Frieren hair lavender", '
    '"good_color": false, "fix_prompt": "Eisen: beard golden-brown", '
    '"regions": ['
    '{"character": "Eisen", "problem": "beard white", '
    '"fix_suggestion": "beard golden-brown", "bbox": [93, 197, 304, 375]}, '
    '{"character": "Frieren", "problem": "hair lavender", '
    '"fix_suggestion": "hair silver-white", "bbox": [781, 154, 878, 240]}, '
    '{"character": "Himmel", "problem": "cloak", '
    '"fix_suggestion": "cloak navy"}'
    ']}'
)


def _image(path: Path, size=(200, 300)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


# ---------------------------------------------------------------------------
# parse_bbox_verdict

def test_parse_bbox_verdict_parity():
    probe = probe_parse_bbox_verdict(VERDICT_TEXT)
    library = parse_bbox_verdict(VERDICT_TEXT)

    assert probe is not None and library is not None
    assert probe["good_color"] == library["good_color"]
    assert probe["analyse"] == library["analyse"]
    assert probe["regions"] == library["regions"]


def test_parse_bbox_verdict_parity_malformed():
    for text in ("", "no json", '{"good_color": "maybe"}', "[1, 2]"):
        assert probe_parse_bbox_verdict(text) is parse_bbox_verdict(text) is None


# ---------------------------------------------------------------------------
# region_instruction

def test_region_instruction_parity():
    assert probe_region_instruction(REGIONS) == region_instruction(REGIONS)
    # edge cases: missing fix_suggestion/problem, empty list
    assert probe_region_instruction([{"character": "?"}]) == region_instruction(
        [{"character": "?"}]
    )
    assert probe_region_instruction([]) == region_instruction([])


# ---------------------------------------------------------------------------
# draw_boxes

def test_draw_boxes_parity(tmp_path):
    """Both copies draw the same boxes on the same image: same pixels."""
    source = _image(tmp_path / "colorized.png")
    probe_out = tmp_path / "probe_annotated.png"
    library_out = tmp_path / "library_annotated.png"

    probe_draw_boxes(source, {"regions": REGIONS}, probe_out)
    draw_boxes(source, REGIONS, library_out)

    with Image.open(probe_out) as a, Image.open(library_out) as b:
        assert a.size == b.size
        assert a.tobytes() == b.tobytes()


def test_draw_boxes_parity_missing_bbox_skipped(tmp_path):
    """Regions without a bbox are skipped by both copies identically."""
    source = _image(tmp_path / "colorized.png")
    probe_out = tmp_path / "probe_annotated.png"
    library_out = tmp_path / "library_annotated.png"
    regions = [{"character": "Himmel", "problem": "p", "fix_suggestion": "s"}]

    probe_draw_boxes(source, {"regions": regions}, probe_out)
    draw_boxes(source, regions, library_out)

    with Image.open(probe_out) as a, Image.open(library_out) as b:
        assert a.tobytes() == b.tobytes()
