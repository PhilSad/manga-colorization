"""Offline unit tests for verify_color.py parsing (no client, no network).

The evaluation itself is tested by the real-network integration suite
(test_integration_*.py, `pytest -m integration`); these tests only pin the
JSON-verdict parsers shared by the L2R and palette verifiers.
"""

from __future__ import annotations

from verify_color import parse_l2r_verdict, parse_palette_verdict


def test_parse_l2r_verdict_ok():
    verdict = parse_l2r_verdict(
        '{"left_to_right_matches": true, '
        '"per_position": [{"position": 1, "character": "Heiter", "matches": true}], '
        '"notes": "all good"}'
    )
    assert verdict["left_to_right_matches"] is True
    assert len(verdict["per_position"]) == 1
    assert verdict["notes"] == "all good"


def test_parse_l2r_verdict_string_bool_and_fenced_json():
    verdict = parse_l2r_verdict(
        "```json\n{\"left_to_right_matches\": \"false\", \"notes\": \"swapped\"}\n```"
    )
    assert verdict["left_to_right_matches"] is False


def test_parse_l2r_verdict_rejects_malformed():
    assert parse_l2r_verdict("") is None
    assert parse_l2r_verdict("no json here") is None
    assert parse_l2r_verdict('{"notes": "missing the flag"}') is None
    assert parse_l2r_verdict('{"left_to_right_matches": "maybe"}') is None


def test_parse_palette_verdict_ok():
    verdict = parse_palette_verdict(
        '{"adheres": true, "missing_required": [], "present_forbidden": [], '
        '"notes": "canonical palette respected"}'
    )
    assert verdict["adheres"] is True
    assert verdict["missing_required"] == []
    assert verdict["present_forbidden"] == []


def test_parse_palette_verdict_lists_are_strings_and_defaulted():
    verdict = parse_palette_verdict(
        '{"adheres": false, "missing_required": ["silver-white hair"], '
        '"present_forbidden": ["magenta hair"], "notes": "off palette"}'
    )
    assert verdict["adheres"] is False
    assert verdict["missing_required"] == ["silver-white hair"]
    assert verdict["present_forbidden"] == ["magenta hair"]
    # missing lists default to empty, never crash
    assert parse_palette_verdict('{"adheres": true}')["missing_required"] == []
    assert parse_palette_verdict('{"adheres": true}')["present_forbidden"] == []


def test_parse_palette_verdict_rejects_malformed():
    assert parse_palette_verdict("") is None
    assert parse_palette_verdict('{"missing_required": []}') is None
    assert parse_palette_verdict('{"adheres": "not-a-bool"}') is None
