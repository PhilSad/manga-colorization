"""Offline unit tests for luna_sanity.py (no client, no network).

These tests pin the structured-output verdict parser, the strict
json_schema contract, the analysis-grid preparation, and the request shape
the checker sends (response_format json_schema +
`provider.require_parameters: true`, no temperature, and no silent
downgrade on a BadRequest rejection). The CLI companion
(scripts/check_luna_sanity.py) is validated by manual runs against real
runs/backends, not by this suite.
"""

from __future__ import annotations

import base64
import io

from PIL import Image
from openai import BadRequestError

from test_characters import FakeClient, FakeResponse, FakeUsage, make_openai_error
from luna_sanity import (
    DEFAULT_MAX_EDGE,
    LINE_ART_VERDICT_SCHEMA,
    RESPONSE_FORMAT,
    LineArtChecker,
    analysis_pair,
    parse_line_art_verdict,
)


def _panel(path, size=(32, 32), color="white"):
    Image.new("RGB", size, color).save(path)
    return path


def _pair(tmp_path, bw_size=(640, 800), col_size=(660, 820)):
    bw = Image.new("RGB", bw_size, "white")
    col = Image.new("RGB", col_size, "white")
    return bw, col


# ---------------------------------------------------------------------------
# Verdict parsing

def test_parse_line_art_verdict_ok():
    verdict = parse_line_art_verdict(
        '{"analyse": "all strokes preserved, color follows the lines", '
        '"line_art_matches": true}'
    )
    assert verdict["line_art_matches"] is True
    assert "preserved" in verdict["analyse"]


def test_parse_line_art_verdict_mismatch():
    verdict = parse_line_art_verdict(
        '{"analyse": "hair strands redrawn", "line_art_matches": false}'
    )
    assert verdict["line_art_matches"] is False
    assert verdict["analyse"] == "hair strands redrawn"


def test_parse_line_art_verdict_string_bool_and_fenced_json():
    verdict = parse_line_art_verdict(
        "```json\n{\"analyse\": \"speed lines dropped\", "
        "\"line_art_matches\": \"false\"}\n```"
    )
    assert verdict["line_art_matches"] is False
    assert verdict["analyse"] == "speed lines dropped"


def test_parse_line_art_verdict_rejects_malformed():
    assert parse_line_art_verdict("") is None
    assert parse_line_art_verdict("no json here") is None
    assert parse_line_art_verdict('{"analyse": "missing the flag"}') is None
    assert parse_line_art_verdict('{"line_art_matches": "maybe"}') is None
    assert parse_line_art_verdict("[1, 2, 3]") is None


def test_parse_line_art_verdict_missing_analyse_defaults():
    verdict = parse_line_art_verdict('{"line_art_matches": true}')
    assert verdict["line_art_matches"] is True
    assert verdict["analyse"] == ""


# ---------------------------------------------------------------------------
# Schema contract

def test_line_art_verdict_schema_is_strict_and_complete():
    """The verdict contract: analyse + line_art_matches, both required, no
    extra properties, strict json_schema mode."""
    schema = LINE_ART_VERDICT_SCHEMA
    assert set(schema["properties"]) == {"analyse", "line_art_matches"}
    assert schema["properties"]["analyse"]["type"] == "string"
    assert schema["properties"]["line_art_matches"]["type"] == "boolean"
    assert schema["required"] == ["analyse", "line_art_matches"]
    assert schema["additionalProperties"] is False
    assert RESPONSE_FORMAT["type"] == "json_schema"
    assert RESPONSE_FORMAT["json_schema"]["name"] == "line_art_verdict"
    assert RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert RESPONSE_FORMAT["json_schema"]["schema"] is LINE_ART_VERDICT_SCHEMA


# ---------------------------------------------------------------------------
# Analysis-grid preparation

def test_analysis_pair_uses_bw_grid():
    bw, col = _pair(object())
    bw_g, col_g, size = analysis_pair(bw, col, max_edge=320)
    # bw 800-longest -> 320 cap -> 256x320; color resampled onto that grid.
    assert size == (256, 320)
    assert (bw_g.width, bw_g.height) == (256, 320)
    assert (col_g.width, col_g.height) == (256, 320)


def test_analysis_pair_noop_below_max_edge():
    bw, col = _pair(object(), bw_size=(300, 400), col_size=(310, 410))
    bw_g, col_g, size = analysis_pair(bw, col, max_edge=1024)
    assert size == (300, 400)
    assert (col_g.width, col_g.height) == (300, 400)


# ---------------------------------------------------------------------------
# Checker client

def _content_images(call):
    return [part for part in call["messages"][0]["content"]
            if part["type"] == "image_url"]


def _decode_image(url):
    assert url.startswith("data:image/png;base64,")
    payload = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def test_check_ok_status_and_cost(tmp_path):
    bw, col = _pair(tmp_path)

    def ok():
        return FakeResponse(
            '{"analyse": "line art matches", "line_art_matches": true}',
            usage=FakeUsage(cost=0.0001234),
        )

    checker = LineArtChecker(client=FakeClient([ok]))
    record = checker.check(col, bw, max_edge=320)

    assert record.status == "ok"
    assert record.line_art_matches is True
    assert record.analysis_size == (256, 320)
    assert record.cost_usd == 0.0001234
    assert record.cost_source == "usage.cost"

    call = checker.client.chat.completions.calls[0]
    assert call["response_format"] == RESPONSE_FORMAT
    assert call["extra_body"] == {"provider": {"require_parameters": True}}
    assert "temperature" not in call
    # Two images: colorized first, B&W second; both on the analysis grid.
    images = _content_images(call)
    assert len(images) == 2
    for part in images:
        image = _decode_image(part["image_url"]["url"])
        assert (image.width, image.height) == (256, 320)


def test_check_mismatch_status(tmp_path):
    bw, col = _pair(tmp_path)

    def bad():
        return FakeResponse(
            '{"analyse": "hair strands redrawn", "line_art_matches": false}',
            usage=FakeUsage(cost=0.00005),
        )

    checker = LineArtChecker(client=FakeClient([bad]))
    record = checker.check(col, bw)

    assert record.status == "mismatch"
    assert record.line_art_matches is False
    assert "hair" in record.analyse


def test_check_as_is_with_prepared_images(tmp_path):
    """max_edge=None sends the (already prepared) images as-is and records
    their size."""
    bw = Image.new("RGB", (256, 320), "white")
    col = Image.new("RGB", (256, 320), "white")

    def ok():
        return FakeResponse(
            '{"analyse": "fine", "line_art_matches": true}',
            usage=FakeUsage(),
        )

    checker = LineArtChecker(client=FakeClient([ok]))
    record = checker.check(col, bw, max_edge=None)

    assert record.analysis_size == (256, 320)
    images = _content_images(checker.client.chat.completions.calls[0])
    assert len(images) == 2
    for part in images:
        image = _decode_image(part["image_url"]["url"])
        assert (image.width, image.height) == (256, 320)


def test_check_unparseable_content(tmp_path):
    bw, col = _pair(tmp_path)

    def garbage():
        return FakeResponse("not json at all", usage=FakeUsage())

    checker = LineArtChecker(client=FakeClient([garbage]))
    record = checker.check(col, bw)

    assert record.status == "unparseable"
    assert record.line_art_matches is None


def test_check_bad_request_no_downgrade(tmp_path):
    """A rejection of the strict json_schema is an error, never retried as
    loose JSON — same convention as verify_color.py."""
    bw, col = _pair(tmp_path)
    bad = make_openai_error(BadRequestError, "bad request", status=400)

    checker = LineArtChecker(client=FakeClient([bad]))
    record = checker.check(col, bw)

    assert record.status == "error"
    assert record.line_art_matches is None
    calls = checker.client.chat.completions.calls
    assert len(calls) == 1
    assert calls[0]["response_format"] == RESPONSE_FORMAT


def test_check_not_found_not_retried(tmp_path):
    """A routing 404 is deterministic — one attempt, no backoff retries."""
    from openai import NotFoundError

    bw, col = _pair(tmp_path)
    not_found = make_openai_error(
        NotFoundError,
        "No endpoints found that can handle the requested parameters",
        status=404,
    )
    checker = LineArtChecker(client=FakeClient([not_found]))
    record = checker.check(col, bw)

    assert record.status == "error"
    assert "NotFoundError" in record.error
    assert len(checker.client.chat.completions.calls) == 1


def test_record_to_dict_shape(tmp_path):
    bw, col = _pair(tmp_path)

    def ok():
        return FakeResponse(
            '{"analyse": "matches", "line_art_matches": true}',
            usage=FakeUsage(cost=0.00001),
        )

    checker = LineArtChecker(client=FakeClient([ok]))
    record = checker.check(col, bw, max_edge=320)
    doc = record.to_dict()
    assert doc["status"] == "ok"
    assert doc["line_art_matches"] is True
    assert doc["analyse"] == "matches"
    assert doc["analysis_size"] == [256, 320]
    assert doc["cost_source"] == "usage.cost"
    assert doc["error"] is None
    assert set(doc) == {
        "status", "line_art_matches", "analyse", "response_text", "usage",
        "cost_usd", "cost_source", "latency_s", "model_returned", "attempts",
        "analysis_size", "error", "finished_at",
    }
