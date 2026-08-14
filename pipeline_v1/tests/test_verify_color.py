"""Offline unit tests for verify_color.py (no client, no network).

The evaluation itself is tested by the real-network integration suite
(test_integration_*.py, `pytest -m integration`); these tests pin the
structured-output verdict parser, the strict json_schema contract, and the
request shape the verifier sends (response_format json_schema +
`provider.require_parameters: true`, and no silent downgrade on a
BadRequest rejection).
"""

from __future__ import annotations

from PIL import Image
from openai import BadRequestError

from test_characters import FakeClient, FakeResponse, FakeUsage, make_openai_error
from verify_color import (
    COLOR_VERDICT_SCHEMA,
    RESPONSE_FORMAT,
    ColorVerifier,
    parse_color_verdict,
)


def _panel(path, size=(32, 32), color="white"):
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------
# Verdict parsing

def test_parse_color_verdict_ok():
    verdict = parse_color_verdict(
        '{"analyse": "Frieren hair is silver-white as expected", '
        '"good_color": true}'
    )
    assert verdict["good_color"] is True
    assert "silver-white" in verdict["analyse"]


def test_parse_color_verdict_string_bool_and_fenced_json():
    verdict = parse_color_verdict(
        "```json\n{\"analyse\": \"everyone is blue\", \"good_color\": \"false\"}\n```"
    )
    assert verdict["good_color"] is False
    assert verdict["analyse"] == "everyone is blue"


def test_parse_color_verdict_rejects_malformed():
    assert parse_color_verdict("") is None
    assert parse_color_verdict("no json here") is None
    assert parse_color_verdict('{"analyse": "missing the flag"}') is None
    assert parse_color_verdict('{"good_color": "maybe"}') is None
    assert parse_color_verdict("[1, 2, 3]") is None


def test_parse_color_verdict_missing_analyse_defaults():
    verdict = parse_color_verdict('{"good_color": true}')
    assert verdict["good_color"] is True
    assert verdict["analyse"] == ""


# ---------------------------------------------------------------------------
# Structured-output contract

def test_color_verdict_schema_is_strict_and_complete():
    """The structured-output contract: exactly analyse + good_color, both
    required, no extra properties, strict mode enabled, descriptive props."""
    schema = COLOR_VERDICT_SCHEMA
    assert set(schema["properties"]) == {"analyse", "good_color"}
    assert schema["properties"]["analyse"]["type"] == "string"
    assert schema["properties"]["good_color"]["type"] == "boolean"
    assert schema["required"] == ["analyse", "good_color"]
    assert schema["additionalProperties"] is False
    assert RESPONSE_FORMAT["type"] == "json_schema"
    assert RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert RESPONSE_FORMAT["json_schema"]["name"] == "color_verdict"
    assert RESPONSE_FORMAT["json_schema"]["schema"] is COLOR_VERDICT_SCHEMA


# ---------------------------------------------------------------------------
# Verifier request shape (fake client)

def test_verify_sends_structured_output_request(tmp_path):
    """One verify call sends the strict json_schema response_format and
    `provider.require_parameters` (never silent degradation), and maps the
    structured verdict onto the record status."""
    colorized = _panel(tmp_path / "colorized.png")
    crop = _panel(tmp_path / "crop.png")

    def ok():
        return FakeResponse(
            '{"analyse": "all canonical palettes", "good_color": true}',
            usage=FakeUsage(cost=0.0001),
        )

    verifier = ColorVerifier(
        model="openai/gpt-5.6-luna", client=FakeClient([ok])
    )
    record = verifier.verify(colorized, crop)

    assert record.status == "verified"
    assert record.good_color is True
    assert record.analyse == "all canonical palettes"
    assert record.cost_source == "usage.cost"

    call = verifier.client.chat.completions.calls[0]
    assert call["response_format"] == RESPONSE_FORMAT
    assert call["extra_body"] == {"provider": {"require_parameters": True}}


def test_verify_mismatch_when_good_color_false(tmp_path):
    colorized = _panel(tmp_path / "colorized.png")

    def bad():
        return FakeResponse(
            '{"analyse": "hair is lavender instead of silver-white", '
            '"good_color": false}',
            usage=FakeUsage(),
        )

    verifier = ColorVerifier(client=FakeClient([bad]))
    record = verifier.verify(colorized, None)
    assert record.status == "mismatch"
    assert record.good_color is False
    assert "lavender" in record.analyse


def test_verify_bad_request_does_not_downgrade(tmp_path):
    """Structured mode must not retry without response_format: a BadRequest
    (e.g. endpoint without structured-output support) is recorded as an
    error after exactly one attempt."""
    colorized = _panel(tmp_path / "colorized.png")

    unsupported = make_openai_error(
        BadRequestError,
        "model does not support structured outputs",
        status=400,
    )
    verifier = ColorVerifier(client=FakeClient([unsupported]))
    record = verifier.verify(colorized, None)

    assert record.status == "error"
    assert "BadRequestError" in record.error
    calls = verifier.client.chat.completions.calls
    assert len(calls) == 1
    # the failing call still carried the strict schema + require_parameters
    assert calls[0]["response_format"] == RESPONSE_FORMAT
    assert calls[0]["extra_body"] == {"provider": {"require_parameters": True}}


def test_verify_unparseable_content(tmp_path):
    colorized = _panel(tmp_path / "colorized.png")

    def garbage():
        return FakeResponse("not json at all", usage=FakeUsage())

    verifier = ColorVerifier(client=FakeClient([garbage]))
    record = verifier.verify(colorized, None)
    assert record.status == "unparseable"
    assert record.good_color is None
