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
    BBOX_RESPONSE_FORMAT,
    BBOX_VERDICT_SCHEMA,
    COLOR_VERDICT_SCHEMA,
    RESPONSE_FORMAT,
    ColorVerifier,
    parse_bbox_verdict,
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
        '"good_color": true, "fix_prompt": ""}'
    )
    assert verdict["good_color"] is True
    assert "silver-white" in verdict["analyse"]
    assert verdict["fix_prompt"] == ""


def test_parse_color_verdict_fix_prompt():
    verdict = parse_color_verdict(
        '{"analyse": "hair is lavender", "good_color": false, '
        '"fix_prompt": "Frieren: hair silver-white, eyes teal"}'
    )
    assert verdict["good_color"] is False
    assert verdict["fix_prompt"] == "Frieren: hair silver-white, eyes teal"


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
# Bbox verdict (--verify-mode bbox): schema + parser

def test_bbox_verdict_schema_is_strict_and_complete():
    """The bbox contract: analyse + good_color + fix_prompt + regions[], all
    required, no extra properties; region items carry character/problem/
    fix_suggestion/bbox in normalized 0-1000 integer coordinates."""
    schema = BBOX_VERDICT_SCHEMA
    assert set(schema["properties"]) == {
        "analyse", "good_color", "fix_prompt", "regions"
    }
    assert schema["properties"]["fix_prompt"]["type"] == "string"
    assert schema["required"] == ["analyse", "good_color", "fix_prompt", "regions"]
    assert schema["additionalProperties"] is False
    region_schema = schema["properties"]["regions"]["items"]
    assert set(region_schema["properties"]) == {
        "character", "problem", "fix_suggestion", "bbox"
    }
    assert region_schema["required"] == [
        "character", "problem", "fix_suggestion", "bbox"
    ]
    assert region_schema["additionalProperties"] is False
    assert BBOX_RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert BBOX_RESPONSE_FORMAT["json_schema"]["schema"] is BBOX_VERDICT_SCHEMA


def test_parse_bbox_verdict_ok():
    verdict = parse_bbox_verdict(
        '{"analyse": "Eisen beard white", "good_color": false, '
        '"fix_prompt": "Eisen: beard golden-brown", '
        '"regions": [{"character": "Eisen", "problem": "beard white", '
        '"fix_suggestion": "beard golden-brown", '
        '"bbox": [93, 197, 304, 375]}]}'
    )
    assert verdict["good_color"] is False
    assert verdict["fix_prompt"] == "Eisen: beard golden-brown"
    assert len(verdict["regions"]) == 1
    region = verdict["regions"][0]
    assert region["character"] == "Eisen"
    assert region["bbox"] == [93, 197, 304, 375]


def test_parse_bbox_verdict_empty_regions_and_fenced_json():
    verdict = parse_bbox_verdict(
        "```json\n{\"analyse\": \"fine\", \"good_color\": \"true\", "
        "\"fix_prompt\": \"\", \"regions\": []}\n```"
    )
    assert verdict["good_color"] is True
    assert verdict["regions"] == []


def test_parse_bbox_verdict_missing_bbox_keeps_none():
    """A region without a usable bbox is kept with bbox=None (draw_boxes
    skips it; the region text is still recorded)."""
    verdict = parse_bbox_verdict(
        '{"analyse": "a", "good_color": false, "fix_prompt": "f", '
        '"regions": [{"character": "Frieren", "problem": "p", '
        '"fix_suggestion": "s"}, '
        '{"character": "Eisen", "problem": "p", "fix_suggestion": "s", '
        '"bbox": [0, 0, 10.4, 5.9]}]}'
    )
    assert verdict["regions"][0]["bbox"] is None
    assert verdict["regions"][1]["bbox"] == [0, 0, 10, 6]


def test_parse_bbox_verdict_out_of_range_not_clamped_by_parser():
    """The parser keeps raw coordinates; clamping to 0-1000 happens when the
    boxes are drawn (draw_boxes)."""
    verdict = parse_bbox_verdict(
        '{"analyse": "a", "good_color": false, "fix_prompt": "f", '
        '"regions": [{"character": "x", "problem": "p", '
        '"fix_suggestion": "s", "bbox": [-50, 0, 1200, 1000]}]}'
    )
    assert verdict["regions"][0]["bbox"] == [-50, 0, 1200, 1000]


def test_parse_bbox_verdict_rejects_malformed():
    assert parse_bbox_verdict("") is None
    assert parse_bbox_verdict("no json") is None
    assert parse_bbox_verdict('{"analyse": "missing flag"}') is None
    assert parse_bbox_verdict('{"good_color": "maybe"}') is None
    assert parse_bbox_verdict('{"good_color": false, "regions": "notalist"}') is None
    assert parse_bbox_verdict("[1, 2]") is None


# ---------------------------------------------------------------------------
# Structured-output contract

def test_color_verdict_schema_is_strict_and_complete():
    """The structured-output contract: analyse + good_color + fix_prompt, all
    required, no extra properties, strict mode enabled, descriptive props.

    fix_prompt is the superset field consumed by the verify loop
    (verify_loop.py); the eval suite reads only the first two fields and
    ignores the third."""
    schema = COLOR_VERDICT_SCHEMA
    assert set(schema["properties"]) == {"analyse", "good_color", "fix_prompt"}
    assert schema["properties"]["analyse"]["type"] == "string"
    assert schema["properties"]["good_color"]["type"] == "boolean"
    assert schema["properties"]["fix_prompt"]["type"] == "string"
    assert schema["required"] == ["analyse", "good_color", "fix_prompt"]
    assert schema["additionalProperties"] is False
    assert RESPONSE_FORMAT["type"] == "json_schema"
    assert RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert RESPONSE_FORMAT["json_schema"]["name"] == "color_verdict"
    assert RESPONSE_FORMAT["json_schema"]["schema"] is COLOR_VERDICT_SCHEMA


# ---------------------------------------------------------------------------
# Verifier request shape (fake client)

def test_verify_sends_structured_output_request(tmp_path):
    """One verify call sends the strict json_schema response_format and
    `provider.require_parameters` (never silent degradation), carries the
    colorized panel + crop + reference atlas as images, and maps the
    structured verdict onto the record status."""
    colorized = _panel(tmp_path / "colorized.png")
    crop = _panel(tmp_path / "crop.png")
    atlas = _panel(tmp_path / "atlas.jpg")

    def ok():
        return FakeResponse(
            '{"analyse": "all canonical palettes", "good_color": true}',
            usage=FakeUsage(cost=0.0001),
        )

    verifier = ColorVerifier(
        model="openai/gpt-5.6-luna", client=FakeClient([ok])
    )
    record = verifier.verify(colorized, crop, atlas=atlas)

    assert record.status == "verified"
    assert record.good_color is True
    assert record.analyse == "all canonical palettes"
    assert record.cost_source == "usage.cost"

    call = verifier.client.chat.completions.calls[0]
    assert call["response_format"] == RESPONSE_FORMAT
    assert call["extra_body"] == {"provider": {"require_parameters": True}}
    # gpt-5.6-luna does not support temperature; sending it would make
    # require_parameters reject every endpoint (routing 404).
    assert "temperature" not in call
    # text + colorized + crop + atlas = 4 content parts, atlas last
    parts = call["messages"][0]["content"]
    assert len(parts) == 4
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_verify_mismatch_when_good_color_false(tmp_path):
    colorized = _panel(tmp_path / "colorized.png")

    def bad():
        return FakeResponse(
            '{"analyse": "hair is lavender instead of silver-white", '
            '"good_color": false, "fix_prompt": "Frieren: hair silver-white"}',
            usage=FakeUsage(),
        )

    verifier = ColorVerifier(client=FakeClient([bad]))
    record = verifier.verify(colorized, None)
    assert record.status == "mismatch"
    assert record.good_color is False
    assert "lavender" in record.analyse
    assert record.fix_prompt == "Frieren: hair silver-white"


def test_verify_fix_prompt_defaults_empty_when_verdict_lacks_it(tmp_path):
    """Backwards compatibility with the pre-loop two-field verdict: a real
    strict output always carries fix_prompt, but the parser defaults it to ''
    when absent so older recorded responses stay parseable."""
    colorized = _panel(tmp_path / "colorized.png")

    def ok():
        return FakeResponse(
            '{"analyse": "fine", "good_color": true}',
            usage=FakeUsage(),
        )

    verifier = ColorVerifier(client=FakeClient([ok]))
    record = verifier.verify(colorized, None)
    assert record.status == "verified"
    assert record.fix_prompt == ""


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


def test_verify_bbox_mode_request_shape_and_regions(tmp_path):
    """--verify-mode bbox: the verifier sends the BBOX verdict schema with
    `reasoning: {effort: "high"}` merged into extra_body (on top of
    require_parameters), and maps the parsed regions onto the record."""
    colorized = _panel(tmp_path / "colorized.png")

    def bad():
        return FakeResponse(
            '{"analyse": "Eisen beard white", "good_color": false, '
            '"fix_prompt": "Eisen: beard golden-brown", '
            '"regions": [{"character": "Eisen", "problem": "beard white", '
            '"fix_suggestion": "beard golden-brown", '
            '"bbox": [93, 197, 304, 375]}]}',
            usage=FakeUsage(),
        )

    verifier = ColorVerifier(
        client=FakeClient([bad]),
        response_format=BBOX_RESPONSE_FORMAT,
        reasoning_effort="high",
        max_tokens=8192,
    )
    record = verifier.verify(colorized, None)

    assert record.status == "mismatch"
    assert record.fix_prompt == "Eisen: beard golden-brown"
    assert len(record.regions) == 1
    assert record.regions[0]["bbox"] == [93, 197, 304, 375]
    call = verifier.client.chat.completions.calls[0]
    assert call["response_format"] == BBOX_RESPONSE_FORMAT
    assert call["max_tokens"] == 8192
    assert call["extra_body"] == {
        "reasoning": {"effort": "high"},
        "provider": {"require_parameters": True},
    }
    assert "temperature" not in call


def test_verify_bbox_mode_verified_records_empty_regions(tmp_path):
    colorized = _panel(tmp_path / "colorized.png")

    def ok():
        return FakeResponse(
            '{"analyse": "all canonical", "good_color": true, '
            '"fix_prompt": "", "regions": []}',
            usage=FakeUsage(),
        )

    verifier = ColorVerifier(
        client=FakeClient([ok]),
        response_format=BBOX_RESPONSE_FORMAT,
        reasoning_effort="high",
    )
    record = verifier.verify(colorized, None)

    assert record.status == "verified"
    assert record.regions == []
    # regions omitted from to_dict when empty (fix-prompt records unchanged)
    assert "regions" not in record.to_dict()
    assert record.to_dict()["fix_prompt"] == ""


def test_verify_unparseable_content(tmp_path):
    colorized = _panel(tmp_path / "colorized.png")

    def garbage():
        return FakeResponse("not json at all", usage=FakeUsage())

    verifier = ColorVerifier(client=FakeClient([garbage]))
    record = verifier.verify(colorized, None)
    assert record.status == "unparseable"
    assert record.good_color is None


def test_call_vlm_merges_extra_body_with_require_parameters(tmp_path):
    """`extra_body` (the bbox probe's `reasoning: {effort: ...}`) is merged
    on top of `provider.require_parameters` for structured calls — the caller
    never has to re-specify the provider pin."""
    colorized = _panel(tmp_path / "colorized.png")

    def ok():
        return FakeResponse(
            '{"analyse": "fine", "good_color": true}',
            usage=FakeUsage(),
        )

    verifier = ColorVerifier(client=FakeClient([ok]))
    verifier.verify(
        colorized, None,
        extra_body={"reasoning": {"effort": "high"}},
    )

    call = verifier.client.chat.completions.calls[0]
    assert call["extra_body"] == {
        "reasoning": {"effort": "high"},
        "provider": {"require_parameters": True},
    }


def test_call_vlm_extra_body_caller_wins_on_conflict(tmp_path):
    """A key the caller passes in `extra_body` overrides the default —
    used to pin a different provider block without duplicating reasoning."""
    colorized = _panel(tmp_path / "colorized.png")

    def ok():
        return FakeResponse(
            '{"analyse": "fine", "good_color": true}',
            usage=FakeUsage(),
        )

    verifier = ColorVerifier(client=FakeClient([ok]))
    verifier.verify(
        colorized, None,
        extra_body={
            "reasoning": {"effort": "high"},
            "provider": {"require_parameters": False},
        },
    )

    call = verifier.client.chat.completions.calls[0]
    assert call["extra_body"]["provider"] == {"require_parameters": False}


def test_verify_not_found_not_retried(tmp_path):
    """A routing 404 (no endpoint supports the required parameters) is
    deterministic — recorded as an error after exactly one attempt, no
    backoff retries."""
    from openai import NotFoundError

    colorized = _panel(tmp_path / "colorized.png")
    not_found = make_openai_error(
        NotFoundError, "No endpoints found that can handle the requested parameters",
        status=404,
    )
    verifier = ColorVerifier(client=FakeClient([not_found]))
    record = verifier.verify(colorized, None)

    assert record.status == "error"
    assert "NotFoundError" in record.error
    assert len(verifier.client.chat.completions.calls) == 1


def test_verify_permission_denied_not_retried(tmp_path):
    """A 403 (e.g. OpenRouter "Key limit exceeded", model not allowed for
    this key) is deterministic — one attempt, no transient backoff retries.
    Regression: PermissionDeniedError is an APIStatusError (an APIError
    subclass) and used to fall into the transient-retry branch, burning 8
    attempts with exponential backoff per call on a permanent key problem."""
    from openai import PermissionDeniedError

    colorized = _panel(tmp_path / "colorized.png")
    denied = make_openai_error(
        PermissionDeniedError,
        "Key limit exceeded (total limit)",
        status=403,
    )
    verifier = ColorVerifier(client=FakeClient([denied]))
    record = verifier.verify(colorized, None)

    assert record.status == "error"
    assert "PermissionDeniedError" in record.error
    assert "Key limit exceeded" in record.error
    assert len(verifier.client.chat.completions.calls) == 1
