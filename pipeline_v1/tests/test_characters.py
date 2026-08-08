"""Tests for characters.py (offline: parsing/validation + fake OpenAI client)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import characters
from characters import (
    OpenRouterCharacterDetector,
    build_prompt,
    canonical_characters,
    parse_characters,
    reference_label,
    validate_characters,
)
from config import PIPELINE_DIR

PROMPT_FILE = PIPELINE_DIR / "prompt.txt"


# ---------------------------------------------------------------------------
# Canonical names & prompt

def make_refs(tmp_path: Path) -> Path:
    refs = tmp_path / "refs"
    refs.mkdir(exist_ok=True)
    for name in ("frieren_reference.webp", "Frieren_anime_profile.webp",
                 "fern_reference.webp", "stark_reference.webp"):
        Image.new("RGB", (4, 4), "gray").save(refs / name)
    return refs


def test_canonical_characters_dedupes(tmp_path):
    refs = make_refs(tmp_path)
    assert canonical_characters(refs) == ["Fern", "Frieren", "Stark"]


def test_reference_label():
    assert reference_label(Path("frieren_reference.webp")) == "Frieren"
    assert reference_label(Path("Frieren_anime_profile.webp")) == "Frieren"
    assert reference_label(Path("aura_reference.webp")) == "Aura"


def test_build_prompt_uses_hints():
    template = "List:\n{characters}\nAnswer json only."
    prompt = build_prompt(template, ["Frieren", "Aura"])
    assert "Frieren: elf, long silver hair, elven ears" in prompt
    assert "Aura: demon general, long black hair, dark dress" in prompt


# ---------------------------------------------------------------------------
# Parsing & validation

@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"characters": ["Frieren", "Fern"]}', ["Frieren", "Fern"]),
        ("```json\n{\"characters\": [\"Frieren\"]}\n```", ["Frieren"]),
        ('here is the answer: {"characters": []} trailing prose', []),
        ("[1, 2]", ["1", "2"]),
        ("", None),
        ("not json at all", None),
    ],
)
def test_parse_characters(text, expected):
    assert parse_characters(text) == expected


def test_validate_characters():
    canonical = ["Frieren", "Fern", "Stark"]
    parsed = ["frieren", "Fern", "Frieren", "Gandalf", ""]
    known, unknown = validate_characters(parsed, canonical)
    assert known == ["Frieren", "Fern"]  # deduped, canonical casing
    assert unknown == ["Gandalf"]
    assert validate_characters(None, canonical) == ([], [])


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible client

class FakeUsage:
    def __init__(self, prompt=10, completion=5, cost=0.0001):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.cost = cost
        self.model_extra = None


class FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class FakeResponse:
    def __init__(self, content, usage=None, model="fake-model"):
        self.choices = [FakeChoice(content)]
        self.usage = usage
        self.model = model
        self.id = "chatcmpl-fake"


class FakeCompletions:
    """Callable stub: returns the next queued response or raises next error."""

    def __init__(self, script):
        self.script = list(script)  # list of callables or exceptions
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step()


class FakeClient:
    def __init__(self, script):
        self.chat = type("C", (), {"completions": FakeCompletions(script)})()


def make_openai_error(cls, message, headers=None, status=500):
    """Build a real openai exception (they require an httpx.Response)."""
    import httpx

    request = httpx.Request(
        "POST", "https://openrouter.ai/api/v1/chat/completions"
    )
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls(message, response=response, body=None)


def make_panel(tmp_path: Path) -> Path:
    panel = tmp_path / "panel_0001.png"
    Image.new("RGB", (32, 32), "white").save(panel)
    return panel


def make_detector(tmp_path, script):
    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it",
        api_key="dummy",
        client=FakeClient(script),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE)
    return detector


def test_detect_success_ok(tmp_path):
    panel = make_panel(tmp_path)

    def ok():
        return FakeResponse(
            '{"characters": ["Frieren", "Fern"]}',
            usage=FakeUsage(cost=0.0001234),
        )

    detector = make_detector(tmp_path, [ok])
    record = detector.detect(panel, make_refs(tmp_path))
    assert record.status == "ok"
    assert record.characters == ["Frieren", "Fern"]
    assert record.cost_usd == 0.0001234
    assert record.cost_source == "usage.cost"
    assert record.usage["total_tokens"] == 15
    assert record.error is None


def test_detect_ok_with_unknown(tmp_path):
    panel = make_panel(tmp_path)

    def ok_unknown():
        return FakeResponse('{"characters": ["Frieren", "Gandalf"]}', usage=FakeUsage())

    detector = make_detector(tmp_path, [ok_unknown])
    record = detector.detect(panel, make_refs(tmp_path))
    assert record.status == "ok-with-unknown"
    assert record.characters == ["Frieren"]
    assert record.unknown_entries == ["Gandalf"]


def test_detect_unparseable(tmp_path):
    panel = make_panel(tmp_path)

    def garbage():
        return FakeResponse("I think Frieren appears", usage=FakeUsage())

    detector = make_detector(tmp_path, [garbage])
    record = detector.detect(panel, make_refs(tmp_path))
    assert record.status == "unparseable"
    assert record.characters == []


def test_detect_cost_missing_is_unpriced(tmp_path):
    panel = make_panel(tmp_path)

    def no_cost():
        return FakeResponse('{"characters": []}', usage=FakeUsage(cost=None))

    detector = make_detector(tmp_path, [no_cost])
    record = detector.detect(panel, make_refs(tmp_path))
    assert record.status == "ok"
    assert record.cost_usd is None
    assert record.cost_source == "unavailable"


def test_detect_rate_limit_retry_then_success(tmp_path):
    from openai import RateLimitError

    panel = make_panel(tmp_path)

    def success():
        return FakeResponse('{"characters": ["Frieren"]}', usage=FakeUsage())

    rate_limited = make_openai_error(
        RateLimitError, "slow down", headers={"retry-after": "0"}, status=429
    )
    detector = make_detector(tmp_path, [rate_limited, success])
    record = detector.detect(panel, make_refs(tmp_path))
    assert record.status == "ok"
    assert record.attempts == 2


def test_detect_rate_limit_exhausted(tmp_path):
    from openai import RateLimitError

    panel = make_panel(tmp_path)
    rate_limited = make_openai_error(RateLimitError, "nope", status=429)
    # MAX_ATTEMPTS failures; sleep is real but small (5s * n) — reduce constant
    # via monkeypatch to keep the test fast.
    characters.BASE_BACKOFF_S = 0.0
    characters.MAX_ATTEMPTS = 3
    try:
        detector = make_detector(tmp_path, [rate_limited] * 10)
        record = detector.detect(panel, make_refs(tmp_path))
    finally:
        characters.BASE_BACKOFF_S = 5.0
        characters.MAX_ATTEMPTS = 8
    assert record.status == "error"
    assert "RateLimitError" in record.error
    assert record.attempts == 3


def test_detect_bad_request_json_format_fallback(tmp_path):
    from openai import BadRequestError

    panel = make_panel(tmp_path)

    def success():
        return FakeResponse('{"characters": ["Fern"]}', usage=FakeUsage())

    bad_request = make_openai_error(
        BadRequestError,
        "response_format json_object not supported for this model",
        status=400,
    )
    detector = make_detector(tmp_path, [bad_request, success])
    record = detector.detect(panel, make_refs(tmp_path))
    assert record.status == "ok"
    # First call had response_format, retry did not.
    calls = detector.client.chat.completions.calls
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_detect_generic_exception_recorded(tmp_path):
    panel = make_panel(tmp_path)
    detector = make_detector(tmp_path, [RuntimeError("boom")])
    record = detector.detect(panel, make_refs(tmp_path))
    assert record.status == "error"
    assert "RuntimeError" in record.error
    assert record.characters == []


def test_record_to_dict_shape(tmp_path):
    panel = make_panel(tmp_path)

    def ok():
        return FakeResponse('{"characters": ["Frieren"]}', usage=FakeUsage(cost=0.0001))

    detector = make_detector(tmp_path, [ok])
    record = detector.detect(panel, make_refs(tmp_path))
    doc = record.to_dict(panel, page="0134-004")
    assert doc["panel"] == "panel_0001.png"
    assert doc["page"] == "0134-004"
    assert doc["status"] == "ok"
    assert len(doc["panel_sha256"]) == 64
    assert doc["cost_usd"] == 0.0001


# ---------------------------------------------------------------------------
# Step-level test (fake detector, offline)

def make_step_fixture(tmp_path):
    """1_panels/<page>/panel_0001.png + panel_0002.png, plus a config."""
    from config import PipelineConfig

    panels_root = tmp_path / "1_panels" / "0134-004"
    panels_root.mkdir(parents=True)
    for name in ("panel_0001.png", "panel_0002.png"):
        Image.new("RGB", (16, 16), "white").save(panels_root / name)
    refs = make_refs(tmp_path)
    config = PipelineConfig(
        input_dir=tmp_path / "pages",
        refs_dir=refs,
        output_root=tmp_path / "output",
        sleep_s=0.0,
        mock=True,
    )
    config.input_dir.mkdir(exist_ok=True)
    return config


class StubCharacterDetector:
    """Returns canned records; records the panels it was called with."""

    def __init__(self):
        self.called: list[Path] = []

    def detect(self, panel, refs_dir):
        self.called.append(panel)
        if panel.name == "panel_0001.png":
            return characters.CharacterRecord(
                status="ok", characters=["Frieren", "Fern"], unknown_entries=[],
                response_text="{}", usage={"total_tokens": 10},
                cost_usd=0.00005, cost_source="usage.cost", latency_s=1.0,
                model_returned="m", attempts=1, finished_at="now",
            )
        return characters.CharacterRecord(
            status="error", characters=[], unknown_entries=[], response_text="",
            usage={}, cost_usd=None, cost_source="unavailable", latency_s=2.0,
            model_returned=None, attempts=1, error="boom", finished_at="now",
        )


def test_characters_step(tmp_path):
    from run_context import RunContext
    from steps.characters import run_characters_step

    config = make_step_fixture(tmp_path)
    # Move the fixtures into a real RunContext layout.
    ctx = RunContext.create(tmp_path / "output", {"status": "running"})
    panels_root = ctx.step_dir("panels") / "0134-004"
    panels_root.mkdir(parents=True)
    for path in (tmp_path / "1_panels" / "0134-004").glob("*.png"):
        (panels_root / path.name).write_bytes(path.read_bytes())

    detector = StubCharacterDetector()
    result = run_characters_step(ctx, config, detector)

    assert result["totals"] == {
        "api_calls": 2,
        "successful_calls": 1,
        "error_calls": 1,
        "unpriced_calls": 1,
        "total_latency_s": 3.0,
        "cost_usd": 0.00005,
    }
    chars_dir = ctx.step_dir("characters") / "0134-004"
    assert (chars_dir / "panel_0001.json").is_file()
    assert (chars_dir / "panel_0002.json").is_file()
    summary = (ctx.step_dir("characters") / "summary.json").is_file()
    assert summary
    assert len(detector.called) == 2
    # load_characters_per_panel returns names keyed by panel stem.
    from steps.characters import load_characters_per_panel

    mapping = load_characters_per_panel(ctx, "0134-004")
    assert mapping["panel_0001"] == ["Frieren", "Fern"]
    assert mapping["panel_0002"] == []
