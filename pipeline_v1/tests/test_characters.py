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
PANEL_PROMPT_FILE = PIPELINE_DIR / "prompt_panel.txt"
PANEL_PAGE_PROMPT_FILE = PIPELINE_DIR / "prompt_panel_page.txt"
PROFILES_FILE = PIPELINE_DIR / "character_profiles.json"


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
    template = "List:\n{characters}{cast_shortlist}\nAnswer json only."
    from profiles import load_profiles

    profiles = load_profiles(PROFILES_FILE)
    prompt = build_prompt(template, ["Frieren", "Aura"], profiles=profiles)
    assert "Frieren: elf, long silver twin-tail hair, elven ears" in prompt
    assert "Aura: demon general, long black hair, dark dress" in prompt
    assert "{cast_shortlist}" not in prompt  # placeholder replaced (empty)


def test_build_prompt_cast_shortlist():
    template = "List:\n{characters}{cast_shortlist}\nAnswer json only."
    prompt = build_prompt(
        template, ["Frieren", "Heiter"],
        cast_shortlist=["Frieren", "Heiter", "Himmel"],
    )
    assert "likely cast is limited to: Frieren, Heiter, Himmel" in prompt
    assert "\"uncertain\": true" in prompt


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
    detector.prepare(
        make_refs(tmp_path),
        prompt_file=PROMPT_FILE,
        panel_prompt_file=PANEL_PROMPT_FILE,
    )
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
        "page_calls": 0,
        "fallback_calls": 0,
        "forced_panels": 0,
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


# ---------------------------------------------------------------------------
# Task 0003: page-level detection

@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"panels": {"panel_0001": {"characters": ["Frieren"], "uncertain": false}}}',
         {"panel_0001": {"characters": ["Frieren"], "uncertain": False}}),
        ("```json\n{\"panels\": {\"panel_0001\": {\"characters\": [], \"uncertain\": true}}}\n```",
         {"panel_0001": {"characters": [], "uncertain": True}}),
        ('trailing prose {"panels": {}} trailing', {}),
        ("", None),
        ("not json", None),
        ('{"characters": ["Frieren"]}', None),  # wrong shape (per-panel answer)
        ('{"panels": {"panel_0001": {"characters": "Frieren"}}}', None),  # malformed
    ],
)
def test_parse_page_mapping(text, expected):
    from characters import parse_page_mapping

    assert parse_page_mapping(text) == expected


def _page_fixture(tmp_path):
    """A run-like layout: 1_panels/<page>/panels.json + panel crops + overlay."""
    from detection import PanelBox
    from extraction import draw_overlay, save_panels
    from PIL import Image

    page_dir = tmp_path / "1_panels" / "0134-004"
    page_dir.mkdir(parents=True)
    page_path = tmp_path / "0134-004.png"
    image = Image.new("RGB", (500, 700), "white")
    image.save(page_path)
    boxes = [PanelBox(20, 20, 480, 300, 0.9), PanelBox(20, 320, 480, 680, 0.9)]
    ordered = [boxes[0], boxes[1]]
    save_panels(image, ordered, page_dir, extension=".png")
    draw_overlay(image, ordered, page_dir / "overlay.png")
    (page_dir / "panels.json").write_text(json.dumps({
        "page": page_path.name,
        "page_path": str(page_path),
        "detections": [
            {"panel_index": 1, "box": [20, 20, 480, 300], "crop": "panel_0001.png"},
            {"panel_index": 2, "box": [20, 320, 480, 680], "crop": "panel_0002.png"},
        ],
    }), encoding="utf-8")
    return page_path, page_dir


def test_detect_page_complete_response_no_fallbacks(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def ok():
        return FakeResponse(
            '{"panels": {"panel_0001": {"characters": ["Frieren"], '
            '"uncertain": false}, "panel_0002": {"characters": ["Fern"], '
            '"uncertain": false}}}',
            usage=FakeUsage(cost=0.0002),
        )

    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([ok]),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE,
                     panel_prompt_file=PANEL_PROMPT_FILE)
    record = detector.detect_page(
        page_path, page_dir, ["panel_0001", "panel_0002"], make_refs(tmp_path)
    )
    assert record.status == "ok"
    assert record.page_calls == 1
    assert record.fallback_calls == 0
    assert record.cost_usd == 0.0002
    assert record.panels["panel_0001"].characters == ["Frieren"]
    assert record.panels["panel_0001"].source == "page"
    assert record.panels["panel_0002"].characters == ["Fern"]
    # Only one paid call for the whole page.
    assert len(detector.client.chat.completions.calls) == 1


def test_detect_page_uncertain_triggers_fallback(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def page_answer():
        return FakeResponse(
            '{"panels": {"panel_0001": {"characters": ["Frieren"], '
            '"uncertain": false}, "panel_0002": {"characters": [], '
            '"uncertain": true}}}',
            usage=FakeUsage(cost=0.0002),
        )

    def fallback_answer():
        return FakeResponse('{"characters": ["Fern"]}', usage=FakeUsage(cost=0.0001))

    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([page_answer, fallback_answer]),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE,
                     panel_prompt_file=PANEL_PROMPT_FILE)
    record = detector.detect_page(
        page_path, page_dir, ["panel_0001", "panel_0002"], make_refs(tmp_path)
    )
    assert record.status == "partial"
    assert record.page_calls == 1
    assert record.fallback_calls == 1
    assert record.cost_usd == pytest.approx(0.0003, abs=1e-9)
    assert record.panels["panel_0001"].source == "page"
    assert record.panels["panel_0002"].source == "fallback"
    assert record.panels["panel_0002"].characters == ["Fern"]
    # One page call + one fallback call.
    assert len(detector.client.chat.completions.calls) == 2


def test_detect_page_missing_panel_triggers_fallback(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def page_answer():
        return FakeResponse(
            '{"panels": {"panel_0001": {"characters": ["Frieren"], '
            '"uncertain": false}}}',  # panel_0002 missing
            usage=FakeUsage(cost=0.0002),
        )

    def fallback_answer():
        return FakeResponse('{"characters": []}', usage=FakeUsage(cost=0.0001))

    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([page_answer, fallback_answer]),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE,
                     panel_prompt_file=PANEL_PROMPT_FILE)
    record = detector.detect_page(
        page_path, page_dir, ["panel_0001", "panel_0002"], make_refs(tmp_path)
    )
    assert record.fallback_calls == 1
    assert record.panels["panel_0002"].source == "fallback"


def test_detect_page_rejects_extra_panel_keys(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def page_answer():
        return FakeResponse(
            '{"panels": {"panel_0001": {"characters": ["Frieren"], '
            '"uncertain": false}, "panel_0099": {"characters": ["Sein"], '
            '"uncertain": false}}}',
            usage=FakeUsage(),
        )

    def fallback_answer():
        return FakeResponse('{"characters": ["Fern"]}', usage=FakeUsage())

    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([page_answer, fallback_answer]),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE,
                     panel_prompt_file=PANEL_PROMPT_FILE)
    record = detector.detect_page(
        page_path, page_dir, ["panel_0001", "panel_0002"], make_refs(tmp_path)
    )
    # panel_0099 is rejected (not accepted into the results); panel_0002
    # missing -> fallback.
    assert "panel_0099" not in record.panels
    assert record.panels["panel_0002"].source == "fallback"


def test_detect_page_retries_once_on_unparseable_answer(tmp_path):
    """An unparseable page-level answer is retried once before per-panel
    fallbacks, instead of exploding into one fallback per panel."""
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def garbage():
        return FakeResponse("I think the heroes are here...", usage=FakeUsage(cost=0.0001))

    def good():
        return FakeResponse(
            '{"panels": {"panel_0001": {"characters": ["Frieren"], '
            '"uncertain": false}, "panel_0002": {"characters": ["Fern"], '
            '"uncertain": false}}}',
            usage=FakeUsage(cost=0.0002),
        )

    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([garbage, good]),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE,
                     panel_prompt_file=PANEL_PROMPT_FILE)
    record = detector.detect_page(
        page_path, page_dir, ["panel_0001", "panel_0002"], make_refs(tmp_path)
    )
    assert record.status == "ok"
    assert record.page_calls == 2       # one retry, no per-panel fallbacks
    assert record.fallback_calls == 0
    assert record.cost_usd == pytest.approx(0.0003, abs=1e-9)
    assert record.panels["panel_0001"].source == "page"
    assert record.panels["panel_0002"].source == "page"
    assert len(detector.client.chat.completions.calls) == 2


def test_detect_page_still_falls_back_after_retry_fails(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def garbage():
        return FakeResponse("not json at all", usage=FakeUsage())

    def fallback_answer():
        return FakeResponse('{"characters": ["Fern"]}', usage=FakeUsage())

    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([garbage, garbage, fallback_answer, fallback_answer]),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE,
                     panel_prompt_file=PANEL_PROMPT_FILE)
    record = detector.detect_page(
        page_path, page_dir, ["panel_0001", "panel_0002"], make_refs(tmp_path)
    )
    assert record.page_calls == 2
    assert record.fallback_calls == 2   # both panels fall back after the retry
    assert record.panels["panel_0001"].source == "fallback"


def test_cast_shortlist_deterministic_and_offline(tmp_path):
    from characters import cast_shortlist_for

    casts_file = tmp_path / "casts.json"
    casts_file.write_text(json.dumps({"casts": {"c001": {"characters": ["A", "B"]}}}),
                          encoding="utf-8")
    assert cast_shortlist_for(casts_file, None) is None
    assert cast_shortlist_for(casts_file, "c001") == ["A", "B"]
    assert cast_shortlist_for(casts_file, "c001") == cast_shortlist_for(casts_file, "c001")
    with pytest.raises(ValueError, match="not found"):
        cast_shortlist_for(casts_file, "nope")


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"characters": ["Frieren"], "uncertain": false}',
         {"characters": ["Frieren"], "uncertain": False}),
        ('{"characters": [], "uncertain": true}',
         {"characters": [], "uncertain": True}),
        ("```json\n{\"characters\": [\"Fern\"]}\n```",
         {"characters": ["Fern"], "uncertain": False}),  # uncertain optional
        ('trailing prose {"characters": ["Stark"]} trailing',
         {"characters": ["Stark"], "uncertain": False}),
        ("", None),
        ("not json", None),
        ('["Frieren"]', None),                             # wrong shape
        ("{\"panels\": {}}", None),                       # page-mode answer
        ("{\"characters\": \"Frieren\"}", None),          # malformed
    ],
)
def test_parse_panel_with_page(text, expected):
    from characters import parse_panel_with_page

    assert parse_panel_with_page(text) == expected


# ---------------------------------------------------------------------------
# V1.2: panel+page detection

def _make_panel_page_detector(tmp_path, script):
    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it",
        api_key="dummy",
        client=FakeClient(script),
    )
    detector.prepare(
        make_refs(tmp_path),
        prompt_file=PROMPT_FILE,
        panel_prompt_file=PANEL_PROMPT_FILE,
        panel_page_prompt_file=PANEL_PAGE_PROMPT_FILE,
    )
    return detector


def test_detect_panels_with_page_all_ok(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def p1():
        return FakeResponse(
            '{"characters": ["Frieren"], "uncertain": false}',
            usage=FakeUsage(cost=0.00015),
        )

    def p2():
        return FakeResponse(
            '{"characters": ["Fern"], "uncertain": false}',
            usage=FakeUsage(cost=0.00018),
        )

    detector = _make_panel_page_detector(tmp_path, [p1, p2])
    record = detector.detect_panels_with_page(
        page_path, page_dir, ["panel_0001", "panel_0002"], make_refs(tmp_path)
    )
    assert record.status == "ok"
    assert record.page_calls == 2       # one panel+page call per panel
    assert record.fallback_calls == 0
    assert record.cost_usd == pytest.approx(0.00033, abs=1e-9)
    assert record.panels["panel_0001"].characters == ["Frieren"]
    assert record.panels["panel_0001"].source == "panel-page"
    assert record.panels["panel_0002"].characters == ["Fern"]
    assert len(detector.client.chat.completions.calls) == 2
    # Each call carries text + full page + panel crop images.
    first_content = detector.client.chat.completions.calls[0]["messages"][0]["content"]
    assert len(first_content) == 3
    assert first_content[0]["type"] == "text"
    assert first_content[1]["type"] == "image_url"
    assert first_content[2]["type"] == "image_url"
    assert first_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    second_content = detector.client.chat.completions.calls[1]["messages"][0][
        "content"
    ]
    assert first_content[1]["image_url"]["url"] != second_content[1]["image_url"]["url"]


def test_detect_panels_with_page_uncertain_falls_back(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def uncertain():
        return FakeResponse(
            '{"characters": [], "uncertain": true}', usage=FakeUsage(cost=0.00015)
        )

    def fallback():
        return FakeResponse('{"characters": ["Frieren"]}', usage=FakeUsage(cost=0.0001))

    detector = _make_panel_page_detector(tmp_path, [uncertain, fallback])
    record = detector.detect_panels_with_page(
        page_path, page_dir, ["panel_0001"], make_refs(tmp_path)
    )
    assert record.status == "partial"
    assert record.page_calls == 1
    assert record.fallback_calls == 1
    assert record.cost_usd == pytest.approx(0.00025, abs=1e-9)
    assert record.panels["panel_0001"].source == "fallback"
    assert record.panels["panel_0001"].characters == ["Frieren"]


def test_detect_panels_with_page_unknown_falls_back(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def unknown():
        return FakeResponse(
            '{"characters": ["Gandalf"], "uncertain": false}',
            usage=FakeUsage(),
        )

    def fallback():
        return FakeResponse('{"characters": ["Frieren"]}', usage=FakeUsage())

    detector = _make_panel_page_detector(tmp_path, [unknown, fallback])
    record = detector.detect_panels_with_page(
        page_path, page_dir, ["panel_0001"], make_refs(tmp_path)
    )
    assert record.status == "partial"
    assert record.fallback_calls == 1
    assert record.panels["panel_0001"].source == "fallback"


def test_detect_panels_with_page_unparseable_falls_back(tmp_path):
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def garbage():
        return FakeResponse("the heroes are here", usage=FakeUsage(cost=0.00015))

    def fallback():
        return FakeResponse('{"characters": ["Fern"]}', usage=FakeUsage(cost=0.0001))

    detector = _make_panel_page_detector(tmp_path, [garbage, fallback])
    record = detector.detect_panels_with_page(
        page_path, page_dir, ["panel_0001"], make_refs(tmp_path)
    )
    assert record.status == "partial"
    assert record.page_calls == 1
    assert record.fallback_calls == 1
    assert record.panels["panel_0001"].source == "fallback"
    assert record.panels["panel_0001"].characters == ["Fern"]


def test_detect_panels_with_page_error_falls_back(tmp_path):
    from openai import RateLimitError

    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def fallback():
        return FakeResponse('{"characters": ["Stark"]}', usage=FakeUsage())

    characters.BASE_BACKOFF_S = 0.0
    characters.MAX_ATTEMPTS = 2
    try:
        rate_limited = make_openai_error(
            RateLimitError, "nope", status=429
        )
        detector = _make_panel_page_detector(
            tmp_path, [rate_limited, rate_limited, fallback]
        )
        record = detector.detect_panels_with_page(
            page_path, page_dir, ["panel_0001"], make_refs(tmp_path)
        )
    finally:
        characters.BASE_BACKOFF_S = 5.0
        characters.MAX_ATTEMPTS = 8
    assert record.status == "partial"
    assert record.page_calls == 1
    assert record.fallback_calls == 1
    assert record.panels["panel_0001"].source == "fallback"
    assert record.panels["panel_0001"].characters == ["Stark"]


def test_detect_panels_with_page_requires_prompt(tmp_path):
    """panel-page detection without a panel-page prompt fails loudly."""
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)
    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([]),
    )
    detector.prepare(make_refs(tmp_path), prompt_file=PROMPT_FILE,
                     panel_prompt_file=PANEL_PROMPT_FILE)  # no panel-page file
    with pytest.raises(ValueError, match="panel-page prompt"):
        detector.detect_panels_with_page(
            page_path, page_dir, ["panel_0001"], make_refs(tmp_path)
        )


def test_characters_step_panel_page_mode(tmp_path):
    """panel-page step: one panel+page call per panel plus per-panel fallbacks."""
    from mock_backends import MockPageCharacterDetector
    from run_context import RunContext
    from steps.characters import run_characters_step

    config = make_step_fixture(tmp_path)
    config.detection_mode = "panel-page"
    ctx = RunContext.create(tmp_path / "output", {"status": "running"})
    panels_root = ctx.step_dir("panels") / "0134-004"
    panels_root.mkdir(parents=True)
    for path in (tmp_path / "1_panels" / "0134-004").glob("*.png"):
        (panels_root / path.name).write_bytes(path.read_bytes())
    (panels_root / "panels.json").write_text(json.dumps({
        "page_path": str(tmp_path / "0134-004.png"), "detections": [],
    }), encoding="utf-8")

    detector = MockPageCharacterDetector({
        "0134-004": {"panel_0001": (["Frieren"], False)},
    })
    result = run_characters_step(ctx, config, detector)

    totals = result["totals"]
    assert totals["api_calls"] == 3      # 2 panel+page calls + 1 fallback
    assert totals["page_calls"] == 2
    assert totals["fallback_calls"] == 1  # panel_0002 uncovered -> fallback
    assert totals["successful_calls"] == 2
    assert totals["cost_usd"] == pytest.approx(0.0005, abs=1e-9)
    assert len(detector.calls) == 1        # one per-page batch of panel+page calls
    # Provenance file for the panel+page calls.
    from run_context import read_json

    provenance = read_json(ctx.step_dir("characters") / "0134-004" / "panel_page_calls.json")
    assert provenance["page_calls"] == 2
    assert provenance["fallback_calls"] == 1
    # Panel records written with sources.
    p1 = read_json(ctx.step_dir("characters") / "0134-004" / "panel_0001.json")
    p2 = read_json(ctx.step_dir("characters") / "0134-004" / "panel_0002.json")
    assert p1["source"] == "panel-page" and p1["characters"] == ["Frieren"]
    assert p2["source"] == "fallback"


def test_characters_step_threaded_matches_sequential(tmp_path):
    """`--workers N` parallelizes pages but produces identical totals and
    per-panel records (page-scoped writes never race)."""
    from mock_backends import MockPageCharacterDetector
    from run_context import RunContext
    from steps.characters import run_characters_step

    panel_map_by_page = {
        "0134-004": {"panel_0001": (["Frieren"], False)},
        "0134-005": {"panel_0001": ([], True)},
    }

    def run(workers: int) -> dict:
        config = make_step_fixture(tmp_path / f"w{workers}")
        config.detection_mode = "panel-page"
        config.workers = workers
        ctx = RunContext.create(tmp_path / f"output{workers}", {"status": "running"})
        for page_stem, panel_map in panel_map_by_page.items():
            panels_root = ctx.step_dir("panels") / page_stem
            panels_root.mkdir(parents=True)
            for name in ("panel_0001.png", "panel_0002.png"):
                Image.new("RGB", (16, 16), "white").save(panels_root / name)
            (panels_root / "panels.json").write_text(json.dumps({
                "page_path": str(tmp_path / f"{page_stem}.png"), "detections": [],
            }), encoding="utf-8")
        detector = MockPageCharacterDetector(panel_map_by_page)
        return run_characters_step(ctx, config, detector)

    seq = run(1)
    thr = run(4)
    assert seq["totals"] == thr["totals"]
    seq_panels = {(r["page"], r["panel"], r["source"]): r["characters"]
                  for r in seq["records"]}
    thr_panels = {(r["page"], r["panel"], r["source"]): r["characters"]
                  for r in thr["records"]}
    assert seq_panels == thr_panels
    assert len(thr_panels) == 4


def test_characters_step_page_mode(tmp_path):
    """Page-level step: one paid call per page covering its panels."""
    from mock_backends import MockPageCharacterDetector
    from run_context import RunContext
    from steps.characters import run_characters_step

    config = make_step_fixture(tmp_path)
    config.detection_mode = "page"
    ctx = RunContext.create(tmp_path / "output", {"status": "running"})
    panels_root = ctx.step_dir("panels") / "0134-004"
    panels_root.mkdir(parents=True)
    for path in (tmp_path / "1_panels" / "0134-004").glob("*.png"):
        (panels_root / path.name).write_bytes(path.read_bytes())
    (panels_root / "panels.json").write_text(json.dumps({
        "page_path": str(tmp_path / "0134-004.png"), "detections": [],
    }), encoding="utf-8")

    detector = MockPageCharacterDetector({
        "0134-004": {"panel_0001": (["Frieren"], False)},
    })
    result = run_characters_step(ctx, config, detector)

    totals = result["totals"]
    assert totals["api_calls"] == 2      # one page call + one fallback
    assert totals["page_calls"] == 1
    assert totals["fallback_calls"] == 1  # panel_0002 uncovered -> fallback
    assert totals["cost_usd"] == pytest.approx(0.0003, abs=1e-9)
    assert len(detector.calls) == 1        # one page-level detection call
    # Panel records written with sources.
    from run_context import read_json

    p1 = read_json(ctx.step_dir("characters") / "0134-004" / "panel_0001.json")
    p2 = read_json(ctx.step_dir("characters") / "0134-004" / "panel_0002.json")
    assert p1["source"] == "page" and p1["characters"] == ["Frieren"]
    assert p2["source"] == "fallback"


# ---------------------------------------------------------------------------
# panel-page-cast: automatic per-chapter cast shortlist

CHAPTER_CASTS_FILE = PIPELINE_DIR / "chapter_casts.json"
CHAPTER_PAGE_MAP_FILE = PIPELINE_DIR.parent / "frieren_wiki_dataset" / "chapter_page_map.json"


def test_cast_key_for_page_filename_tag(tmp_path):
    """Correctly-tagged volume page: map + tag agree on c005 (ch. 5, p130)."""
    from characters import cast_key_for_page

    page = tmp_path / ("Frieren - Beyond Journey's End v01 (2021) (Digital) "
                       "(1r0n) (f2)") / ("Frieren - Beyond Journey's End - c005 "
                       "(v01) - p130 [VIZ Media] [Digital] [1r0n].png")
    key = cast_key_for_page(page, CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE)
    assert key == "c005"


def test_cast_key_for_page_v09_mislabeled_uses_map(tmp_path):
    """v09 filenames say c078 everywhere but p130 belongs to ch. 85; the
    chapter_page_map must win over the misleading tag."""
    from characters import cast_key_for_page

    page = tmp_path / ("Frieren - Beyond Journey's End v09 (2023) (Digital) "
                       "(1r0n)") / ("Frieren - Beyond Journey's End - c078 "
                       "(v09) - p130 [VIZ Media] [Digital] [1r0n].png")
    key = cast_key_for_page(page, CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE)
    assert key == "c085"


def test_cast_key_for_page_chapter_134_prefix(tmp_path):
    """data/chapter_134/0134-004.png has no volume dir or c-tag: the leading
    `NNN-` prefix must yield c134."""
    from characters import cast_key_for_page

    page = tmp_path / "0134-004.png"
    key = cast_key_for_page(page, CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE)
    assert key == "c134"


def test_cast_key_for_page_none(tmp_path):
    """Pages outside any chapter (padding/preview) get no cast -> full roster."""
    from characters import cast_key_for_page

    assert cast_key_for_page(tmp_path / "cover.jpg",
                             CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE) is None
    assert cast_key_for_page(tmp_path / "unrelated.png",
                             CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE) is None


def test_cast_key_for_page_missing_cast_falls_back_to_roster(tmp_path):
    """A derived chapter whose shortlist is absent from the casts file must
    return None (full roster), not an invalid key."""
    from characters import cast_key_for_page

    casts = tmp_path / "casts.json"
    casts.write_text(json.dumps({"casts": {"c001": {"characters": ["Frieren"]}}}),
                     encoding="utf-8")
    page = tmp_path / ("Frieren - Beyond Journey's End - c005 (v01) - "
                       "p130 [VIZ Media] [Digital] [1r0n].png")
    assert cast_key_for_page(page, casts, CHAPTER_PAGE_MAP_FILE) is None


def test_set_cast_rebuilds_prompts(tmp_path):
    """set_cast switches the cast shortlist in all prompts without re-reading
    templates; the roster section stays, the cast section changes."""
    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([]),
        chapter_casts_file=CHAPTER_CASTS_FILE,
    )
    detector.prepare(
        make_refs(tmp_path),
        prompt_file=PROMPT_FILE,
        panel_prompt_file=PANEL_PROMPT_FILE,
        panel_page_prompt_file=PANEL_PAGE_PROMPT_FILE,
    )
    assert "limited to: Frieren, Fern" not in detector.panel_page_prompt
    detector.set_cast("c005")
    assert "limited to: Frieren, Fern" in detector.panel_page_prompt
    assert "Flamme" not in detector.panel_page_prompt  # not in ch. 5's cast
    assert "Flamme" not in detector.panel_prompt
    detector.set_cast(None)  # back to the full roster
    assert "limited to:" not in detector.panel_page_prompt


def test_detect_panels_with_page_cast_key_renders_prompt(tmp_path):
    """detect_panels_with_page(cast_key=...) renders the panel-page prompt for
    that cast per call (thread-safe) — the shortlist sentence appears in the
    text content."""
    from characters import OpenRouterCharacterDetector

    page_path, page_dir = _page_fixture(tmp_path)

    def answer():
        return FakeResponse(
            '{"characters": ["Frieren"], "uncertain": false}',
            usage=FakeUsage(cost=0.00015),
        )

    detector = OpenRouterCharacterDetector(
        model="google/gemma-4-31b-it", api_key="dummy",
        client=FakeClient([answer]),
        chapter_casts_file=CHAPTER_CASTS_FILE,
    )
    detector.prepare(
        make_refs(tmp_path),
        prompt_file=PROMPT_FILE,
        panel_prompt_file=PANEL_PROMPT_FILE,
        panel_page_prompt_file=PANEL_PAGE_PROMPT_FILE,
    )
    detector.detect_panels_with_page(
        page_path, page_dir, ["panel_0001"], make_refs(tmp_path),
        cast_key="c005",
    )
    content = detector.client.chat.completions.calls[0]["messages"][0]["content"]
    text = content[0]["text"]
    assert "limited to: Frieren, Fern" in text
    assert "Flamme" not in text.split("{characters}")[0]


def test_characters_step_panel_page_cast_mode(tmp_path):
    """panel-page-cast step: per-page auto cast derived from the page (the
    0134-004 fixture -> c134) is recorded in set_cast + forwarded to the
    panel+page call + written into the provenance."""
    from mock_backends import MockPageCharacterDetector
    from run_context import RunContext, read_json
    from steps.characters import run_characters_step

    config = make_step_fixture(tmp_path)
    config.detection_mode = "panel-page-cast"
    ctx = RunContext.create(tmp_path / "output", {"status": "running"})
    panels_root = ctx.step_dir("panels") / "0134-004"
    panels_root.mkdir(parents=True)
    for path in (tmp_path / "1_panels" / "0134-004").glob("*.png"):
        (panels_root / path.name).write_bytes(path.read_bytes())
    (panels_root / "panels.json").write_text(json.dumps({
        "page_path": str(tmp_path / "0134-004.png"), "detections": [],
    }), encoding="utf-8")

    detector = MockPageCharacterDetector({
        "0134-004": {"panel_0001": (["Frieren"], False)},
    })
    result = run_characters_step(ctx, config, detector)

    assert result["totals"]["page_calls"] == 2
    assert detector.current_cast == "c134"            # set_cast per page
    assert detector.cast_keys[-1] == "c134"           # forwarded to the call
    provenance = read_json(
        ctx.step_dir("characters") / "0134-004" / "panel_page_calls.json"
    )
    assert provenance["cast_key"] == "c134"
