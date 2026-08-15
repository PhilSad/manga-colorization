"""Tests for gpt_colorizer.py (full-page gpt-image-2 backend): request
payload (minimal size + fixed medium quality), usage/cost accounting, and
retry behaviour against a fake OpenAI client (same pattern as
fake_flux_server.py). Fully offline."""

from __future__ import annotations

import base64
import io
import types
from pathlib import Path

import pytest
from PIL import Image

from colorizer import (
    ATLAS_INSTRUCTION,
    NO_ATLAS_INSTRUCTION,
    NO_PROFILE_INSTRUCTION,
)
from config import GPT_IMAGE_QUALITY
from gpt_colorizer import GptImage2Colorizer

_TEMPLATE = (
    "colorize a {width}x{height} manga page in full color. "
    "{atlas_instruction} {character_profiles}"
)


def make_png(size=(1500, 2250), color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_response(b64: str, usage: object | None = None):
    return types.SimpleNamespace(
        data=[types.SimpleNamespace(b64_json=b64)],
        usage=usage,
    )


def make_usage(*, input_tokens, output_tokens, input_image, input_text,
               output_image, output_text):
    return types.SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=types.SimpleNamespace(
            image_tokens=input_image, text_tokens=input_text,
        ),
        output_tokens_details=types.SimpleNamespace(
            image_tokens=output_image, text_tokens=output_text,
        ),
    )


class FakeImagesAPI:
    """Records every edit call, snapshots the uploaded payloads (the real
    client closes its handles after the call), and pops responses/exceptions
    from a queue in order."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.payloads: list[list[bytes]] = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        payloads = []
        for handle in kwargs.get("image", []):
            try:
                payloads.append(handle.read())
            except Exception:
                payloads.append(None)
        self.payloads.append(payloads)
        if not self.responses:
            raise AssertionError("edit called more times than responses queued")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, responses: list):
        self.api_key = "test-key"
        self.timeout = None
        self.images = FakeImagesAPI(responses)


def install_fake_openai(monkeypatch, responses: list) -> FakeClient:
    client = FakeClient(responses)
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    return client


def make_colorizer(monkeypatch, **overrides) -> GptImage2Colorizer:
    """Colorizer instance with the fake OpenAI client installed; defaults
    keep tests fast (no retries, tiny backoff)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    defaults = dict(prompt_template=_TEMPLATE, retries=0, retry_backoff_s=0.001)
    defaults.update(overrides)
    return GptImage2Colorizer(**defaults)


def make_page(tmp_path: Path) -> Path:
    page = tmp_path / "page.png"
    page.write_bytes(make_png())
    return page


def make_atlas(tmp_path: Path, size=(360, 480)) -> Path:
    atlas = tmp_path / "atlas.jpg"
    atlas.write_bytes(make_png(size, (10, 200, 30)))
    return atlas


# ---------------------------------------------------------------------------
# Request payload

def test_edit_request_carries_minimal_size_and_medium(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    atlas = make_atlas(tmp_path)
    output = tmp_path / "out" / "panel_0001.png"
    b64 = base64.b64encode(make_png(size=(672, 1008))).decode()
    client = install_fake_openai(monkeypatch, [make_response(b64)])
    colorizer = make_colorizer(monkeypatch)

    record = colorizer.colorize(page, atlas, output)

    call = client.images.calls[0]
    assert call["model"] == "gpt-image-2"
    assert call["size"] == "672x1008"          # minimal size for 1500x2250
    assert call["quality"] == "medium"         # fixed (no quality flag)
    assert call["output_format"] == "png"
    assert call["n"] == 1
    assert len(call["image"]) == 2             # page + atlas
    prompt = call["prompt"]
    assert "672x1008" in prompt
    assert ATLAS_INSTRUCTION in prompt

    assert record.status == "ok"
    assert record.requested_size == (672, 1008)
    assert record.model == "gpt-image-2"
    assert record.quality == GPT_IMAGE_QUALITY == "medium"
    assert output.exists()
    with Image.open(output) as image:
        assert image.size == (672, 1008)


def test_no_atlas_sends_single_image(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    output = tmp_path / "panel_0001.png"
    b64 = base64.b64encode(make_png()).decode()
    client = install_fake_openai(monkeypatch, [make_response(b64)])
    colorizer = make_colorizer(monkeypatch)

    record = colorizer.colorize(page, None, output)

    call = client.images.calls[0]
    assert len(call["image"]) == 1
    assert NO_ATLAS_INSTRUCTION in call["prompt"]
    assert NO_PROFILE_INSTRUCTION in call["prompt"]   # no palette passed
    assert record.status == "ok"


def test_palette_instruction_overrides_profile_fallback(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    b64 = base64.b64encode(make_png()).decode()
    client = install_fake_openai(monkeypatch, [make_response(b64)])
    colorizer = make_colorizer(monkeypatch)

    colorizer.colorize(page, None, tmp_path / "panel_0001.png",
                       palette_instruction="Frieren: white hair, green eyes")
    prompt = client.images.calls[0]["prompt"]
    assert "Frieren: white hair, green eyes" in prompt
    assert NO_PROFILE_INSTRUCTION not in prompt


def test_gpt_size_override_used(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    b64 = base64.b64encode(make_png()).decode()
    client = install_fake_openai(monkeypatch, [make_response(b64)])
    colorizer = make_colorizer(monkeypatch, size=(1024, 1536))

    record = colorizer.colorize(page, None, tmp_path / "panel_0001.png")

    assert client.images.calls[0]["size"] == "1024x1536"
    assert record.requested_size == (1024, 1536)


def test_atlas_scale_downscales_upload(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    atlas = make_atlas(tmp_path, size=(200, 400))
    b64 = base64.b64encode(make_png()).decode()
    client = install_fake_openai(monkeypatch, [make_response(b64)])
    colorizer = make_colorizer(monkeypatch, atlas_scale=0.5)

    colorizer.colorize(page, atlas, tmp_path / "panel_0001.png")

    upload = client.images.payloads[0][1]
    with Image.open(io.BytesIO(upload)) as image:
        assert image.size == (100, 200)


# ---------------------------------------------------------------------------
# Usage / cost accounting

def test_usage_and_cost_accounting(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    b64 = base64.b64encode(make_png()).decode()
    usage = make_usage(
        input_tokens=3456, output_tokens=1396,
        input_image=3456, input_text=501,
        output_image=1296, output_text=100,
    )
    install_fake_openai(monkeypatch, [make_response(b64, usage)])
    colorizer = make_colorizer(monkeypatch)

    record = colorizer.colorize(page, None, tmp_path / "panel_0001.png")

    assert record.usage["input_tokens"] == 3456
    assert record.usage["input_tokens_details"] == {
        "image_tokens": 3456, "text_tokens": 501,
    }
    assert record.usage["output_tokens_details"] == {
        "image_tokens": 1296, "text_tokens": 100,
    }
    expected = (
        3456 / 1e6 * 8.0 + 501 / 1e6 * 5.0
        + 1296 / 1e6 * 30.0 + 100 / 1e6 * 30.0
    )
    assert record.est_cost_usd == pytest.approx(expected, abs=1e-6)


def test_missing_usage_means_no_cost(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    b64 = base64.b64encode(make_png()).decode()
    install_fake_openai(monkeypatch, [make_response(b64)])  # usage omitted
    colorizer = make_colorizer(monkeypatch)

    record = colorizer.colorize(page, None, tmp_path / "panel_0001.png")

    assert record.status == "ok"
    assert record.usage == {}
    assert record.est_cost_usd is None


# ---------------------------------------------------------------------------
# Retry policy

def _transient(message="boom"):
    return type("InternalServerError", (Exception,), {})(message)


def test_transient_error_retried_then_succeeds(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    b64 = base64.b64encode(make_png()).decode()
    client = install_fake_openai(
        monkeypatch, [_transient(), make_response(b64)]
    )
    colorizer = make_colorizer(monkeypatch, retries=2)

    record = colorizer.colorize(page, None, tmp_path / "panel_0001.png")

    assert len(client.images.calls) == 2   # initial + one retry
    assert record.status == "ok"
    assert record.error is None


def test_persistent_failure_fails_loudly(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    output = tmp_path / "panel_0001.png"
    client = install_fake_openai(
        monkeypatch, [_transient(), _transient(), _transient()]
    )
    colorizer = make_colorizer(monkeypatch, retries=2)

    record = colorizer.colorize(page, None, output)

    assert len(client.images.calls) == 3   # initial + 2 retries, then give up
    assert record.status == "error"
    assert "InternalServerError" in record.error
    assert record.output is None
    assert not output.exists()             # no silent partial output


def test_non_retryable_error_fails_immediately(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    permanent = type("BadRequestError", (Exception,), {"status_code": 400})("nope")
    client = install_fake_openai(monkeypatch, [permanent])
    colorizer = make_colorizer(monkeypatch, retries=3)

    record = colorizer.colorize(page, None, tmp_path / "panel_0001.png")

    assert len(client.images.calls) == 1
    assert record.status == "error"
    assert "BadRequestError" in record.error


def test_rate_limit_retried(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    rate = type("RateLimitError", (Exception,), {"status_code": 429})("slow down")
    b64 = base64.b64encode(make_png()).decode()
    client = install_fake_openai(monkeypatch, [rate, make_response(b64)])
    colorizer = make_colorizer(monkeypatch, retries=2)

    record = colorizer.colorize(page, None, tmp_path / "panel_0001.png")

    assert len(client.images.calls) == 2
    assert record.status == "ok"


def test_http_500_retried_by_status_code(tmp_path, monkeypatch):
    page = make_page(tmp_path)
    server_error = type("APIStatusError", (Exception,), {"status_code": 500})("gpu")
    b64 = base64.b64encode(make_png()).decode()
    client = install_fake_openai(monkeypatch, [server_error, make_response(b64)])
    colorizer = make_colorizer(monkeypatch, retries=2)

    record = colorizer.colorize(page, None, tmp_path / "panel_0001.png")

    assert len(client.images.calls) == 2
    assert record.status == "ok"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        GptImage2Colorizer(prompt_template=_TEMPLATE)
