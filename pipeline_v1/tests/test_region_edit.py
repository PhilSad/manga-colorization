"""Offline unit tests for region_edit.py (bbox-guided region editing).

Pins the pure helpers (draw_boxes mapping/numbering, region_instruction
rendering) and the GptImage2RegionEditor request contract (images.edit with
the boxed image first and the atlas uploaded as a filename-carrying tuple,
prompt slots filled, cost recorded) against a fake OpenAI client — the same
pattern as test_gpt_colorizer.py. Fully offline.
"""

from __future__ import annotations

import base64
import io
import types
from pathlib import Path

import pytest
from PIL import Image

from config import GPT_IMAGE_QUALITY
from region_edit import GptImage2RegionEditor, draw_boxes, region_instruction

_TEMPLATE = (
    "fix regions on the {width}x{height} page. {region_instruction} "
    "{palette_instruction}"
)

REGIONS = [
    {
        "character": "Eisen",
        "problem": "beard was colored white",
        "fix_suggestion": "Eisen: beard golden-brown/blond",
        "bbox": [0, 0, 500, 500],
    },
    {
        "character": "Frieren",
        "problem": "hair lavender, should be silver-white",
        "fix_suggestion": "Frieren: hair silver-white",
        "bbox": [500, 500, 1000, 1000],
    },
]


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
    """Records every edit call and snapshots uploaded payloads (same pattern
    as test_gpt_colorizer.FakeImagesAPI)."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.payloads: list[list[bytes]] = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        payloads = []
        for handle in kwargs.get("image", []):
            if isinstance(handle, tuple):   # ("atlas.jpg", buffer) upload
                handle = handle[1]
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


def make_editor(monkeypatch, **overrides) -> GptImage2RegionEditor:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    defaults = dict(prompt_template=_TEMPLATE, retries=0, retry_backoff_s=0.001)
    defaults.update(overrides)
    return GptImage2RegionEditor(**defaults)


def make_boxed(tmp_path: Path, size=(672, 1008)) -> Path:
    path = tmp_path / "panel_0001.attempt_1.boxed.png"
    Image.new("RGB", size, "white").save(path)
    return path


def make_atlas(tmp_path: Path, size=(360, 480)) -> Path:
    atlas = tmp_path / "atlas.jpg"
    Image.new("RGB", size, (10, 200, 30)).save(atlas)
    return atlas


# ---------------------------------------------------------------------------
# draw_boxes

def test_draw_boxes_normalized_mapping_and_numbering(tmp_path):
    boxed = make_boxed(tmp_path, size=(200, 100))
    out = tmp_path / "boxed_out.png"

    draw_boxes(boxed, REGIONS, out)

    with Image.open(out) as image:
        assert image.size == (200, 100)


def test_draw_boxes_draws_at_requested_size(tmp_path):
    """The resolution rule: boxes are drawn on the image at the resolution
    actually sent to gpt-image-2 (normalized 0-1000 coords scale exactly)."""
    boxed = make_boxed(tmp_path, size=(100, 50))
    out = tmp_path / "boxed_out.png"

    draw_boxes(boxed, REGIONS, out, size=(200, 100))

    with Image.open(out) as image:
        assert image.size == (200, 100)
    # half of the first box (0-500 of 1000) lands exactly at x=100 of 200
    with Image.open(out) as image:
        assert image.getpixel((100, 50)) != image.getpixel((0, 0))


def test_draw_boxes_skips_missing_bbox(tmp_path):
    boxed = make_boxed(tmp_path)
    out = tmp_path / "boxed_out.png"
    regions = [
        {"character": "Frieren", "problem": "p", "fix_suggestion": "s"},
        *REGIONS,
    ]

    draw_boxes(boxed, regions, out)

    assert out.is_file()


def test_draw_boxes_clamps_out_of_range(tmp_path):
    boxed = make_boxed(tmp_path, size=(100, 100))
    out = tmp_path / "boxed_out.png"
    regions = [
        {"character": "x", "problem": "p", "fix_suggestion": "s",
         "bbox": [-50, -50, 1500, 1500]}
    ]

    draw_boxes(boxed, regions, out)

    assert out.is_file()


# ---------------------------------------------------------------------------
# region_instruction

def test_region_instruction_numbers_regions_in_order():
    text = region_instruction(REGIONS)
    assert "Region 0 (Eisen): Eisen: beard golden-brown/blond" in text
    assert "Region 1 (Frieren): Frieren: hair silver-white" in text


def test_region_instruction_falls_back_to_problem():
    text = region_instruction(
        [{"character": "Frieren", "problem": "hair lavender, should be silver"}]
    )
    assert "Region 0 (Frieren): hair lavender, should be silver" in text


# ---------------------------------------------------------------------------
# GptImage2RegionEditor request contract

def test_edit_request_shape(tmp_path, monkeypatch):
    """images.edit: boxed image first, atlas uploaded as the filename-carrying
    tuple (the mimetype regression), prompt slots filled, size/quality set,
    output written, cost recorded."""
    boxed = make_boxed(tmp_path)
    atlas = make_atlas(tmp_path)
    output = tmp_path / "out" / "panel_0001.attempt_2.png"
    b64 = base64.b64encode(
        Image.new("RGB", (672, 1008), "red").tobytes()
    ).decode()
    client = install_fake_openai(monkeypatch, [make_response(b64)])
    editor = make_editor(monkeypatch)

    instruction = region_instruction(REGIONS)
    record = editor.edit(
        boxed, atlas, output, instruction,
        palette_instruction="Frieren: silver-white hair, teal eyes",
    )

    call = client.images.calls[0]
    assert call["model"] == "gpt-image-2"
    assert call["size"] == "672x1008"
    assert call["quality"] == GPT_IMAGE_QUALITY == "medium"
    assert call["output_format"] == "png"
    assert call["n"] == 1
    assert len(call["image"]) == 2
    # boxed image is a plain file handle; atlas is the named tuple upload
    assert not isinstance(call["image"][0], tuple)
    atlas_upload = call["image"][1]
    assert isinstance(atlas_upload, tuple)
    assert atlas_upload[0] == "atlas.jpg"
    upload = client.images.payloads[0][1]
    with Image.open(io.BytesIO(upload)) as image:
        assert image.size == (360, 480)
    # prompt slots: region instruction + palette both rendered
    prompt = call["prompt"]
    assert "672x1008" in prompt
    assert "Region 0 (Eisen)" in prompt
    assert "silver-white hair" in prompt
    assert "{region_instruction}" not in prompt

    assert record.status == "ok"
    assert record.model == "gpt-image-2"
    assert record.requested_size == (672, 1008)
    assert output.exists()


def test_edit_records_usage_and_cost(tmp_path, monkeypatch):
    boxed = make_boxed(tmp_path)
    b64 = base64.b64encode(Image.new("RGB", (672, 1008), "red").tobytes()).decode()
    usage = make_usage(
        input_tokens=3456, output_tokens=1396,
        input_image=3456, input_text=501,
        output_image=1296, output_text=100,
    )
    install_fake_openai(monkeypatch, [make_response(b64, usage)])
    editor = make_editor(monkeypatch)

    record = editor.edit(boxed, None, tmp_path / "edited.png", "Region 0: fix")

    assert record.est_cost_usd == pytest.approx(
        3456 / 1e6 * 8.0 + 501 / 1e6 * 5.0
        + 1296 / 1e6 * 30.0 + 100 / 1e6 * 30.0,
        abs=1e-6,
    )


def test_edit_without_atlas_sends_single_image(tmp_path, monkeypatch):
    boxed = make_boxed(tmp_path)
    b64 = base64.b64encode(Image.new("RGB", (672, 1008), "red").tobytes()).decode()
    client = install_fake_openai(monkeypatch, [make_response(b64)])
    editor = make_editor(monkeypatch)

    editor.edit(boxed, None, tmp_path / "edited.png", "Region 0: fix")

    assert len(client.images.calls[0]["image"]) == 1


def test_edit_error_fails_loudly(tmp_path, monkeypatch):
    boxed = make_boxed(tmp_path)
    output = tmp_path / "edited.png"
    transient = type("InternalServerError", (Exception,), {})( "boom")
    client = install_fake_openai(monkeypatch, [transient, transient])
    editor = make_editor(monkeypatch, retries=1)

    record = editor.edit(boxed, None, output, "Region 0: fix")

    assert len(client.images.calls) == 2   # initial + 1 retry
    assert record.status == "error"
    assert record.output is None
    assert not output.exists()


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        GptImage2RegionEditor(prompt_template=_TEMPLATE)
