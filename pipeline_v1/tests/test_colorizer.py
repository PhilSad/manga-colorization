"""Tests for colorizer.py against the fake FLUX /edit server (offline)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from colorizer import FluxColorizer
from config import PIPELINE_DIR
from tests.fake_flux_server import FakeFluxServer

PROMPT_TEMPLATE = (PIPELINE_DIR / "colorizer_prompt.txt").read_text(encoding="utf-8")


@pytest.fixture
def server():
    fake = FakeFluxServer()
    fake.start()
    yield fake
    fake.stop()


def make_panel(path: Path, size=(340, 500)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


def make_atlas(path: Path) -> Path:
    Image.new("RGB", (360, 480), "lightgray").save(path, format="JPEG")
    return path


def test_colorize_sends_panel_and_atlas(server, tmp_path):
    panel = make_panel(tmp_path / "panel_0001.png")
    atlas = make_atlas(tmp_path / "atlas.jpg")
    colorizer = FluxColorizer(
        endpoint=server.url,
        prompt_template=PROMPT_TEMPLATE,
        num_inference_steps=20,
        guidance_scale=4.0,
        lora_scale=1.0,
        seed=42,
        output_format="png",
        timeout=60,
    )
    output = tmp_path / "out.png"
    record = colorizer.colorize(panel, atlas, output)

    assert record.status == "ok"
    assert record.error is None
    assert record.requested_size == (336, 496)  # nearest multiples of 16
    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (336, 496)

    assert len(server.requests) == 1
    request = server.requests[0]
    fields = request["fields"]
    assert fields["width"] == "336"
    assert fields["height"] == "496"
    assert fields["num_inference_steps"] == "20"
    assert fields["guidance_scale"] == "4.0"
    assert fields["lora_scale"] == "1.0"
    assert fields["seed"] == "42"
    assert fields["output_format"] == "png"
    assert "mngclranm" in fields["prompt"]
    assert "reference atlas in #2" in fields["prompt"]
    assert len(request["images"]) == 2
    # Image order: panel first, atlas second.
    assert request["images_sizes"][0] == [340, 500]
    assert request["images_sizes"][1] == [360, 480]


def test_colorize_without_atlas_sends_single_image(server, tmp_path):
    panel = make_panel(tmp_path / "p.png", size=(100, 200))
    colorizer = FluxColorizer(
        endpoint=server.url, prompt_template=PROMPT_TEMPLATE, timeout=60
    )
    record = colorizer.colorize(panel, None, tmp_path / "out.png")
    assert record.status == "ok"
    assert len(server.requests[0]["images"]) == 1
    assert "No reference atlas is provided" in server.requests[0]["fields"]["prompt"]
    assert server.requests[0]["fields"]["width"] == "96"  # 100 -> 96
    assert server.requests[0]["fields"]["height"] == "192"  # 200/16=12.5 -> 12


def test_size_policy_nearest_multiple_of_16(server, tmp_path):
    # 200/16 = 12.5 -> round half even -> 12 -> 192
    panel = make_panel(tmp_path / "p.png", size=(200, 200))
    colorizer = FluxColorizer(endpoint=server.url, prompt_template=PROMPT_TEMPLATE, timeout=60)
    colorizer.colorize(panel, None, tmp_path / "out.png")
    fields = server.requests[0]["fields"]
    assert (fields["width"], fields["height"]) == ("192", "192")


def test_colorize_server_error_recorded(server, tmp_path):
    panel = make_panel(tmp_path / "p.png")
    colorizer = FluxColorizer(endpoint=server.url, prompt_template=PROMPT_TEMPLATE, timeout=60)
    # Point at a URL that will 404 -> the fake server handler only knows POST
    # /edit; a different path returns 404 via BaseHTTPRequestHandler? It returns
    # 501 for unknown methods and 404 for unknown paths -> our handler only
    # implements do_POST, so GET /edit would 501. Instead, shut the server down
    # to force a connection error.
    server.stop()
    record = colorizer.colorize(panel, None, tmp_path / "out.png")
    assert record.status == "error"
    assert record.error is not None
    assert not (tmp_path / "out.png").exists()


def test_record_to_dict(tmp_path):
    panel = make_panel(tmp_path / "p.png")
    record = None
    from colorizer import ColorizeRecord

    record = ColorizeRecord(
        status="ok",
        output=tmp_path / "out.png",
        requested_size=(336, 496),
        latency_s=1.5,
        error=None,
        seed=7,
    )
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "out.png")
    doc = record.to_dict(panel, atlas=None)
    assert doc["status"] == "ok"
    assert doc["panel"] == "p.png"
    assert doc["atlas"] is None
    assert doc["requested_size"] == {"width": 336, "height": 496}
    assert doc["seed"] == 7


# ---------------------------------------------------------------------------
# V1.1 (task 0002): explicit palette conditioning reaches the /edit request

def test_palette_instruction_reaches_edit_request(server, tmp_path):
    panel = make_panel(tmp_path / "panel_0001.png")
    colorizer = FluxColorizer(
        endpoint=server.url,
        prompt_template=PROMPT_TEMPLATE,
        timeout=60,
    )
    palette = (
        "Canonical colors to apply to the characters below (from the official "
        "character profiles):\n- Frieren: silver-white hair; green eyes; "
        "white coat with gold trim."
    )
    colorizer.colorize(panel, None, tmp_path / "out.png",
                       palette_instruction=palette)
    fields = server.requests[0]["fields"]
    assert "silver-white hair" in fields["prompt"]
    assert "Frieren" in fields["prompt"]


def test_colorize_without_palette_uses_generic_instruction(server, tmp_path):
    panel = make_panel(tmp_path / "p.png")
    colorizer = FluxColorizer(
        endpoint=server.url,
        prompt_template=PROMPT_TEMPLATE,
        timeout=60,
    )
    colorizer.colorize(panel, None, tmp_path / "out.png")
    prompt = server.requests[0]["fields"]["prompt"]
    assert "No explicit character palette profiles are provided" in prompt
    assert "silver-white hair" not in prompt


# ---------------------------------------------------------------------------
# V1.1 (task 0004): oversized request capping

def test_bounded_size_spread_matches_fixture():
    from config import bounded_requested_size

    # SIZE-001: detected box [23,0,2918,2250] -> crop 2895x2250, cap 2.0 MP.
    width, height = bounded_requested_size(2895, 2250, 2.0)
    assert (width, height) == (1600, 1248)
    assert width % 16 == 0 and height % 16 == 0
    assert width * height <= 2_000_000


def test_bounded_size_never_upscales_ordinary_panels():
    from config import bounded_requested_size

    assert bounded_requested_size(340, 500, 2.0) == (336, 496)
    assert bounded_requested_size(1200, 900, 2.0) == (1200, 896)
    # 1.5 MP stays uncapped; nearest multiples of 16 per axis (V1 policy).
    assert bounded_requested_size(1500, 1000, 2.0) == (1504, 992)


def test_bounded_size_preserves_aspect_within_tolerance():
    from config import bounded_requested_size

    for width, height in ((2895, 2250), (3000, 2250), (2500, 1200), (2000, 2000)):
        requested_w, requested_h = bounded_requested_size(width, height, 2.0)
        original_ratio = width / height
        requested_ratio = requested_w / requested_h
        assert abs(requested_ratio - original_ratio) / original_ratio < 0.02
        assert requested_w * requested_h <= 2_000_000
        assert requested_w % 16 == 0 and requested_h % 16 == 0


def test_cap_applied_to_edit_request(server, tmp_path):
    panel = make_panel(tmp_path / "spread.png", size=(2895, 2250))
    colorizer = FluxColorizer(
        endpoint=server.url,
        prompt_template=PROMPT_TEMPLATE,
        timeout=60,
        max_megapixels=2.0,
    )
    record = colorizer.colorize(panel, None, tmp_path / "out.png")
    assert record.status == "ok"
    assert record.requested_size == (1600, 1248)
    assert record.original_size == (2895, 2250)
    assert record.cap_applied is True
    assert record.scale == pytest.approx(0.5537, abs=1e-3)
    fields = server.requests[0]["fields"]
    assert fields["width"] == "1600"
    assert fields["height"] == "1248"
    with Image.open(tmp_path / "out.png") as image:
        assert image.size == (1600, 1248)


def test_no_cap_for_ordinary_panels(server, tmp_path):
    panel = make_panel(tmp_path / "p.png", size=(340, 500))
    colorizer = FluxColorizer(
        endpoint=server.url,
        prompt_template=PROMPT_TEMPLATE,
        timeout=60,
        max_megapixels=2.0,
    )
    record = colorizer.colorize(panel, None, tmp_path / "out.png")
    assert record.cap_applied is False
    assert record.requested_size == (336, 496)
    assert record.scale == 1.0
