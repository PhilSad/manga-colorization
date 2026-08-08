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
