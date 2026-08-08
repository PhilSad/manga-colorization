#!/usr/bin/env python3
"""fal-compatible client backed by the local BentoML FLUX.2 Klein server.

Implements the subset of the fal client API used by run.py (`upload_file`,
`submit`, `Handler.request_id`, `Handler.get`) so the sequential colorization
pipeline can run against the self-hosted server with `--endpoint <url>` and
zero changes to the manifest/provenance logic.

HTTP contract (see server/service.py at the repo root):
    POST /edit   multipart/form-data
      images                 repeated file parts, all named "images", in order
                             [current_page, reference_atlas, previous_page?]
      prompt                 str field
      width, height          int fields
      num_inference_steps    int field
      guidance_scale         float field (optional; ~4-5 for the LoRA base model)
      lora_scale             float field (optional; LoRA weight override)
      seed                   int field (omitted when None)
      output_format          str field
    response: raw image bytes in the requested format.

Output is written to a temp file and returned as a `file://` URL so run.py's
existing download_file() (urllib) works unchanged.

Stdlib only — no extra dependencies for the method's venv.
"""

from __future__ import annotations

import mimetypes
import os
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

ENDPOINT = os.environ.get("BENTOML_ENDPOINT", "").rstrip("/")
_TIMEOUT_SECONDS = 1800

_registry: dict[str, str] = {}  # local://<n> -> absolute path of the image file
_counter = 0


def configure(endpoint: str) -> None:
    """Point the client at a local BentoML server (e.g. http://spark:3000)."""
    global ENDPOINT
    ENDPOINT = endpoint.rstrip("/")


def upload_file(path: str) -> str:
    """Record a local image for a later submit(). Returns a local:// reference.

    The image is only attached to the request at submit() time; nothing is
    uploaded ahead of time (the atlas/previous-page reuse pattern in run.py is
    preserved at the code level).
    """
    global _counter
    key = f"local://{_counter}"
    _counter += 1
    _registry[key] = str(path)
    return key


def submit(model: str, arguments: dict | None = None, **kwargs) -> "LocalHandler":
    return LocalHandler(arguments or {})


def _fetch_to_temp(url: str) -> str:
    """Download a remote URL into a temp file (for non-local:// image refs)."""
    fd, path = tempfile.mkstemp(prefix="flux2-klein-remote-", suffix=".img")
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
        with os.fdopen(fd, "wb") as handle:
            handle.write(response.read())
    return path


def _encode_multipart(
    fields: dict[str, str], files: list[tuple[str, Path]]
) -> tuple[bytes, str]:
    """Build a multipart/form-data body using only the stdlib."""
    boundary = "----flux2local" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")
    for name, path in files:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode()
        )
        chunks.append(f"Content-Type: {mime}\r\n\r\n".encode())
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class LocalHandler:
    """Mimics fal_client's request handler: .request_id + .get()."""

    def __init__(self, arguments: dict) -> None:
        self.request_id = str(uuid.uuid4())
        self._arguments = arguments

    def get(self) -> dict:
        if not ENDPOINT:
            raise RuntimeError(
                "local endpoint not configured: call configure(endpoint) or "
                "pass --endpoint <url> to run.py"
            )
        image_urls = self._arguments.get("image_urls") or []
        if not image_urls:
            raise ValueError("no image_urls in arguments")
        files: list[tuple[str, Path]] = []
        for url in image_urls:
            if url.startswith("local://"):
                path = _registry[url]
            else:
                path = _fetch_to_temp(url)
            files.append(("images", Path(path)))

        image_size = self._arguments.get("image_size") or {}
        fields: dict[str, str] = {
            "prompt": self._arguments["prompt"],
            "width": str(image_size.get("width", 1216)),
            "height": str(image_size.get("height", 1824)),
            "num_inference_steps": str(self._arguments.get("num_inference_steps", 4)),
            "output_format": self._arguments.get("output_format", "png"),
        }
        seed = self._arguments.get("seed")
        if seed is not None:
            fields["seed"] = str(seed)
        guidance_scale = self._arguments.get("guidance_scale")
        if guidance_scale is not None:
            fields["guidance_scale"] = str(guidance_scale)
        lora_scale = self._arguments.get("lora_scale")
        if lora_scale is not None:
            fields["lora_scale"] = str(lora_scale)

        body, content_type = _encode_multipart(fields, files)
        request = urllib.request.Request(
            f"{ENDPOINT}/edit",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        started = time.time()
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            payload = response.read()

        fd, output_path = tempfile.mkstemp(prefix="flux2-klein-", suffix=".png")
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)

        elapsed_ms = (time.time() - started) * 1000.0
        return {
            "images": [{"url": Path(output_path).as_uri()}],
            "seed": seed,
            "timings": {"total_ms": round(elapsed_ms, 1)},
            "has_nsfw_concepts": [False],
            "prompt": self._arguments["prompt"],
        }
