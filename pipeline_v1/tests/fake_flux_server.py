"""A tiny threaded HTTP server implementing the FLUX `POST /edit` contract.

Used by offline tests as a stand-in for the BentoML server on Spark. It parses
the multipart request, records every field and image it receives, and replies
with a deterministically tinted copy of the first image at the requested size.
"""

from __future__ import annotations

import email
import io
import threading
from email import policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image


class FakeFluxServer:
    """Records incoming /edit requests; returns tinted panels."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._httpd is not None
        host, port = self._httpd.server_address
        return f"http://127.0.0.1:{port}"

    def start(self) -> None:
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def _make_handler(server: FakeFluxServer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                fields, images = _parse_multipart(body, self.headers.get("Content-Type", ""))
                record = {
                    "fields": fields,
                    "images": images,
                    "images_sizes": [
                        list(Image.open(io.BytesIO(data)).size) for data in images
                    ],
                }
                server.requests.append(record)
                payload = _tint_response(images, fields)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:  # noqa: BLE001
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                payload = str(error).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, *args):  # silence stderr
            pass

    return Handler


def _parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], list[bytes]]:
    """Parse a multipart/form-data body into (fields, image bytes in order)."""
    message = email.message_from_bytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body,
        policy=policy.default,
    )
    fields: dict[str, str] = {}
    images: list[bytes] = []
    for part in message.iter_parts():
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            images.append(payload)
        else:
            name = part.get_param("name", header="content-disposition")
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, images


def _tint_response(images: list[bytes], fields: dict[str, str]) -> bytes:
    """Reply image: the request's first image resized to width x height with a
    red tint (deterministic, clearly 'colorized')."""
    width = int(fields.get("width", "256"))
    height = int(fields.get("height", "256"))
    source = Image.open(io.BytesIO(images[0])).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    tint = Image.new("RGB", (width, height), (220, 60, 60))
    blended = Image.blend(source, tint, 0.45)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return buffer.getvalue()
