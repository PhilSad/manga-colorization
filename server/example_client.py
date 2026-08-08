#!/usr/bin/env python3
"""Example client for the local FLUX.2 Klein 9B inference server.

Minimal, dependency-free (stdlib only) demonstration of the `POST /edit`
multipart contract implemented by `service.py`:

    images:            repeated file parts, all named "images", in order
                       [current_page, reference_atlas, previous_page?]
    prompt:            str field
    width, height      int fields
    num_inference_steps int field (Klein distilled default 4; the LoRA's base
                       model wants ~20-50)
    guidance_scale     float field (optional; ~4-5 for the LoRA's base model,
                       ignored by the distilled model)
    lora_scale         float field (optional; override the LoRA weight, 0.8-1.0)
    seed               int field (omit for random)
    output_format      "png" | "jpeg" | "webp"

Response: raw image bytes in the requested format, written to --output.

Examples
--------
One-page colorization smoke test (run from the repo root, server on spark):

    python server/example_client.py \
      --endpoint http://spark:3000 \
      --images data/chapter_134/0134-001.png data/refs/frieren_reference.webp \
      --prompt "Add flat anime-style color to the manga page. The character with
                silver hair is Frieren: silver-white hair, emerald green eyes,
                white and dark-blue robe." \
      --output colorized-page1.png --width 1216 --height 1824 --steps 4

Same page through the manga-colorization LoRA (base model + trigger word):

    python server/example_client.py \
      --endpoint http://spark:3000 \
      --images data/chapter_134/0134-001.png data/refs/frieren_reference.webp \
      --prompt "mngclranm, flat anime-style color, silver-white hair, emerald
                green eyes" \
      --output lora-page1.png --width 1216 --height 1824 \
      --steps 20 --guidance-scale 4.0 --lora-scale 1.0

Equivalent curl (same contract, useful for debugging):

    curl -s -o colorized-page1.png -X POST http://spark:3000/edit \
      -F images=@data/chapter_134/0134-001.png \
      -F images=@data/refs/frieren_reference.webp \
      -F "prompt=Add flat anime-style color to the manga page" \
      -F width=1216 -F height=1824 -F num_inference_steps=4 \
      -F output_format=png
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

DEFAULT_PROMPT = (
    "Add flat anime-style color to the black-and-white manga page. Preserve the "
    "line art, panels, text, and margins. Use the other supplied images as "
    "character/color references where they apply, and invent a restrained, "
    "coherent palette elsewhere."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("BENTOML_ENDPOINT", "http://localhost:3000"),
        help="Base URL of the local server (default: $BENTOML_ENDPOINT or "
        "http://localhost:3000).",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more image files. The first is the edit target; the rest "
        "are references (sent in this order, all as 'images' multipart parts).",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Prompt text (default: manga colorization prompt)."
    )
    parser.add_argument("--prompt-file", type=Path, help="Read the prompt from a file instead.")
    parser.add_argument("--width", type=int, default=1216, help="Output width (multiple of 16).")
    parser.add_argument("--height", type=int, default=1824, help="Output height (multiple of 16).")
    parser.add_argument("--steps", "--num-inference-steps", dest="steps", type=int, default=4)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG scale (base model with LoRA: ~4-5; ignored by the distilled model).",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA weight override, e.g. 0.8-1.0 (ignored when the server has no LoRA).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility.")
    parser.add_argument(
        "--output-format", choices=("png", "jpeg", "webp"), default="png",
    )
    parser.add_argument("--output", type=Path, default=Path("colorized.png"))
    parser.add_argument("--timeout", type=int, default=1800, help="HTTP timeout in seconds.")
    return parser.parse_args()


def _prepare_image(path: Path) -> Path:
    """Decode the source and re-encode it as a true PNG.

    The server opens uploaded files with PIL restricted to the mime type of the
    multipart part, so source files with misleading extensions (e.g. chapter
    pages named .png but containing JPEG data) must be normalized first — the
    same reason run.py writes normalized-inputs/. Returns a temp PNG path.
    """
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        tmp = Path(tempfile.mkstemp(prefix="example-input-", suffix=".png")[1])
        rgb.save(tmp, format="PNG")
    return tmp


def _encode_multipart(
    fields: dict[str, str], files: list[tuple[str, Path]]
) -> tuple[bytes, str]:
    boundary = "----flux2example" + uuid.uuid4().hex
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


def main() -> int:
    args = parse_args()
    for path in args.images:
        if not Path(path).is_file():
            print(f"error: image not found: {path}", file=sys.stderr)
            return 2
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt.strip():
        print("error: prompt is empty", file=sys.stderr)
        return 2

    files = [("images", _prepare_image(Path(path))) for path in args.images]
    fields: dict[str, str] = {
        "prompt": prompt,
        "width": str(args.width),
        "height": str(args.height),
        "num_inference_steps": str(args.steps),
        "output_format": args.output_format,
    }
    if args.seed is not None:
        fields["seed"] = str(args.seed)
    if args.guidance_scale is not None:
        fields["guidance_scale"] = str(args.guidance_scale)
    if args.lora_scale is not None:
        fields["lora_scale"] = str(args.lora_scale)

    body, content_type = _encode_multipart(fields, files)
    url = args.endpoint.rstrip("/") + "/edit"
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": content_type}, method="POST"
    )

    print(f"POST {url}  ({len(body)} bytes, {len(files)} image(s))", flush=True)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        print(f"error: server returned HTTP {exc.code}: {exc.read().decode(errors='replace')}",
              file=sys.stderr)
        return 1
    except Exception as exc:  # network errors
        print(f"error: request failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote {args.output} ({len(payload)} bytes, "
          f"{time.time() - started:.1f}s wall time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
