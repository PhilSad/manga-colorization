#!/usr/bin/env python3
"""Sequential manga colorization with GPT Image 1 Mini at low quality."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import shlex
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps


METHOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = METHOD_DIR.parents[1]
DEFAULT_MODEL = "gpt-image-1-mini"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

PRICING = {
    "model": "gpt-image-1-mini",
    "date": "2026-08-08",
    "currency": "USD",
    "source": "https://developers.openai.com/api/docs/models/gpt-image-1-mini",
    "text_input_per_million_tokens": 2.00,
    "cached_text_input_per_million_tokens": 0.20,
    "image_input_per_million_tokens": 2.50,
    "cached_image_input_per_million_tokens": 0.25,
    "output_image_each": {
        "low": {"1024x1024": 0.005, "1024x1536": 0.006, "1536x1024": 0.006},
        "medium": {"1024x1024": 0.011, "1024x1536": 0.015, "1536x1024": 0.015},
        "high": {"1024x1024": 0.036, "1024x1536": 0.052, "1536x1024": 0.052},
    },
    "notes": "Standard pricing; image-edit input text/image tokens are additional.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Colorize manga pages sequentially with GPT Image 1 Mini, a labelled "
            "character-reference atlas, and the previous generated page."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, default=REPO_ROOT / "data" / "chapter_134"
    )
    parser.add_argument("--refs-dir", type=Path, default=REPO_ROOT / "data" / "refs")
    parser.add_argument(
        "--output-root", type=Path, default=METHOD_DIR / "output"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--quality", choices=("low", "medium", "high"), default="low"
    )
    parser.add_argument(
        "--size",
        choices=("1024x1024", "1024x1536", "1536x1024"),
        default="1024x1536",
    )
    parser.add_argument(
        "--output-format", choices=("jpeg", "png", "webp"), default="jpeg"
    )
    parser.add_argument(
        "--output-compression",
        type=int,
        default=95,
        help="JPEG/WebP output quality from 0 to 100.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-file", type=Path, default=METHOD_DIR / "prompt.txt")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 <= args.output_compression <= 100:
        parser.error("--output-compression must be between 0 and 100")
    return args


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_mime_and_size(path: Path) -> tuple[str, list[int]]:
    with Image.open(path) as image:
        image_format = (image.format or "").upper()
        dimensions = [image.width, image.height]
    mime = Image.MIME.get(image_format)
    if not mime:
        raise ValueError(f"Unsupported image format for {path}")
    return mime, dimensions


def file_record(path: Path) -> dict[str, Any]:
    mime, dimensions = image_mime_and_size(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "mime_type": mime,
        "dimensions": dimensions,
    }


def reference_label(path: Path) -> str:
    name = path.stem
    for suffix in ("_reference", "_anime_profile"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip().title()


def build_reference_atlas(refs: list[Path], destination: Path) -> None:
    columns = 4
    cell_width = 360
    cell_height = 480
    label_height = 36
    rows = math.ceil(len(refs) / columns)
    atlas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(atlas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = (
        ImageFont.truetype(str(font_path), 20)
        if font_path.exists()
        else ImageFont.load_default()
    )

    for index, ref_path in enumerate(refs):
        column = index % columns
        row = index // columns
        left = column * cell_width
        top = row * cell_height
        with Image.open(ref_path) as source:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            fitted = ImageOps.contain(
                background.convert("RGB"),
                (cell_width - 20, cell_height - label_height - 20),
                Image.Resampling.LANCZOS,
            )
        image_left = left + (cell_width - fitted.width) // 2
        image_top = top + label_height + (cell_height - label_height - fitted.height) // 2
        atlas.paste(fitted, (image_left, image_top))
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline="#777777",
            width=2,
        )
        draw.text(
            (left + 10, top + 7), reference_label(ref_path), fill="black", font=font
        )

    atlas.save(destination, format="JPEG", quality=94, subsampling=0)


def normalize_page_for_upload(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG", optimize=True)


def get_field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def usage_record(response: Any) -> dict[str, Any]:
    usage = get_field(response, "usage")
    input_details = get_field(usage, "input_tokens_details")
    return {
        "input_tokens": get_field(usage, "input_tokens"),
        "output_tokens": get_field(usage, "output_tokens"),
        "total_tokens": get_field(usage, "total_tokens"),
        "input_tokens_details": {
            "text_tokens": get_field(input_details, "text_tokens"),
            "image_tokens": get_field(input_details, "image_tokens"),
            "cached_tokens": get_field(input_details, "cached_tokens"),
        },
    }


def estimated_cost(
    usage: dict[str, Any], quality: str, size: str
) -> dict[str, Any]:
    output_cost = PRICING["output_image_each"][quality][size]
    details = usage["input_tokens_details"]
    text_tokens = details.get("text_tokens")
    image_tokens = details.get("image_tokens")
    input_tokens = usage.get("input_tokens")

    known = output_cost
    unpriced_input_tokens = input_tokens
    if text_tokens is not None and image_tokens is not None:
        known += text_tokens * PRICING["text_input_per_million_tokens"] / 1_000_000
        known += image_tokens * PRICING["image_input_per_million_tokens"] / 1_000_000
        classified = text_tokens + image_tokens
        unpriced_input_tokens = max(0, (input_tokens or classified) - classified)

    upper_bound = known
    if unpriced_input_tokens:
        upper_bound += (
            unpriced_input_tokens
            * PRICING["image_input_per_million_tokens"]
            / 1_000_000
        )
    return {
        "known_usd": round(known, 8),
        "upper_bound_usd": round(upper_bound, 8),
        "unpriced_input_tokens": unpriced_input_tokens,
        "note": (
            "Includes the fixed output-image price. Input token cost is exact when "
            "the response separates text and image tokens; otherwise the upper bound "
            "prices unclassified input tokens at the image-input rate."
        ),
    }


def extension_and_mime(output_format: str) -> tuple[str, str]:
    if output_format == "jpeg":
        return ".jpg", "image/jpeg"
    if output_format == "webp":
        return ".webp", "image/webp"
    return ".png", "image/png"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("openai", "Pillow", "python-dotenv"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def run() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key: set {args.api_key_env} or add it to {REPO_ROOT / '.env'}"
        )

    input_dir = args.input_dir.resolve()
    refs_dir = args.refs_dir.resolve()
    prompt_path = args.prompt_file.resolve()
    pages = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    refs = sorted(
        path
        for path in refs_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        raise SystemExit(f"No supported page images found in {input_dir}")
    if not refs:
        raise SystemExit(f"No supported reference images found in {refs_dir}")
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    run_dir = create_run_dir(args.output_root.resolve())
    manifest_path = run_dir / "manifest.json"
    atlas_path = run_dir / "reference-atlas.jpg"
    normalized_dir = run_dir / "normalized-inputs"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    build_reference_atlas(refs, atlas_path)
    extension, output_mime = extension_and_mime(args.output_format)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "openai-gpt-image-1-mini-low-sequential",
        "status": "running",
        "started_at": iso_now(),
        "finished_at": None,
        "run_directory": str(run_dir),
        "command": shlex.join(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
        "configuration": {
            "model": args.model,
            "quality": args.quality,
            "size": args.size,
            "output_format": args.output_format,
            "output_compression": args.output_compression,
            "input_order_first_page": ["current_black_and_white", "reference_atlas"],
            "input_order_later_pages": [
                "current_black_and_white",
                "reference_atlas",
                "previous_colorized_page",
            ],
            "input_dir": str(input_dir),
            "refs_dir": str(refs_dir),
            "prompt_file": str(prompt_path),
            "prompt": prompt,
        },
        "preprocessing": {
            "reference_atlas": "4-column labelled JPEG atlas",
            "source_pages": (
                "Decoded and saved losslessly as PNG because chapter files have .png "
                "names but contain JPEG data. Originals remain unchanged."
            ),
        },
        "dependencies": {"python": sys.version, **package_versions()},
        "pricing_assumptions": PRICING,
        "inputs": [file_record(path) for path in pages],
        "references": [
            {"label": reference_label(path), **file_record(path)} for path in refs
        ],
        "reference_atlas": file_record(atlas_path),
        "pages": [],
        "totals": {
            "api_calls": 0,
            "successful_pages": 0,
            "known_estimated_cost_usd": 0.0,
            "upper_bound_estimated_cost_usd": 0.0,
            "unpriced_input_tokens": 0,
        },
    }
    write_json(manifest_path, manifest)

    client = OpenAI(api_key=api_key)
    previous_output: Path | None = None

    try:
        for index, page_path in enumerate(pages):
            page_number = index + 1
            normalized_page = normalized_dir / f"{page_path.stem}.png"
            normalize_page_for_upload(page_path, normalized_page)

            if previous_output is None:
                input_paths = [normalized_page, atlas_path]
                input_description = (
                    "Input image 1 is the CURRENT BLACK-AND-WHITE TARGET PAGE. "
                    "Input image 2 is the LABELLED CHARACTER REFERENCE ATLAS. "
                    "There is no previous page; invent a coherent palette where the "
                    "atlas does not apply."
                )
            else:
                input_paths = [normalized_page, atlas_path, previous_output]
                input_description = (
                    "Input image 1 is the CURRENT BLACK-AND-WHITE TARGET PAGE and is "
                    "the image to colorize. Input image 2 is the LABELLED CHARACTER "
                    "REFERENCE ATLAS. Input image 3 is the PREVIOUS COLORIZED PAGE and "
                    "is continuity guidance only."
                )
            contextual_prompt = f"{prompt}\n\n{input_description}"

            manifest["totals"]["api_calls"] += 1
            write_json(manifest_path, manifest)
            request_options: dict[str, Any] = {
                "model": args.model,
                "prompt": contextual_prompt,
                "quality": args.quality,
                "size": args.size,
                "output_format": args.output_format,
                "n": 1,
            }
            if args.output_format in {"jpeg", "webp"}:
                request_options["output_compression"] = args.output_compression

            with ExitStack() as stack:
                image_files = [
                    stack.enter_context(path.open("rb")) for path in input_paths
                ]
                response = client.images.edit(image=image_files, **request_options)

            if not response.data or not response.data[0].b64_json:
                raise RuntimeError("OpenAI returned no base64 image data")
            image_bytes = base64.b64decode(response.data[0].b64_json)
            output_path = run_dir / f"{page_path.stem}{extension}"
            output_path.write_bytes(image_bytes)
            with Image.open(io.BytesIO(image_bytes)) as generated:
                output_dimensions = [generated.width, generated.height]

            usage = usage_record(response)
            cost = estimated_cost(usage, args.quality, args.size)
            page_record = {
                "sequence": page_number,
                "input": file_record(page_path),
                "normalized_input": file_record(normalized_page),
                "request_image_order": [str(path) for path in input_paths],
                "previous_colorized_page": str(previous_output) if previous_output else None,
                "output": {
                    "path": str(output_path),
                    "sha256": sha256(output_path),
                    "bytes": output_path.stat().st_size,
                    "mime_type": output_mime,
                    "dimensions": output_dimensions,
                },
                "usage_metadata": usage,
                "estimated_cost": cost,
                "request_id": get_field(response, "_request_id"),
                "revised_prompt": get_field(response.data[0], "revised_prompt"),
                "completed_at": iso_now(),
            }
            manifest["pages"].append(page_record)
            manifest["totals"]["successful_pages"] += 1
            manifest["totals"]["known_estimated_cost_usd"] = round(
                manifest["totals"]["known_estimated_cost_usd"]
                + cost["known_usd"],
                8,
            )
            manifest["totals"]["upper_bound_estimated_cost_usd"] = round(
                manifest["totals"]["upper_bound_estimated_cost_usd"]
                + cost["upper_bound_usd"],
                8,
            )
            manifest["totals"]["unpriced_input_tokens"] += (
                cost["unpriced_input_tokens"] or 0
            )
            write_json(manifest_path, manifest)
            previous_output = output_path
            print(f"[{page_number}/{len(pages)}] wrote {output_path}", flush=True)

        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = iso_now()
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    run()
