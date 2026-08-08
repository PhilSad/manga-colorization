#!/usr/bin/env python3
"""Sequential manga colorization with character references and page continuity."""

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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageOps


METHOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = METHOD_DIR.parents[1]
DEFAULT_MODEL = "gemini-3.1-flash-lite-image"
# No Gemini image-generation model has been verified to accept explicit
# createCachedContent calls through the Gemini Developer API. The model pages'
# broad "caching" capability can refer to implicit caching.
CACHE_SUPPORTED_MODELS: set[str] = set()
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

SYSTEM_INSTRUCTION = """You are a meticulous manga colorization artist. The reference atlas contains labelled canonical character colors. Apply those colors whenever a referenced character appears. Preserve the source page's exact panels, line art, lettering, speech bubbles, composition, and aspect ratio. The previous colorized page is continuity guidance only; never copy its composition into the current page."""

PRICING = {
    "gemini-3.1-flash-lite-image": {
        "date": "2026-08-08",
        "currency": "USD",
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
        "input_per_million_tokens": 0.25,
        "text_and_thinking_output_per_million_tokens": 1.50,
        "output_image_each": 0.0336,
        "notes": "Standard paid tier, 1K output. Explicit context caching is unsupported.",
    },
    "gemini-2.5-flash-image": {
        "date": "2026-08-08",
        "currency": "USD",
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
        "input_per_million_tokens": 0.30,
        "output_image_each": 0.039,
        "notes": (
            "Standard paid tier, 1K output. Google's model pricing table does not "
            "separately state this image model's cached-token/storage rate; the "
            "manifest therefore leaves cached-token cost unpriced."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Colorize manga pages sequentially with a character-reference atlas and "
            "the previous generated page as continuity context."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "chapter_134",
        help="Directory containing monochrome pages (sorted by filename).",
    )
    parser.add_argument(
        "--refs-dir",
        type=Path,
        default=REPO_ROOT / "data" / "refs",
        help="Directory containing labelled character reference images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=METHOD_DIR / "output",
        help="Parent directory for fresh timestamped run directories.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reference-mode",
        choices=("auto", "cache", "inline-atlas"),
        default="auto",
        help=(
            "auto uses a cache only for known cache-capable image models; "
            "inline-atlas sends the atlas on every request."
        ),
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=int,
        default=7200,
        help="Lifetime of the temporary reference cache.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not delete the server-side cache and uploaded atlas after the run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N sorted pages (useful for a paid test run).",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N sorted pages before applying --limit (default 0).",
    )
    parser.add_argument("--aspect-ratio", default="2:3")
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "high"),
        default="minimal",
        help="Used by Gemini 3.1 image models; ignored by Gemini 2.5 Flash Image.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=METHOD_DIR / "prompt.txt",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable containing the Gemini API key.",
    )
    args = parser.parse_args()

    if args.cache_ttl_seconds < 300:
        parser.error("--cache-ttl-seconds must be at least 300")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.skip_first < 0:
        parser.error("--skip-first must be non-negative")

    normalized_model = args.model.removeprefix("models/")
    if args.reference_mode == "cache" and normalized_model not in CACHE_SUPPORTED_MODELS:
        parser.error(
            f"{args.model} has not been verified to support explicit "
            "createCachedContent for image generation. Use "
            "--reference-mode inline-atlas."
        )
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
        size = [image.width, image.height]
    mime = Image.MIME.get(image_format)
    if not mime:
        raise ValueError(f"Unsupported image format for {path}")
    return mime, size


def file_record(path: Path) -> dict[str, Any]:
    mime, dimensions = image_mime_and_size(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "mime_type": mime,
        "dimensions": dimensions,
    }


def image_part(path: Path) -> types.Part:
    mime, _ = image_mime_and_size(path)
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


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
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()

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
        draw.text((left + 10, top + 7), reference_label(ref_path), fill="black", font=font)

    atlas.save(destination, format="JPEG", quality=94, subsampling=0)


def effective_reference_mode(requested: str, model: str) -> str:
    if requested != "auto":
        return requested
    normalized_model = model.removeprefix("models/")
    return "cache" if normalized_model in CACHE_SUPPORTED_MODELS else "inline-atlas"


def wait_for_uploaded_file(client: genai.Client, uploaded: Any) -> Any:
    while True:
        state = getattr(uploaded, "state", None)
        state_name = getattr(state, "name", str(state or "ACTIVE")).upper()
        if state_name not in {"PROCESSING", "STATE_PROCESSING"}:
            if "FAILED" in state_name:
                raise RuntimeError(f"Reference atlas upload failed: {state_name}")
            return uploaded
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)


def create_reference_cache(
    client: genai.Client,
    model: str,
    atlas_path: Path,
    ttl_seconds: int,
) -> tuple[Any, Any]:
    uploaded = wait_for_uploaded_file(client, client.files.upload(file=atlas_path))
    uploaded_mime = getattr(uploaded, "mime_type", None) or "image/jpeg"
    content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="LABELLED CANONICAL CHARACTER COLOR REFERENCE ATLAS:"
            ),
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded_mime),
        ],
    )
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=f"manga-character-refs-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            system_instruction=SYSTEM_INSTRUCTION,
            contents=[content],
            ttl=f"{ttl_seconds}s",
        ),
    )
    return cache, uploaded


def serialize_cache(cache: Any) -> dict[str, Any]:
    usage = getattr(cache, "usage_metadata", None)
    return {
        "name": getattr(cache, "name", None),
        "model": getattr(cache, "model", None),
        "display_name": getattr(cache, "display_name", None),
        "create_time": str(getattr(cache, "create_time", None) or "") or None,
        "expire_time": str(getattr(cache, "expire_time", None) or "") or None,
        "usage_metadata": {
            "total_token_count": getattr(usage, "total_token_count", None),
            "image_count": getattr(usage, "image_count", None),
            "text_count": getattr(usage, "text_count", None),
        },
    }


def usage_record(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None)
    names = (
        "prompt_token_count",
        "cached_content_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
    )
    return {name: getattr(usage, name, None) if usage else None for name in names}


def estimated_cost(model: str, usage: dict[str, int | None]) -> dict[str, Any]:
    normalized_model = model.removeprefix("models/")
    pricing = PRICING.get(normalized_model)
    if not pricing:
        return {
            "known_usd": None,
            "unpriced_cached_tokens": usage.get("cached_content_token_count"),
            "note": "No pricing formula is configured for this model.",
        }

    prompt_tokens = usage.get("prompt_token_count") or 0
    cached_tokens = usage.get("cached_content_token_count") or 0
    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    known = uncached_tokens * pricing["input_per_million_tokens"] / 1_000_000
    known += pricing["output_image_each"]
    thoughts_tokens = usage.get("thoughts_token_count") or 0
    if "text_and_thinking_output_per_million_tokens" in pricing:
        known += (
            thoughts_tokens
            * pricing["text_and_thinking_output_per_million_tokens"]
            / 1_000_000
        )
    return {
        "known_usd": round(known, 8),
        "unpriced_cached_tokens": cached_tokens if cached_tokens else 0,
        "note": (
            "Known estimate includes one output image and measured uncached input. "
            "Cached-token/storage charges are excluded when not published separately."
        ),
    }


def generation_config(
    args: argparse.Namespace, mode: str, cache_name: str | None
) -> types.GenerateContentConfig:
    normalized_model = args.model.removeprefix("models/")
    image_options: dict[str, str] = {"aspect_ratio": args.aspect_ratio}
    config_options: dict[str, Any] = {
        "response_modalities": ["IMAGE"],
        "image_config": types.ImageConfig(**image_options),
    }
    if normalized_model.startswith("gemini-3.1-"):
        image_options["image_size"] = "1K"
        config_options["image_config"] = types.ImageConfig(**image_options)
        config_options["thinking_config"] = types.ThinkingConfig(
            thinking_level=args.thinking_level
        )
    if mode == "cache":
        config_options["cached_content"] = cache_name
    else:
        config_options["system_instruction"] = SYSTEM_INSTRUCTION
    return types.GenerateContentConfig(**config_options)


def extract_generated_image(response: Any) -> tuple[bytes, str, list[str]]:
    images: list[tuple[bytes, str]] = []
    notes: list[str] = []
    for part in response.parts or []:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            notes.append(text)
        inline = getattr(part, "inline_data", None)
        if not inline or not str(getattr(inline, "mime_type", "")).startswith("image/"):
            continue
        data = inline.data
        if isinstance(data, str):
            data = base64.b64decode(data)
        images.append((bytes(data), inline.mime_type))
    if not images:
        raise RuntimeError("Gemini returned no final image part")
    image_bytes, mime_type = images[-1]
    return image_bytes, mime_type, notes


def extension_for_mime(mime_type: str) -> str:
    return {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime_type, ".png")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("google-genai", "Pillow", "python-dotenv"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def run() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv(args.api_key_env) or (
        os.getenv("GOOGLE_API_KEY") if args.api_key_env == "GEMINI_API_KEY" else None
    )
    if not api_key:
        raise SystemExit(
            f"Missing API key: set {args.api_key_env} or add it to {REPO_ROOT / '.env'}"
        )

    input_dir = args.input_dir.resolve()
    refs_dir = args.refs_dir.resolve()
    prompt_path = args.prompt_file.resolve()
    pages = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    refs = sorted(
        path for path in refs_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if args.skip_first:
        pages = pages[args.skip_first :]
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        raise SystemExit(f"No supported page images found in {input_dir}")
    if not refs:
        raise SystemExit(f"No supported reference images found in {refs_dir}")
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    mode = effective_reference_mode(args.reference_mode, args.model)
    run_dir = create_run_dir(args.output_root.resolve())
    manifest_path = run_dir / "manifest.json"
    atlas_path = run_dir / "reference-atlas.jpg"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    build_reference_atlas(refs, atlas_path)

    normalized_model = args.model.removeprefix("models/")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "gemini-reference-cache-sequential",
        "status": "running",
        "started_at": iso_now(),
        "finished_at": None,
        "run_directory": str(run_dir),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "configuration": {
            "model": args.model,
            "requested_reference_mode": args.reference_mode,
            "effective_reference_mode": mode,
            "cache_ttl_seconds": args.cache_ttl_seconds if mode == "cache" else None,
            "keep_cache": args.keep_cache,
            "aspect_ratio": args.aspect_ratio,
            "thinking_level": args.thinking_level if normalized_model.startswith("gemini-3.1-") else None,
            "skip_first": args.skip_first,
            "input_dir": str(input_dir),
            "refs_dir": str(refs_dir),
            "prompt_file": str(prompt_path),
            "prompt": prompt,
            "system_instruction": SYSTEM_INSTRUCTION,
        },
        "dependencies": {"python": sys.version, **package_versions()},
        "pricing_assumptions": PRICING.get(normalized_model, {"date": "unknown"}),
        "inputs": [file_record(path) for path in pages],
        "references": [
            {"label": reference_label(path), **file_record(path)} for path in refs
        ],
        "reference_atlas": file_record(atlas_path),
        "cache": None,
        "pages": [],
        "totals": {
            "api_calls": 0,
            "successful_pages": 0,
            "known_estimated_cost_usd": 0.0,
            "unpriced_cached_tokens": 0,
        },
        "cleanup": {"cache_deleted": None, "uploaded_atlas_deleted": None, "errors": []},
    }
    write_json(manifest_path, manifest)

    client = genai.Client(api_key=api_key)
    cache = None
    uploaded_atlas = None
    previous_output: Path | None = None

    try:
        if mode == "cache":
            cache, uploaded_atlas = create_reference_cache(
                client, args.model, atlas_path, args.cache_ttl_seconds
            )
            manifest["cache"] = serialize_cache(cache)
            write_json(manifest_path, manifest)

        for index, page_path in enumerate(pages):
            page_number = index + 1
            contextual_prompt = prompt + (
                "\n\nThis is the first page. There is no previous colorized page. "
                "Use canonical colors from the reference atlas for known characters, "
                "and invent a coherent palette for all other elements."
                if previous_output is None
                else "\n\nUse the supplied PREVIOUS COLORIZED PAGE only to maintain "
                "color continuity for recurring characters, clothing, objects, lighting, "
                "and locations. Colorize the CURRENT BLACK-AND-WHITE PAGE."
            )
            parts: list[types.Part] = [types.Part.from_text(text=contextual_prompt)]
            if mode == "inline-atlas":
                parts.extend(
                    [
                        types.Part.from_text(text="REFERENCE ATLAS:"),
                        image_part(atlas_path),
                    ]
                )
            if previous_output is not None:
                parts.extend(
                    [
                        types.Part.from_text(text="PREVIOUS COLORIZED PAGE:"),
                        image_part(previous_output),
                    ]
                )
            parts.extend(
                [
                    types.Part.from_text(text="CURRENT BLACK-AND-WHITE PAGE:"),
                    image_part(page_path),
                ]
            )

            manifest["totals"]["api_calls"] += 1
            write_json(manifest_path, manifest)
            response = client.models.generate_content(
                model=args.model,
                contents=[types.Content(role="user", parts=parts)],
                config=generation_config(
                    args, mode, getattr(cache, "name", None) if cache else None
                ),
            )
            image_bytes, mime_type, response_notes = extract_generated_image(response)
            output_path = run_dir / f"{page_path.stem}{extension_for_mime(mime_type)}"
            output_path.write_bytes(image_bytes)
            with Image.open(io.BytesIO(image_bytes)) as generated:
                output_dimensions = [generated.width, generated.height]

            usage = usage_record(response)
            cost = estimated_cost(args.model, usage)
            page_record = {
                "sequence": page_number,
                "input": file_record(page_path),
                "previous_colorized_page": str(previous_output) if previous_output else None,
                "output": {
                    "path": str(output_path),
                    "sha256": sha256(output_path),
                    "bytes": output_path.stat().st_size,
                    "mime_type": mime_type,
                    "dimensions": output_dimensions,
                },
                "usage_metadata": usage,
                "estimated_cost": cost,
                "response_notes": response_notes,
                "completed_at": iso_now(),
            }
            manifest["pages"].append(page_record)
            manifest["totals"]["successful_pages"] += 1
            if cost["known_usd"] is not None:
                manifest["totals"]["known_estimated_cost_usd"] = round(
                    manifest["totals"]["known_estimated_cost_usd"] + cost["known_usd"],
                    8,
                )
            manifest["totals"]["unpriced_cached_tokens"] += (
                cost["unpriced_cached_tokens"] or 0
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
        if not args.keep_cache:
            if cache is not None:
                try:
                    client.caches.delete(name=cache.name)
                    manifest["cleanup"]["cache_deleted"] = True
                except Exception as cleanup_error:
                    manifest["cleanup"]["cache_deleted"] = False
                    manifest["cleanup"]["errors"].append(str(cleanup_error))
            if uploaded_atlas is not None:
                try:
                    client.files.delete(name=uploaded_atlas.name)
                    manifest["cleanup"]["uploaded_atlas_deleted"] = True
                except Exception as cleanup_error:
                    manifest["cleanup"]["uploaded_atlas_deleted"] = False
                    manifest["cleanup"]["errors"].append(str(cleanup_error))
        manifest["finished_at"] = iso_now()
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    run()
