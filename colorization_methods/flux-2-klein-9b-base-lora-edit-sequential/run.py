#!/usr/bin/env python3
"""Sequential manga colorization with FLUX.2 Klein 9B base + manga LoRA.

Runs the same sequential-reference pipeline as the fal FLUX.2 Klein 9B Edit
method (current page + labelled reference atlas + previous colorized page)
but against the self-hosted BentoML server (--endpoint), which serves the
undistilled `black-forest-labs/FLUX.2-klein-base-9B` checkpoint with the
thedeoxen manga-colorization-by-reference LoRA loaded (trigger word
`mngclranm` in the prompt). The base model is NOT step-distilled: use
~20-50 num_inference_steps and guidance_scale ~4-5.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import fal_client
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


METHOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = METHOD_DIR.parents[1]
METHOD_NAME = "flux-2-klein-9b-base-lora-edit-sequential"
DEFAULT_MODEL = "fal-ai/flux-2/klein/9b/edit"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

PRICING = {
    "model": DEFAULT_MODEL,
    "date": "2026-08-08",
    "currency": "USD",
    "source": "https://fal.ai/models/fal-ai/flux-2/klein/9b/edit",
    "usd_per_megapixel_input_and_output": 0.011,
    "input_image_billing": "Each input image is resized to 1 MP for billing.",
    "notes": (
        "Cost is estimated from the documented $0.011 per megapixel of input and "
        "output. The endpoint response does not include billed cost."
    ),
}

LOCAL_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
LOCAL_PRICING = {
    "model": LOCAL_MODEL_ID,
    "hosting": "self-hosted via BentoML on the DGX Spark (see server/ at repo root)",
    "date": "2026-08-08",
    "currency": "USD",
    "usd_per_megapixel_input_and_output": 0.0,
    "notes": (
        "Self-hosted: the server loads the thedeoxen manga-colorization-by-reference "
        "LoRA (trigger word mngclranm) on top of the undistilled base checkpoint. "
        "No per-call fee; estimates in this run are therefore $0. Electricity "
        "estimate: DGX Spark ~350-400 W during inference; an 18-page run at ~10-20 min "
        "is roughly $0.01-0.02 at $0.15/kWh. The base model at 20 steps is ~5x slower "
        "per page than the step-distilled 4-step model. See server/README.md (repo root)."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Colorize manga pages sequentially with fal FLUX.2 Klein 9B Edit, "
            "a labelled character-reference atlas, and the previous generated page."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, default=REPO_ROOT / "data" / "chapter_134"
    )
    parser.add_argument("--refs-dir", type=Path, default=REPO_ROOT / "data" / "refs")
    parser.add_argument("--output-root", type=Path, default=METHOD_DIR / "output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    # Base-model defaults: the undistilled checkpoint wants 1216x1824 (FLUX
    # VAE multiples of 16, parity with fal outputs) and ~20-50 steps.
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--height", type=int, default=1824)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help=(
            "CFG scale. Sent only when --endpoint is set (the fal schema has no "
            "guidance_scale): the LoRA's undistilled base model wants ~4-5, the "
            "plain distilled model ignores it."
        ),
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        help=(
            "LoRA weight override (0.8-1.0 recommended). Sent only when "
            "--endpoint is set and the server has the manga-colorization LoRA."
        ),
    )
    parser.add_argument(
        "--output-format", choices=("jpeg", "png", "webp"), default="png"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help=(
            "One-based chapter page index at which to start, applied after "
            "--skip-first (default 1)."
        ),
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N pages of the input folder before applying --start-at (default 0).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--previous-page",
        type=Path,
        help="Existing colorized page to upload as continuity context for the first selected page.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable fal's optional safety checker for this run.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=(
            "Base URL of a local BentoML FLUX.2 Klein server (e.g. "
            "http://spark:3000). When set, requests go to the local server "
            "instead of fal and no API key is required. Use --width 1216 "
            "--height 1824 for parity with fal outputs (FLUX VAE needs "
            "multiples of 16)."
        ),
    )
    parser.add_argument("--prompt-file", type=Path, default=METHOD_DIR / "prompt.txt")
    parser.add_argument("--api-key-env", default="FAL_API_KEY")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.start_at < 1:
        parser.error("--start-at must be at least 1")
    if args.skip_first < 0:
        parser.error("--skip-first must be non-negative")
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
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


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    return str(value)


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


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("fal-client", "Pillow", "python-dotenv", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def extension_for_format(output_format: str) -> str:
    return {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}[output_format]


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "manga-colorization/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def estimate_cost(input_count: int, output_dimensions: list[int]) -> dict[str, Any]:
    input_mp = float(input_count)
    output_mp = output_dimensions[0] * output_dimensions[1] / 1_000_000
    total_mp = input_mp + output_mp
    estimated_usd = total_mp * PRICING["usd_per_megapixel_input_and_output"]
    return {
        "input_image_count": input_count,
        "billed_input_megapixels_estimate": round(input_mp, 6),
        "output_megapixels": round(output_mp, 6),
        "total_megapixels_estimate": round(total_mp, 6),
        "estimated_usd": round(estimated_usd, 8),
        "note": (
            "Fal documents each input as resized to 1 MP for billing. Output cost "
            "is estimated from the downloaded output dimensions."
        ),
    }


def run() -> None:
    global PRICING, fal_client
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    if args.endpoint:
        import local_fal_client as fal_client

        fal_client.configure(args.endpoint)
        PRICING = LOCAL_PRICING
        if args.model == DEFAULT_MODEL:
            args.model = LOCAL_MODEL_ID
    else:
        api_key = os.getenv(args.api_key_env) or os.getenv("FAL_KEY")
        if not api_key:
            raise SystemExit(
                f"Missing API key: set {args.api_key_env} or FAL_KEY in {REPO_ROOT / '.env'}"
            )
        os.environ["FAL_KEY"] = api_key

    input_dir = args.input_dir.resolve()
    refs_dir = args.refs_dir.resolve()
    prompt_path = args.prompt_file.resolve()
    all_pages = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    refs = sorted(
        path
        for path in refs_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    pages = all_pages[args.skip_first + args.start_at - 1 :]
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        raise SystemExit(f"No supported page images found in {input_dir}")
    if not refs:
        raise SystemExit(f"No supported reference images found in {refs_dir}")
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    previous_page_path = args.previous_page.resolve() if args.previous_page else None
    if previous_page_path is not None and not previous_page_path.is_file():
        raise SystemExit(f"Previous colorized page not found: {previous_page_path}")

    run_dir = create_run_dir(args.output_root.resolve())
    manifest_path = run_dir / "manifest.json"
    atlas_path = run_dir / "reference-atlas.jpg"
    normalized_dir = run_dir / "normalized-inputs"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    build_reference_atlas(refs, atlas_path)
    output_extension = extension_for_format(args.output_format)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD_NAME,
        "status": "preparing",
        "started_at": iso_now(),
        "finished_at": None,
        "run_directory": str(run_dir),
        "command": shlex.join(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
        "configuration": {
            "model": args.model,
            "image_size": {"width": args.width, "height": args.height},
            "num_inference_steps": args.num_inference_steps,
            "output_format": args.output_format,
            "seed": args.seed,
            "start_at": args.start_at,
            "skip_first": args.skip_first,
            "initial_previous_page": (
                str(previous_page_path) if previous_page_path else None
            ),
            "enable_safety_checker": not args.disable_safety_checker,
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
            "reference_atlas": "4-column labelled JPEG atlas, uploaded once per run",
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
        "uploads": [],
        "pages": [],
        "totals": {
            "api_calls": 0,
            "uploads": 0,
            "successful_pages": 0,
            "safety_blocked_pages": 0,
            "estimated_cost_usd": 0.0,
        },
    }
    write_json(manifest_path, manifest)

    previous_remote_url: str | None = None
    previous_output: Path | None = previous_page_path

    try:
        atlas_url = fal_client.upload_file(str(atlas_path))
        manifest["uploads"].append(
            {"role": "reference_atlas", "path": str(atlas_path), "url": atlas_url}
        )
        manifest["totals"]["uploads"] += 1
        if previous_page_path is not None:
            previous_remote_url = fal_client.upload_file(str(previous_page_path))
            manifest["uploads"].append(
                {
                    "role": "initial_previous_colorized_page",
                    "path": str(previous_page_path),
                    "url": previous_remote_url,
                }
            )
            manifest["totals"]["uploads"] += 1
        manifest["status"] = "running"
        write_json(manifest_path, manifest)

        progress = tqdm(
            total=len(pages),
            desc="Colorizing",
            unit="page",
            dynamic_ncols=True,
        )
        try:
            for index, page_path in enumerate(pages):
                page_number = index + 1
                chapter_sequence = args.skip_first + args.start_at + index
                progress.set_description(f"chapter page {chapter_sequence}")
                progress.set_postfix_str(page_path.stem)
                normalized_page = normalized_dir / f"{page_path.stem}.png"
                normalize_page_for_upload(page_path, normalized_page)
                current_url = fal_client.upload_file(str(normalized_page))
                manifest["uploads"].append(
                    {
                        "role": "current_black_and_white",
                        "sequence": chapter_sequence,
                        "path": str(normalized_page),
                        "url": current_url,
                    }
                )
                manifest["totals"]["uploads"] += 1

                image_urls = [current_url, atlas_url]
                input_roles = ["current_black_and_white", "reference_atlas"]
                if previous_remote_url is None:
                    input_description = (
                        "#1 is the CURRENT BLACK-AND-WHITE TARGET PAGE. #2 is the "
                        "LABELLED CHARACTER REFERENCE ATLAS. There is no previous page; "
                        "invent a coherent palette where #2 does not apply."
                    )
                else:
                    image_urls.append(previous_remote_url)
                    input_roles.append("previous_colorized_page")
                    input_description = (
                        "#1 is the CURRENT BLACK-AND-WHITE TARGET PAGE and is the only "
                        "image to edit. #2 is the LABELLED CHARACTER REFERENCE ATLAS. "
                        "#3 is the PREVIOUS COLORIZED PAGE and is continuity guidance only."
                    )
                contextual_prompt = f"{prompt}\n\n{input_description}"
                arguments: dict[str, Any] = {
                    "prompt": contextual_prompt,
                    "image_urls": image_urls,
                    "image_size": {"width": args.width, "height": args.height},
                    "num_inference_steps": args.num_inference_steps,
                    "num_images": 1,
                    "enable_safety_checker": not args.disable_safety_checker,
                    "output_format": args.output_format,
                }
                if args.endpoint:
                    if args.guidance_scale is not None:
                        arguments["guidance_scale"] = args.guidance_scale
                    if args.lora_scale is not None:
                        arguments["lora_scale"] = args.lora_scale
                if args.seed is not None:
                    arguments["seed"] = args.seed + chapter_sequence - 1

                manifest["totals"]["api_calls"] += 1
                write_json(manifest_path, manifest)
                handler = fal_client.submit(args.model, arguments=arguments)
                request_id = handler.request_id
                manifest["active_request"] = {
                    "sequence": chapter_sequence,
                    "request_id": request_id,
                    "submitted_at": iso_now(),
                }
                write_json(manifest_path, manifest)
                result = handler.get()

                images = result.get("images") if isinstance(result, dict) else None
                if not images or not images[0].get("url"):
                    raise RuntimeError("fal returned no output image URL")
                remote_image = images[0]
                remote_url = remote_image["url"]
                output_path = run_dir / f"{page_path.stem}{output_extension}"
                download_file(remote_url, output_path)
                with Image.open(output_path) as generated:
                    output_dimensions = [generated.width, generated.height]
                    output_mime = Image.MIME.get((generated.format or "").upper())

                cost = estimate_cost(len(image_urls), output_dimensions)
                safety_flags = json_safe(result.get("has_nsfw_concepts"))
                safety_blocked = bool(
                    isinstance(safety_flags, list) and any(safety_flags)
                )
                page_record = {
                    "sequence": chapter_sequence,
                    "status": "safety_blocked" if safety_blocked else "completed",
                    "input": file_record(page_path),
                    "normalized_input": file_record(normalized_page),
                    "request_image_roles": input_roles,
                    "request_image_urls": image_urls,
                    "previous_colorized_page": (
                        str(previous_output) if previous_output else None
                    ),
                    "request_id": request_id,
                    "request_arguments": arguments,
                    "output": {
                        **file_record(output_path),
                        "mime_type": output_mime,
                        "remote": json_safe(remote_image),
                    },
                    "fal_result": {
                        "seed": result.get("seed"),
                        "timings": json_safe(result.get("timings")),
                        "has_nsfw_concepts": safety_flags,
                        "prompt": result.get("prompt"),
                    },
                    "estimated_cost": cost,
                    "completed_at": iso_now(),
                }
                manifest["pages"].append(page_record)
                manifest["totals"]["estimated_cost_usd"] = round(
                    manifest["totals"]["estimated_cost_usd"] + cost["estimated_usd"],
                    8,
                )
                manifest.pop("active_request", None)
                if safety_blocked:
                    manifest["totals"]["safety_blocked_pages"] += 1
                    write_json(manifest_path, manifest)
                    raise RuntimeError(
                        f"fal safety checker blocked chapter page {chapter_sequence}; "
                        f"the blocked output was preserved at {output_path}"
                    )
                manifest["totals"]["successful_pages"] += 1
                write_json(manifest_path, manifest)
                previous_remote_url = remote_url
                previous_output = output_path
                progress.update(1)
                tqdm.write(f"[{page_number}/{len(pages)}] wrote {output_path}")
        finally:
            progress.close()

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
