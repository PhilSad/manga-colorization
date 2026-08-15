#!/usr/bin/env python3
"""research-v2: convert cast reference images to manga line art style via the
Spark FLUX.2 Klein image-edit server.

The color references in `data/refs/` are full-color anime portraits; the
YOLOE visual-prompt detection fails because of the color -> B&W-manga domain
gap. This script edits each reference into black-and-white manga line art
(same character, pose, and identity; clean line work on white) through the
self-hosted BentoML `POST /edit` endpoint, so the next detection run prompts
YOLOE with references in the same visual domain as the panels.

Request (see server/service.py):
    POST /edit multipart/form-data
      images   [current=color reference, atlas=B&W manga style exemplar]
      prompt   style-conversion instruction
      width, height, num_inference_steps, guidance_scale, lora_scale, seed,
      output_format

The server runs the step-distilled FLUX.2 Klein 9B (4 steps, CFG ignored);
the manga-colorization LoRA is loaded but neutralized with `lora_scale=0`.

Output: `research-v2/data/refs_manga/<name>_reference.png` per cast member,
plus a manifest.json recording prompt, seed, and cost.

Usage:
    .venv/bin/python research-v2/convert_refs_to_manga.py
    .venv/bin/python research-v2/convert_refs_to_manga.py --names himmel frieren eisen heiter
    .venv/bin/python research-v2/convert_refs_to_manga.py --style-ref <manga-page.png> --seed 7
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

from detect_characters_yoloe import CASTS_JSON, DEFAULT_REFS_DIR, load_cast

DEFAULT_ENDPOINT = "http://spark:3000"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "refs_manga"
DEFAULT_CAST_KEY = "c001"
# B&W manga style exemplar passed as the edit atlas (a 4-character panel of
# the same series, so FLUX.2 Klein sees the target line-art style).
DEFAULT_STYLE_REF = (
    Path(__file__).resolve().parent / "data" / "panels"
    / "Frieren - Beyond Journey's End - c001 (v01) - p007 [VIZ Media] [Digital] [1r0n]"
    / "panel_0002.png"
)
TIMEOUT_SECONDS = 1800  # first request pays the model-load cost (~1-3 min)

PROMPT = (
    "Convert this colored anime character illustration into clean black and "
    "white manga line art in the style of the reference manga page. Keep the "
    "character's identity, pose, hair style, facial features, clothing and "
    "expression exactly the same. Use crisp black ink lines on a pure white "
    "background with no colors and no gray shading."
)


def _mime(path: Path) -> str:
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}[
        path.suffix.lower().lstrip(".")
    ]


def _edit_multiple_of_16(size: tuple[int, int]) -> tuple[int, int]:
    """FLUX VAE requires multiples of 16 (the server floors to a valid size)."""
    return (size[0] // 16 * 16, size[1] // 16 * 16)


def convert_reference(
    ref_path: Path,
    style_ref: Path,
    out_path: Path,
    *,
    endpoint: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int | None,
    output_format: str,
) -> dict:
    from PIL import Image

    with Image.open(ref_path) as im:
        width, height = _edit_multiple_of_16(im.size)
    fields = {
        "prompt": PROMPT,
        "width": str(width),
        "height": str(height),
        "num_inference_steps": str(num_inference_steps),
        "guidance_scale": str(guidance_scale),
        "lora_scale": "0",  # neutralize the loaded colorization LoRA
        "output_format": output_format,
    }
    if seed is not None:
        fields["seed"] = str(seed)
    files = [
        ("images", (ref_path.name, open(ref_path, "rb"), _mime(ref_path))),
        ("images", (style_ref.name, open(style_ref, "rb"), _mime(style_ref))),
    ]
    started = time.monotonic()
    try:
        response = requests.post(
            f"{endpoint.rstrip('/')}/edit", data=fields, files=files, timeout=TIMEOUT_SECONDS
        )
    except Exception as error:  # noqa: BLE001 - connection errors etc.
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    finally:
        for _, (_, handle, _) in files:
            handle.close()

    latency_s = round(time.monotonic() - started, 1)
    if response.status_code != 200:
        return {
            "status": "error",
            "error": f"HTTP {response.status_code}: {response.text[:500]}",
            "latency_s": latency_s,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return {"status": "ok", "latency_s": latency_s, "size": list(im.size)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cast reference images to manga line art via Spark FLUX edit."
    )
    parser.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS_DIR,
                        help="source color references (default: data/refs)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="manga-converted output dir (default: data/refs_manga)")
    parser.add_argument("--names", nargs="*", default=None,
                        help="characters to convert (default: the cast for --cast-key)")
    parser.add_argument("--cast-key", default=DEFAULT_CAST_KEY,
                        help="chapter cast when --names is not given (default: c001)")
    parser.add_argument("--style-ref", type=Path, default=DEFAULT_STYLE_REF,
                        help="B&W manga style exemplar (atlas slot)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="Spark FLUX edit endpoint (default: http://spark:3000)")
    parser.add_argument("--num-inference-steps", type=int, default=4,
                        help="Klein step-distilled default is 4")
    parser.add_argument("--guidance-scale", type=float, default=4.0,
                        help="ignored by the distilled model, kept > 0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-format", default="png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = args.names or load_cast(args.cast_key)
    style_ref = args.style_ref
    if not style_ref.is_file():
        raise SystemExit(f"style exemplar not found: {style_ref}")

    records = []
    for name in names:
        ref_path = args.refs_dir / f"{name.lower()}_reference.webp"
        if not ref_path.is_file():
            raise SystemExit(f"no reference image for {name}: {ref_path}")
        out_path = args.out_dir / f"{name.lower()}_reference.{args.output_format}"
        print(f"converting {name} ({ref_path.name}) ...", flush=True)
        record = convert_reference(
            ref_path, style_ref, out_path,
            endpoint=args.endpoint,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_format=args.output_format,
        )
        record["name"] = name
        record["output"] = str(out_path.relative_to(Path(__file__).resolve().parents[1]))
        records.append(record)
        print(f"  -> {record['status']}" + (f" ({record['latency_s']}s)" if "latency_s" in record else ""))

    manifest = {
        "command": "research-v2/convert_refs_to_manga.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt": PROMPT,
        "config": {
            "endpoint": args.endpoint,
            "style_ref": str(style_ref),
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "lora_scale": 0,
            "seed": args.seed,
            "output_format": args.output_format,
        },
        "backend": {
            "model": "FLUX.2 Klein 9B (step-distilled) via Spark BentoML /edit",
            "cost": "self-hosted on Spark, $0 per call (electricity only)",
        },
        "records": records,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    failed = [r for r in records if r["status"] != "ok"]
    print(f"\n{len(records) - len(failed)}/{len(records)} converted -> {args.out_dir}")
    for r in failed:
        print(f"  FAILED {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
