#!/usr/bin/env python3
"""Colorize a manga panel with OpenAI gpt-image-2 (Image API edits endpoint).

The test inputs live in data/patch/:

  orig.png   — the black & white manga panel (the image to colorize)
  patch.png  — the same panel with the character reference composited on top
  prompt.txt — the edit prompt ("colorize the manga panel using the character
               reference ... adapt the orientation ... use the correct
               characters colors")

Both images are sent as input images (no mask -> they act as references) and
the script runs the edit once per requested quality (default: low, medium,
high) so the quality settings can be compared on identical inputs.

Output: research-v2/output/<YYYYMMDD-HHMMSS>/ with one PNG per quality
(quality_<low|medium|high>.png) and a manifest.json recording the prompt,
config, per-quality timestamps/duration, the API's `usage` (if returned) and
cost notes.

Cost notes (OpenAI pricing page / image generation guide, 2026-06):
  gpt-image-2: image input $8.00 / 1M image tokens, text input $5.00 / 1M text
  tokens, image output $30.00 / 1M output tokens (standard tier). Output token
  count scales with size and quality (docs list e.g. 1024x1536: low $0.005 /
  medium $0.041 / high $0.165). gpt-image-2 always processes image inputs at
  high fidelity, so edit requests with reference images use more input tokens.
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def load_env(env_path: Path) -> dict:
    """Minimal .env loader (KEY=VALUE lines, no quoting magic)."""
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default="gpt-image-2", help="OpenAI image model (default gpt-image-2)"
    )
    ap.add_argument(
        "--quality",
        action="append",
        choices=["low", "medium", "high", "auto"],
        help="quality setting(s) to test; repeatable (default: low medium high)",
    )
    ap.add_argument(
        "--size",
        default="2880x2240",
        help="output size WxH, multiples of 16, max edge <= 3840, ratio <= 3:1, "
        "655,360..8,294,400 px (default 2880x2240 ~ the source panel aspect)",
    )
    ap.add_argument("--output-format", default="png", choices=["png", "jpeg", "webp"])
    ap.add_argument("--input-dir", default=str(HERE / "data" / "patch"))
    ap.add_argument("--prompt-file", default=str(HERE / "data" / "patch" / "prompt.txt"))
    ap.add_argument("--output-root", default=str(HERE / "output"))
    ap.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    args = ap.parse_args()

    qualities = args.quality or ["low", "medium", "high"]

    env = load_env(Path(args.env_file))
    if "OPENAI_API_KEY" not in env and "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not found in env or .env", file=sys.stderr)
        return 2
    os.environ.setdefault("OPENAI_API_KEY", env["OPENAI_API_KEY"])

    from openai import OpenAI

    client = OpenAI(timeout=600)

    input_dir = Path(args.input_dir)
    orig_path = input_dir / "orig.png"
    patch_path = input_dir / "patch.png"
    prompt_path = Path(args.prompt_file)
    if not all(p.exists() for p in (orig_path, patch_path, prompt_path)):
        print(
            f"error: expected {orig_path.name}, {patch_path.name}, {prompt_path.name} "
            f"in {input_dir}",
            file=sys.stderr,
        )
        return 2

    prompt = prompt_path.read_text().strip()
    images = []
    inputs_info = []
    for p in (orig_path, patch_path):
        with Image.open(p) as im:
            dims = im.size
            mode = im.mode
        images.append(open(p, "rb"))
        inputs_info.append(
            {"file": p.name, "path": str(p), "width": dims[0], "height": dims[1], "mode": mode}
        )

    run_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    size = args.size
    w, h = (int(v) for v in size.lower().split("x"))
    if w % 16 or h % 16:
        print(f"warning: {size} is not a multiple of 16 (API constraint)", file=sys.stderr)

    records = {}
    for q in qualities:
        print(f"\n=== quality={q} size={size} ===", flush=True)
        started = utcnow_iso()
        t0 = time.monotonic()
        try:
            resp = client.images.edit(
                model=args.model,
                image=images,
                prompt=prompt,
                quality=q,
                size=size,
                output_format=args.output_format,
                n=1,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue other qualities
            finished = utcnow_iso()
            records[q] = {
                "started": started,
                "finished": finished,
                "duration_s": round(time.monotonic() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"ERROR quality={q}: {type(exc).__name__}: {exc}", flush=True)
            continue

        finished = utcnow_iso()
        duration_s = round(time.monotonic() - t0, 1)
        data = resp.data[0]
        out_name = f"quality_{q}.{args.output_format}"
        out_path = run_dir / out_name
        out_path.write_bytes(base64.b64decode(data.b64_json))

        usage = None
        cost = None
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            input_details = getattr(raw_usage, "input_tokens_details", None)
            output_details = getattr(raw_usage, "output_tokens_details", None)
            input_image_tokens = getattr(input_details, "image_tokens", None) if input_details else None
            input_text_tokens = getattr(input_details, "text_tokens", None) if input_details else None
            output_image_tokens = getattr(output_details, "image_tokens", None) if output_details else None
            output_text_tokens = getattr(output_details, "text_tokens", None) if output_details else None
            usage = {
                "input_tokens": getattr(raw_usage, "input_tokens", None),
                "output_tokens": getattr(raw_usage, "output_tokens", None),
                "total_tokens": getattr(raw_usage, "total_tokens", None),
                "input_tokens_details": {
                    "image_tokens": input_image_tokens,
                    "text_tokens": input_text_tokens,
                },
                "output_tokens_details": {
                    "image_tokens": output_image_tokens,
                    "text_tokens": output_text_tokens,
                },
            }
            # standard tier: image input $8/1M, text input $5/1M, image output $30/1M
            it_img = (input_image_tokens or 0) / 1e6 * 8.0
            it_txt = (input_text_tokens or 0) / 1e6 * 5.0
            ot_img = (output_image_tokens or 0) / 1e6 * 30.0
            ot_txt = (output_text_tokens or 0) / 1e6 * 30.0
            cost = round(it_img + it_txt + ot_img + ot_txt, 6)

        with Image.open(out_path) as im:
            out_dims = im.size

        records[q] = {
            "started": started,
            "finished": finished,
            "duration_s": duration_s,
            "output": out_name,
            "output_width": out_dims[0],
            "output_height": out_dims[1],
            "usage": usage,
            "est_cost_usd": cost,
            "revised_prompt": getattr(data, "revised_prompt", None),
        }
        print(
            f"OK quality={q}: {out_path.name} ({out_dims[0]}x{out_dims[1]}) "
            f"in {duration_s}s, usage={usage}, est_cost_usd={cost}",
            flush=True,
        )

    manifest = {
        "command": " ".join(sys.argv),
        "timestamp": utcnow_iso(),
        "config": {
            "model": args.model,
            "size": size,
            "output_format": args.output_format,
            "qualities": qualities,
            "input_images": inputs_info,
            "prompt_file": str(prompt_path),
        },
        "prompt": prompt,
        "cost_notes": {
            "standard_tier_per_1M_tokens": {
                "image_input_usd": 8.0,
                "text_input_usd": 5.0,
                "image_output_usd": 30.0,
            },
            "doc_reference_at_1024x1536_usd": {
                "low": 0.005,
                "medium": 0.041,
                "high": 0.165,
            },
            "note": "gpt-image-2 output is billed per output token (quality/size "
            "dependent); image inputs are always processed at high fidelity. "
            "est_cost_usd is computed from the API's usage details (input image "
            "$8/1M, input text $5/1M, output image $30/1M, standard tier).",
        },
        "records": records,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for img in images:
        img.close()

    print(f"\nrun dir: {run_dir}")
    print(f"manifest: {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
