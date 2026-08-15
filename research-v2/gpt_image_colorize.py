#!/usr/bin/env python3
"""Colorize a manga panel with OpenAI gpt-image-2 (Image API edits endpoint).

Three input modes (method A/B on the same orig.png):

  patch (default)   — data/patch/: orig.png (the B&W panel) + patch.png (the
                      same panel with the character reference composited on
                      top) + prompt.txt. Both images act as references.
  atlas             — --atlas-chars: orig.png + a labelled reference atlas
                      (pipeline_v1/atlas.py, 360x480 labelled cells) built at
                      run time from data/refs/ for the given characters.
  no-reference      --no-reference: orig.png alone (model baseline, no
                      reference conditioning).

The image(s) are sent as input images (no mask -> additional images act as
references) and the script runs the edit once per requested quality (default:
low, medium, high) so the quality settings can be compared on identical
inputs.

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
# pipeline_v1 modules (atlas builder) are imported as a library.
PIPELINE_V1 = REPO_ROOT / "pipeline_v1"


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
    ap.add_argument("--prompt-file", default=None,
                    help="edit prompt file (default: data/patch/prompt.txt for the "
                    "patch mode, data/atlas/prompt.txt for --atlas-chars)")
    ap.add_argument("--output-root", default=str(HERE / "output"))
    ap.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    ap.add_argument("--atlas-chars", nargs="+", default=None, metavar="NAME",
                    help="run the atlas method: build a labelled reference atlas "
                    "from data/refs/ for these characters and send it as the second "
                    "input image instead of patch.png")
    ap.add_argument("--refs-dir", default=str(REPO_ROOT / "data" / "refs"),
                    help="reference images dir for --atlas-chars (default: data/refs)")
    ap.add_argument("--no-reference", action="store_true",
                    help="send only orig.png (no patch/atlas reference image)")
    args = ap.parse_args()

    if args.atlas_chars and args.no_reference:
        print("error: --atlas-chars and --no-reference are mutually exclusive",
              file=sys.stderr)
        return 2

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
    prompt_path = Path(args.prompt_file) if args.prompt_file else (
        Path(HERE / "data" / "atlas" / "prompt.txt")
        if args.atlas_chars else Path(input_dir / "prompt.txt")
    )
    if not orig_path.exists() or not prompt_path.exists():
        print(
            f"error: expected {orig_path.name} in {input_dir} and {prompt_path.name} "
            f"in {prompt_path.parent}",
            file=sys.stderr,
        )
        return 2

    # Atlas mode: build the labelled reference atlas (pipeline_v1 builder).
    atlas_built = None
    if args.atlas_chars:
        sys.path.insert(0, str(PIPELINE_V1))
        from atlas import build_labelled_atlas, refs_for_characters  # noqa: E402

        refs = refs_for_characters(args.atlas_chars, Path(args.refs_dir))
        if not refs:
            print("error: no reference images found for --atlas-chars "
                  f"{args.atlas_chars} in {args.refs_dir}", file=sys.stderr)
            return 2
        run_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        atlas_path = run_dir / "atlas.jpg"
        build_labelled_atlas(refs, atlas_path)
        atlas_built = {
            "file": atlas_path.name,
            "characters": args.atlas_chars,
            "refs": [str(r) for r in refs],
        }
        print(f"atlas: built {atlas_path} from {[r.name for r in refs]}", flush=True)
    else:
        run_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)

    prompt = prompt_path.read_text().strip()
    images = []
    inputs_info = []
    second_paths = [] if args.no_reference else ([atlas_path] if atlas_built else [input_dir / "patch.png"])
    for p in [orig_path] + second_paths:
        if not p.exists():
            print(f"error: expected {p.name} in {p.parent}", file=sys.stderr)
            return 2
        with Image.open(p) as im:
            dims = im.size
            mode = im.mode
        images.append(open(p, "rb"))
        inputs_info.append(
            {"file": p.name, "path": str(p), "width": dims[0], "height": dims[1], "mode": mode}
        )

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
            "mode": "no-reference" if args.no_reference else (
                "atlas" if atlas_built else "patch"),
            "atlas": atlas_built,
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
