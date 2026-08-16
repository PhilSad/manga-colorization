#!/usr/bin/env python3
"""Probe whether Luna (high reasoning effort) can output edit-need bounding boxes.

Replays a previously REJECTED colorization (one the verifier already judged
"not ok" in a real pipeline run — see `*.verify.json`), asks `openai/gpt-5.6-luna`
on OpenRouter to return, as strict structured output, the bounding boxes of the
regions in the colorized image where palette edits are needed, then draws those
boxes on a copy of the colorized image for visual inspection and records
cost/latency. This answers the open question behind a planned verify-loop
improvement ("fix only these regions") before any pipeline code changes.

Usage:
  .venv/bin/python pipeline_v1/scripts/probe_luna_bboxes.py \
      --run-dir pipeline_v1/output/20260815-174713 --page p010 --attempt 2
  .venv/bin/python pipeline_v1/scripts/probe_luna_bboxes.py --help

Inputs (resolved from a completed run):
  3_colorized/<page>/<stem>.verify.json   previous verdicts (ground truth fix_prompt)
  3_colorized/<page>/<stem>.attempt_N.png rejected colorization under review
  3_colorized/<page>/<stem>_atlas.jpg     reference atlas (canonical palette)
  1_panels/<page>/<stem>.png              original monochrome image (reference)

Output (default pipeline_v1/output/<ts>-luna-bbox/):
  request.json      exact prompt + schema + image inventory sent
  response.json     raw model response + usage + cost
  parsed.json       parsed regions/verdict
  manifest.json     provenance (model, reasoning effort/mode, images, ground truth)
  annotated.png     colorized image + drawn boxes (needs Pillow)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Make the pipeline package importable when run as a script from anywhere.
PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from dotenv import load_dotenv  # noqa: E402

from characters import _iso_now, call_vlm  # noqa: E402
from verify_color import _content_with_images  # noqa: E402
from config import REPO_ROOT  # noqa: E402

DEFAULT_MODEL = "openai/gpt-5.6-luna"
PROMPT_FILE = Path(__file__).resolve().parent / "probe_luna_bbox_prompt.txt"
API_KEY_ENV = "OPENROUTER_API_KEY"

# Strict json_schema structured output (same convention as verify_color.py).
BBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analyse": {
            "type": "string",
            "description": (
                "Which characters' palettes are correct or wrong, and where "
                "the wrong regions are."
            ),
        },
        "good_color": {
            "type": "boolean",
            "description": "True if every character has its canonical palette.",
        },
        "regions": {
            "type": "array",
            "description": (
                "One entry per region of the colorized image that needs a "
                "color edit; empty when good_color is true."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "character": {
                        "type": "string",
                        "description": (
                            "Canonical character name from the atlas, or a "
                            "short description."
                        ),
                    },
                    "problem": {
                        "type": "string",
                        "description": (
                            "What is wrong and what the canonical color "
                            "should be."
                        ),
                    },
                    "fix_suggestion": {
                        "type": "string",
                        "description": (
                            "Exact corrective instruction for the colorizer."
                        ),
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "[x1, y1, x2, y2] in normalized 0-1000 integer "
                            "coordinates; (0,0) top-left, (1000,1000) "
                            "bottom-right of the colorized image."
                        ),
                    },
                },
                "required": [
                    "character", "problem", "fix_suggestion", "bbox",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analyse", "good_color", "regions"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "bbox_verdict",
        "strict": True,
        "schema": BBOX_SCHEMA,
    },
}

# Colors for drawing boxes (cycling), high-contrast on both light and dark.
BOX_COLORS = [
    (255, 0, 0),      # red
    (0, 0, 255),      # blue
    (0, 200, 0),      # green
    (255, 165, 0),    # orange
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
]


@dataclass
class ProbeCase:
    """Resolved inputs for one probe: the previously rejected colorization."""

    verify_json: Path
    stem: str
    colorized: Path              # rejected attempt image under review
    atlas: Path | None
    monochrome: Path | None
    ground_truth_fix: str        # recorded fix_prompt from the real run
    verdict_status: str          # e.g. "mismatch" (attempt N of the real run)

    @property
    def page_dir(self) -> str:
        return self.verify_json.parent.name


def resolve_case(run_dir: Path, page_filter: str, attempt: int) -> ProbeCase:
    """Find the verify.json for the page, then the attempt image to probe."""
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")

    verify_files = sorted(run_dir.glob(f"3_colorized/*/*.verify.json"))
    matches = [
        v for v in verify_files if page_filter in v.parent.name
    ]
    if not matches:
        raise SystemExit(
            f"no verify.json matching --page {page_filter!r} in {run_dir}"
        )
    if len(matches) > 1:
        listing = "\n  ".join(str(v.parent.name) for v in matches)
        raise SystemExit(
            f"--page {page_filter!r} matches {len(matches)} pages:\n"
            f"  {listing}\nbe more specific"
        )
    verify_json = matches[0]
    doc = json.loads(verify_json.read_text(encoding="utf-8"))
    out_dir = verify_json.parent

    # Attempt image: prefer <stem>.attempt_<N><ext>, fall back to <stem><ext>.
    stems: set[str] = {
        Path(a["colorize"]["output"]["filename"]).stem
        for a in doc.get("attempts", [])
        if a.get("colorize", {}).get("output", {}).get("filename")
    } | {verify_json.stem.replace(".verify", "")}
    stem = sorted(stems)[0] if stems else verify_json.stem
    extension = next(
        (
            Path(a["colorize"]["output"]["filename"]).suffix
            for a in doc.get("attempts", [])
            if a.get("colorize", {}).get("output", {}).get("filename")
        ),
        ".png",
    )
    colorized = out_dir / f"{stem}.attempt_{attempt}{extension}"
    if not colorized.is_file():
        colorized = out_dir / f"{stem}{extension}"
    if not colorized.is_file():
        raise SystemExit(f"no colorized image to probe in {out_dir}")

    atlas = out_dir / f"{stem}_atlas.jpg"
    if not atlas.is_file():
        atlas = out_dir / f"{stem}_atlas.png"
    if not atlas.is_file():
        atlas = None

    monochrome = run_dir / "1_panels" / out_dir.name / f"{stem}{extension}"
    if not monochrome.is_file():
        monochrome = None

    # Ground truth: the last fix_prompt from the real run ('' if none).
    fix = ""
    status = "unknown"
    for a in doc.get("attempts", []):
        v = a.get("verify", {})
        if v.get("fix_prompt"):
            fix = v["fix_prompt"]
        if a.get("attempt") == attempt:
            status = v.get("status", "unknown")

    return ProbeCase(
        verify_json=verify_json,
        stem=stem,
        colorized=colorized,
        atlas=atlas,
        monochrome=monochrome,
        ground_truth_fix=fix,
        verdict_status=status,
    )


def parse_bbox_verdict(text: str) -> dict | None:
    """Parse `{analyse, good_color, regions}`; returns None when malformed."""
    from characters import _extract_json_object

    if not text:
        return None
    data = _extract_json_object(text.strip())
    if not isinstance(data, dict):
        return None
    raw = data.get("good_color")
    if isinstance(raw, bool):
        good = raw
    elif isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        good = raw.strip().lower() == "true"
    else:
        return None
    regions = data.get("regions")
    if regions is None:
        regions = []
    elif not isinstance(regions, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            bbox = [int(round(float(v))) for v in bbox]
        else:
            bbox = None
        cleaned.append(
            {
                "character": str(region.get("character") or ""),
                "problem": str(region.get("problem") or ""),
                "fix_suggestion": str(region.get("fix_suggestion") or ""),
                "bbox": bbox,
            }
        )
    return {
        "analyse": str(data.get("analyse") or ""),
        "good_color": good,
        "regions": cleaned,
    }


def draw_boxes(colorized: Path, parsed: dict, out: Path) -> Path:
    """Overlay each region's bbox (normalized 0-1000) on the colorized image.

    Returns the annotated image path. Requires Pillow; missing bboxes are
    skipped (the region text is still logged)."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(colorized).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:  # noqa: BLE001 - fall back to the default bitmap font
        font = ImageFont.load_default()

    for index, region in enumerate(parsed.get("regions", [])):
        bbox = region.get("bbox")
        if bbox is None or len(bbox) != 4:
            print(f"  [skip] region {index} has no usable bbox: {region!r}")
            continue
        x1, y1, x2, y2 = [
            max(0, min(1000, int(v))) for v in bbox
        ]
        color = BOX_COLORS[index % len(BOX_COLORS)]
        rect = (
            int(x1 / 1000 * width),
            int(y1 / 1000 * height),
            int(x2 / 1000 * width),
            int(y2 / 1000 * height),
        )
        draw.rectangle(rect, outline=color, width=4)
        label = f"{index}: {region.get('character') or '?'}"
        draw.text((rect[0], max(0, rect[1] - 24)), label, fill=color, font=font)

    image.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="probe_luna_bboxes",
        description=(
            "Ask Luna (high reasoning effort) for the bboxes of regions "
            "needing edits in a previously rejected colorization."
        ),
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="pipeline run dir containing 3_colorized/ (e.g. "
             "pipeline_v1/output/20260815-174713)",
    )
    parser.add_argument(
        "--page", default="p010",
        help="substring matching the page dir name (default: p010)",
    )
    parser.add_argument(
        "--attempt", type=int, default=2,
        help="attempt number of the rejected image to probe "
             "(default: 2 -> <stem>.attempt_2.<ext>)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenRouter model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--reasoning-effort", default="high",
        help="OpenRouter reasoning.effort (low/medium/high/...; default: high)",
    )
    parser.add_argument(
        "--reasoning-mode", default=None,
        help="OpenRouter reasoning.mode (e.g. 'pro' for the *-pro variant)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="max output tokens (default: 2048; high effort needs headroom)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PIPELINE_DIR / "output",
        help="output root for the timestamped probe dir (default: "
             "pipeline_v1/output)",
    )
    parser.add_argument(
        "--no-annotate", action="store_true",
        help="skip drawing boxes (still writes parsed.json etc.)",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = __import__("os").getenv(API_KEY_ENV)
    if not api_key:
        raise SystemExit(
            f"missing {API_KEY_ENV}: set it in the environment or add it to "
            f"{REPO_ROOT / '.env'}"
        )

    case = resolve_case(args.run_dir, args.page, args.attempt)
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    images = (case.colorized, case.monochrome, case.atlas)
    content = _content_with_images(prompt, images)

    reasoning: dict[str, Any] = {"effort": args.reasoning_effort}
    if args.reasoning_mode:
        reasoning["mode"] = args.reasoning_mode

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    result = call_vlm(
        client, args.model, content,
        max_tokens=args.max_tokens,
        response_format=RESPONSE_FORMAT,
        extra_body={"reasoning": reasoning},
    )

    # Fresh timestamped output dir, never overwriting.
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_root / f"{stamp}-luna-bbox"
    suffix = 1
    while out_dir.exists():
        out_dir = args.output_root / f"{stamp}-luna-bbox-{suffix:02d}"
        suffix += 1
    out_dir.mkdir()

    parsed = parse_bbox_verdict(result.text) if result.error is None else None

    # Request provenance: what exactly was sent.
    request_doc: dict[str, Any] = {
        "model": args.model,
        "reasoning": reasoning,
        "max_tokens": args.max_tokens,
        "prompt": prompt,
        "schema": BBOX_SCHEMA,
        "images": {
            "colorized": str(case.colorized),
            "monochrome": str(case.monochrome) if case.monochrome else None,
            "atlas": str(case.atlas) if case.atlas else None,
        },
    }
    (out_dir / "request.json").write_text(
        json.dumps(request_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    annotated = None
    if parsed is not None and not args.no_annotate:
        annotated = draw_boxes(case.colorized, parsed, out_dir / "annotated.png")

    parsed_doc = {
        "parsed": parsed,
        "raw_text": result.text,
        "error": result.error,
        "usage": result.usage,
        "cost_usd": result.cost_usd,
        "cost_source": result.cost_source,
        "latency_s": round(result.latency_s, 3),
        "model_returned": result.model_returned,
        "attempts": result.attempts,
    }
    (out_dir / "response.json").write_text(
        json.dumps(parsed_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "parsed.json").write_text(
        json.dumps(
            {
                "analyse": parsed["analyse"] if parsed else None,
                "good_color": parsed["good_color"] if parsed else None,
                "regions": parsed["regions"] if parsed else [],
            },
            indent=2, ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _iso_now(),
        "purpose": (
            "probe: can Luna (high reasoning effort) output edit-need "
            "bounding boxes for a rejected colorization?"
        ),
        "case": {
            "run_dir": str(case.verify_json.parents[2]),
            "page_dir": case.page_dir,
            "stem": case.stem,
            "attempt_probed": args.attempt,
            "previous_verdict_status": case.verdict_status,
            "colorized": str(case.colorized),
            "monochrome": str(case.monochrome) if case.monochrome else None,
            "atlas": str(case.atlas) if case.atlas else None,
            "ground_truth_fix_prompt": case.ground_truth_fix,
        },
        "model": args.model,
        "reasoning": reasoning,
        "max_tokens": args.max_tokens,
        "result": {
            "error": result.error,
            "good_color": parsed["good_color"] if parsed else None,
            "regions": len(parsed["regions"]) if parsed else 0,
            "cost_usd": result.cost_usd,
            "cost_source": result.cost_source,
            "latency_s": round(result.latency_s, 3),
            "model_returned": result.model_returned,
            "usage": result.usage,
        },
        "outputs": {
            "request_json": str(out_dir / "request.json"),
            "response_json": str(out_dir / "response.json"),
            "parsed_json": str(out_dir / "parsed.json"),
            "annotated": str(annotated) if annotated else None,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Console summary.
    print(f"\ncase       : {case.page_dir} (attempt {args.attempt}, "
          f"previous verdict: {case.verdict_status})")
    print(f"colorized  : {case.colorized}")
    if case.monochrome:
        print(f"monochrome : {case.monochrome}")
    if case.atlas:
        print(f"atlas      : {case.atlas}")
    print(f"model      : {args.model}  reasoning={reasoning}")
    if result.error:
        print(f"ERROR      : {result.error}")
    elif parsed is None:
        print("ERROR      : unparseable model output")
    else:
        print(f"verdict    : {'GOOD' if parsed['good_color'] else 'NEEDS EDITS'}")
        print(f"regions    : {len(parsed['regions'])}")
        for i, region in enumerate(parsed["regions"]):
            print(
                f"  [{i}] {region['character'] or '?'} bbox={region['bbox']}\n"
                f"      problem: {region['problem']}\n"
                f"      fix   : {region['fix_suggestion']}"
            )
        if annotated:
            print(f"annotated  : {annotated}")
    print(f"cost       : {result.cost_usd} ({result.cost_source})  "
          f"latency: {result.latency_s:.1f}s  tokens: "
          f"{result.usage.get('total_tokens')}")
    print(f"ground truth fix_prompt from real run:\n  {case.ground_truth_fix or '(none)'}")
    print(f"output dir : {out_dir}")

    return 0 if (result.error is None and parsed is not None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
