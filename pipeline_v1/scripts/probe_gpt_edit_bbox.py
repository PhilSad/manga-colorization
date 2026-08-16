#!/usr/bin/env python3
"""Probe: does gpt-image-2 apply a bbox-guided color edit correctly?

Follow-up to `probe_luna_bboxes.py`: that probe proved Luna (high reasoning
effort) can output the bounding boxes of the regions where a rejected
colorization is palette-wrong. This probe tests the edit half of the loop —
take the colorized page with Luna's boxes **drawn on it**, hand it to
gpt-image-2 (the full-page OpenAI method, `GptImage2Colorizer`) with a prompt
that says "recolor ONLY inside the red rectangles to these canonical colors",
and check whether the edits actually landed.

This is a behavior probe, not a pipeline change: nothing in verify_loop.py /
verify_color.py / gpt_colorizer.py is modified. The boxed image is the only
locator (no mask — gpt-image-2 mask support is unverified and out of scope).

Usage:
  .venv/bin/python pipeline_v1/scripts/probe_gpt_edit_bbox.py \
      --probe-dir pipeline_v1/output/20260816-110358-luna-bbox \
      --run-dir pipeline_v1/output/20260815-174713 --page p010 --attempt 2
  .venv/bin/python pipeline_v1/scripts/probe_gpt_edit_bbox.py \
      ... --verify        # also re-probe the edited result with Luna bboxes
  .venv/bin/python pipeline_v1/scripts/probe_gpt_edit_bbox.py --help

Inputs:
  --probe-dir/parsed.json     Luna regions (from the bbox probe)
  --probe-dir/annotated.png   colorized page + boxes (re-drawn if missing)
  --run-dir/--page/--attempt  resolve the original colorized page + atlas

Output (default pipeline_v1/output/<ts>-gpt-edit-bbox/):
  request.json       exact prompt + regions + image inventory sent
  response.json      gpt-image-2 result (usage + est_cost_usd)
  edited.png         the gpt-image-2 output (boxes removed per prompt)
  manifest.json      provenance + cost + region in/out summary
  verify2/           (only with --verify) Luna re-probe of the edited image:
                     parsed.json + annotated.png + manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Make the pipeline package importable when run as a script from anywhere.
PIPELINE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from dotenv import load_dotenv  # noqa: E402

from characters import _iso_now, call_vlm  # noqa: E402
from config import REPO_ROOT  # noqa: E402
from gpt_colorizer import GptImage2Colorizer  # noqa: E402
from probe_luna_bboxes import (  # noqa: E402
    BBOX_SCHEMA,
    PROMPT_FILE as BBOX_PROMPT_FILE,
    RESPONSE_FORMAT as BBOX_RESPONSE_FORMAT,
    draw_boxes,
    parse_bbox_verdict,
    resolve_case,
)
from verify_color import _content_with_images  # noqa: E402

EDIT_PROMPT_FILE = Path(__file__).resolve().parent / "probe_gpt_edit_prompt.txt"
VERIFY_MODEL = "openai/gpt-5.6-luna"


def region_instruction(regions: list[dict[str, Any]]) -> str:
    """Render the numbered region fixes for the edit prompt's
    `{character_profiles}` slot, matching the boxes drawn on the image."""
    lines = ["Regions to fix (numbered in order, matching the red rectangles):"]
    for i, region in enumerate(regions):
        character = region.get("character") or "?"
        fix = (region.get("fix_suggestion") or region.get("problem") or "").strip()
        lines.append(f"- Region {i} ({character}): {fix}")
    return "\n".join(lines)


def fresh_output_dir(output_root: Path, label: str) -> Path:
    """Fresh timestamped output dir, never overwriting (repo convention)."""
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / f"{stamp}-{label}"
    suffix = 1
    while out_dir.exists():
        out_dir = output_root / f"{stamp}-{label}-{suffix:02d}"
        suffix += 1
    out_dir.mkdir()
    return out_dir


def write_json(path: Path, doc: dict) -> None:
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_verify_probe(
    edited: Path,
    monochrome: Path | None,
    atlas: Path | None,
    api_key: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Re-probe the edited image with Luna bboxes (high effort, 8192 tokens —
    the probe showed 2048 is fully consumed by reasoning). Writes verify2/
    inside `out_dir`. Returns the result summary."""
    from openai import OpenAI

    verify_dir = out_dir / "verify2"
    verify_dir.mkdir(exist_ok=True)

    prompt = BBOX_PROMPT_FILE.read_text(encoding="utf-8")
    content = _content_with_images(prompt, (edited, monochrome, atlas))
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    result = call_vlm(
        client, VERIFY_MODEL, content,
        max_tokens=8192,
        response_format=BBOX_RESPONSE_FORMAT,
        extra_body={"reasoning": {"effort": "high"}},
    )
    parsed = parse_bbox_verdict(result.text) if result.error is None else None

    annotated = None
    if parsed is not None:
        annotated = draw_boxes(edited, parsed, verify_dir / "annotated.png")

    write_json(
        verify_dir / "parsed.json",
        {
            "analyse": parsed["analyse"] if parsed else None,
            "good_color": parsed["good_color"] if parsed else None,
            "regions": parsed["regions"] if parsed else [],
        },
    )
    write_json(
        verify_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at": _iso_now(),
            "purpose": "re-probe of the bbox-guided gpt-image-2 edit",
            "model": VERIFY_MODEL,
            "reasoning": {"effort": "high"},
            "max_tokens": 8192,
            "edited": str(edited),
            "monochrome": str(monochrome) if monochrome else None,
            "atlas": str(atlas) if atlas else None,
            "result": {
                "error": result.error,
                "good_color": parsed["good_color"] if parsed else None,
                "regions": len(parsed["regions"]) if parsed else 0,
                "cost_usd": result.cost_usd,
                "cost_source": result.cost_source,
                "latency_s": round(result.latency_s, 3),
                "usage": result.usage,
            },
            "outputs": {
                "parsed_json": str(verify_dir / "parsed.json"),
                "annotated": str(annotated) if annotated else None,
            },
        },
    )
    return {
        "good_color": parsed["good_color"] if parsed else None,
        "regions": parsed["regions"] if parsed else [],
        "regions_count": len(parsed["regions"]) if parsed else 0,
        "cost_usd": result.cost_usd,
        "latency_s": round(result.latency_s, 3),
        "error": result.error,
        "dir": str(verify_dir),
        "annotated": str(annotated) if annotated else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="probe_gpt_edit_bbox",
        description=(
            "Ask gpt-image-2 to re-color only the Luna-bbox regions of a "
            "rejected colorization, then (optionally) re-probe with Luna."
        ),
    )
    parser.add_argument(
        "--probe-dir", type=Path, required=True,
        help="luna-bbox output dir containing parsed.json (regions) and "
             "annotated.png (colorized page + boxes)",
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="pipeline run dir (e.g. pipeline_v1/output/20260815-174713)",
    )
    parser.add_argument(
        "--page", default="p010",
        help="substring matching the page dir name (default: p010)",
    )
    parser.add_argument(
        "--attempt", type=int, default=2,
        help="attempt number of the rejected image (default: 2)",
    )
    parser.add_argument(
        "--atlas-scale", type=float, default=1.0,
        help="atlas downscale for the gpt-image-2 upload (default: 1.0)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="re-probe the edited image with Luna bboxes (extra ~$0.002-0.004)",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PIPELINE_DIR / "output",
        help="output root for the timestamped probe dir (default: "
             "pipeline_v1/output)",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    import os

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit(
            f"missing OPENAI_API_KEY: set it in the environment or add it to "
            f"{REPO_ROOT / '.env'}"
        )

    # --- inputs -------------------------------------------------------------
    parsed_path = args.probe_dir / "parsed.json"
    if not parsed_path.is_file():
        raise SystemExit(f"no parsed.json in probe dir: {args.probe_dir}")
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    regions = parsed.get("regions") or []
    if not regions:
        raise SystemExit("probe dir has no regions to edit — nothing to do")

    case = resolve_case(args.run_dir, args.page, args.attempt)

    # Boxed image: prefer the probe's annotated.png; re-draw if missing.
    annotated = args.probe_dir / "annotated.png"
    if not annotated.is_file():
        print("  [note] annotated.png missing in probe dir — re-drawing boxes")
        annotated = draw_boxes(case.colorized, parsed, args.probe_dir / "annotated.png")

    # --- gpt-image-2 edit ---------------------------------------------------
    prompt_template = EDIT_PROMPT_FILE.read_text(encoding="utf-8")
    instruction = region_instruction(regions)
    colorizer = GptImage2Colorizer(
        prompt_template=prompt_template,
        atlas_scale=args.atlas_scale,
    )

    out_dir = fresh_output_dir(args.output_root, "gpt-edit-bbox")
    edited = out_dir / "edited.png"
    record = colorizer.colorize(
        panel=annotated,
        atlas=case.atlas,
        output=edited,
        palette_instruction=instruction,
    )

    # --- optional Luna re-probe --------------------------------------------
    verify_summary = None
    if args.verify and record.status == "ok":
        verify_summary = run_verify_probe(
            edited, case.monochrome, case.atlas, os.getenv("OPENROUTER_API_KEY"),
            out_dir,
        )

    # --- provenance ---------------------------------------------------------
    request_doc = {
        "model": colorizer.model,
        "quality": getattr(record, "quality", None),
        "size": list(record.requested_size) if record.requested_size else None,
        "prompt_template": prompt_template,
        "regions": regions,
        "region_instruction": instruction,
        "images": {
            "edited_target": str(annotated),
            "atlas": str(case.atlas) if case.atlas else None,
            "monochrome": str(case.monochrome) if case.monochrome else None,
        },
    }
    write_json(out_dir / "request.json", request_doc)

    manifest = {
        "schema_version": 1,
        "created_at": _iso_now(),
        "purpose": (
            "probe: can gpt-image-2 apply a bbox-guided region edit to a "
            "rejected colorization?"
        ),
        "case": {
            "run_dir": str(case.verify_json.parents[2]),
            "page_dir": case.page_dir,
            "stem": case.stem,
            "attempt_probed": args.attempt,
            "previous_verdict_status": case.verdict_status,
            "ground_truth_fix_prompt": case.ground_truth_fix,
            "colorized": str(case.colorized),
        },
        "probe_dir": str(args.probe_dir),
        "regions_in": regions,
        "edit": {
            "status": record.status,
            "error": record.error,
            "requested_size": record.requested_size,
            "original_size": record.original_size,
            "latency_s": round(record.latency_s, 3),
            "model": record.model,
            "quality": getattr(record, "quality", None),
            "usage": record.usage,
            "est_cost_usd": record.est_cost_usd,
            "output": str(record.output) if record.output else None,
        },
        "verify2": verify_summary,
        "outputs": {
            "request_json": str(out_dir / "request.json"),
            "edited": str(edited),
            "manifest_json": str(out_dir / "manifest.json"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    # --- console summary ----------------------------------------------------
    print(f"\ncase       : {case.page_dir} (attempt {args.attempt}, "
          f"previous verdict: {case.verdict_status})")
    print(f"target     : {annotated}  ({len(regions)} regions in)")
    for i, region in enumerate(regions):
        print(f"  in [{i}] {region.get('character')} bbox={region.get('bbox')}")
    print(f"model      : gpt-image-2 ({getattr(record, 'quality', None)})  "
          f"size={record.requested_size}")
    if record.status == "error":
        print(f"ERROR      : {record.error}")
    else:
        print(f"edited     : {edited}")
        print(f"est cost   : ${record.est_cost_usd}  "
              f"latency: {record.latency_s:.1f}s  "
              f"usage: {record.usage}")
    if verify_summary is not None:
        print("\n-- verify2 (Luna re-probe of the edited image) --")
        if verify_summary["error"]:
            print(f"  ERROR   : {verify_summary['error']}")
        else:
            verdict = ("GOOD" if verify_summary["good_color"]
                       else "STILL NEEDS EDITS")
            print(f"  verdict : {verdict}  regions: "
                  f"{verify_summary['regions_count']}")
            for i, region in enumerate(verify_summary["regions"]):
                print(f"  out [{i}] {region.get('character')} "
                      f"bbox={region.get('bbox')} fix={region.get('fix_suggestion')}")
            print(f"  cost    : ${verify_summary['cost_usd']}  "
                  f"latency: {verify_summary['latency_s']:.1f}s")
            if verify_summary["annotated"]:
                print(f"  view    : {verify_summary['annotated']}")
    print(f"output dir : {out_dir}")

    return 0 if record.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
