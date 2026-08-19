#!/usr/bin/env python3
"""Luna-based semantic line-art check of a completed pipeline run.

Companion of the sanity step (7_sanity/) and of its offline tool
scripts/check_sanity.py: where the local metrics score line-art fidelity
with hand-crafted geometry (thin-stroke line maps), this tool asks a
vision-language model — `openai/gpt-5.6-luna` via OpenRouter, one paid call
per panel — to judge, in plain sight of the colorized panel and its B&W
original, whether the line art matches (`luna_sanity.py`, strict
json_schema structured output). Each call costs money (OpenRouter usage.cost
is recorded per panel and totalled); no Spark server or pipeline rerun is
needed, only `OPENROUTER_API_KEY` in `.env`.

Input layout (a completed run, e.g. pipeline_v1/output/YYYYMMDD-HHMMSS/):
  1_panels/<page>/panel_000N.png     black & white panel crops
  1_panels/<page>/panels.json        panel boxes
  4_stitched/<page>.png              final pages (colorized panels at box positions)
  3_colorized/<page>/                fallback source for the colorized panels
  manifest.json                      B&W-fallback panels are skipped

Output (default <run-dir>/7_sanity_luna/):
  inputs/<page>/<stem>.bw.png        the exact analysis-grid images the model
  inputs/<page>/<stem>.colorized.png saw (long edge <= --max-edge)
  <page>.json                        per-panel verdicts + costs
  <page>_mismatch.png                contact sheet of the mismatch panels
  summary.json                       run totals + cost + mismatch list

Usage:
  .venv/bin/python pipeline_v1/scripts/check_luna_sanity.py \
      --run-dir pipeline_v1/output/20260819-202719
  .venv/bin/python pipeline_v1/scripts/check_luna_sanity.py \
      --run-dir pipeline_v1/output/20260819-202719 --max-edge 1024 \
      --workers 4 --page p006
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make the pipeline package importable when run as a script from anywhere.
PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from dotenv import load_dotenv  # noqa: E402

from config import REPO_ROOT  # noqa: E402
from luna_sanity import (  # noqa: E402
    DEFAULT_LUNA_SANITY_MODEL,
    DEFAULT_MAX_EDGE,
    LineArtChecker,
    MATCHES,
    MISMATCH,
    analysis_pair,
)

# Reuse the sanity step's per-run loading helpers (same panels/boxes/colorized
# resolution as 7_sanity/).
from steps.sanity import _crop_box, _find_colorized, _load_stitched  # noqa: E402
from steps.debug import load_fallback_map  # noqa: E402
from tqdm import tqdm  # noqa: E402
from util import load_font  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

API_KEY_ENV = "OPENROUTER_API_KEY"


def _build_panel_tasks(run_dir: Path, page_substrings: tuple[str, ...]):
    """(page, crop_name, box, provenance, bw_path) for every panel to check.

    Mirrors the sanity step's iteration: panels with a B&W-fallback record
    are returned with `bw_fallback=True` (skipped by the caller), blank
    pages are skipped.
    """
    from config import PipelineConfig
    from run_context import RunContext

    ctx = RunContext.load(run_dir)
    panels_root = run_dir / "1_panels"
    stitched_dir = run_dir / "4_stitched"
    colorized_root = run_dir / "3_colorized"
    if not panels_root.is_dir():
        raise ValueError("no 1_panels/ dir in the run; run the 'panels' step first")
    config = PipelineConfig()
    stored = ctx.manifest.get("configuration", {})
    config.resume = stored.get("resume")
    fallback_map = load_fallback_map(ctx, config)

    page_dirs = sorted(path for path in panels_root.iterdir() if path.is_dir())
    if page_substrings:
        page_dirs = [
            path for path in page_dirs
            if any(sub in path.name for sub in page_substrings)
        ]
    if not page_dirs:
        raise ValueError(f"no pages selected in {panels_root}")

    tasks = []
    for page_dir in page_dirs:
        page = page_dir.name
        geometry_path = page_dir / "panels.json"
        if not geometry_path.is_file():
            continue
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        if geometry.get("blank_page"):
            continue
        stitched = _load_stitched(stitched_dir, page)
        for detection in geometry["detections"]:
            crop_name = detection["crop"]
            stem = Path(crop_name).stem
            bw_path = page_dir / crop_name
            if not bw_path.is_file():
                continue
            tasks.append({
                "page": page,
                "crop": crop_name,
                "stem": stem,
                "box": detection["box"],
                "provenance": detection.get("provenance"),
                "bw_path": bw_path,
                "bw_fallback": stem in fallback_map.get(page, set()),
                "stitched": stitched,
                "colorized_root": colorized_root,
            })
    return tasks


def _load_color_image(task: dict) -> Image.Image | None:
    """The colorized panel: cropped from the stitched page, else from
    3_colorized/<page>/ (same resolution as the sanity step)."""
    stitched = task["stitched"]
    if stitched is not None:
        color_image = _crop_box(stitched, task["box"])
        if color_image is not None:
            return color_image
    colorized_path = _find_colorized(
        task["colorized_root"] / task["page"], task["crop"]
    )
    if colorized_path is None:
        return None
    with Image.open(colorized_path) as image:
        return image.convert("RGB")


def _check_one(
    checker: LineArtChecker,
    inputs_dir: Path,
    task: dict,
    max_edge: int,
) -> dict:
    """One panel's Luna check; returns the page-JSON record."""
    page, stem = task["page"], task["stem"]
    record: dict = {
        "panel": task["crop"],
        "box": task["box"],
        "provenance": task["provenance"],
    }
    if task["bw_fallback"]:
        record.update({
            "bw_fallback": True,
            "skipped": True,
            "note": "stitched from B&W crop; skipped (trivially identical)",
        })
        return record

    with Image.open(task["bw_path"]) as bw_image:
        bw_image = bw_image.convert("RGB")
    color_image = _load_color_image(task)
    if color_image is None:
        record.update({
            "bw_fallback": False,
            "skipped": True,
            "note": "no colorized output available for comparison",
        })
        return record

    bw_grid, color_grid, size = analysis_pair(bw_image, color_image, max_edge)
    page_inputs = inputs_dir / page
    page_inputs.mkdir(parents=True, exist_ok=True)
    bw_grid.save(page_inputs / f"{stem}.bw.png")
    color_grid.save(page_inputs / f"{stem}.colorized.png")

    result = checker.check(color_grid, bw_grid, max_edge=None)
    record.update({
        "bw_fallback": False,
        "skipped": False,
        "analysis_size": list(size),
        **result.to_dict(),
    })
    return record


def _contact_sheet(out_dir: Path, page: str, page_record: dict) -> Path | None:
    """Side-by-side B&W | colorized tiles for every mismatch panel of a page,
    stacked vertically, red border + verdict label; one PNG per page."""
    tiles: list[Image.Image] = []
    for panel in page_record["panels"]:
        if panel.get("skipped") or panel.get("status") != MISMATCH:
            continue
        stem = Path(panel["panel"]).stem
        bw_path = out_dir / "inputs" / page / f"{stem}.bw.png"
        color_path = out_dir / "inputs" / page / f"{stem}.colorized.png"
        if not bw_path.is_file() or not color_path.is_file():
            continue
        with Image.open(bw_path) as bw, Image.open(color_path) as color:
            bw_tile = bw.convert("RGB")
            color_tile = color.convert("RGB")
        gap = 12
        label_height = 30
        pair = Image.new(
            "RGB",
            (bw_tile.width * 2 + gap * 3, bw_tile.height + label_height),
            "white",
        )
        pair.paste(bw_tile, (gap, label_height))
        pair.paste(color_tile, (bw_tile.width + gap * 2, label_height))
        draw = ImageDraw.Draw(pair)
        draw.rectangle([0, 0, pair.width - 1, pair.height - 1],
                       outline=(200, 20, 20), width=4)
        label = (
            f"{stem}  line_art_matches=false  "
            f"{panel.get('analyse', '')}"
        )
        font = load_font(24)
        max_width = pair.width - 2 * gap
        while font.getlength(label) > max_width and len(label) > 8:
            label = label[:-1]
        draw.text((gap, 4), label, fill=(150, 0, 0), font=font)
        tiles.append(pair)

    if not tiles:
        return None
    sheet = Image.new("RGB", (max(t.width for t in tiles),
                              sum(t.height for t in tiles)), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    path = out_dir / f"{page}_mismatch.png"
    sheet.save(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="check_luna_sanity",
        description=(
            "Luna-based line-art check of a completed run: one paid "
            "OpenRouter VLM call per panel (openai/gpt-5.6-luna, strict "
            "structured output) asking whether the colorized line art "
            "matches the B&W original. Needs OPENROUTER_API_KEY in .env."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed pipeline run directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: <run-dir>/7_sanity_luna).")
    parser.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE,
                        help="Analysis-grid long-edge cap in px (default: "
                             f"{DEFAULT_MAX_EDGE}).")
    parser.add_argument("--model", type=str, default=DEFAULT_LUNA_SANITY_MODEL,
                        help="OpenRouter vision model (default: "
                             f"{DEFAULT_LUNA_SANITY_MODEL}).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel panels per run (default: 1).")
    parser.add_argument("--api-key-env", type=str, default=API_KEY_ENV,
                        help=f"Environment variable with the API key "
                             f"(default: {API_KEY_ENV}).")
    parser.add_argument("--page", action="append", default=[], metavar="SUBSTR",
                        help="Only check pages whose name contains SUBSTR "
                             "(repeatable).")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(
            f"missing {args.api_key_env} (load it into {REPO_ROOT / '.env'})"
        )

    run_dir = args.run_dir.resolve()
    out_dir = (args.output_dir or run_dir / "7_sanity_luna").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_panel_tasks(run_dir, tuple(args.page))
    if not tasks:
        raise SystemExit(f"no panels selected in {run_dir}")

    checker = LineArtChecker(model=args.model, api_key=api_key)
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(tqdm(
                pool.map(
                    lambda task: _check_one(
                        checker, inputs_dir, task, args.max_edge
                    ),
                    tasks,
                ),
                total=len(tasks), desc="luna_sanity: panels", unit="panel",
            ))
    else:
        records = [
            _check_one(checker, inputs_dir, task, args.max_edge)
            for task in tqdm(tasks, desc="luna_sanity: panels", unit="panel")
        ]

    checked = 0
    mismatched = 0
    errored = 0
    total_cost = 0.0
    cost_source = "usage.cost"
    pages: dict[str, list[dict]] = {}
    for task, record in zip(tasks, records):
        pages.setdefault(task["page"], []).append(record)
        if record.get("skipped"):
            continue
        checked += 1
        status = record.get("status")
        if status == MISMATCH:
            mismatched += 1
        elif status != MATCHES:
            errored += 1
        if record.get("cost_source") == "usage.cost" and record.get("cost_usd"):
            total_cost += record["cost_usd"]
        if status == MISMATCH or status not in (MATCHES,):
            print(
                f"  {'MISMATCH' if status == MISMATCH else status.upper()} "
                f"{task['page']}/{record['panel']}: "
                f"{record.get('analyse') or record.get('error') or ''}",
                flush=True,
            )

    page_records = []
    for page in sorted(pages):
        page_record = {
            "page": page,
            "panels": pages[page],
            "mismatch_panels": [
                p["panel"] for p in pages[page] if p.get("status") == MISMATCH
            ],
            "contact_sheet": None,
        }
        sheet = _contact_sheet(out_dir, page, page_record)
        if sheet is not None:
            page_record["contact_sheet"] = str(sheet)
        write_json(out_dir / f"{page}.json", page_record)
        page_records.append(page_record)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "model": args.model,
        "max_edge": args.max_edge,
        "pages_checked": len(page_records),
        "panels_checked": checked,
        "panels_matches": checked - mismatched - errored,
        "panels_mismatch": mismatched,
        "panels_error_or_unparseable": errored,
        "total_cost_usd": round(total_cost, 6),
        "total_cost_source": cost_source,
        "pages": [
            {
                "page": r["page"],
                "mismatch_panels": r["mismatch_panels"],
                "contact_sheet": r["contact_sheet"],
            }
            for r in page_records
        ],
    }
    write_json(out_dir / "summary.json", summary)
    print(
        f"checked {checked} panels; {mismatched} mismatch, "
        f"{checked - mismatched - errored} match, {errored} error/unparseable "
        f"| cost ${total_cost:.6f}",
        flush=True,
    )
    print(f"summary: {out_dir / 'summary.json'}", flush=True)
    return 0


def write_json(path: Path, value: dict) -> None:
    """Atomic JSON write (same as run_context.write_json)."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - friendly failure output
        print(f"check_luna_sanity failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        raise SystemExit(1)
