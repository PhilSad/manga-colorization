#!/usr/bin/env python3
"""Debug annotation of a completed pipeline run's stitched pages.

Reads a pipeline run directory and writes a debug copy of `4_stitched/` with,
per panel: a bounding box and a label with the panel name + the characters
detected for it (from `2_characters/<page>/<panel>.json`).

This is the standalone, offline companion of the pipeline's `debug` stage
(step 5, `5_debug/`): the stage runs automatically at the end of every
pipeline run with the same rendering; this script re-annotates any *completed*
run with custom options (page filter, font size, bbox width, output dir)
without re-running the pipeline. Both share `steps.debug.run_debug_step`.

Input layout (a completed run, e.g. pipeline_v1/output/YYYYMMDD-HHMMSS/):
  4_stitched/<page>.png            final pages
  1_panels/<page>/panels.json      panel geometry (boxes + crop names)
  2_characters/<page>/<panel>.json detected characters per panel
  manifest.json                    optional: B&W fallback panels (steps.stitch)

Output (default <run-dir>/5_debug/):
  <page>.png                       same page + bbox + character label per panel
  summary.json                     per-page provenance records

Panels that were stitched from their original black & white crop
(--stitch-bw-fallback) get an orange bbox and a "[B&W fallback]" tag.

Usage:
  .venv/bin/python pipeline_v1/scripts/annotate_stitch.py \
      --run-dir pipeline_v1/output/20260809-125148
  .venv/bin/python pipeline_v1/scripts/annotate_stitch.py \
      --run-dir pipeline_v1/output/20260809-125148 --page p077 --font-size 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the pipeline package importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="annotate_stitch",
        description=(
            "Debug copy of a run's stitched pages: panel bboxes + detected "
            "characters per panel (same rendering as the pipeline's debug "
            "stage, 5_debug/)."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed pipeline run directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: <run-dir>/5_debug).")
    parser.add_argument("--page", action="append", default=[], metavar="SUBSTR",
                        help="Only annotate pages whose name contains SUBSTR "
                             "(repeatable).")
    parser.add_argument("--font-size", type=int, default=None,
                        help="Label font size in px (default: the run's "
                             "config, i.e. 42).")
    parser.add_argument("--bbox-width", type=int, default=None,
                        help="Bounding-box stroke width in px (default: the "
                             "run's config, i.e. 5).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    stitched_dir = run_dir / "4_stitched"
    if not stitched_dir.is_dir():
        raise SystemExit(f"no 4_stitched/ dir in run: {run_dir}")

    from config import PipelineConfig
    from run_context import RunContext
    from steps.debug import run_debug_step

    config = PipelineConfig()
    ctx = RunContext.load(run_dir)
    try:
        record = run_debug_step(
            ctx,
            config,
            output_dir=args.output_dir,
            page_substrings=tuple(args.page),
            font_size=args.font_size,
            bbox_width=args.bbox_width,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out_dir = Path(record["output_dir"])
    print(
        f"wrote {record['pages_annotated']} annotated pages to {out_dir}",
        flush=True,
    )
    print(f"summary: {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - friendly failure output
        print(f"annotate_stitch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
