#!/usr/bin/env python3
"""PDF export of a completed pipeline run's stitched pages.

Reads a pipeline run directory and packs `4_stitched/` into a single
multi-page PDF (one page per manga page, filename order = reading order)
using Pillow's native PDF writer — no extra dependency.

This is the standalone, offline companion of the pipeline's `pdf` stage
(step 6, `6_pdf/`): the stage runs automatically at the end of every
pipeline run; this script re-exports any *completed* run with custom
options (page filter, output name, DPI, output dir) without re-running
the pipeline. Both share `steps.pdf.run_pdf_step`.

Input (a completed run, e.g. pipeline_v1/output/YYYYMMDD-HHMMSS/):
  4_stitched/<page>.png            final pages (filename order = reading order)

Output (default <run-dir>/6_pdf/):
  colorized.pdf                    the multi-page PDF (or --name)
  summary.json                     per-page provenance records

Usage:
  .venv/bin/python pipeline_v1/scripts/make_pdf.py \
      --run-dir pipeline_v1/output/20260815-124816
  .venv/bin/python pipeline_v1/scripts/make_pdf.py \
      --run-dir pipeline_v1/output/20260815-124816 --name volume-1.pdf --dpi 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the pipeline package importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="make_pdf",
        description=(
            "Multi-page PDF of a run's stitched pages (same export as the "
            "pipeline's pdf stage, 6_pdf/)."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed pipeline run directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: <run-dir>/6_pdf).")
    parser.add_argument("--page", action="append", default=[], metavar="SUBSTR",
                        help="Only export pages whose name contains SUBSTR "
                             "(repeatable).")
    parser.add_argument("--name", default=None,
                        help="Output PDF filename (default: the run's "
                             "config, i.e. colorized.pdf).")
    parser.add_argument("--dpi", type=int, default=None,
                        help="PDF embedding resolution (default: the run's "
                             "config, i.e. 72; page size in points = "
                             "pixel size * 72 / dpi).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    stitched_dir = run_dir / "4_stitched"
    if not stitched_dir.is_dir():
        raise SystemExit(f"no 4_stitched/ dir in run: {run_dir}")

    from config import PipelineConfig
    from run_context import RunContext
    from steps.pdf import run_pdf_step

    config = PipelineConfig(
        pdf_name=args.name if args.name is not None else "colorized.pdf",
        pdf_dpi=args.dpi if args.dpi is not None else 72,
    )
    ctx = RunContext.load(run_dir)
    try:
        record = run_pdf_step(
            ctx,
            config,
            output_dir=args.output_dir,
            page_substrings=tuple(args.page),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out_dir = Path(record["output_dir"])
    print(
        f"wrote {record['pages_in_pdf']} pages to {out_dir / config.pdf_name}",
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
        print(f"make_pdf failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
