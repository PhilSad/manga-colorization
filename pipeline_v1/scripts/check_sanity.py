#!/usr/bin/env python3
"""Line-art fidelity sanity check of a completed pipeline run.

Reads a pipeline run directory and writes `7_sanity/` with per-panel
line-fidelity metrics (structural line maps: IoU / chamfer / large
components / drift, see sanity.py) and flags the panels whose colorized
line art drifted below the threshold for review. Pure local compute
(numpy + OpenCV, no backends, no network).

This is the standalone, offline companion of the pipeline's `sanity` stage
(step 7, `7_sanity/`): the stage runs automatically at the end of every
pipeline run with the same scoring; this script re-checks any *completed*
run with custom options (threshold, page filter, output dir) without
re-running the pipeline. Both share `steps.sanity.run_sanity_step`.

Input layout (a completed run, e.g. pipeline_v1/output/YYYYMMDD-HHMMSS/):
  1_panels/<page>/panel_000N.png     black & white panel crops
  1_panels/<page>/panels.json        panel boxes
  4_stitched/<page>.png              final pages (colorized panels at box positions)
  3_colorized/<page>/                fallback source for the colorized panels
  manifest.json                      B&W-fallback panels are skipped

Output (default <run-dir>/7_sanity/):
  <page>.json                        per-panel metrics + verdicts
  <page>_flagged.png                 contact sheet of the flagged panels
  summary.json                       run totals + flagged list

Usage:
  .venv/bin/python pipeline_v1/scripts/check_sanity.py \
      --run-dir pipeline_v1/output/20260819-202719
  .venv/bin/python pipeline_v1/scripts/check_sanity.py \
      --run-dir pipeline_v1/output/20260819-202719 --threshold 0.5 --page p006
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the pipeline package importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="check_sanity",
        description=(
            "Line-art fidelity sanity check of a completed run: compares "
            "each colorized panel with its B&W original and flags panels "
            "whose line art drifted below the threshold (same scoring as "
            "the pipeline's sanity stage, 7_sanity/)."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed pipeline run directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: <run-dir>/7_sanity).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Line-fidelity flag threshold (default: the "
                             "run's config, i.e. 0.45).")
    parser.add_argument("--max-edge", type=int, default=None,
                        help="Analysis grid long-edge cap in px (default: "
                             "the run's config, i.e. 1024).")
    parser.add_argument("--page", action="append", default=[], metavar="SUBSTR",
                        help="Only check pages whose name contains SUBSTR "
                             "(repeatable).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    panels_dir = run_dir / "1_panels"
    if not panels_dir.is_dir():
        raise SystemExit(f"no 1_panels/ dir in run: {run_dir}")

    from config import PipelineConfig
    from run_context import RunContext
    from steps.sanity import run_sanity_step

    config = PipelineConfig()
    ctx = RunContext.load(run_dir)
    # Reuse the run's own sanity settings unless overridden on the CLI.
    stored = ctx.manifest.get("configuration", {})
    config.sanity_threshold = stored.get("sanity_threshold",
                                         config.sanity_threshold)
    config.sanity_max_edge = stored.get("sanity_max_edge",
                                        config.sanity_max_edge)
    try:
        record = run_sanity_step(
            ctx,
            config,
            threshold=args.threshold,
            max_edge=args.max_edge,
            page_substrings=tuple(args.page),
            output_dir=args.output_dir,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out_dir = Path(record["output_dir"])
    print(
        f"checked {record['pages_checked']} pages / {record['panels_checked']} "
        f"panels; flagged {record['panels_flagged']} for review "
        f"(threshold {record['threshold']:.3f})",
        flush=True,
    )
    print(f"summary: {out_dir / 'summary.json'}", flush=True)
    if record["panels_flagged"]:
        for entry in record["outputs"]:
            for panel in entry["panels"]:
                if panel.get("flagged"):
                    print(
                        f"  FLAG {entry['page']}/{panel['panel']}: "
                        f"fidelity={panel.get('line_fidelity')} "
                        f"iou={panel.get('line_iou')} "
                        f"chamfer={panel.get('chamfer_px')}px "
                        f"drift={panel.get('drift_px')}px | "
                        f"{'; '.join(panel.get('reasons') or [panel.get('note')])}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - friendly failure output
        print(f"check_sanity failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
