"""Pipeline stage 4: per-panel colorization with a filtered atlas.

For each extracted panel: load the characters detected for it (stage 3), build
a labelled atlas containing only those characters, and call the colorizer with
the panel (+ atlas). Outputs go to `3_colorized/<page>/`; the atlas used per
panel is saved next to the output for provenance.
"""

from __future__ import annotations

import time
from pathlib import Path

from config import PipelineConfig
from run_context import RunContext, write_json
from util import SUPPORTED_IMAGE_SUFFIXES


def _panel_images(page_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in page_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and path.stem != "overlay"
        and "_atlas" not in path.stem
    )


def run_colorize_step(
    ctx: RunContext,
    config: PipelineConfig,
    colorizer,  # Colorizer
) -> dict:
    """Run stage 4 for all panels of all pages. Returns per-call records."""
    from atlas import build_filtered_atlas
    from steps.characters import load_characters_per_panel

    panels_root = ctx.step_dir("panels")
    colorized_root = ctx.step_dir("colorize")
    page_dirs = sorted(path for path in panels_root.iterdir() if path.is_dir())
    if not page_dirs:
        raise ValueError("no extracted panels; run the 'panels' step first")

    extension = _extension(config)
    records: list[dict] = []
    totals = {
        "api_calls": 0,
        "successful_calls": 0,
        "error_calls": 0,
        "total_latency_s": 0.0,
    }
    for page_dir in page_dirs:
        page = page_dir.name
        try:
            names_by_panel = load_characters_per_panel(ctx, page)
        except Exception:  # noqa: BLE001 - stage 3 not run: colorize without atlas
            names_by_panel = {}
            print(
                f"  colorize: no character records for {page}, "
                "using panel-only colorization",
                flush=True,
            )
        out_dir = colorized_root / page
        out_dir.mkdir(parents=True, exist_ok=True)
        for panel_path in _panel_images(page_dir):
            stem = panel_path.stem
            characters = names_by_panel.get(stem, [])
            atlas_path = out_dir / f"{stem}_atlas.jpg"
            atlas = build_filtered_atlas(
                characters, config.refs_dir, atlas_path,
                columns=config.atlas_columns,
            )
            output_path = out_dir / f"{stem}{extension}"
            print(
                f"  colorize: {page}/{panel_path.name} characters={characters} "
                f"atlas={'yes' if atlas else 'no'} ...",
                flush=True,
            )
            record = colorizer.colorize(panel_path, atlas, output_path)
            doc = record.to_dict(panel_path, atlas)
            doc["characters"] = characters
            records.append(doc)
            totals["api_calls"] += 1
            if record.status == "ok":
                totals["successful_calls"] += 1
            else:
                totals["error_calls"] += 1
            totals["total_latency_s"] = round(
                totals["total_latency_s"] + record.latency_s, 3
            )

    write_json(colorized_root / "summary.json", {"records": records, "totals": totals})
    return {"records": records, "totals": totals}


def _extension(config: PipelineConfig) -> str:
    return {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[config.output_format]
