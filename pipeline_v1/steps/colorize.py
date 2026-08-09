"""Pipeline stage 4: per-panel colorization with a filtered atlas and explicit
canonical palettes.

For each extracted panel: load the characters detected for it (stage 3), build
a labelled atlas containing only those characters, render the explicit
canonical-palette instruction from the shared character profiles (task 0002),
and call the colorizer with the panel (+ atlas). Outputs go to
`3_colorized/<page>/`; the atlas used per panel is saved next to the output for
provenance.

`--only-panel` (task 0001) restricts which panels are colorized; when resuming,
the non-selected panels' colorized outputs are copied from the resume run so
the page can still be stitched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from config import STEP_DIRS, PipelineConfig
from run_context import RunContext, write_json
from selection import page_selected, panel_selected
from tqdm import tqdm
from util import SUPPORTED_IMAGE_SUFFIXES


def _panel_images(page_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in page_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and path.stem != "overlay"
        and path.stem != "detection_annotated"
        and "_atlas" not in path.stem
    )


def run_colorize_step(
    ctx: RunContext,
    config: PipelineConfig,
    colorizer,  # Colorizer
) -> dict:
    """Run stage 4 for all panels of all pages. Returns per-call records."""
    from atlas import build_filtered_atlas
    from profiles import load_profiles, palette_instruction, profiles_sha256, unknown_names
    from steps.characters import load_characters_per_panel

    profiles = {}
    try:
        profiles = load_profiles(config.profiles_file)
    except (OSError, ValueError):
        profiles = {}
    profiles_sha = profiles_sha256(config.profiles_file)

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
    for page_dir in tqdm(
        page_dirs, desc="colorize: pages", unit="page", leave=False
    ):
        page = page_dir.name
        if config.only_panels and not page_selected(page, config.only_panels):
            continue
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

        all_panels = _panel_images(page_dir)
        if config.only_panels:
            panels = [
                p for p in all_panels
                if panel_selected(page, p.stem, config.only_panels)
            ]
        else:
            panels = all_panels

        fresh_stems: set[str] = set()
        panels_bar = tqdm(
            panels, desc=f"colorize: {page}", unit="panel", leave=False
        )
        for panel_path in panels_bar:
            stem = panel_path.stem
            characters = names_by_panel.get(stem, [])
            atlas_path = out_dir / f"{stem}_atlas.jpg"
            atlas = build_filtered_atlas(
                characters, config.refs_dir, atlas_path,
                columns=config.atlas_columns,
            )
            output_path = out_dir / f"{stem}{extension}"
            palette = palette_instruction(characters, profiles)
            panels_bar.set_postfix(
                panel=panel_path.name,
                characters=",".join(characters) or "-",
                atlas="yes" if atlas else "no",
                palette="yes" if palette else "no",
            )
            record = colorizer.colorize(
                panel_path, atlas, output_path, palette_instruction=palette
            )
            doc = record.to_dict(panel_path, atlas)
            doc["characters"] = characters
            doc["palette_instruction"] = palette
            doc["unknown_characters"] = unknown_names(characters, profiles)
            doc["profiles_sha256"] = profiles_sha
            doc["page"] = page
            records.append(doc)
            fresh_stems.add(stem)
            totals["api_calls"] += 1
            if record.status == "ok":
                totals["successful_calls"] += 1
            else:
                totals["error_calls"] += 1
            totals["total_latency_s"] = round(
                totals["total_latency_s"] + record.latency_s, 3
            )

        _copy_resumed_panels(ctx, config, page, fresh_stems)

    write_json(colorized_root / "summary.json", {"records": records, "totals": totals})
    return {"records": records, "totals": totals}


def _copy_resumed_panels(
    ctx: RunContext, config: PipelineConfig, page: str, fresh_stems: set[str]
) -> None:
    """Copy colorized panels for non-selected panels from the resume run so a
    targeted rerun can still stitch the full page (task 0001)."""
    if not config.resume:
        return
    source = Path(config.resume) / STEP_DIRS["colorize"] / page
    if not source.is_dir():
        return
    target = ctx.step_dir("colorize") / page
    for path in sorted(source.iterdir()):
        if path.name == "summary.json" or path.stem in fresh_stems:
            continue
        if path.is_dir():
            continue
        shutil.copy2(path, target / path.name)
    print(f"  colorize: reused {page} panels from resume run", flush=True)


def _extension(config: PipelineConfig) -> str:
    return {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[config.output_format]
