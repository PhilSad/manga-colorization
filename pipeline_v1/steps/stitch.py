"""Pipeline stage 5: stitch colorized panels back onto the original pages.

Reads the geometry from `1_panels/<page>/panels.json`, the colorized panels
from `3_colorized/<page>/`, and writes the final pages to `4_stitched/`.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from config import PipelineConfig
from detection import PanelBox
from run_context import RunContext, write_json
from selection import page_selected
from stitching import stitch_page
from util import SUPPORTED_IMAGE_SUFFIXES, file_record


def run_stitch_step(
    ctx: RunContext,
    config: PipelineConfig,
    colorized_ext: str | None = None,
) -> dict:
    """Run stage 5. `colorized_ext` overrides the panel file extension lookup
    (defaults to the config's output format)."""
    panels_root = ctx.step_dir("panels")
    colorized_root = ctx.step_dir("colorize")
    stitched_dir = ctx.step_dir("stitch")
    extension = colorized_ext or _extension(config)

    page_dirs = sorted(path for path in panels_root.iterdir() if path.is_dir())
    if not page_dirs:
        raise ValueError("no panels to stitch; run the 'panels' step first")

    outputs: list[dict] = []
    for page_dir in page_dirs:
        page = page_dir.name
        if config.only_panels and not page_selected(page, config.only_panels):
            continue
        geometry_path = page_dir / "panels.json"
        if not geometry_path.is_file():
            raise ValueError(f"missing {geometry_path}; run the 'panels' step first")
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))

        colorized_page_dir = colorized_root / page
        if not colorized_page_dir.is_dir():
            raise ValueError(
                f"no colorized panels for {page}; run the 'colorize' step first"
            )

        source_page = Path(geometry["page_path"])
        with Image.open(source_page) as page_image:
            page_image = page_image.convert("RGB")
            pairs: list[tuple[PanelBox, Image.Image]] = []
            for detection in geometry["detections"]:
                crop_name = detection["crop"]
                colorized_path = _find_colorized(colorized_page_dir, crop_name, extension)
                if colorized_path is None:
                    raise ValueError(f"missing colorized panel for {crop_name}")
                with Image.open(colorized_path) as colorized_image:
                    pairs.append(
                        (
                            PanelBox(*detection["box"], detection["confidence"]),
                            colorized_image.convert("RGB"),
                        )
                    )
            stitched = stitch_page(page_image, pairs, inset=config.panel_inset)
            output_path = stitched_dir / f"{page}{extension}"
            stitched.save(output_path)

        outputs.append({
            "page": page,
            "input": geometry["page"],
            "output": file_record(output_path),
            "panels_stitched": len(pairs),
        })
    return {"outputs": outputs}


def _find_colorized(page_dir: Path, crop_name: str, extension: str) -> Path | None:
    """Colorized file for a crop: same stem, any supported extension, then the
    exact extension."""
    stem = Path(crop_name).stem
    candidates = [
        page_dir / f"{stem}{extension}",
        *(page_dir / f"{stem}{path.suffix}"
          for path in page_dir.iterdir()
          if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
    ]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _extension(config: PipelineConfig) -> str:
    return {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[config.output_format]
