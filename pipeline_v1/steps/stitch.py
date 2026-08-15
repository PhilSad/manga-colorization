"""Pipeline stage 5: stitch colorized panels back onto the original pages.

Reads the geometry from `1_panels/<page>/panels.json`, the colorized panels
from `3_colorized/<page>/`, and writes the final pages to `4_stitched/`.

With `--stitch-bw-fallback`, a panel whose colorized output is missing (e.g. a
FLUX call that errored) is stitched from its original black & white crop
instead of failing the whole step; every such fallback is logged (stderr) and
recorded per page (`panels_bw_fallback`) and in the step totals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from config import PipelineConfig
from detection import PanelBox
from run_context import RunContext
from selection import page_selected, panel_selected
from stitching import stitch_page
from tqdm import tqdm
from util import SUPPORTED_IMAGE_SUFFIXES, file_record


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def run_stitch_step(
    ctx: RunContext,
    config: PipelineConfig,
    colorized_ext: str | None = None,
) -> dict:
    """Run stage 5. `colorized_ext` overrides the panel file extension lookup
    (defaults to the config's output format).

    Missing colorized panels: with `config.stitch_bw_fallback` the original
    crop is pasted instead (B&W, logged + recorded); without it a ValueError is
    raised, except for un-selected panels of a targeted rerun (`--only-panel`),
    which always stay B&W silently.
    """
    panels_root = ctx.step_dir("panels")
    colorized_root = ctx.step_dir("colorize")
    stitched_dir = ctx.step_dir("stitch")
    extension = colorized_ext or _extension(config)

    page_dirs = sorted(path for path in panels_root.iterdir() if path.is_dir())
    if not page_dirs:
        raise ValueError("no panels to stitch; run the 'panels' step first")

    outputs: list[dict] = []
    totals_bw_fallback = 0
    for page_dir in tqdm(
        page_dirs, desc="stitch: pages", unit="page", leave=False
    ):
        page = page_dir.name
        if config.only_panels and not page_selected(page, config.only_panels):
            continue
        geometry_path = page_dir / "panels.json"
        if not geometry_path.is_file():
            raise ValueError(f"missing {geometry_path}; run the 'panels' step first")
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))

        colorized_page_dir = colorized_root / page
        if not colorized_page_dir.is_dir():
            if not config.stitch_bw_fallback:
                raise ValueError(
                    f"no colorized panels for {page}; run the 'colorize' step first"
                )
            _warn(
                f"no colorized panels for {page}; stitching all panels from "
                "their original black & white crops (--stitch-bw-fallback)"
            )

        if config.full_page:
            # Full-page passthrough: the single colorized panel already covers
            # the whole page, so copy it straight to the stitched output (no
            # inset, no re-paste). Missing colorized output falls back to the
            # original page with --stitch-bw-fallback.
            bw_fallback: list[str] = []
            crop_name = geometry["detections"][0]["crop"] \
                if geometry["detections"] else "panel_0001"
            colorized_path = _find_colorized(colorized_page_dir, crop_name, extension)
            if colorized_path is None:
                if not config.stitch_bw_fallback:
                    raise ValueError(f"missing colorized panel for {crop_name}")
                colorized_path = page_dir / crop_name
                if not colorized_path.is_file():
                    raise ValueError(
                        f"missing colorized panel for {crop_name} (and no "
                        f"original crop at {colorized_path} to fall back to)"
                    )
                bw_fallback = [crop_name]
                _warn(
                    f"missing colorized panel for {page}/{crop_name}; copying "
                    "the original black & white page (--stitch-bw-fallback)"
                )
            output_path = stitched_dir / f"{page}{extension}"
            with Image.open(colorized_path) as colorized_image:
                colorized_image.convert("RGB").save(output_path)
            totals_bw_fallback += len(bw_fallback)
            outputs.append({
                "page": page,
                "input": geometry["page"],
                "output": file_record(output_path),
                "panels_stitched": 0 if bw_fallback else 1,
                "panels_skipped_black_white": [],
                "panels_bw_fallback": bw_fallback,
            })
            continue

        source_page = Path(geometry["page_path"])
        with Image.open(source_page) as page_image:
            page_image = page_image.convert("RGB")
            pairs: list[tuple[PanelBox, Image.Image]] = []
            skipped: list[str] = []
            bw_fallback: list[str] = []
            for detection in geometry["detections"]:
                crop_name = detection["crop"]
                colorized_path = _find_colorized(colorized_page_dir, crop_name, extension)
                if colorized_path is not None:
                    with Image.open(colorized_path) as colorized_image:
                        pairs.append(
                            (
                                PanelBox(*detection["box"], detection["confidence"]),
                                colorized_image.convert("RGB"),
                            )
                        )
                    continue
                # Targeted rerun: un-selected panels of the page stay B&W
                # (only when --only-panel was used); otherwise this is a
                # missing colorize output.
                if config.only_panels and not panel_selected(
                    page, Path(crop_name).stem, config.only_panels
                ):
                    skipped.append(crop_name)
                    continue
                if not config.stitch_bw_fallback:
                    raise ValueError(f"missing colorized panel for {crop_name}")
                original_path = page_dir / crop_name
                if not original_path.is_file():
                    raise ValueError(
                        f"missing colorized panel for {crop_name} "
                        f"(and no original crop at {original_path} to fall back to)"
                    )
                _warn(
                    f"missing colorized panel for {page}/{crop_name}; "
                    f"stitching the original black & white crop "
                    f"(--stitch-bw-fallback)"
                )
                with Image.open(original_path) as original_image:
                    pairs.append(
                        (
                            PanelBox(*detection["box"], detection["confidence"]),
                            original_image.convert("RGB"),
                        )
                    )
                bw_fallback.append(crop_name)
            stitched = stitch_page(page_image, pairs, inset=config.panel_inset)
            output_path = stitched_dir / f"{page}{extension}"
            stitched.save(output_path)

        totals_bw_fallback += len(bw_fallback)
        outputs.append({
            "page": page,
            "input": geometry["page"],
            "output": file_record(output_path),
            "panels_stitched": len(pairs),
            "panels_skipped_black_white": skipped,
            "panels_bw_fallback": bw_fallback,
        })
    return {"outputs": outputs, "panels_bw_fallback": totals_bw_fallback}


def _find_colorized(page_dir: Path, crop_name: str, extension: str) -> Path | None:
    """Colorized file for a crop: same stem, any supported extension, then the
    exact extension. Returns None when the page has no colorized dir at all."""
    if not page_dir.is_dir():
        return None
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
