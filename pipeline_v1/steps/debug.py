"""Pipeline stage 5: debug annotation of the colorized (stitched) pages.

For each stitched page in `4_stitched/`, renders a debug copy with, per
panel: the detected bounding box (from `1_panels/<page>/panels.json`) and a
label with the panel name + the characters detected for it (from
`2_characters/<page>/<panel>.json`). Writes `5_debug/<page>.<ext>` +
`5_debug/summary.json` with per-page provenance records.

Panels that were stitched from their original black & white crop
(--stitch-bw-fallback) get an orange bbox and a "[B&W fallback]" tag; the
fallback list comes from the stitch step record in the run manifest (current
run, or the --resume run's manifest when resuming from the stitch stage).

The standalone offline tool `scripts/annotate_stitch.py` delegates to
`run_debug_step` so a completed run can be re-annotated with custom options
(--page filter, --font-size, --bbox-width, --output-dir) without re-running
the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from config import STEP_DIRS, PipelineConfig
from run_context import RunContext, write_json
from selection import page_selected
from tqdm import tqdm
from util import SUPPORTED_IMAGE_SUFFIXES, load_font

NORMAL_COLOR = (220, 30, 30)      # red bbox + badge outline
FALLBACK_COLOR = (230, 110, 0)    # orange: stitched from the B&W crop
TEXT_COLOR = (150, 0, 0)
FALLBACK_TEXT_COLOR = (180, 80, 0)
BADGE_PAD = 8
BADGE_OFFSET = 6


def label_text(crop_name: str, record: dict) -> str:
    stem = Path(crop_name).stem
    names = record.get("characters", [])
    if names:
        return f"{stem}: {', '.join(names)}"
    if record.get("uncertain"):
        return f"{stem}: (uncertain)"
    if record:
        return f"{stem}: (no characters)"
    return f"{stem}: (no record)"


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> list[str]:
    """Word-wrap `text` to `max_width` px."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    text: str,
    font,
    outline_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    """White badge with the panel label, anchored at the box's top-left."""
    x1, y1, x2, y2 = (round(value) for value in box)
    box_w = max(120, x2 - x1 - 2 * BADGE_OFFSET)
    lines = wrap_lines(draw, text, font, box_w - 2 * BADGE_PAD)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 4
    widths = [draw.textlength(line, font=font) for line in lines]
    label_w = max(widths) + 2 * BADGE_PAD
    label_h = line_h * len(lines) + 2 * BADGE_PAD
    bx1 = x1 + BADGE_OFFSET
    by1 = y1 + BADGE_OFFSET
    draw.rectangle(
        (bx1, by1, bx1 + label_w, by1 + label_h),
        fill="white", outline=outline_color, width=3,
    )
    for index, line in enumerate(lines):
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        draw.text(
            (bx1 + BADGE_PAD, by1 + BADGE_PAD + index * line_h - top),
            line, fill=text_color, font=font,
        )


def load_character_record(run_dir: Path, page: str, crop_stem: str) -> dict:
    """Character record for one panel ({} when missing/unreadable)."""
    path = run_dir / "2_characters" / page / f"{crop_stem}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_fallback_map(ctx: RunContext, config: PipelineConfig) -> dict[str, set[str]]:
    """page stem -> set of panel stems stitched from the B&W crop.

    Sources: the current run's manifest (stitch runs right before debug in
    the same run, so its record is already in ctx.manifest / manifest.json),
    and the --resume run's manifest (--from-step debug --resume RUN: the
    fresh run has no stitch record, the resumed one does).
    """
    result: dict[str, set[str]] = {}

    def _collect(manifest: dict | None) -> None:
        if not manifest:
            return
        for record in manifest.get("steps", {}).get("stitch", {}).get("outputs", []):
            fallback = record.get("panels_bw_fallback") or []
            result.setdefault(record["page"], set()).update(
                Path(crop).stem for crop in fallback
            )

    _collect(ctx.manifest)
    if config.resume:
        manifest_path = Path(config.resume) / "manifest.json"
        try:
            _collect(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return result


def run_debug_step(
    ctx: RunContext,
    config: PipelineConfig,
    *,
    output_dir: Path | None = None,
    page_substrings: tuple[str, ...] = (),
    font_size: int | None = None,
    bbox_width: int | None = None,
) -> dict:
    """Run stage 5: annotate every stitched page into `5_debug/`.

    Pure image processing (no backends, no network). Optional overrides let
    scripts/annotate_stitch.py re-annotate a completed run with a page filter
    and custom rendering options. Returns the per-page records for the
    manifest (also written as `5_debug/summary.json`).
    """
    stitched_dir = ctx.run_dir / STEP_DIRS["stitch"]
    if not stitched_dir.is_dir():
        raise ValueError(
            "no stitched pages to annotate; run the 'stitch' step first"
        )
    pages = sorted(
        path for path in stitched_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if config.only_panels:
        pages = [p for p in pages if page_selected(p.stem, config.only_panels)]
    if page_substrings:
        pages = [p for p in pages if any(s in p.stem for s in page_substrings)]
    if not pages:
        raise ValueError(f"no stitched pages to annotate in {stitched_dir}")

    out_dir = output_dir or ctx.step_dir("debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(font_size if font_size is not None else config.debug_font_size)
    stroke = bbox_width if bbox_width is not None else config.debug_bbox_width
    fallback_map = load_fallback_map(ctx, config)

    records: list[dict] = []
    for page_path in tqdm(
        pages, desc="debug: annotate", unit="page", leave=False
    ):
        page = page_path.stem
        geometry_path = ctx.run_dir / STEP_DIRS["panels"] / page / "panels.json"
        detections: list[dict] = []
        if geometry_path.is_file():
            try:
                detections = json.loads(
                    geometry_path.read_text(encoding="utf-8")
                )["detections"]
            except (OSError, ValueError, KeyError):
                detections = []

        with Image.open(page_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            panels: list[dict] = []
            for detection in detections:
                crop = detection["crop"]
                stem = Path(crop).stem
                record = load_character_record(ctx.run_dir, page, stem)
                is_fallback = stem in fallback_map.get(page, set())
                color = FALLBACK_COLOR if is_fallback else NORMAL_COLOR
                text_color = FALLBACK_TEXT_COLOR if is_fallback else TEXT_COLOR
                draw.rectangle(
                    tuple(round(value) for value in detection["box"]),
                    outline=color, width=stroke,
                )
                text = label_text(crop, record)
                if is_fallback:
                    text += " [B&W fallback]"
                draw_label(draw, detection["box"], text, font, color, text_color)
                panels.append({
                    "panel": crop,
                    "characters": record.get("characters", []),
                    "bw_fallback": is_fallback,
                })
            output_path = out_dir / page_path.name
            image.save(output_path)

        records.append({
            "page": page,
            "output": str(output_path.resolve()),
            "panels": panels,
        })

    summary = {
        "run_dir": str(ctx.run_dir),
        "output_dir": str(out_dir),
        "pages_annotated": len(records),
        "records": records,
    }
    write_json(out_dir / "summary.json", summary)
    return {
        "outputs": records,
        "pages_annotated": len(records),
        "output_dir": str(out_dir),
    }
