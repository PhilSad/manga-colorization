"""Reference atlas builder filtered to detected characters.

Port of `build_reference_atlas` from
`research/colorization_methods/flux-2-klein-9b-base-lora-edit-sequential/run.py`
(360x480 labelled cells, DejaVu bold labels, JPEG quality 94), parameterized to
a subset of the reference images so each panel gets an atlas containing only the
characters detected in it.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from util import SUPPORTED_IMAGE_SUFFIXES, load_font

CELL_WIDTH = 360
CELL_HEIGHT = 480
LABEL_HEIGHT = 36
IMAGE_MARGIN = 10

# Suffix preferences when a canonical name maps to several files.
_SUFFIX_PREFERENCE = ("_reference", "_anime_profile", "")


def reference_label(path: Path) -> str:
    """Canonical display name for a reference file (shared with characters.py;
    kept here to avoid a dependency loop)."""
    name = path.stem
    for suffix in ("_reference", "_anime_profile"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip().title()


def _canonical_key(name: str) -> str:
    return name.replace(" ", "").replace("_", "").strip().lower()


def refs_for_characters(characters: list[str], refs_dir: Path) -> list[Path]:
    """Map canonical character names back to their reference files.

    - A character is matched case/space-insensitively against the reference
      labels (`Frieren` matches `frieren_reference.webp`).
    - When several files match, the first by suffix preference
      (`_reference` > `_anime_profile` > plain), then filename order, wins.
    - Names without a matching file are reported (printed) and skipped.
    - Result order follows the input character list; duplicates are dropped.
    """
    by_key: dict[str, list[Path]] = {}
    for path in sorted(refs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        label = reference_label(path)
        by_key.setdefault(_canonical_key(label), []).append(path)

    result: list[Path] = []
    seen: set[str] = set()
    for name in characters:
        key = _canonical_key(name)
        candidates = by_key.get(key, [])
        if not candidates:
            print(f"  atlas: no reference image for character {name!r}, skipped")
            continue
        candidates.sort(key=lambda p: _suffix_rank(p))
        chosen = candidates[0]
        if chosen not in seen:
            seen.add(chosen)
            result.append(chosen)
    return result


def _suffix_rank(path: Path) -> int:
    lowered = path.stem.lower()
    for rank, suffix in enumerate(_SUFFIX_PREFERENCE):
        if suffix and lowered.endswith(suffix):
            return rank
    return len(_SUFFIX_PREFERENCE) - 1


def build_labelled_atlas(
    refs: list[Path],
    destination: Path,
    columns: int | None = None,
) -> Path:
    """Build a labelled atlas grid and save it as JPEG.

    `columns` defaults to ceil(sqrt(len(refs))) so the atlas reads as a
    square-ish grid (4 refs -> 2x2). Cells are 360x480 with a 36 px label
    strip; labels come from `reference_label`.
    """
    if not refs:
        raise ValueError("cannot build an atlas from an empty refs list")
    if columns is None:
        columns = max(1, math.ceil(math.sqrt(len(refs))))
    rows = math.ceil(len(refs) / columns)
    atlas = Image.new("RGB", (columns * CELL_WIDTH, rows * CELL_HEIGHT), "white")
    draw = ImageDraw.Draw(atlas)
    font = load_font(20)

    for index, ref_path in enumerate(refs):
        column = index % columns
        row = index // columns
        left = column * CELL_WIDTH
        top = row * CELL_HEIGHT
        with Image.open(ref_path) as source:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            fitted = ImageOps.contain(
                background.convert("RGB"),
                (CELL_WIDTH - 2 * IMAGE_MARGIN,
                 CELL_HEIGHT - LABEL_HEIGHT - 2 * IMAGE_MARGIN),
                Image.Resampling.LANCZOS,
            )
        image_left = left + (CELL_WIDTH - fitted.width) // 2
        image_top = top + LABEL_HEIGHT + (CELL_HEIGHT - LABEL_HEIGHT - fitted.height) // 2
        atlas.paste(fitted, (image_left, image_top))
        draw.rectangle(
            (left, top, left + CELL_WIDTH - 1, top + CELL_HEIGHT - 1),
            outline="#777777",
            width=2,
        )
        draw.text(
            (left + 10, top + 7), reference_label(ref_path), fill="black", font=font
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(destination, format="JPEG", quality=94, subsampling=0)
    return destination


def build_filtered_atlas(
    characters: list[str],
    refs_dir: Path,
    destination: Path,
    columns: int | None = None,
) -> Path | None:
    """Build an atlas containing only `characters`' reference images.

    Returns None when `characters` is empty (the colorize step then sends the
    panel without an atlas — user-confirmed policy).
    """
    refs = refs_for_characters(characters, refs_dir)
    if not refs:
        return None
    return build_labelled_atlas(refs, destination, columns=columns)
