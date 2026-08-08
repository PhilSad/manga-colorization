#!/usr/bin/env python3
"""Extract pages from CBZ manga volumes into a per-volume folder layout.

Reads .cbz archives (zip files) from a source directory (default:
``data/volumes/``) and writes each volume's pages into its own directory
under the destination (default: ``data/page_per_volume/``)::

    data/page_per_volume/<volume-name>/<original-page-filename>

Pages keep their original filenames (they embed chapter/page numbers and are
zero-padded, so natural sorting yields reading order). Extraction is
resumable: pages that already exist with the correct size are skipped unless
``--force`` is given.

Dependencies: Python 3 stdlib only (zipfile, re).

Examples::

    # Extract every .cbz from data/volumes into data/page_per_volume
    python3 script/extract_pages.py

    # Only volumes whose name matches a substring
    python3 script/extract_pages.py --volume v05

    # Re-extract everything, overwriting existing files
    python3 script/extract_pages.py --force
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif", ".tif", ".tiff",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE = REPO_ROOT / "data" / "volumes"
DEFAULT_DEST = REPO_ROOT / "data" / "page_per_volume"


def natural_sort_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically (p2 < p10)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def iter_cbz_files(source: Path, volume_filter: str | None) -> list[Path]:
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = sorted(source.glob("*.cbz"), key=lambda p: p.name.lower())
    else:
        sys.exit(f"error: source {source} is neither a file nor a directory")

    if volume_filter:
        matched = [p for p in candidates if volume_filter.lower() in p.stem.lower()]
        if not matched:
            sys.exit(f"error: no .cbz under {source} matches --volume {volume_filter!r}")
        return matched
    return candidates


def image_entries(cbz: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Image entries of a cbz, ordered by natural filename sort."""
    entries = [
        info for info in cbz.infolist()
        if Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        and not info.is_dir()
    ]
    entries.sort(key=lambda info: natural_sort_key(Path(info.filename).name))
    return entries


def extract_volume(cbz_path: Path, dest: Path, force: bool) -> tuple[int, int, int]:
    """Extract one volume; return (pages, bytes written, skipped)."""
    out_dir = dest / cbz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cbz_path) as cbz:
        entries = image_entries(cbz)
        if not entries:
            print(f"  WARNING: no image files inside {cbz_path.name}")

        # Disambiguate entries that would collide on basename.
        used_names: dict[str, int] = {}
        targets: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in entries:
            base = Path(info.filename).name
            count = used_names.get(base, 0)
            used_names[base] = count + 1
            name = base if count == 0 else f"{Path(base).stem}-{count + 1}{Path(base).suffix}"
            targets.append((info, out_dir / name))

        written = skipped = 0
        for info, target in targets:
            if not force and target.is_file() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            with cbz.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written += 1

    total_bytes = sum(t.stat().st_size for _, t in targets if t.is_file())
    return len(targets), written, skipped, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pages from CBZ manga volumes into per-volume folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help="Directory of .cbz files, or a single .cbz file.",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help="Destination directory; one subdirectory per volume is created here.",
    )
    parser.add_argument(
        "--volume", default=None,
        help="Only process volumes whose name contains this substring.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract pages even if a same-sized file already exists.",
    )
    args = parser.parse_args()

    cbz_files = iter_cbz_files(args.source.resolve(), args.volume)
    dest = args.dest.resolve()

    grand_total = {"volumes": 0, "pages": 0, "bytes": 0}
    for cbz_path in cbz_files:
        print(f"[{cbz_path.name}]")
        pages, written, skipped, total_bytes = extract_volume(cbz_path, dest, args.force)
        out_dir = dest / cbz_path.stem
        status = f"{pages} pages ({written} extracted, {skipped} already present), {total_bytes / 1e6:.1f} MB"
        print(f"  -> {out_dir}\n  {status}")
        grand_total["volumes"] += 1
        grand_total["pages"] += pages
        grand_total["bytes"] += total_bytes

    print(
        f"Done: {grand_total['volumes']} volumes, {grand_total['pages']} pages, "
        f"{grand_total['bytes'] / 1e6:.1f} MB in {dest}"
    )


if __name__ == "__main__":
    main()
