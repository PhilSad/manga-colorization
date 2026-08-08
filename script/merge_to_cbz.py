#!/usr/bin/env python3
"""Merge a folder of images back into a single CBZ archive.

Reads all image files from a directory (recursively) and packs them into a
.cbz (zip) file in natural filename order, which yields reading order for
volume pages extracted by ``extract_pages.py`` (zero-padded page numbers).

Only image files are included by default; pass ``--all`` to include every
file in the folder. Non-image metadata files such as ``manifest.txt`` are
skipped unless ``--all`` is given.

Dependencies: Python 3 stdlib only (zipfile, re).

Examples::

    # Merge data/page_per_volume/v01 back into a cbz in the current directory
    python3 script/merge_to_cbz.py data/page_per_volume/v01

    # Write to an explicit path
    python3 script/merge_to_cbz.py data/page_per_volume/v01 \
        --output "data/volumes/Frieren v01 colorized.cbz"

    # Pack everything in the folder, not just images
    python3 script/merge_to_cbz.py some/folder --all
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif", ".tif", ".tiff",
}


def natural_sort_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically (p2 < p10)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def collect_files(input_dir: Path, include_all: bool) -> list[Path]:
    """Image files (or all files) in natural order of relative path."""
    files = [
        p for p in input_dir.rglob("*")
        if p.is_file()
        and (include_all or p.suffix.lower() in IMAGE_SUFFIXES)
    ]
    files.sort(key=lambda p: natural_sort_key(str(p.relative_to(input_dir))))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a folder of images into a CBZ archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Folder of pages to pack.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .cbz path (default: <input-folder-name>.cbz in the "
             "current directory).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include every file in the folder, not just images.",
    )
    parser.add_argument(
        "--level", type=int, default=6, choices=range(0, 10),
        help="Zip compression level (0 = store, 9 = max).",
    )
    args = parser.parse_args()

    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        sys.exit(f"error: {input_dir} is not a directory")

    output = (args.output or Path.cwd() / f"{input_dir.name}.cbz").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(input_dir, args.all)
    if not files:
        sys.exit(f"error: no files to pack in {input_dir}")

    # CBZ readers expect flat page entries; detect basename collisions.
    basenames: dict[str, Path] = {}
    for path in files:
        if path.name in basenames:
            sys.exit(
                f"error: {path} and {basenames[path.name]} share the basename "
                f"{path.name!r}; rename one of them before merging"
            )
        basenames[path.name] = path

    total_bytes = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=args.level) as cbz:
        for path in files:
            cbz.write(path, arcname=path.name)
            total_bytes += path.stat().st_size

    archive_size = output.stat().st_size
    print(
        f"Merged {len(files)} files into {output} "
        f"({total_bytes / 1e6:.1f} MB unpacked, {archive_size / 1e6:.1f} MB packed)"
    )


if __name__ == "__main__":
    main()
