"""Small shared helpers: hashing, image/file metadata records."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(path: Path) -> list[int]:
    with Image.open(path) as image:
        return [image.width, image.height]


def image_mime(path: Path) -> str:
    with Image.open(path) as image:
        image_format = (image.format or "").upper()
    mime = Image.MIME.get(image_format)
    if not mime:
        raise ValueError(f"Unsupported image format for {path}")
    return mime


def file_record(path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance record for an image file (repo convention)."""
    record: dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "filename": Path(path).name,
        "sha256": sha256(path),
        "bytes": Path(path).stat().st_size,
        "mime_type": image_mime(path),
        "dimensions": image_dimensions(path),
    }
    if extra:
        record.update(extra)
    return record


# --------------------------------------------------------------------------
# Fonts

_FONT_CANDIDATES: list[Path] = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
]


def _bundled_font_candidates() -> list[Path]:
    """Fonts shipped with installed packages (e.g. matplotlib bundles DejaVu)."""
    candidates: list[Path] = []
    try:
        import matplotlib

        fonts_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if fonts_dir.is_dir():
            candidates.extend(sorted(fonts_dir.glob("DejaVuSans-Bold.ttf")))
            candidates.extend(sorted(fonts_dir.glob("*.ttf")))
    except Exception:  # noqa: BLE001 - matplotlib optional
        pass
    return candidates


def load_font(size: int):
    """Return a bold TrueType font of `size`, preferring system fonts and
    falling back to fonts bundled with installed packages, then Pillow's
    default font. Never fails."""
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES + _bundled_font_candidates():
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: no size argument
        return ImageFont.load_default()
