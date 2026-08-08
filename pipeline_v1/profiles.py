"""Canonical character profiles (task 0002): the single source of truth for
monochrome identity hints (detection) and canonical palette descriptions
(FLUX prompt conditioning).

Loaded from `character_profiles.json`. Validation is strict: duplicate
names/aliases and missing reference files fail with a clear error. Prompt
rendering names only the characters assigned to a panel; unknown characters
get a neutral invented palette and are reported rather than silently mapped
to another character.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from atlas import reference_label
from util import SUPPORTED_IMAGE_SUFFIXES

DEFAULT_PROFILES_FILE = Path(__file__).resolve().parent / "character_profiles.json"


@dataclass(frozen=True)
class CharacterProfile:
    name: str
    aliases: tuple[str, ...] = ()
    identity_cues: tuple[str, ...] = ()
    canonical_palette: tuple[str, ...] = ()
    reference_files: tuple[str, ...] = ()
    variants: dict = field(default_factory=dict)

    @property
    def palette_text(self) -> str:
        return "; ".join(self.canonical_palette)

    @property
    def hint_text(self) -> str:
        return ", ".join(self.identity_cues)


def _key(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def _parse_entry(entry: dict) -> CharacterProfile:
    missing = [key for key in ("name", "canonical_palette") if key not in entry]
    if missing:
        raise ValueError(f"profile entry missing keys {missing}: {entry.get('name')!r}")
    name = str(entry["name"]).strip()
    if not name:
        raise ValueError("profile entry has an empty name")
    return CharacterProfile(
        name=name,
        aliases=tuple(str(a).strip() for a in entry.get("aliases", []) if str(a).strip()),
        identity_cues=tuple(str(c).strip() for c in entry.get("identity_cues", []) if str(c).strip()),
        canonical_palette=tuple(str(c).strip() for c in entry["canonical_palette"] if str(c).strip()),
        reference_files=tuple(str(r) for r in entry.get("reference_files", []) if str(r).strip()),
        variants=dict(entry.get("variants", {}) or {}),
    )


def load_profiles(path: Path = DEFAULT_PROFILES_FILE) -> dict[str, CharacterProfile]:
    """Load and validate the profile file. Returns profiles keyed by lowercase
    canonical name."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("profiles", data) if isinstance(data, dict) else data
    profiles: dict[str, CharacterProfile] = {}
    seen_names: dict[str, str] = {}
    seen_aliases: dict[str, str] = {}
    for entry in entries:
        profile = _parse_entry(entry)
        key = _key(profile.name)
        if key in seen_names:
            raise ValueError(
                f"duplicate profile name {profile.name!r} "
                f"(already defined as {seen_names[key]!r})"
            )
        for alias in profile.aliases:
            alias_key = _key(alias)
            if alias_key in seen_names or alias_key in seen_aliases:
                raise ValueError(
                    f"duplicate alias {alias!r} for {profile.name!r} "
                    f"(already used by {seen_aliases.get(alias_key) or seen_names.get(alias_key)})"
                )
            seen_aliases[alias_key] = profile.name
        seen_names[key] = profile.name
        profiles[key] = profile
    return profiles


def validate_reference_files(
    profiles: dict[str, CharacterProfile], refs_dir: Path
) -> None:
    """Every reference file must exist under refs_dir (task 0002)."""
    available = {
        _key(reference_label(path)): path.name
        for path in refs_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    for profile in profiles.values():
        for filename in profile.reference_files:
            key = _key(reference_label(Path(filename)))
            if key not in available:
                raise ValueError(
                    f"profile {profile.name!r} references {filename!r}, "
                    f"but no matching file exists in {refs_dir}"
                )


def validate_profiles(
    path: Path = DEFAULT_PROFILES_FILE, refs_dir: Path | None = None
) -> dict[str, CharacterProfile]:
    """Load + validate; optionally check reference files against refs_dir."""
    profiles = load_profiles(path)
    if refs_dir is not None:
        validate_reference_files(profiles, refs_dir)
    return profiles


def profile_for(
    profiles: dict[str, CharacterProfile], name: str
) -> CharacterProfile | None:
    key = _key(name)
    if key in profiles:
        return profiles[key]
    for profile in profiles.values():
        if any(_key(alias) == key for alias in profile.aliases):
            return profile
    return None


def hint_text(name: str, profiles: dict[str, CharacterProfile]) -> str | None:
    """Detection hint for `name` (identity cues from the shared profiles)."""
    profile = profile_for(profiles, name)
    if profile is None or not profile.identity_cues:
        return None
    return profile.hint_text


def unknown_names(
    names: list[str], profiles: dict[str, CharacterProfile]
) -> list[str]:
    return [name for name in names if profile_for(profiles, name) is None]


def palette_instruction(
    names: list[str], profiles: dict[str, CharacterProfile]
) -> str:
    """Explicit canonical-colour instruction for the FLUX prompt.

    Names only the selected characters; unprofiled names are recorded as
    needing a neutral invented palette (they are never mapped to another
    character). Returns an empty string when no profile applies, so the
    colorizer falls back to its generic instruction.
    """
    lines: list[str] = []
    for name in names:
        profile = profile_for(profiles, name)
        if profile is not None and profile.canonical_palette:
            lines.append(f"- {profile.name}: {profile.palette_text}.")
    if profiles:
        unknown = unknown_names(names, profiles)
        if unknown:
            lines.append(
                "- Unprofiled characters ("
                + ", ".join(unknown)
                + "): neutral invented palette consistent with the series, do not map them to another character."
            )
    if not lines:
        return ""
    return (
        "Canonical colors to apply to the characters below (from the official "
        "character profiles):\n" + "\n".join(lines)
    )


def profiles_sha256(path: Path = DEFAULT_PROFILES_FILE) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
