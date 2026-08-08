"""Tests for profiles.py (task 0002): schema validation, shared identity
hints, explicit palette prompt rendering, provenance hashes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from config import PIPELINE_DIR, REPO_ROOT
from profiles import (
    CharacterProfile,
    hint_text,
    load_profiles,
    palette_instruction,
    profile_for,
    profiles_sha256,
    unknown_names,
    validate_profiles,
    validate_reference_files,
)

PROFILES_FILE = PIPELINE_DIR / "character_profiles.json"
REFS_DIR = REPO_ROOT / "data" / "refs"


def make_refs(tmp_path: Path) -> Path:
    refs = tmp_path / "refs"
    refs.mkdir(exist_ok=True)
    for name in ("frieren_reference.webp", "heiter_reference.webp",
                 "sein_reference.webp", "fern_reference.webp"):
        Image.new("RGB", (4, 4), "gray").save(refs / name)
    return refs


def write_profiles(tmp_path: Path, profiles: list[dict]) -> Path:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": profiles}),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Ship data

def test_ship_profiles_load_and_validate_against_refs():
    profiles = validate_profiles(PROFILES_FILE, refs_dir=REFS_DIR)
    assert len(profiles) == 17
    for name in ("Frieren", "Fern", "Stark", "Himmel", "Heiter", "Eisen",
                 "Sein", "Aura", "Denken", "Flamme", "Serie", "Kanne",
                 "Laufen", "Lawine", "Richter", "Wirbel", "Uebel"):
        assert name.lower() in profiles, name


def test_ship_profiles_have_canonical_palettes():
    profiles = load_profiles(PROFILES_FILE)
    frieren = profiles["frieren"]
    assert "silver-white hair" in frieren.canonical_palette
    heiter = profiles["heiter"]
    assert "light green hair" in heiter.canonical_palette
    assert "glasses" in heiter.canonical_palette
    sein = profiles["sein"]
    assert any("purple" in c or "blue" in c for c in sein.canonical_palette)


# ---------------------------------------------------------------------------
# Validation

def test_duplicate_names_fail(tmp_path):
    path = write_profiles(tmp_path, [
        {"name": "Frieren", "canonical_palette": ["x"]},
        {"name": "frieren", "canonical_palette": ["y"]},
    ])
    with pytest.raises(ValueError, match="duplicate profile name"):
        load_profiles(path)


def test_duplicate_aliases_fail(tmp_path):
    path = write_profiles(tmp_path, [
        {"name": "Frieren", "aliases": ["Slayer"], "canonical_palette": ["x"]},
        {"name": "Fern", "aliases": ["slayer"], "canonical_palette": ["y"]},
    ])
    with pytest.raises(ValueError, match="duplicate alias"):
        load_profiles(path)


def test_alias_colliding_with_name_fails(tmp_path):
    path = write_profiles(tmp_path, [
        {"name": "Frieren", "canonical_palette": ["x"]},
        {"name": "Fern", "aliases": ["frieren"], "canonical_palette": ["y"]},
    ])
    with pytest.raises(ValueError, match="duplicate alias"):
        load_profiles(path)


def test_missing_reference_files_fail(tmp_path):
    profiles = load_profiles(PROFILES_FILE)
    with pytest.raises(ValueError, match="no matching file exists"):
        validate_reference_files(profiles, make_refs(tmp_path))


# ---------------------------------------------------------------------------
# Prompt rendering

def test_palette_instruction_names_only_selected_characters():
    profiles = load_profiles(PROFILES_FILE)
    text = palette_instruction(["Frieren", "Heiter"], profiles)
    assert "Frieren: silver-white hair" in text
    assert "Heiter: light green hair" in text
    # Unselected characters never leak into the instruction.
    assert "Fern" not in text
    assert "Sein" not in text
    assert "Stark" not in text


def test_palette_instruction_unknown_character_neutral():
    profiles = load_profiles(PROFILES_FILE)
    text = palette_instruction(["Frieren", "Gandalf"], profiles)
    assert "Gandalf" in text
    assert "neutral invented palette" in text
    assert unknown_names(["Gandalf"], profiles) == ["Gandalf"]
    assert unknown_names(["Frieren"], profiles) == []


def test_palette_instruction_empty_when_no_profiles():
    assert palette_instruction(["Gandalf"], {}) == ""
    assert palette_instruction([], {}) == ""


def test_hint_text_from_shared_profiles():
    profiles = load_profiles(PROFILES_FILE)
    assert "elf" in hint_text("Frieren", profiles)
    assert hint_text("Gandalf", profiles) is None


def test_profile_for_alias_and_case():
    profiles = load_profiles(PROFILES_FILE)
    assert profile_for(profiles, "frieren").name == "Frieren"
    assert profile_for(profiles, "Frieren the Slayer").name == "Frieren"


def test_profiles_sha256_stable():
    digest = profiles_sha256(PROFILES_FILE)
    assert digest is not None
    assert len(digest) == 64
    assert profiles_sha256(PROFILES_FILE) == digest
