"""Tests for atlas.py: refs filtering + labelled atlas builder."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from atlas import (
    build_filtered_atlas,
    build_labelled_atlas,
    refs_for_characters,
    reference_label,
)


def make_refs(tmp_path: Path, names: list[str]) -> Path:
    refs = tmp_path / "refs"
    refs.mkdir(exist_ok=True)
    for name in names:
        Image.new("RGB", (10, 10), "gray").save(refs / name)
    return refs


def test_reference_label():
    assert reference_label(Path("frieren_reference.webp")) == "Frieren"
    assert reference_label(Path("Frieren_anime_profile.webp")) == "Frieren"


def test_refs_for_characters_prefers_reference_suffix(tmp_path):
    refs = make_refs(
        tmp_path,
        ["frieren_reference.webp", "Frieren_anime_profile.webp", "fern_reference.webp"],
    )
    picked = refs_for_characters(["Frieren", "Fern"], refs)
    assert [p.name for p in picked] == [
        "frieren_reference.webp",
        "fern_reference.webp",
    ]


def test_refs_for_characters_dedupes_and_orders(tmp_path):
    refs = make_refs(tmp_path, ["frieren_reference.webp", "fern_reference.webp"])
    picked = refs_for_characters(["Fern", "Frieren", "Fern"], refs)
    assert [p.name for p in picked] == ["fern_reference.webp", "frieren_reference.webp"]


def test_refs_for_characters_missing_skipped(capsys, tmp_path):
    refs = make_refs(tmp_path, ["frieren_reference.webp"])
    picked = refs_for_characters(["Frieren", "Gandalf"], refs)
    assert [p.name for p in picked] == ["frieren_reference.webp"]
    assert "Gandalf" in capsys.readouterr().out


def test_refs_for_characters_empty(tmp_path):
    refs = make_refs(tmp_path, ["frieren_reference.webp"])
    assert refs_for_characters([], refs) == []


def test_build_labelled_atlas_grid_sizes(tmp_path):
    refs_dir = make_refs(
        tmp_path,
        ["a_reference.webp", "b_reference.webp", "c_reference.webp", "d_reference.webp"],
    )
    refs = sorted(refs_dir.iterdir())
    out = tmp_path / "atlas.jpg"
    build_labelled_atlas(refs, out, columns=None)
    assert out.is_file()
    with Image.open(out) as image:
        assert image.size == (720, 960)  # 2x2 grid of 360x480 cells


def test_build_labelled_atlas_single_cell(tmp_path):
    refs_dir = make_refs(tmp_path, ["a_reference.webp"])
    refs = sorted(refs_dir.iterdir())
    out = tmp_path / "atlas.jpg"
    build_labelled_atlas(refs, out, columns=1)
    with Image.open(out) as image:
        assert image.size == (360, 480)


def test_build_labelled_atlas_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        build_labelled_atlas([], tmp_path / "atlas.jpg")


def test_build_filtered_atlas_none_on_empty_characters(tmp_path):
    refs = make_refs(tmp_path, ["frieren_reference.webp"])
    out = tmp_path / "atlas.jpg"
    assert build_filtered_atlas([], refs, out) is None
    assert not out.exists()


def test_build_filtered_atlas_unknown_only_returns_none(tmp_path):
    refs = make_refs(tmp_path, ["frieren_reference.webp"])
    out = tmp_path / "atlas.jpg"
    assert build_filtered_atlas(["Gandalf"], refs, out) is None


def test_build_filtered_atlas_roundtrip_labels(tmp_path):
    """The atlas for [Frieren, Fern] contains exactly those two cells."""
    refs = make_refs(tmp_path, ["frieren_reference.webp", "fern_reference.webp"])
    out = tmp_path / "atlas.jpg"
    build_filtered_atlas(["Frieren", "Fern"], refs, out)
    with Image.open(out) as image:
        assert image.size == (720, 480)  # 2 cells in one row
