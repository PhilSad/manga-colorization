#!/usr/bin/env python3
"""Generate pipeline_v1/chapter_casts.json from the Frieren wiki dataset.

The pipeline's character-detection stage can be constrained with a per-chapter
cast shortlist (--cast-key cNNN): the VLM prompt then only allows naming
characters in the shortlist, everything else must be reported as unknown
(never guessed). See pipeline_v1/characters.py `cast_shortlist_for`.

This script builds those shortlists from the wiki data in
`frieren_wiki_dataset/chapters.json` ("Characters in Order of Appearance"),
intersected with the canonical reference roster derived from `data/refs/`
(the characters that actually have reference images):

* characters are kept in wiki appearance order;
* a character is only listed if it has a reference image, so the detector
  cannot name characters it cannot match to a reference;
* mentioned-only entries (type "Mentioned", "Mentioned (Indirectly)",
  "Flashback Mentioned", "Mentioned Debut") are excluded: the character is
  referenced in dialogue but not drawn on the pages;
* name matching normalises German umlauts (Übel -> Uebel) and expands clone
  entries ("Fern and clone" / "Sense's clone" -> the character appears).

Every wiki character that ends up excluded is listed in the entry's `note`,
with its type, so nothing is silently dropped.

Output schema matches what `cast_shortlist_for` expects:

    {"schema_version": 1, "casts": {"c001": {"label", "characters", "note"}, ...}}

Usage:
    python3 build_chapter_casts.py [--dataset frieren_wiki_dataset/chapters.json]
                                   [--refs-dir data/refs]
                                   [--out pipeline_v1/chapter_casts.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
DEFAULT_DATASET = REPO_ROOT / "frieren_wiki_dataset" / "chapters.json"
DEFAULT_REFS_DIR = REPO_ROOT / "data" / "refs"
DEFAULT_OUT = REPO_ROOT / "pipeline_v1" / "chapter_casts.json"

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# Entry types that reference the character in dialogue without drawing them.
MENTIONED_TYPES = frozenset({
    "Mentioned", "Mentioned (Indirectly)", "Flashback Mentioned", "Mentioned Debut",
})
# "Fern and clone" / "Sense's clone" -> the character itself appears.
CLONE_RE = re.compile(r"^(.*) (?:and|&|'s) clone$", re.IGNORECASE)

UMLAUTS = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def normalize(name: str) -> str:
    """Case-folded, umlaut-expanded, alphanumerics-only key for name matching."""
    lowered = name.lower()
    for umlaut, expansion in UMLAUTS.items():
        lowered = lowered.replace(umlaut, expansion)
    return re.sub(r"[^a-z0-9]+", "", lowered)


def reference_label(path: Path) -> str:
    """Canonical character name from a reference filename (mirrors
    pipeline_v1.characters.reference_label): uebel_reference.webp -> Uebel."""
    name = path.stem
    for suffix in ("_reference", "_anime_profile"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip().title()


def canonical_roster(refs_dir: Path) -> dict[str, str]:
    """{normalized_name: canonical_name} for every reference image."""
    roster: dict[str, str] = {}
    for path in sorted(refs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        canonical = reference_label(path)
        roster.setdefault(normalize(canonical), canonical)
    return roster


def resolve_entry_name(entry: dict, roster: dict[str, str]) -> str | None:
    """Map a wiki character entry to a canonical roster name, or None."""
    name = entry["name"]
    canonical = roster.get(normalize(name))
    if canonical is None:
        clone = CLONE_RE.match(name)
        if clone:
            canonical = roster.get(normalize(clone.group(1)))
    return canonical


def build_cast(entry: dict, roster: dict[str, str]):
    """Return (characters_in_wiki_order, excluded_entries).

    `excluded_entries` are the wiki entries that did not make the shortlist:
    (display_name, type) pairs, in wiki order.
    """
    characters: list[str] = []
    excluded: list[tuple[str, str]] = []
    for entry in entry["characters"]:
        entry_type = entry.get("type", "")
        canonical = resolve_entry_name(entry, roster)
        mentioned = entry_type in MENTIONED_TYPES
        if canonical is not None and not mentioned:
            if canonical not in characters:
                characters.append(canonical)
        else:
            excluded.append((entry["name"], entry_type))
    return characters, excluded


def entry_note(excluded: list[tuple[str, str]], cast: list[str]) -> str:
    if not cast and not excluded:
        return "Chapter has no listed characters."
    if not cast:
        listed = ", ".join(f"{name} ({typ})" if typ else name
                           for name, typ in excluded)
        return ("None of the chapter's characters has a reference image; "
                f"the detector must report them as unknown. Wiki cast: {listed}.")
    if not excluded:
        return "Every listed character has a reference image."
    listed = ", ".join(f"{name} ({typ})" if typ else name
                       for name, typ in excluded)
    return ("Restricted to characters with a reference image; the rest are "
            f"excluded so they cannot be guessed: {listed}.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="frieren_wiki_dataset/chapters.json")
    parser.add_argument("--refs-dir", default=str(DEFAULT_REFS_DIR),
                        help="directory with reference character images")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="output chapter_casts.json path")
    parser.add_argument("--quiet", action="store_true",
                        help="only print the summary")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    roster = canonical_roster(Path(args.refs_dir))
    if not roster:
        print("error: no reference images found", file=sys.stderr)
        return 1

    casts = {}
    empty_casts = []
    n_excluded = 0
    for chapter in dataset["chapters"]:
        number = chapter["number"]
        characters, excluded = build_cast(chapter, roster)
        n_excluded += len(excluded)
        if not characters:
            empty_casts.append(number)
        casts[f"c{number:03d}"] = {
            "label": f"Chapter {number} — {chapter['title']} "
                     f"(volume {chapter['volume']})",
            "characters": characters,
            "note": entry_note(excluded, characters),
        }
        if not args.quiet:
            tag = " [EMPTY]" if not characters else ""
            print(f"c{number:03d}: {len(characters):>2} reference character(s)"
                  f"{tag} — {chapter['title']}", file=sys.stderr)

    payload = {
        "schema_version": 1,
        "description": (
            "Per-chapter cast shortlists generated from the Frieren wiki dataset "
            "(Characters in Order of Appearance) intersected with the reference "
            "roster in data/refs/. Selected deterministically by --cast-key "
            "('cNNN'); never fetched remotely during a run. Only characters with "
            "a reference image are listed (the detector must report everything "
            "else as unknown, never guess); mentioned-only characters are "
            "excluded too. Generated by build_chapter_casts.py; do not edit by "
            "hand, regenerate instead."
        ),
        "generated_by": "build_chapter_casts.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "chapters": str(Path(args.dataset)),
            "refs": str(Path(args.refs_dir)),
        },
        "casts": casts,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(f"wrote {len(casts)} casts to {out_path}", file=sys.stderr)
    print(f"roster: {len(roster)} reference characters "
          f"({', '.join(sorted(roster.values()))})", file=sys.stderr)
    print(f"excluded wiki entries: {n_excluded}", file=sys.stderr)
    if empty_casts:
        print(f"note: chapters with no reference characters: {empty_casts}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
