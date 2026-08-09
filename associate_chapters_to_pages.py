#!/usr/bin/env python3
"""Associate manga chapters with page indices in data/page_per_volume/.

For every chapter of Frieren: Beyond Journey's End this script computes the
range of pages (0-based file indices into the natural-sorted listing of the
chapter's volume directory) that contain the chapter's content, and validates
that the per-volume page counts add up.

How the layout is computed
--------------------------
The extracted pages carry chapter tags in their filenames::

    ... - c007 (v01) - p186-p187 [VIZ Media] ... .png
          ^^^^ ^^^^^   ^^^^ ^^^^^
          chapter     volume  p-number range (a spread covers two p-numbers)

For volumes whose chapter tags are consistent, each chapter's pages are the
contiguous run of files tagged with its number (files are the ground truth:
the tag runs partition the volume exactly). Padding is removed first: the
first 3 files and the last file of each volume are front/back matter (cover,
title, credits, preview), not chapter content.

For volumes with inconsistent tags (v09 is mislabeled as c078 everywhere),
the layout falls back to the wiki page counts laid out cumulatively, and the
result is reported as unverified.

Page counts
-----------
* Chapters in correctly-tagged volumes: count taken from the files themselves.
  This includes chapters whose wiki count is missing (105, 106, 119) - the
  counts are printed explicitly so they can be compared against manual counts.
* Volume 9 chapters: wiki counts (or --pages overrides); marked unverified.
* Any chapter can be overridden with --pages "N=M" (repeatable).

Validation ("the numbers add up")
---------------------------------
* Per volume: sum of chapter file counts must equal content files
  (total files - padding-start - padding-end).
* Per chapter: wiki page count vs actual file count is reported; deltas stem
  from the VIZ release adding recap/ads pages relative to the Japanese
  tankobon counts (largest at the first chapter of each volume).

Outputs (in --out-dir, default frieren_wiki_dataset/):
  * chapter_page_map.json - per chapter: volume, file index range, p-number
    range, page count + its source, wiki count, verified flag, notes.
  * chapter_pages.csv - long form: one row per (chapter, page file).

Usage:
    python3 associate_chapters_to_pages.py [--pages "105=18" "106=18"]
                                           [--padding-start 3] [--padding-end 1]
                                           [--quiet]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
DEFAULT_VOLUMES_DIR = REPO_ROOT / "data" / "page_per_volume"
DEFAULT_DATASET = REPO_ROOT / "frieren_wiki_dataset" / "chapters.json"
DEFAULT_OUT_DIR = REPO_ROOT / "frieren_wiki_dataset"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif"}
FILENAME_RE = re.compile(
    r"c(\d{3,4}) \(v(\d{1,3})\) - p(\d+)(?:-p(\d+))?"
)
VOLUME_TAG_RE = re.compile(r" v(\d{1,3}) ")


def natural_sort_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically (p2 < p10)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_wiki_dataset(path: Path) -> dict:
    """Return {chapter_number: {volume, pages, title, ...}} from chapters.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {rec["number"]: rec for rec in data["chapters"]}


def load_volumes(volumes_dir: Path):
    """Yield (volume_number, dir_path, [file_path, ...]) for every volume dir.

    Files are natural-sorted (reading order). Non-image files are skipped.
    """
    volumes = []
    for path in sorted(volumes_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        m = VOLUME_TAG_RE.search(path.name)
        if not m:
            print(f"warning: skipping dir without volume tag: {path.name}",
                  file=sys.stderr)
            continue
        files = sorted(
            (f for f in path.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda f: natural_sort_key(f.name),
        )
        volumes.append((int(m.group(1)), path, files))
    return sorted(volumes, key=lambda v: v[0])


def parse_filename(name: str):
    """Return (chapter, p_start, p_end) or None for non-conforming filenames."""
    m = FILENAME_RE.search(name)
    if not m:
        return None
    chapter = int(m.group(1))
    p_start = int(m.group(3))
    p_end = int(m.group(4) if m.group(4) is not None else m.group(3))
    return chapter, p_start, p_end


# --------------------------------------------------------------------------
# Per-volume chapter layout
# --------------------------------------------------------------------------

def chapters_from_file_tags(files: list[Path]):
    """Map contiguous runs of identical chapter tags to (start, end) file slices.

    Returns ([(chapter, start, end), ...], consistent) where `consistent` is
    False when a chapter appears in more than one run (e.g. v09, tagged c078
    everywhere).
    """
    runs: list[tuple[int, int, int]] = []
    cur_chapter = None
    run_start = 0
    for i, f in enumerate(files):
        parsed = parse_filename(f.name)
        chapter = parsed[0] if parsed else None
        if chapter != cur_chapter:
            if cur_chapter is not None:
                runs.append((cur_chapter, run_start, i))
            cur_chapter = chapter
            run_start = i
    if cur_chapter is not None:
        runs.append((cur_chapter, run_start, len(files)))
    chapters = [r for r in runs if r[0] is not None]
    consistent = len(chapters) == len({c for c, _, _ in chapters})
    return chapters, consistent


def layout_by_file_tags(volume_num, files, padding_start, padding_end, wiki):
    """Layout a correctly-tagged volume from its filename chapter tags.

    Returns (records, errors) where records are per-chapter dicts with
    file-indices in [start, end) (0-based, end-exclusive). Returns
    (None, errors) when the tags are inconsistent (e.g. v09, where every
    file is tagged c078).
    """
    runs, _ = chapters_from_file_tags(files)
    tag_chapters = {c for c, _, _ in runs}
    wiki_chapters = {c["number"] for c in wiki.values()
                     if c["volume"] == volume_num}
    if tag_chapters != wiki_chapters:
        return None, [
            f"filename chapter tags {sorted(tag_chapters)} do not match "
            f"wiki chapters {sorted(wiki_chapters)} for this volume"
        ]
    records = []
    errors = []
    content_start = padding_start
    content_end = len(files) - padding_end
    for chapter, run_start, run_end in runs:
        # Clip the padding files off the first/last chapter's run.
        start = max(run_start, content_start)
        end = min(run_end, content_end)
        chapter_files = files[start:end]
        if start >= end:
            errors.append(f"chapter {chapter}: empty content run after padding")
            continue
        p_start = parse_filename(chapter_files[0].name)[1]
        p_end = parse_filename(chapter_files[-1].name)[2]
        wiki_rec = wiki.get(chapter, {})
        records.append({
            "number": chapter,
            "volume": volume_num,
            "volume_dir": None,  # filled by caller
            "start_idx": start,
            "end_idx": end,
            "file_count": end - start,
            "p_start": p_start,
            "p_end": p_end,
            "page_count": end - start,
            "page_count_source": "files",
            "wiki_pages": wiki_rec.get("pages"),
            "verified": True,
            "notes": [],
        })
    return records, errors


def layout_by_wiki_counts(volume_num, files, padding_start, padding_end,
                          wiki, overrides):
    """Fallback layout: cumulative file counts from wiki pages (+ overrides).

    Used for volumes with inconsistent filename tags (v09). Every chapter in
    the volume must be resolvable (override > wiki) or the volume is skipped.
    """
    chapters = sorted(
        (c for c in wiki.values() if c["volume"] == volume_num),
        key=lambda c: c["number"],
    )
    records = []
    errors = []
    content_start = padding_start
    content_end = len(files) - padding_end
    cursor = content_start
    unknown = []
    for chap in chapters:
        n = chap["number"]
        if n in overrides:
            count = overrides[n]
        elif chap["pages"] is not None:
            count = chap["pages"]
        else:
            unknown.append(n)
            count = None
        if count is None:
            continue
        end = cursor + count
        if end > content_end:
            errors.append(
                f"chapter {n}: count {count} overflows volume content "
                f"({cursor} + {count} > {content_end})"
            )
            return None, errors
        records.append({
            "number": n,
            "volume": volume_num,
            "volume_dir": None,
            "start_idx": cursor,
            "end_idx": end,
            "file_count": count,
            "p_start": parse_filename(files[cursor].name)[1],
            "p_end": parse_filename(files[end - 1].name)[2],
            "page_count": count,
            "page_count_source": ("override" if n in overrides else "wiki"),
            "wiki_pages": chap["pages"],
            "verified": False,
            "notes": ["volume has inconsistent filename tags; "
                      "layout from wiki page counts, verify manually"],
        })
        cursor = end
    # Single remaining unknown in the volume -> infer from the remainder.
    if unknown and cursor < content_end:
        remaining = content_end - cursor
        if len(unknown) == 1 and remaining > 0:
            n = unknown[0]
            end = cursor + remaining
            records.append({
                "number": n,
                "volume": volume_num,
                "volume_dir": None,
                "start_idx": cursor,
                "end_idx": end,
                "file_count": remaining,
                "p_start": parse_filename(files[cursor].name)[1],
                "p_end": parse_filename(files[end - 1].name)[2],
                "page_count": remaining,
                "page_count_source": "inferred",
                "wiki_pages": None,
                "verified": False,
                "notes": ["count inferred from remaining volume pages"],
            })
            cursor = end
            unknown = []
    if unknown:
        errors.append(
            f"cannot resolve page counts for chapters {unknown} "
            f"(use --pages \"N=M\")"
        )
        return None, errors
    # A residual means the wiki counts do not tile the volume exactly (VIZ
    # volumes carry extra pages). Keep the approximate layout but flag it:
    # boundaries are unverified and must be checked manually.
    if cursor != content_end:
        residual = content_end - cursor
        print(f"v{volume_num:02d}: wiki-count layout leaves "
              f"{residual:+d} file(s) unassigned; boundaries are "
              f"approximate, verify manually", file=sys.stderr)
        note = (f"wiki page counts leave {residual:+d} file(s) unassigned; "
                f"chapter boundaries are approximate (verify manually)")
        if records:
            records[-1]["notes"].append(note)
        else:
            errors.append("no chapters laid out: " + note)
            return None, errors
    records.sort(key=lambda r: r["number"])
    return records, errors


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def validate_volume(records, volume_files, padding_start, padding_end):
    """Check that chapter file ranges tile the content pages exactly."""
    content_start = padding_start
    content_end = len(volume_files) - padding_end
    errors = []
    prev_end = content_start
    for rec in sorted(records, key=lambda r: r["start_idx"]):
        if rec["start_idx"] != prev_end:
            errors.append(f"gap/overlap before chapter {rec['number']}: "
                          f"expected {prev_end}, got {rec['start_idx']}")
        prev_end = rec["end_idx"]
    if prev_end != content_end:
        errors.append(f"content ends at {prev_end}, expected {content_end}")
    return errors


def wiki_delta_report(records, wiki):
    """Count chapters whose wiki page count differs from the file count."""
    deltas = []
    for rec in records:
        w = rec["wiki_pages"]
        if w is None or rec["page_count_source"] != "files":
            continue
        if w != rec["file_count"]:
            deltas.append((rec["number"], w, rec["file_count"], w - rec["file_count"]))
    return deltas


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volumes-dir", default=str(DEFAULT_VOLUMES_DIR),
                        help="directory with per-volume page folders")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="frieren_wiki_dataset/chapters.json")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="output directory (default: frieren_wiki_dataset)")
    parser.add_argument("--pages", action="append", default=[], metavar="N=M",
                        help="override chapter page count, repeatable "
                             "(e.g. --pages \"105=18\" --pages \"106=19\")")
    parser.add_argument("--padding-start", type=int, default=3,
                        help="leading padding files per volume (default: 3)")
    parser.add_argument("--padding-end", type=int, default=1,
                        help="trailing padding files per volume (default: 1)")
    parser.add_argument("--quiet", action="store_true",
                        help="only print warnings and the final summary")
    args = parser.parse_args()

    overrides = {}
    for spec in args.pages:
        m = re.match(r"^(\d+)[=:](\d+)$", spec)
        if not m:
            parser.error(f"invalid --pages spec: {spec!r} (expected N=M)")
        overrides[int(m.group(1))] = int(m.group(2))

    wiki = load_wiki_dataset(Path(args.dataset))
    volumes = load_volumes(Path(args.volumes_dir))
    if not volumes:
        print("error: no volumes found", file=sys.stderr)
        return 1

    all_records = []
    wiki_chapter_nums = set(wiki)
    missing_volumes = []
    problems = []

    for volume_num, vol_dir, files in volumes:
        records, errors = layout_by_file_tags(
            volume_num, files, args.padding_start, args.padding_end, wiki)
        if records is None:
            print(f"v{volume_num:02d}: filename tags inconsistent "
                  f"({errors[0]}); falling back to wiki page counts",
                  file=sys.stderr)
            records, errors = layout_by_wiki_counts(
                volume_num, files, args.padding_start, args.padding_end,
                wiki, overrides)
        for rec in records:
            rec["volume_dir"] = vol_dir.name
            # --pages overrides on tag-based volumes are a verification aid:
            # the files are the ground truth, so a mismatch is reported.
            if rec["number"] in overrides:
                want = overrides[rec["number"]]
                if rec["page_count_source"] == "files" and want != rec["file_count"]:
                    print(f"warning: --pages {rec['number']}={want} conflicts "
                          f"with the {rec['file_count']} file(s) tagged "
                          f"c{rec['number']:03d} in the volume; keeping the "
                          f"file-derived count", file=sys.stderr)
            # Wiki count sanity note for file-derived chapters.
            if rec["page_count_source"] == "files" and rec["wiki_pages"] is not None:
                if rec["wiki_pages"] != rec["file_count"]:
                    rec["notes"].append(
                        f"wiki page count {rec['wiki_pages']} differs from "
                        f"files ({rec['file_count']})")
            elif (rec["page_count_source"] == "files"
                  and rec["wiki_pages"] is None):
                rec["notes"].append(
                    "no wiki page count; taken from the volume files")
            if not args.quiet:
                src = rec["page_count_source"]
                tag = "" if rec["verified"] else " [UNVERIFIED]"
                print(f"  ch{rec['number']:>3} v{rec['volume']:>2} "
                      f"files[{rec['start_idx']:>3}:{rec['end_idx']:>3}) "
                      f"count={rec['file_count']:>2} ({src}){tag}",
                      file=sys.stderr)
        all_records.extend(records)
        for err in errors:
            problems.append(f"v{volume_num:02d}: {err}")
        # Tile validation only makes sense for tag-based layouts; the wiki
        # fallback is inherently approximate and already flagged above.
        if all(rec["verified"] for rec in records) and records:
            tile_errors = validate_volume(records, files,
                                          args.padding_start, args.padding_end)
            for err in tile_errors:
                problems.append(f"v{volume_num:02d}: {err}")

    # Chapters whose volume has not been extracted (v15: 138-147).
    extracted = {v[0] for v in volumes}
    for chap in sorted(wiki.values(), key=lambda c: c["number"]):
        if chap["volume"] not in extracted:
            missing_volumes.append(chap["number"])
            all_records.append({
                "number": chap["number"],
                "volume": chap["volume"],
                "volume_dir": None,
                "start_idx": None,
                "end_idx": None,
                "file_count": None,
                "p_start": None,
                "p_end": None,
                "page_count": chap["pages"],
                "page_count_source": "wiki" if chap["pages"] is not None else None,
                "wiki_pages": chap["pages"],
                "verified": False,
                "notes": ["volume not extracted in data/page_per_volume"],
            })

    # ---- reports ---------------------------------------------------------
    mapped = [r for r in all_records if r["start_idx"] is not None]
    unmapped = [r for r in all_records if r["start_idx"] is None]

    if missing_volumes:
        print(f"note: chapters {missing_volumes[0]}..{missing_volumes[-1]} "
              f"(vol {wiki[missing_volumes[0]]['volume']}) have no extracted "
              f"volume pages; left unmapped.", file=sys.stderr)

    # Wiki-vs-files discrepancy summary.
    deltas = wiki_delta_report(mapped, wiki)
    if deltas:
        import collections
        dist = collections.Counter(d[3] for d in deltas)
        worst = sorted(deltas, key=lambda d: abs(d[3]), reverse=True)[:8]
        print(f"note: wiki page count differs from files for "
              f"{len(deltas)}/{len(mapped)} chapters "
              f"(delta distribution {dict(sorted(dist.items()))}). "
              f"Largest: {[(d[0], d[2] - d[1]) for d in worst]}",
              file=sys.stderr)

    # Explicit note for the chapters the user asked about.
    for n in (105, 106, 119):
        rec = next((r for r in mapped if r["number"] == n), None)
        if rec:
            files = Path(args.volumes_dir, rec["volume_dir"]).iterdir()
            first = sorted(f for f in files
                           if f.suffix.lower() in IMAGE_SUFFIXES)
            names = [f.name for f in first[rec["start_idx"]:rec["end_idx"]]]
            print(f"chapter {n}: {rec['file_count']} file(s) "
                  f"(from filenames, {rec['page_count_source']}) - "
                  f"files[{rec['start_idx']}:{rec['end_idx']}) "
                  f"in '{rec['volume_dir'][:40]}...'", file=sys.stderr)
            if not args.quiet:
                for nm in names:
                    print(f"    {nm}", file=sys.stderr)

    if problems:
        print("PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)

    # ---- write outputs ---------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": {
            "volumes": str(Path(args.volumes_dir)),
            "dataset": str(Path(args.dataset)),
            "layout": "filename chapter tags (ground truth); wiki page counts "
                      "fallback for inconsistently tagged volumes",
            "padding": {"start_files": args.padding_start,
                        "end_files": args.padding_end},
            "indexing": "file_indices are 0-based into the natural-sorted "
                        "listing of the volume dir; end-exclusive; "
                        "p-numbers are the VIZ page numbers from filenames "
                        "(spread files cover two p-numbers)",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "chapters": all_records,
    }
    (out_dir / "chapter_page_map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(out_dir / "chapter_pages.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chapter", "volume", "file_index", "filename", "p_numbers"])
        for rec in mapped:
            vol_dir = Path(args.volumes_dir) / rec["volume_dir"]
            files = sorted((f for f in vol_dir.iterdir()
                            if f.suffix.lower() in IMAGE_SUFFIXES),
                           key=lambda f: natural_sort_key(f.name))
            for idx in range(rec["start_idx"], rec["end_idx"]):
                parsed = parse_filename(files[idx].name)
                pnums = f"{parsed[1]}-{parsed[2]}" if parsed else ""
                writer.writerow([rec["number"], rec["volume"], idx,
                                 files[idx].name, pnums])

    print(f"wrote {len(mapped)} mapped + {len(unmapped)} unmapped chapters to "
          f"{out_dir}/chapter_page_map.json and chapter_pages.csv",
          file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
