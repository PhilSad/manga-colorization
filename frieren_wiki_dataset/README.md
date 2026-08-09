# Frieren wiki chapter dataset

Scraped from the [Frieren: Beyond Journey's End wiki](https://frieren.fandom.com)
(Fandom) via the MediaWiki API (`api.php`, `action=parse`, `prop=wikitext`),
which is reachable even though the raw HTML pages sit behind a Cloudflare
challenge. Reproduce with:

```bash
python3 scrape_frieren_wiki.py
```

## Files

- `chapters.json` — full dataset, one object per chapter, ordered by chapter number.
- `chapters.csv` — flat table; the `characters` column is a JSON-encoded list.
- `characters.csv` — long form: one row per character appearance
  (`chapter`, `position`, `name`, `type`, `link`).
- `chapter_page_map.json` + `chapter_pages.csv` — chapter → page-file mapping
  for `data/page_per_volume/`, produced by `associate_chapters_to_pages.py`
  (see below).

## Fields per chapter

| field       | description                                                                   |
|-------------|-------------------------------------------------------------------------------|
| `number`    | chapter number as listed on the wiki (1–147)                                  |
| `title`     | chapter title from the infobox                                                |
| `url`       | canonical wiki page URL                                                       |
| `pages`     | number of manga pages for the chapter (infobox `\|pages = N`)                 |
| `summary`   | text of the `Summary` section, wikitext cleaned (links unwrapped, refs/templates dropped) |
| `characters`| ordered list from `Characters in Order of Appearance`; each entry has `name`, plus `type` (Debut / Mentioned / Flashback / Flashback Debut / Imagined / Pictured / Chapter Cover / Appear) and `link` when the wiki provides them |

## Completeness (as fetched)

Fetched: 2025-08-09 (all 147 chapters).

- `pages`: 144/147 — the wiki lists no page count for chapters 105, 106, 119
  (`\|pages =` empty in the infobox).
- `summary`: 113/147 — 34 chapters (75, 76, 97, 99–102, 104–106, 115, 117–119,
  121, 125–129, 131–132, 134, 136, 138–147) have an empty `Summary` section or a
  `{{Stub/Section}}` placeholder; the wiki has not written summaries for them yet.
  Missing entries are `null`/absent, not empty strings.
- `characters`: 147/147.

## Chapter → page mapping (chapter_page_map.json / chapter_pages.csv)

`associate_chapters_to_pages.py` associates every chapter with the page files
of its volume in `data/page_per_volume/`:

- layout is driven by the **filename chapter tags** (`c001 (v01) - p003 ...`),
  which partition each volume exactly; the first 3 files and the last file of
  each volume are padding (cover/title/credits/preview) and are excluded;
- volume 9's files are mislabeled (every page tagged `c078`), so it falls back
  to wiki page counts and is marked `verified: false` — its wiki counts leave
  +6 files unassigned (VIZ extra pages), so its boundaries are approximate;
- chapters 138–147 (vol 15) have no extracted volume yet and are unmapped;
- chapters 105/106/119 have no wiki page count; the counts are taken from the
  files (18 each) and can be overridden/verified with `--pages "105=18"`;
- wiki counts differ from the actual files for 48/137 chapters (VIZ adds recap
  and ad pages, mostly at the first chapter of each volume) — the mapping uses
  the files and records the deltas in each chapter's `notes`.

```bash
python3 associate_chapters_to_pages.py [--pages "105=18"]
```

`file_indices` are 0-based into the natural-sorted listing of the volume
directory, end-exclusive. `p_numbers` are the VIZ page numbers from the
filenames (a spread file covers two p-numbers).

## Notes

- Summaries are the wiki's own prose (one or more paragraphs, newline-separated).
  They are not translations of the raw manga text and were not generated here.
- `type` values are kept exactly as on the wiki (e.g. `Flashback Debut`, `Chapter Cover`).
