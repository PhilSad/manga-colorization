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

## Notes

- Summaries are the wiki's own prose (one or more paragraphs, newline-separated).
  They are not translations of the raw manga text and were not generated here.
- `type` values are kept exactly as on the wiki (e.g. `Flashback Debut`, `Chapter Cover`).
