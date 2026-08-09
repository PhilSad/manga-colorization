#!/usr/bin/env python3
"""Scrape the Frieren: Beyond Journey's End wiki (frieren.fandom.com) into a dataset.

For every manga chapter (Chapter 1 .. Chapter N) this script extracts, from the
chapter's wikitext (fetched through the MediaWiki API, which is not blocked by
Cloudflare the way the raw HTML pages are):

  * pages            - how many manga pages the chapter has (infobox `|pages = N`)
  * summary          - the text of the "Summary" section (wikitext cleaned up)
  * characters       - the list from "Characters in Order of Appearance", in order,
                       with their appearance `type` (Debut / Mentioned / Flashback /
                       Flashback Debut / Imagined / Pictured) when the wiki provides it

Outputs (written to --out-dir, default `frieren_wiki_dataset/`):
  * chapters.json  - full dataset, one object per chapter, ordered by chapter number
  * chapters.csv   - flat table (characters column is a JSON-encoded list)
  * characters.csv - long-form table, one row per character appearance
  * README.md      - provenance notes for the dataset

Python 3 stdlib only. Usage:
    python3 scrape_frieren_wiki.py [--out-dir DIR] [--limit N] [--start N]
                                   [--delay SEC] [--quiet]
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

WIKI = "https://frieren.fandom.com"
API_URL = WIKI + "/api.php"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36 frieren-wiki-dataset-scraper/1.0"
)
CHAPTER_TITLE_RE = re.compile(r"^Chapter (\d+(?:\.\d+)?)$")
SECTION_HEADER_RE = re.compile(r"^==+\s*(.*?)\s*==+\s*$")
PAGES_RE = re.compile(r"^\|\s*pages\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.MULTILINE)
CHAPTER_TITLE_FIELD_RE = re.compile(r"^\|\s*chapter title\s*=\s*(.+)$", re.MULTILINE)
CHAR_TEMPLATE_RE = re.compile(r"\{\{Character Appearance\|(.*?)\}\}", re.DOTALL)
LINK_RE = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\]")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*?\}\}")
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/>]*/>|<references\s*/>", re.DOTALL)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)

# Required dataset fields per chapter; used for the completeness report.
REQUIRED = ("pages", "summary", "characters")


# --------------------------------------------------------------------------
# MediaWiki API
# --------------------------------------------------------------------------

def api_get(params, retries=5, timeout=60):
    """GET the MediaWiki API with retries + exponential backoff.

    Returns the parsed JSON. Raises RuntimeError after `retries` failures.
    """
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
            last_err = err
            backoff = 1.5 * (2 ** attempt)
            print(f"  ! request failed ({err}); retrying in {backoff:.1f}s "
                  f"({attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError(f"API request failed after {retries} attempts: {url} ({last_err})")


def discover_chapters():
    """Return the sorted list of chapter page titles (Chapter 1, Chapter 2, ...)."""
    titles = []
    cont = {}
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": "0",
            "apprefix": "Chapter ",
            "apfilterredir": "nonredirects",
            "aplimit": "500",
            "format": "json",
        }
        params.update(cont)
        data = api_get(params)
        for page in data["query"]["allpages"]:
            if CHAPTER_TITLE_RE.match(page["title"]):
                titles.append(page["title"])
        if "continue" in data:
            cont = data["continue"]
        else:
            break

    def chapter_number(title):
        return float(CHAPTER_TITLE_RE.match(title).group(1))

    return sorted(titles, key=chapter_number)


# --------------------------------------------------------------------------
# Wikitext parsing
# --------------------------------------------------------------------------

def split_sections(wikitext):
    """Split wikitext into {section_title: body_lines} for == level == headers."""
    sections = {}
    current = None
    for line in wikitext.splitlines():
        m = SECTION_HEADER_RE.match(line)
        if m:
            current = m.group(1).strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {title: "\n".join(lines) for title, lines in sections.items()}


def clean_text(text):
    """Turn wikitext prose into plain text: drop refs/templates, unwrap links."""
    if not text:
        return ""
    text = REF_RE.sub("", text)
    text = BR_RE.sub(" ", text)

    # Drop templates. {{Ruby|A|B}} and {{Nihongo|A|...}} keep their first
    # argument (the visible text); anything else is removed entirely.
    def replace_template(m):
        inner = m.group(0)[2:-2]
        name, _, rest = inner.partition("|")
        if name.strip() in ("Ruby", "Nihongo") and rest:
            return rest.split("|", 1)[0]
        return ""

    text = TEMPLATE_RE.sub(replace_template, text)

    # Unwrap [[Target|Display]] -> Display and [[Target]] -> Target.
    # Drop [[File:...]]/[[Image:...]] links entirely: they are images, not prose.
    def replace_link(m):
        target = m.group(1)
        if target.lower().startswith(("file:", "image:")):
            return ""
        return m.group(2) if m.group(2) else target

    text = LINK_RE.sub(replace_link, text)

    # Collapse runs of spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text


def parse_characters(section):
    """Parse the characters section into ordered [{name, type?, link?}, ...]."""
    characters = []
    for match in CHAR_TEMPLATE_RE.finditer(section):
        params = {}
        positional = None
        for part in match.group(1).split("|"):
            part = part.strip()
            if "=" in part:
                key, _, value = part.partition("=")
                params[key.strip()] = value.strip()
            elif part:
                positional = part
        # `name` is the display name when both `name` and `chara` are given.
        name = params.get("name") or params.get("chara") or positional
        if not name:
            continue
        record = {"name": name}
        if params.get("type"):
            record["type"] = params["type"]
        if params.get("link"):
            record["link"] = params["link"]
        characters.append(record)
    return characters


def parse_chapter(title, wikitext):
    """Extract the dataset record for one chapter page."""
    number = float(CHAPTER_TITLE_RE.match(title).group(1))
    sections = split_sections(wikitext)

    pages_match = PAGES_RE.search(wikitext)
    title_field_match = CHAPTER_TITLE_FIELD_RE.search(wikitext)

    summary = clean_text(sections.get("Summary", ""))
    characters = parse_characters(
        sections.get("Characters in Order of Appearance", "")
    )

    return {
        "number": int(number) if number.is_integer() else number,
        "title": clean_text(title_field_match.group(1)) if title_field_match else None,
        "url": f"{WIKI}/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
        "pages": int(pages_match.group(1)) if pages_match else None,
        "summary": summary or None,
        "characters": characters,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_outputs(out_dir, records, fetched_at):
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = {
        "source": "https://frieren.fandom.com (Frieren: Beyond Journey's End wiki)",
        "api": "https://frieren.fandom.com/api.php (MediaWiki action=parse, prop=wikitext)",
        "fetched_at": fetched_at,
        "scraper": "scrape_frieren_wiki.py",
        "fields": {
            "number": "chapter number as listed on the wiki",
            "title": "chapter title from the infobox",
            "url": "canonical wiki page URL",
            "pages": "number of manga pages for the chapter (infobox `|pages = N`)",
            "summary": "text of the `Summary` section, wikitext cleaned",
            "characters": "ordered list from `Characters in Order of Appearance`; "
                          "each entry has a `name`, and `type`/`link` when the wiki provides them",
        },
        "chapters": records,
    }

    (out_dir / "chapters.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with open(out_dir / "chapters.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["number", "title", "url", "pages", "summary", "characters"])
        for rec in records:
            writer.writerow([
                rec["number"], rec["title"], rec["url"], rec["pages"],
                rec["summary"], json.dumps(rec["characters"], ensure_ascii=False),
            ])

    with open(out_dir / "characters.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chapter", "position", "name", "type", "link"])
        for rec in records:
            for pos, char in enumerate(rec["characters"], start=1):
                writer.writerow([
                    rec["number"], pos, char["name"],
                    char.get("type", ""), char.get("link", ""),
                ])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="frieren_wiki_dataset",
                        help="output directory (default: frieren_wiki_dataset)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only scrape the first N chapters (0 = all)")
    parser.add_argument("--start", type=int, default=0,
                        help="0-based index of the first chapter to scrape")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="seconds to wait between API calls (default: 0.25)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print per-chapter progress")
    args = parser.parse_args()

    print("Discovering chapter pages on the wiki ...", file=sys.stderr)
    titles = discover_chapters()
    print(f"Found {len(titles)} chapters (Chapter 1 .. {titles[-1]}).",
          file=sys.stderr)

    titles = titles[args.start:]
    if args.limit > 0:
        titles = titles[:args.limit]

    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records = []
    errors = []

    for i, title in enumerate(titles, start=1):
        try:
            data = api_get({"action": "parse", "page": title,
                            "prop": "wikitext", "format": "json"})
            wikitext = data["parse"]["wikitext"]["*"]
            records.append(parse_chapter(title, wikitext))
            if not args.quiet:
                pages = records[-1]["pages"]
                nchars = len(records[-1]["characters"])
                print(f"  [{i}/{len(titles)}] {title}: pages={pages}, "
                      f"characters={nchars}", file=sys.stderr)
        except Exception as err:  # noqa: BLE001 - keep going, report at the end
            errors.append((title, str(err)))
            print(f"  [{i}/{len(titles)}] {title}: ERROR {err}", file=sys.stderr)
        time.sleep(args.delay)

    # Completeness report.
    missing = {}
    for rec in records:
        for field in REQUIRED:
            value = rec[field]
            if value is None or value == "" or value == []:
                missing.setdefault(field, []).append(rec["number"])
    for field, chapters in sorted(missing.items()):
        print(f"WARNING: {len(chapters)} chapter(s) without '{field}': "
              f"{chapters}", file=sys.stderr)
    if errors:
        print(f"ERROR: {len(errors)} chapter(s) could not be fetched: {errors}",
              file=sys.stderr)

    if not records:
        print("No records scraped; nothing written.", file=sys.stderr)
        return 1

    from pathlib import Path
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, records, fetched_at)
    print(f"Wrote {len(records)} chapters to {out_dir}/ "
          "(chapters.json, chapters.csv, characters.csv).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
