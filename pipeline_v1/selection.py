"""Selector helpers for targeted reruns (task 0001).

`--only-panel PAGE:PANEL` selects which panels a run processes; a page
selector matches a page stem by (in order):

1. exact stem equality;
2. a V1.1 fixture alias (e.g. `P003`) resolved through
   `evaluation/v1_1_cases.json`;
3. an unambiguous substring of the stem.

A panel selector matches a panel stem exactly (`panel_0006`). `--force-characters`
uses the same page/panel selector syntax.
"""

from __future__ import annotations

import json
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = PIPELINE_DIR / "evaluation" / "v1_1_cases.json"


def _aliases() -> dict[str, str]:
    try:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in data.get("aliases", {}).items()}


def alias_page_stems() -> dict[str, str]:
    """Alias (uppercased, e.g. `P003`) -> resolved source-page stem."""
    result: dict[str, str] = {}
    for alias, relative in _aliases().items():
        result[alias.upper()] = Path(relative).stem
    return result


def page_matches(page_stem: str, selector: str) -> bool:
    """True when the page selector matches this page stem."""
    if page_stem == selector:
        return True
    if selector in page_stem:
        return True
    return page_stem == alias_page_stems().get(selector.upper(), "")


def panel_matches(panel_stem: str, selector: str) -> bool:
    """True when the panel selector matches this panel stem."""
    return panel_stem == selector


def parse_only_panel(value: str) -> tuple[str, str]:
    """Parse `PAGE:PANEL` -> (page selector, panel selector)."""
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            f"invalid --only-panel {value!r} (expected PAGE:PANEL, e.g. P003:panel_0006)"
        )
    return parts[0].strip(), parts[1].strip()


def parse_force_characters(value: str) -> tuple[str, str, list[str]]:
    """Parse `PAGE:PANEL=Name1,Name2` -> (page selector, panel selector, names)."""
    if "=" not in value:
        raise ValueError(
            f"invalid --force-characters {value!r} "
            "(expected PAGE:PANEL=Name1,Name2)"
        )
    key, names = value.split("=", 1)
    page_sel, panel_sel = parse_only_panel(key)
    parsed = [name.strip() for name in names.split(",") if name.strip()]
    if not parsed:
        raise ValueError(f"--force-characters {value!r} has no character names")
    return page_sel, panel_sel, parsed


def page_selected(page_stem: str, selectors: tuple[str, ...]) -> bool:
    """True when at least one selector's page part matches `page_stem`."""
    for selector in selectors:
        page_sel, _panel_sel = parse_only_panel(selector)
        if page_matches(page_stem, page_sel):
            return True
    return False


def panel_selected(
    page_stem: str, panel_stem: str, selectors: tuple[str, ...]
) -> bool:
    """True when at least one selector matches this page+panel."""
    for selector in selectors:
        page_sel, panel_sel = parse_only_panel(selector)
        if page_matches(page_stem, page_sel) and panel_matches(panel_stem, panel_sel):
            return True
    return False


def forced_names_for(
    forced_characters: dict[str, list[str]],
    page_stem: str,
    panel_stem: str,
) -> list[str] | None:
    """Ground-truth identity for (page, panel), or None when not forced."""
    for key, names in forced_characters.items():
        try:
            page_sel, panel_sel = parse_only_panel(key)
        except ValueError:
            continue
        if page_matches(page_stem, page_sel) and panel_matches(panel_stem, panel_sel):
            return list(names)
    return None
