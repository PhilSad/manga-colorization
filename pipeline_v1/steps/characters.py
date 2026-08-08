"""Pipeline stage 3: character detection (per panel or per page).

Default (`detection_mode="page"`, task 0003): one paid call per page mapping
numbered panels to canonical characters, with cropped-panel fallbacks for
missing/invalid/uncertain results. `detection_mode="panel"` keeps the V1
one-call-per-panel behaviour. Panels with forced ground-truth identities
(`--force-characters`, task 0001) never make a paid call. `--only-panel`
restricts which panels are processed.

Writes one JSON record per panel into `2_characters/<page>/` plus a flat
`summary.json` with call/cost totals split into page and fallback calls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import PipelineConfig
from run_context import RunContext, write_json
from selection import forced_names_for, page_selected, panel_selected
from util import SUPPORTED_IMAGE_SUFFIXES


def _panel_images(page_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in page_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and path.stem != "overlay"
        and path.stem != "detection_annotated"
    )


def _new_totals() -> dict:
    return {
        "api_calls": 0,
        "successful_calls": 0,
        "error_calls": 0,
        "unpriced_calls": 0,
        "total_latency_s": 0.0,
        "cost_usd": 0.0,
        "page_calls": 0,
        "fallback_calls": 0,
        "forced_panels": 0,
    }


def run_characters_step(
    ctx: RunContext,
    config: PipelineConfig,
    detector,  # CharacterDetector | PageCharacterDetector
) -> dict:
    """Run stage 3 for all panels extracted by stage 1+2."""
    from characters import forced_record
    from profiles import load_profiles

    panels_root = ctx.step_dir("panels")
    page_dirs = sorted(path for path in panels_root.iterdir() if path.is_dir())
    if not page_dirs:
        raise ValueError(
            f"no extracted panels under {panels_root}; run the 'panels' step first"
        )

    profiles = {}
    try:
        profiles = load_profiles(config.profiles_file)
    except (OSError, ValueError):
        profiles = {}

    characters_dir = ctx.step_dir("characters")
    records: list[dict] = []
    totals = _new_totals()
    for page_dir in page_dirs:
        page = page_dir.name
        if config.only_panels and not page_selected(page, config.only_panels):
            continue
        all_panels = _panel_images(page_dir)
        if config.only_panels:
            panels = [
                p for p in all_panels
                if panel_selected(page, p.stem, config.only_panels)
            ]
        else:
            panels = all_panels
        if not panels:
            continue

        out_page_dir = characters_dir / page
        out_page_dir.mkdir(parents=True, exist_ok=True)

        forced: dict[str, list[str]] = {}
        needed: list[Path] = []
        for panel_path in panels:
            names = forced_names_for(
                config.forced_characters, page, panel_path.stem
            )
            if names is not None:
                forced[panel_path.stem] = names
            else:
                needed.append(panel_path)

        page_capable = config.detection_mode == "page" and hasattr(
            detector, "detect_page"
        )
        if page_capable and needed:
            docs, page_totals = _detect_page(
                ctx, config, detector, page, page_dir, needed, out_page_dir
            )
            _merge_totals(totals, page_totals)
            records.extend(docs)
        elif needed:
            for panel_path in needed:
                print(
                    f"  characters: {page}/{panel_path.name} ...", flush=True
                )
                record = detector.detect(panel_path, config.refs_dir)
                doc = record.to_dict(panel_path, page=page)
                write_json(out_page_dir / f"{panel_path.stem}.json", doc)
                records.append(doc)
                _count_record(totals, record)

        for stem, names in forced.items():
            panel_path = _find_panel(page_dir, stem)
            if panel_path is None:
                continue
            record = forced_record(panel_path, names, profiles=profiles)
            doc = record.to_dict(panel_path, page=page)
            write_json(out_page_dir / f"{stem}.json", doc)
            records.append(doc)
            totals["forced_panels"] += 1

        if config.sleep_s and (needed or forced):
            time.sleep(config.sleep_s)

    write_json(characters_dir / "summary.json", {"records": records, "totals": totals})
    return {"records": records, "totals": totals}


def _detect_page(
    ctx: RunContext,
    config: PipelineConfig,
    detector,
    page: str,
    page_dir: Path,
    needed: list[Path],
    out_page_dir: Path,
) -> tuple[list[dict], dict]:
    """One page-level detection call for `needed` panels + fallbacks."""
    from run_context import read_json

    totals = _new_totals()
    geometry = read_json(page_dir / "panels.json")
    page_image = Path(geometry["page_path"])
    expected = [panel.stem for panel in needed]
    print(
        f"  characters: {page} page-level detection "
        f"(panels {expected}) ...",
        flush=True,
    )
    page_record = detector.detect_page(
        page_image, page_dir, expected, config.refs_dir
    )
    totals["page_calls"] += page_record.page_calls
    totals["fallback_calls"] += page_record.fallback_calls
    totals["cost_usd"] = round(
        totals["cost_usd"] + page_record.cost_usd, 8
    )
    totals["total_latency_s"] = round(
        totals["total_latency_s"] + page_record.total_latency_s, 3
    )
    totals["api_calls"] += page_record.page_calls + page_record.fallback_calls
    totals["unpriced_calls"] += page_record.unpriced_calls
    if page_record.error is not None:
        totals["error_calls"] += page_record.page_calls

    # Provenance: the raw page-level answer + parse outcome.
    write_json(out_page_dir / "page_call.json", {
        "page": page,
        "expected_panels": expected,
        "status": page_record.status,
        "page_calls": page_record.page_calls,
        "fallback_calls": page_record.fallback_calls,
        "cost_usd": page_record.cost_usd,
        "latency_s": round(page_record.total_latency_s, 3),
        "parse_ok": page_record.page_parse_ok,
        "response_text": page_record.page_response_text,
        "error": page_record.error,
    })

    docs: list[dict] = []
    for panel_path in needed:
        stem = panel_path.stem
        record = page_record.panels.get(stem)
        if record is None:
            continue
        doc = record.to_dict(panel_path, page=page)
        write_json(out_page_dir / f"{stem}.json", doc)
        docs.append(doc)
        if record.status in ("ok", "ok-with-unknown"):
            totals["successful_calls"] += 1
        elif record.status == "error":
            totals["error_calls"] += 1
    return docs, totals


def _count_record(totals: dict, record) -> None:
    totals["api_calls"] += 1
    if record.status in ("ok", "ok-with-unknown"):
        totals["successful_calls"] += 1
    else:
        totals["error_calls"] += 1
    totals["total_latency_s"] = round(
        totals["total_latency_s"] + record.latency_s, 3
    )
    if record.cost_usd is not None:
        totals["cost_usd"] = round(totals["cost_usd"] + record.cost_usd, 8)
    else:
        totals["unpriced_calls"] += 1


def _merge_totals(target: dict, source: dict) -> None:
    for key in ("api_calls", "successful_calls", "error_calls",
                "unpriced_calls", "page_calls", "fallback_calls",
                "forced_panels"):
        target[key] += source[key]
    target["total_latency_s"] = round(
        target["total_latency_s"] + source["total_latency_s"], 3
    )
    target["cost_usd"] = round(target["cost_usd"] + source["cost_usd"], 8)


def _find_panel(page_dir: Path, stem: str) -> Path | None:
    for path in page_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES and path.stem == stem:
            return path
    return None


def load_characters_per_panel(
    ctx: RunContext, page: str
) -> dict[str, list[str]]:
    """Convenience for the colorize step: map panel stem -> character names,
    from the records written by run_characters_step."""
    out_page_dir = ctx.step_dir("characters") / page
    result: dict[str, list[str]] = {}
    for path in sorted(out_page_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        result[path.stem] = doc.get("characters", [])
    return result
