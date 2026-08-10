"""Pipeline stage 3: character detection (per panel or per page).

The detection algorithm is selected by `--detection-mode` and implemented as
one strategy per mode (see `characters.DETECTION_STRATEGIES`):

- `panel`: one call per panel, the crop alone (V1 prompt).
- `page` (V1.1, task 0003): one paid call per page mapping numbered panels to
  canonical characters, with cropped-panel fallbacks for missing/invalid/
  uncertain results.
- `panel-page` (V1.2, default): one call per panel — the full page as context
  plus the target panel, with the same cropped-panel fallback as page mode.
- `panel-page-prev2`: panel-page that also sends the two preceding pages in
  reading order as extra story context (fewer when they do not exist; blank
  pages are skipped), so the model can use recent story events to disambiguate.
- `panel-page-cast`: panel-page with an automatically derived per-chapter cast
  shortlist (the page's chapter via `chapter_page_map.json`; `--cast-key`
  overrides the derivation): the panel-page prompt is rendered for that cast
  per page, thread-safely, so look-alike characters outside the chapter cast
  cannot be guessed (e.g. Flamme on p130 of ch. 5).

Every strategy returns a per-page `PageCharacterRecord`; the step writes one
JSON record per panel into `2_characters/<page>/` plus a flat `summary.json`
with call/cost totals split into primary and fallback calls. Panels with
forced ground-truth identities (`--force-characters`, task 0001) never make a
paid call. `--only-panel` restricts which panels are processed.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import PipelineConfig
from run_context import RunContext, write_json
from selection import forced_names_for, page_selected, panel_selected
from tqdm import tqdm
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
    detector,  # anything with strategy_for(mode) -> DetectionStrategy
) -> dict:
    """Run stage 3 for all panels extracted by stage 1+2.

    Pages are independent units of work: with `--workers N`
    (config.workers) they are processed concurrently by a
    ThreadPoolExecutor. Every per-page write goes to its own
    `2_characters/<page>/` directory, so worker threads never race on the
    same files; per-page results/totals are merged back in the main thread
    as futures complete. workers=1 keeps the original sequential
    behaviour, including per-panel progress bars and the `--sleep` page
    throttle (ignored when workers > 1).
    """
    from profiles import load_profiles
    print(f"characters: detection mode {config.detection_mode}", flush=True)

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

    strategy = detector.strategy_for(config.detection_mode)
    records: list[dict] = []
    totals = _new_totals()

    pages_bar = tqdm(
        total=len(page_dirs), desc="characters: pages", unit="page",
        leave=False,
    )
    try:
        if config.workers <= 1:
            for page_dir in page_dirs:
                page_records, page_totals, worked = _process_page(
                    ctx, config, strategy, page_dir, profiles
                )
                records.extend(page_records)
                _merge_totals(totals, page_totals)
                if config.sleep_s and worked:
                    time.sleep(config.sleep_s)
                pages_bar.update(1)
        else:
            with ThreadPoolExecutor(
                max_workers=config.workers, thread_name_prefix="characters"
            ) as pool:
                futures = {
                    pool.submit(
                        _process_page, ctx, config, strategy, page_dir, profiles
                    ): page_dir
                    for page_dir in page_dirs
                }
                for future in as_completed(futures):
                    page_records, page_totals, _ = future.result()
                    records.extend(page_records)
                    _merge_totals(totals, page_totals)
                    pages_bar.update(1)
    finally:
        pages_bar.close()

    write_json(
        ctx.step_dir("characters") / "summary.json",
        {"records": records, "totals": totals},
    )
    return {"records": records, "totals": totals}


def _process_page(
    ctx: RunContext,
    config: PipelineConfig,
    strategy,  # DetectionStrategy (characters.DETECTION_STRATEGIES[mode])
    page_dir: Path,
    profiles: dict,
) -> tuple[list[dict], dict, bool]:
    """Full per-page detection work: API calls, forced panels, JSON writes.

    Safe to run from worker threads — every write goes to the page-specific
    `2_characters/<page>/` directory. Returns (page records, page totals
    delta, worked); `worked` is True when the page had panels to process
    (used by the sequential-mode `--sleep` throttle).
    """
    from characters import forced_record

    page = page_dir.name
    page_totals = _new_totals()
    docs: list[dict] = []
    if config.only_panels and not page_selected(page, config.only_panels):
        return docs, page_totals, False
    all_panels = _panel_images(page_dir)
    if config.only_panels:
        panels = [
            p for p in all_panels
            if panel_selected(page, p.stem, config.only_panels)
        ]
    else:
        panels = all_panels
    if not panels:
        return docs, page_totals, False

    out_page_dir = ctx.step_dir("characters") / page
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
    worked = bool(needed or forced)

    if needed:
        page_docs, page_det_totals = _detect_page(
            ctx, config, strategy, page, page_dir, needed, out_page_dir
        )
        _merge_totals(page_totals, page_det_totals)
        docs.extend(page_docs)
    for stem, names in forced.items():
        panel_path = _find_panel(page_dir, stem)
        if panel_path is None:
            continue
        record = forced_record(panel_path, names, profiles=profiles)
        doc = record.to_dict(panel_path, page=page)
        write_json(out_page_dir / f"{stem}.json", doc)
        docs.append(doc)
        page_totals["forced_panels"] += 1

    return docs, page_totals, worked


def _detect_page(
    ctx: RunContext,
    config: PipelineConfig,
    strategy,
    page: str,
    page_dir: Path,
    needed: list[Path],
    out_page_dir: Path,
) -> tuple[list[dict], dict]:
    """Detect `needed` panels of one page through the mode's strategy (one
    uniform path for every `--detection-mode`; the strategy carries its
    provenance file name, progress label, and cast handling).

    Returns (per-panel docs, page totals delta)."""
    from run_context import read_json

    totals = _new_totals()
    geometry = read_json(page_dir / "panels.json")
    page_image = Path(geometry["page_path"])
    expected = [panel.stem for panel in needed]
    print(
        f"  characters: {page} {strategy.label} detection "
        f"(panels {expected}) ...",
        flush=True,
    )
    page_record = strategy.detect(
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

    # Provenance: the raw page-level answer + parse outcome (page-context
    # strategies only; panel mode writes no provenance file).
    if strategy.provenance is not None:
        write_json(out_page_dir / strategy.provenance, {
            "page": page,
            "expected_panels": expected,
            "cast_key": page_record.cast_key,
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
