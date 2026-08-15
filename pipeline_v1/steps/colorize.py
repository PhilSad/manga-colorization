"""Pipeline stage 4: colorization with a filtered atlas and explicit
canonical palettes.

For each extracted panel: load the characters detected for it (stage 3), build
a labelled atlas containing only those characters, render the explicit
canonical-palette instruction from the shared character profiles (task 0002),
and call the colorizer with the panel (+ atlas). Outputs go to
`3_colorized/<page>/`; the atlas used per panel is saved next to the output for
provenance.

`--only-panel` (task 0001) restricts which panels are colorized; when resuming,
the non-selected panels' colorized outputs are copied from the resume run so
the page can still be stitched.

Full-page gpt-image-2 mode (`--full-page`): each page has one synthetic panel
(`panel_0001` = the whole page) and the colorizer is `GptImage2Colorizer`.
The atlas characters come from `--atlas-source`: `detected` uses the
page-level VLM record; `cast` derives the full chapter cast
(`cast_key_for_page` / `--cast-key`) with zero VLM calls, falling back to the
full canonical roster when no chapter cast is derivable.

Verification loop (`--verify-attempts N`, see verify_loop.py): after each
colorization the panel is checked by the Luna verifier (strict structured
output). On a mismatch the fix prompt is output (console + per-panel files)
and the panel is re-colorized with it up to N attempts total. Every attempt
and every verdict is recorded in `<panel>.verify.json` (+ `attempt_<n>`
images, `<panel>.fix_prompt.txt`); the final attempt is copied to the
canonical output name so the stitch step is untouched.

Parallel colorization (`--worker-colorization N`): pages are independent units
of work — each page writes only to its own `3_colorized/<page>/` dir, so
worker threads never race on files. With N > 1 the per-page body runs in a
ThreadPoolExecutor and records/totals are merged back in the main thread as
futures complete (same pattern as the characters step). N=1 keeps the original
sequential path unchanged (per-panel progress bars + postfix).
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import STEP_DIRS, PipelineConfig
from run_context import RunContext, write_json
from selection import page_selected, panel_selected
from tqdm import tqdm
from util import SUPPORTED_IMAGE_SUFFIXES


def _panel_images(page_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in page_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and path.stem != "overlay"
        and path.stem != "detection_annotated"
        and "_atlas" not in path.stem
    )


def _new_totals() -> dict:
    totals = {
        "api_calls": 0,
        "successful_calls": 0,
        "error_calls": 0,
        "total_latency_s": 0.0,
    }
    # Verification loop counters (--verify-attempts); all zero when off.
    totals.update({
        "verify_calls": 0,
        "successful_verify_calls": 0,
        "verified_panels": 0,
        "mismatch_panels": 0,
        "verifier_error_panels": 0,
        "colorization_retries": 0,
        "verify_cost_usd": 0.0,
    })
    return totals


def run_colorize_step(
    ctx: RunContext,
    config: PipelineConfig,
    colorizer,  # Colorizer
    verifier=None,  # ColorVerifier (None -> no verification loop)
) -> dict:
    """Run stage 4 for all panels of all pages. Returns per-call records.

    When `verifier` is set and `config.verify_attempts > 0`, each panel goes
    through the verify loop (verify_loop.py) instead of a single colorize
    call."""
    from atlas import build_filtered_atlas  # noqa: F401 (kept for _process_page imports)
    from profiles import load_profiles, profiles_sha256

    profiles = {}
    try:
        profiles = load_profiles(config.profiles_file)
    except (OSError, ValueError):
        profiles = {}
    profiles_sha = profiles_sha256(config.profiles_file)

    panels_root = ctx.step_dir("panels")
    colorized_root = ctx.step_dir("colorize")
    page_dirs = sorted(path for path in panels_root.iterdir() if path.is_dir())
    if not page_dirs:
        raise ValueError("no extracted panels; run the 'panels' step first")

    extension = _extension(config)
    records: list[dict] = []
    totals = _new_totals()

    if config.worker_colorization <= 1:
        # Sequential path (unchanged behaviour: per-page progress bars).
        for page_dir in tqdm(
            page_dirs, desc="colorize: pages", unit="page", leave=False
        ):
            page_records, page_totals, _ = _process_page(
                ctx, config, colorizer, page_dir,
                profiles, profiles_sha, extension, verifier=verifier,
                progress=True,
            )
            records.extend(page_records)
            _merge_totals(totals, page_totals)
    else:
        # Parallel path: one worker per page; every write is page-scoped so
        # threads never race. Records/totals merged in completion order.
        pages_bar = tqdm(
            total=len(page_dirs), desc="colorize: pages", unit="page",
            leave=False,
        )
        try:
            with ThreadPoolExecutor(
                max_workers=config.worker_colorization,
                thread_name_prefix="colorize",
            ) as pool:
                futures = {
                    pool.submit(
                        _process_page, ctx, config, colorizer, page_dir,
                        profiles, profiles_sha, extension, verifier=verifier,
                        progress=False,
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

    write_json(colorized_root / "summary.json", {"records": records, "totals": totals})
    return {"records": records, "totals": totals}


def _process_page(
    ctx: RunContext,
    config: PipelineConfig,
    colorizer,
    page_dir: Path,
    profiles: dict,
    profiles_sha: str,
    extension: str,
    *,
    verifier=None,
    progress: bool,
) -> tuple[list[dict], dict, set[str]]:
    """Full per-page colorization work: character lookup, atlas build, calls,
    JSON writes. Safe to run from worker threads — every write goes to the
    page-specific `3_colorized/<page>/` directory (and `_copy_resumed_panels`
    only touches that page's dirs). Returns (page records, page totals delta,
    fresh panel stems)."""
    from atlas import build_filtered_atlas
    from profiles import palette_instruction, unknown_names
    from steps.characters import load_characters_per_panel

    page = page_dir.name
    page_totals = _new_totals()
    docs: list[dict] = []
    fresh_stems: set[str] = set()
    if config.only_panels and not page_selected(page, config.only_panels):
        return docs, page_totals, fresh_stems

    if config.full_page and config.atlas_source == "cast":
        # --atlas-source cast: the atlas is the full chapter cast (derived per
        # page from the chapter map / filename, or --cast-key); no VLM calls.
        # No derivable cast -> full canonical roster (characters.py convention).
        names_by_panel = _cast_names_by_panel(config, page_dir)
    else:
        try:
            names_by_panel = load_characters_per_panel(ctx, page)
        except Exception:  # noqa: BLE001 - stage 3 not run: colorize without atlas
            names_by_panel = {}
            if not config.full_page:
                print(
                    f"  colorize: no character records for {page}, "
                    "using panel-only colorization",
                    flush=True,
                )

    out_dir = ctx.step_dir("colorize") / page
    out_dir.mkdir(parents=True, exist_ok=True)

    all_panels = _panel_images(page_dir)
    if config.only_panels:
        panels = [
            p for p in all_panels
            if panel_selected(page, p.stem, config.only_panels)
        ]
    else:
        panels = all_panels

    panels_bar = None
    if progress:
        panels_bar = tqdm(
            panels, desc=f"colorize: {page}", unit="panel", leave=False
        )
    try:
        iterator = panels_bar or panels
        for panel_path in iterator:
            stem = panel_path.stem
            characters = names_by_panel.get(stem, [])
            atlas_path = out_dir / f"{stem}_atlas.jpg"
            atlas = build_filtered_atlas(
                characters, config.refs_dir, atlas_path,
                columns=config.atlas_columns,
            )
            output_path = out_dir / f"{stem}{extension}"
            palette = palette_instruction(characters, profiles)
            if panels_bar is not None:
                panels_bar.set_postfix(
                    panel=panel_path.name,
                    characters=",".join(characters) or "-",
                    atlas="yes" if atlas else "no",
                    palette="yes" if palette else "no",
                )
            if config.verify_attempts > 0 and verifier is not None:
                record, doc, verify_doc = _colorize_with_verify_loop(
                    ctx, config, colorizer, verifier, panel_path, atlas,
                    output_path, page, stem, palette, characters, profiles,
                    profiles_sha,
                )
                page_totals["api_calls"] += verify_doc["api_calls"]
                page_totals["successful_calls"] += verify_doc["successful_calls"]
                page_totals["error_calls"] += verify_doc["error_calls"]
                page_totals["total_latency_s"] = round(
                    page_totals["total_latency_s"] + verify_doc["total_latency_s"], 3
                )
                page_totals["verify_calls"] += verify_doc["verify_calls"]
                page_totals["successful_verify_calls"] += (
                    verify_doc["successful_verify_calls"]
                )
                page_totals["colorization_retries"] += verify_doc["colorization_retries"]
                page_totals["verify_cost_usd"] = round(
                    page_totals["verify_cost_usd"] + verify_doc["verify_cost_usd"], 8
                )
                outcome = verify_doc["outcome"]
                if outcome == "verified":
                    page_totals["verified_panels"] += 1
                elif outcome == "mismatch":
                    page_totals["mismatch_panels"] += 1
                elif outcome == "verifier_error":
                    page_totals["verifier_error_panels"] += 1
            else:
                record = colorizer.colorize(
                    panel_path, atlas, output_path, palette_instruction=palette
                )
                doc = record.to_dict(panel_path, atlas)
                doc["characters"] = characters
                doc["palette_instruction"] = palette
                doc["unknown_characters"] = unknown_names(characters, profiles)
                doc["profiles_sha256"] = profiles_sha
                doc["page"] = page
                page_totals["api_calls"] += 1
                if record.status == "ok":
                    page_totals["successful_calls"] += 1
                else:
                    page_totals["error_calls"] += 1
                page_totals["total_latency_s"] = round(
                    page_totals["total_latency_s"] + record.latency_s, 3
                )
            docs.append(doc)
            fresh_stems.add(stem)
    finally:
        if panels_bar is not None:
            panels_bar.close()

    _copy_resumed_panels(ctx, config, page, fresh_stems)
    return docs, page_totals, fresh_stems


def _colorize_with_verify_loop(
    ctx: RunContext,
    config: PipelineConfig,
    colorizer,
    verifier,
    panel_path: Path,
    atlas: Path | None,
    output_path: Path,
    page: str,
    stem: str,
    palette: str,
    characters: list[str],
    profiles: dict,
    profiles_sha: str,
) -> tuple[object, dict, dict]:
    """Run the per-panel verify loop (verify_loop.py) and persist the extra
    provenance: `<stem>.verify.json` (outcome, every attempt's colorize +
    verify records, fix prompt, verify cost) and `<stem>.fix_prompt.txt`
    (last fix prompt, when one was produced). Returns
    (final ColorizeRecord, pipeline doc dict, verify counters dict)."""
    from profiles import unknown_names  # noqa: F401 (used below; _process_page
                                        # imports palette_instruction itself)
    from verify_loop import run_verify_loop

    out_dir = output_path.parent
    result = run_verify_loop(
        colorizer,
        verifier,
        panel_path,
        atlas,
        output_path,
        palette_instruction=palette,
        max_attempts=max(1, config.verify_attempts),
    )

    record = result.colorize
    doc = record.to_dict(panel_path, atlas)
    doc["characters"] = characters
    doc["palette_instruction"] = palette
    doc["unknown_characters"] = unknown_names(characters, profiles)
    doc["profiles_sha256"] = profiles_sha
    doc["page"] = page
    doc["verify_loop"] = {
        "outcome": result.outcome,
        "max_attempts": max(1, config.verify_attempts),
        "fix_prompt": result.fix_prompt,
        "verify_calls": result.verify_calls,
        "successful_verify_calls": result.successful_verify_calls,
        "verify_cost_usd": result.verify_cost_usd,
        "attempts": result.attempts,
    }

    write_json(out_dir / f"{stem}.verify.json", doc["verify_loop"])
    if result.fix_prompt:
        (out_dir / f"{stem}.fix_prompt.txt").write_text(
            result.fix_prompt + "\n", encoding="utf-8"
        )

    # Per-attempt colorize counters so totals reflect every call (not just
    # the final one): attempts == colorize calls, one record each.
    api_calls = len(result.attempts)
    successful_calls = sum(
        1 for a in result.attempts if a["colorize"].get("status") == "ok"
    )
    total_latency_s = sum(
        a["colorize"].get("latency_s") or 0.0 for a in result.attempts
    )
    verify_doc = {
        "outcome": result.outcome,
        "api_calls": api_calls,
        "successful_calls": successful_calls,
        "error_calls": api_calls - successful_calls,
        "total_latency_s": round(total_latency_s, 3),
        "verify_calls": result.verify_calls,
        "successful_verify_calls": result.successful_verify_calls,
        "colorization_retries": result.colorization_retries,
        "verify_cost_usd": result.verify_cost_usd,
    }
    return record, doc, verify_doc


def _cast_names_by_panel(
    config: PipelineConfig, page_dir: Path
) -> dict[str, list[str]]:
    """--atlas-source cast: `{panel_0001: <chapter cast names>}` for a full-page
    run. The cast comes from --cast-key or cast_key_for_page; no derivable cast
    -> the full canonical roster (same convention as the detection strategies)."""
    from characters import canonical_characters, cast_key_for_page, cast_shortlist_for
    from run_context import read_json

    geometry = read_json(page_dir / "panels.json")
    page_path = Path(geometry["page_path"])
    key = config.cast_key or cast_key_for_page(
        page_path, config.chapter_casts_file, config.chapter_page_map_file
    )
    canonical = canonical_characters(config.refs_dir)
    if key is None:
        names = canonical
    else:
        canonical_lower = {name.lower(): name for name in canonical}
        names = [
            canonical_lower[name.lower()]
            for name in cast_shortlist_for(config.chapter_casts_file, key)
            if name.lower() in canonical_lower
        ]
        if not names:
            names = canonical
    panels = _panel_images(page_dir)
    stem = panels[0].stem if panels else "panel_0001"
    return {stem: names}


def _copy_resumed_panels(
    ctx: RunContext, config: PipelineConfig, page: str, fresh_stems: set[str]
) -> None:
    """Copy colorized panels for non-selected panels from the resume run so a
    targeted rerun can still stitch the full page (task 0001)."""
    if not config.resume:
        return
    source = Path(config.resume) / STEP_DIRS["colorize"] / page
    if not source.is_dir():
        return
    target = ctx.step_dir("colorize") / page
    for path in sorted(source.iterdir()):
        if path.name == "summary.json" or path.stem in fresh_stems:
            continue
        if path.is_dir():
            continue
        shutil.copy2(path, target / path.name)
    print(f"  colorize: reused {page} panels from resume run", flush=True)


def _merge_totals(target: dict, source: dict) -> None:
    for key in ("api_calls", "successful_calls", "error_calls"):
        target[key] += source[key]
    for key in (
        "verify_calls",
        "successful_verify_calls",
        "verified_panels",
        "mismatch_panels",
        "verifier_error_panels",
        "colorization_retries",
    ):
        target[key] += source.get(key, 0)
    target["total_latency_s"] = round(
        target["total_latency_s"] + source["total_latency_s"], 3
    )
    target["verify_cost_usd"] = round(
        target["verify_cost_usd"] + source.get("verify_cost_usd", 0.0), 8
    )


def _extension(config: PipelineConfig) -> str:
    return {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[config.output_format]
