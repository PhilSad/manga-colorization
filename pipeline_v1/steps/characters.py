"""Pipeline stage 3: per-panel character detection.

Reads the extracted panels from `1_panels/<page>/`, calls the character
detector once per panel, and writes one JSON record per panel into
`2_characters/<page>/` plus a flat `summary.json`.
"""

from __future__ import annotations

import time
from pathlib import Path

from config import PipelineConfig
from run_context import RunContext, write_json
from util import SUPPORTED_IMAGE_SUFFIXES


def _panel_images(page_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in page_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def run_characters_step(
    ctx: RunContext,
    config: PipelineConfig,
    detector,  # CharacterDetector
) -> dict:
    """Run stage 3 for all panels extracted by stage 1+2."""
    panels_root = ctx.step_dir("panels")
    page_dirs = sorted(
        path for path in panels_root.iterdir() if path.is_dir()
    )
    if not page_dirs:
        raise ValueError(
            f"no extracted panels under {panels_root}; run the 'panels' step first"
        )

    characters_dir = ctx.step_dir("characters")
    records: list[dict] = []
    totals = {
        "api_calls": 0,
        "successful_calls": 0,
        "error_calls": 0,
        "unpriced_calls": 0,
        "total_latency_s": 0.0,
        "cost_usd": 0.0,
    }
    for page_dir in page_dirs:
        page = page_dir.name
        out_page_dir = characters_dir / page
        out_page_dir.mkdir(parents=True, exist_ok=True)
        for panel_path in _panel_images(page_dir):
            print(f"  characters: {page}/{panel_path.name} ...", flush=True)
            record = detector.detect(panel_path, config.refs_dir)
            doc = record.to_dict(panel_path, page=page)
            write_json(out_page_dir / f"{panel_path.stem}.json", doc)
            records.append(doc)
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
            if config.sleep_s:
                time.sleep(config.sleep_s)

    write_json(characters_dir / "summary.json", {"records": records, "totals": totals})
    return {"records": records, "totals": totals}


def load_characters_per_panel(
    ctx: RunContext, page: str
) -> dict[str, list[str]]:
    """Convenience for the colorize step: map panel stem -> character names,
    from the records written by run_characters_step."""
    import json

    out_page_dir = ctx.step_dir("characters") / page
    result: dict[str, list[str]] = {}
    for path in sorted(out_page_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        result[path.stem] = doc.get("characters", [])
    return result
