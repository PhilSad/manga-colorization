#!/usr/bin/env python3
"""Detection model + mode sweep over the integration-suite DET/OOV cases.

Runs the same stage-isolated panel detection the integration test performs
(committed pre-cropped panel -> real OpenRouter call, V1 panel prompt) and
optionally the two page-context modes the pipeline supports, across several
OpenRouter vision models, each repeated N times to measure run-to-run
stability. Reports per-case pass counts, modal (most frequent) detections,
aggregate precision/recall, fallback rates, cost, and latency.

Modes (mirror pipeline_v1 `--detection-mode`):
  - panel:            V1, one call per panel, the committed crop alone
  - page:             V1.1, one call per page (numbered panels), per-panel
                      cropped fallbacks for missing/uncertain/unknown entries
  - panel-page:       V1.2, one call per panel: full page (target highlighted)
                      as context + the crop, same cropped fallbacks
  - panel-page-cast:  V1.2 with an automatically derived per-chapter cast
                      shortlist (cast_key_for_page: chapter_page_map.json ->
                      filename tag -> NNN- prefix), mirroring the pipeline's
                      `--detection-mode panel-page-cast`

Output goes to `tests/output/YYYYMMDD-HHMMSS/` (gitignored, same convention
as integration sessions):
  - manifest.json  per-call records + totals
  - summary.json   per-(mode, model) aggregation
  - results.md     markdown tables ready to paste into pipelines.md

Usage:
    .venv/bin/python pipeline_v1/tests/sweep_detection_models.py \
        [--models google/gemma-4-31b-it,openai/gpt-5.6-luna] \
        [--modes panel,page,panel-page] [--reps 4] \
        [--output-dir tests/output/YYYYMMDD-HHMMSS]
    .venv/bin/python pipeline_v1/tests/sweep_detection_models.py \
        --re-render tests/output/YYYYMMDD-HHMMSS/summary.json   # no API calls

Requires OPENROUTER_API_KEY in .env (paid calls: one per case per rep in
panel/panel-page mode, one per page per rep in page mode, plus fallbacks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TESTS_DIR.parent
REPO_ROOT = PIPELINE_DIR.parent
for path in (PIPELINE_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from characters import OpenRouterCharacterDetector  # noqa: E402
from detection import YoloPanelDetector  # noqa: E402
from extraction import save_panels  # noqa: E402
from integration_support import (  # noqa: E402
    PAGE_PROMPT_FILE,
    PANEL_PROMPT_FILE,
    REFS_DIR,
    case_by_id,
    crop_path,
    load_fixture,
    write_json,
)
from panel_ordering import reading_order  # noqa: E402
from PIL import Image  # noqa: E402
from test_integration_detection import DETECTION_CASES  # noqa: E402
from util import sha256  # noqa: E402

DEFAULT_MODELS = ["google/gemma-4-31b-it", "openai/gpt-5.6-luna"]
DEFAULT_MODES = ["panel", "page", "panel-page", "panel-page-cast"]
DEFAULT_REPS = 4
OUTPUT_ROOT = TESTS_DIR / "output"
PANEL_PAGE_PROMPT_FILE = PIPELINE_DIR / "prompt_panel_page.txt"
CHAPTER_CASTS_FILE = PIPELINE_DIR / "chapter_casts.json"
CHAPTER_PAGE_MAP_FILE = PIPELINE_DIR.parent / "frieren_wiki_dataset" / "chapter_page_map.json"


def eval_case(case: dict, record) -> dict:
    """Same assertions as test_integration_detection.py."""
    expected = set(case["expected"]["characters"])
    expected_unknown_present = case["expected"]["unknown_present"]
    expected_unknown = set(case["expected"].get("expected_unknown_characters", []))
    detected = set(record.characters)
    chars_match = (
        record.status not in ("error", "unparseable") and detected == expected
    )
    unknown_ok = bool(record.unknown_entries) == expected_unknown_present
    unknowns_ok = expected_unknown <= set(record.unknown_entries)
    tp = len(detected & expected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    return {
        "chars_match": chars_match,
        "unknown_ok": unknown_ok,
        "unknowns_ok": unknowns_ok,
        "pass": chars_match and unknown_ok and unknowns_ok,
        "tp": tp, "fp": fp, "fn": fn,
        "expected_count": len(expected),
    }


def panel_keys_for(n_panels: int) -> list[str]:
    return [f"panel_{i:04d}" for i in range(1, n_panels + 1)]


def build_page_dirs(fixture: dict, work_dir: Path) -> dict[str, Path]:
    """Per-alias page dirs with the real reading-order extraction (YOLO +
    `panel_ordering.reading_order` + `save_panels` + panels.json), the same
    code path `prepare_integration_data.py` and the pipeline use."""
    aliases: dict[str, list[dict]] = {}
    for cid in DETECTION_CASES:
        case = case_by_id(fixture, cid)
        aliases.setdefault(case["input"]["source_page"], []).append(case)

    detector = YoloPanelDetector()
    page_dirs: dict[str, Path] = {}
    for alias, _cases in aliases.items():
        page_path = (REPO_ROOT / fixture["aliases"][alias]).resolve()
        boxes = detector.detect(page_path)
        order = reading_order(boxes)
        ordered = [boxes[i] for i in order]
        if not ordered:
            raise RuntimeError(f"{alias}: 0 panels detected")
        page_dir = work_dir / alias
        page_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(page_path) as image:
            page = image.convert("RGB")
        records = save_panels(page, ordered, page_dir, inset=0)
        detections = [
            {
                "panel_index": index,
                "box": [round(b.x1), round(b.y1), round(b.x2), round(b.y2)],
                "confidence": round(b.confidence, 4),
                "crop": rec["filename"],
                "provenance": "yolo",
            }
            for index, (b, rec) in enumerate(zip(ordered, records), start=1)
        ]
        geometry = {
            "page": page_path.name,
            "page_path": str(page_path.resolve()),
            "page_sha256": sha256(page_path),
            "detection_order_into_reading_order": order,
            "detections": detections,
            "reading_order": [d["panel_index"] for d in detections],
            "blank_page": False,
            "skip_reason": None,
            "full_page_fallback": False,
        }
        write_json(page_dir / "panels.json", geometry)
        page_dirs[alias] = page_dir
        print(f"  page dir {alias}: {len(ordered)} panels")
    return page_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated OpenRouter model ids")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES),
                        help="comma-separated modes: panel,page,panel-page,"
                             "panel-page-cast")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel detection threads (1 = sequential); items "
                             "within one rep run concurrently, reps stay ordered")
    parser.add_argument("--output-dir", default=None,
                        help="explicit output dir (default: timestamped)")
    parser.add_argument("--re-render", default=None,
                        help="regenerate results.md from an existing summary.json "
                             "(no API calls)")
    parser.add_argument("--recompute-from", default=None,
                        help="rebuild summary.json + results.md from a manifest.json "
                             "(recomputes aggregates with the current logic; no API calls)")
    args = parser.parse_args()

    if args.recompute_from:
        return _recompute(Path(args.recompute_from))

    if args.re_render:
        summary = json.loads(Path(args.re_render).read_text(encoding="utf-8"))
        fixture = load_fixture()
        cases = [case_by_id(fixture, cid) for cid in summary["cases"]]
        out = Path(args.re_render).parent / "results.md"
        out.write_text(render_markdown(summary, cases), encoding="utf-8")
        print(f"re-rendered {out} from {args.re_render}")
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    fixture = load_fixture()
    cases = [case_by_id(fixture, cid) for cid in DETECTION_CASES]
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set in .env; aborting", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else \
        OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "detection-model-mode-sweep",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "models": models,
            "modes": modes,
            "reps": args.reps,
            "workers": args.workers,
            "cases": [c["id"] for c in cases],
            "prompt": ("V1 panel prompt (panel), page prompt (page), "
                       "panel-page prompt (panel-page); same as the pipeline"),
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        "records": [],
        "totals": {},
    }
    print(f"output dir: {output_dir}")
    print(f"cases: {[c['id'] for c in cases]} | models: {models} | modes: {modes}")

    page_dirs = build_page_dirs(fixture, output_dir / ".pages")
    # Map each case to its alias page (page/panel-page modes call per page).
    cases_by_alias: dict[str, list[dict]] = {}
    for case in cases:
        cases_by_alias.setdefault(case["input"]["source_page"], []).append(case)
    aliases = sorted(cases_by_alias)
    per_mode_model: dict[tuple[str, str], dict] = {}

    for mode in modes:
        for model in models:
            print(f"\n== mode={mode} model={model} "
                  f"({args.reps} reps x {len(cases)} cases, "
                  f"workers={args.workers})", flush=True)
            detector = OpenRouterCharacterDetector(
                model=model, api_key=api_key,
                chapter_casts_file=CHAPTER_CASTS_FILE,
                workers=args.workers,
            )
            detector.prepare(
                REFS_DIR,
                prompt_file=PAGE_PROMPT_FILE,
                panel_prompt_file=PANEL_PROMPT_FILE,
                panel_page_prompt_file=PANEL_PAGE_PROMPT_FILE,
            )
            rows: list[dict] = []
            page_totals: dict[str, dict] = {}

            pool = (ThreadPoolExecutor(
                max_workers=args.workers, thread_name_prefix="sweep"
            ) if args.workers > 1 else None)
            try:
                for rep in range(1, args.reps + 1):
                    # Items within one rep are independent (distinct crops, or
                    # distinct pages with their own page dirs), so reps stay
                    # ordered and no two threads touch the same annotated page.
                    items: list = list(cases) if mode == "panel" else list(aliases)
                    if pool is None:
                        for item in items:
                            item_rows, pt = _run_item(
                                mode, model, rep, item, detector,
                                fixture, page_dirs, cases_by_alias,
                            )
                            rows.extend(item_rows)
                            _merge_page_totals(page_totals, pt)
                    else:
                        futures = {
                            pool.submit(
                                _run_item, mode, model, rep, item, detector,
                                fixture, page_dirs, cases_by_alias,
                            ): item for item in items
                        }
                        for future in as_completed(futures):
                            item_rows, pt = future.result()
                            rows.extend(item_rows)
                            _merge_page_totals(page_totals, pt)
            finally:
                if pool is not None:
                    pool.shutdown()

            manifest["records"].extend(rows)
            per_mode_model[(mode, model)] = _aggregate(rows)
            manifest["totals"][f"{mode}|{model}"] = \
                per_mode_model[(mode, model)]["aggregate"]
            write_json(output_dir / "manifest.json", manifest)

            agg = per_mode_model[(mode, model)]["aggregate"]
            print(f"  -> {agg['passes']}/{agg['case_reps']} pass "
                  f"(P={agg['precision']} R={agg['recall']}, "
                  f"${agg['cost_usd']:.4f}, {agg['latency_avg_s']:.1f}s avg)",
                  flush=True)
            for case in cases:
                info = per_mode_model[(mode, model)]["by_case"][case["id"]]
                print(f"     {case['id']}: {info['passes']}/{args.reps} "
                      f"modal={info['modal_detection'] or '∅'}", flush=True)

    summary = {
        "kind": "detection-model-mode-sweep",
        "models": models,
        "modes": modes,
        "reps": args.reps,
        "cases": [c["id"] for c in cases],
        "page_totals": {f"{mode}|{model}|{alias}": pt
                        for mode in modes for model in models
                        for alias, pt in page_totals.items()},
        "per_mode_model": {
            f"{mode}|{model}": agg for (mode, model), agg in per_mode_model.items()
        },
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "results.md").write_text(
        render_markdown(summary, cases), encoding="utf-8"
    )
    print(f"\nwrote {output_dir / 'summary.json'} and {output_dir / 'results.md'}")
    return 0


def _recompute(manifest_path: Path) -> int:
    """Rebuild summary.json + results.md from a saved manifest.json, using
    the current aggregation logic (e.g. after a bug fix). No API calls."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    rows = manifest["records"]
    fixture = load_fixture()
    cases = [case_by_id(fixture, cid) for cid in config["cases"]]
    models, modes = config["models"], config["modes"]
    per = {}
    for mode in modes:
        for model in models:
            key = f"{mode}|{model}"
            per[key] = _aggregate(
                [r for r in rows if r["mode"] == mode and r["model"] == model]
            )
    summary = {
        "kind": manifest.get("kind", "detection-model-mode-sweep"),
        "models": models,
        "modes": modes,
        "reps": config["reps"],
        "cases": config["cases"],
        "per_mode_model": per,
    }
    out_dir = manifest_path.parent
    write_json(out_dir / "summary.json", summary)
    (out_dir / "results.md").write_text(
        render_markdown(summary, cases), encoding="utf-8"
    )
    print(f"recomputed {out_dir / 'summary.json'} + {out_dir / 'results.md'}"
          f" from {manifest_path.name}")
    return 0


def _missing_record(panel_key: str):
    from characters import CharacterRecord
    return CharacterRecord(
        status="error", characters=[], unknown_entries=[], response_text="",
        usage={}, cost_usd=None, cost_source="missing", latency_s=0.0,
        model_returned=None, attempts=0,
        error=f"missing panel record {panel_key}", source="missing",
    )


def _merge_page_totals(page_totals: dict, pt: dict | None) -> None:
    """Merge one item's page-call totals into the mode-level map (main thread)."""
    if not pt:
        return
    for alias, delta in pt.items():
        entry = page_totals.setdefault(alias, {
            "page_calls": 0, "fallback_calls": 0, "cost_usd": 0.0, "latency_s": 0.0,
        })
        entry["page_calls"] += delta["page_calls"]
        entry["fallback_calls"] += delta["fallback_calls"]
        entry["cost_usd"] = round(entry["cost_usd"] + delta["cost_usd"], 8)
        entry["latency_s"] = round(entry["latency_s"] + delta["latency_s"], 3)


def _run_item(
    mode: str,
    model: str,
    rep: int,
    item,
    detector,
    fixture: dict,
    page_dirs: dict[str, Path],
    cases_by_alias: dict[str, list[dict]],
) -> tuple[list[dict], dict | None]:
    """One work item: a (rep, case) panel call or a (rep, alias) page-scoped
    call. Returns (row dicts, page-totals delta or None). Runs in a worker
    thread: items of one rep touch distinct crops / distinct page dirs, so no
    two threads write the same annotated page; the detector's prompt reads are
    lock-guarded and panel-page prompts render per call."""
    rows: list[dict] = []
    if mode == "panel":
        case = item
        cid = case["id"]
        record = detector.detect(crop_path(cid), REFS_DIR)
        verdict = eval_case(case, record)
        rows.append({
            "mode": mode, "model": model, "rep": rep, "case": cid,
            "expected": sorted(case["expected"]["characters"]),
            "detected": sorted(record.characters),
            "unknown_entries": record.unknown_entries,
            "status": record.status, "source": record.source,
            "page_parse_ok": None, "fallback": False, "cast_key": None,
            **verdict,
            "cost_usd": record.cost_usd, "cost_source": record.cost_source,
            "latency_s": round(record.latency_s, 3),
            "model_returned": record.model_returned, "error": record.error,
        })
        return rows, None

    alias = item
    page_dir = page_dirs[alias]
    page_path = (REPO_ROOT / fixture["aliases"][alias]).resolve()
    geometry = json.loads((page_dir / "panels.json").read_text(encoding="utf-8"))
    expected = panel_keys_for(len(geometry["detections"]))
    cast_key = None
    if mode == "panel-page-cast":
        from characters import cast_key_for_page

        cast_key = cast_key_for_page(
            page_path, CHAPTER_CASTS_FILE, CHAPTER_PAGE_MAP_FILE
        )
        if hasattr(detector, "set_cast"):
            detector.set_cast(cast_key)
    if cast_key is not None:
        page_record = detector.detect_panels_with_page(
            page_path, page_dir, expected, REFS_DIR, cast_key=cast_key
        )
    else:
        method = "detect_page" if mode == "page" else "detect_panels_with_page"
        page_record = getattr(detector, method)(
            page_path, page_dir, expected, REFS_DIR
        )
    pt = {alias: {
        "page_calls": page_record.page_calls,
        "fallback_calls": page_record.fallback_calls,
        "cost_usd": page_record.cost_usd or 0.0,
        "latency_s": page_record.total_latency_s,
    }}
    for case in cases_by_alias[alias]:
        cid = case["id"]
        panel_key = case["input"]["panel"]
        record = page_record.panels.get(panel_key)
        if record is None:
            record = _missing_record(panel_key)
        verdict = eval_case(case, record)
        if mode == "page":
            # cost/latency attributed at page level: share across the page's
            # panels (the page call serves all of them).
            share = max(1, len(expected))
            cost = (page_record.cost_usd or 0.0) / share
            lat = page_record.total_latency_s / share
        else:
            cost = record.cost_usd
            lat = record.latency_s
        rows.append({
            "mode": mode, "model": model, "rep": rep, "case": cid,
            "expected": sorted(case["expected"]["characters"]),
            "detected": sorted(record.characters),
            "unknown_entries": record.unknown_entries,
            "status": record.status, "source": record.source,
            "page_parse_ok": page_record.page_parse_ok,
            "fallback": record.source == "fallback",
            "cast_key": cast_key,
            **verdict,
            "cost_usd": round(cost, 8) if cost is not None else None,
            "cost_source": (record.cost_source if mode == "panel-page"
                            else "page-level-share"),
            "latency_s": round(lat, 3),
            "model_returned": record.model_returned, "error": record.error,
        })
    return rows, pt


def _aggregate(rows: list[dict]) -> dict:
    by_case: dict[str, dict] = {}
    for case_id in DETECTION_CASES:
        r = [x for x in rows if x["case"] == case_id]
        if not r:
            continue
        detections = Counter(tuple(x["detected"]) for x in r)
        modal = max(detections.items(), key=lambda kv: (kv[1], kv[0]))[0]
        by_case[case_id] = {
            "passes": sum(x["pass"] for x in r),
            "chars_matches": sum(x["chars_match"] for x in r),
            "modal_detection": list(modal),
            "modal_reps": detections[modal],
            "detection_histogram": {
                ", ".join(d) or "(none)": n for d, n in
                sorted(detections.items(), key=lambda kv: -kv[1])
            },
            "unknown_ok": sum(x["unknown_ok"] for x in r),
            "fallbacks": sum(1 for x in r if x.get("fallback")),
            "errors": sum(1 for x in r if x["status"] in ("error", "unparseable")
                          or (x.get("mode") == "page"
                              and x.get("page_parse_ok") is False)),
            "cost_usd": sum(x["cost_usd"] or 0 for x in r),
        }
    tp = sum(x["tp"] for x in rows)
    fp = sum(x["fp"] for x in rows)
    fn = sum(x["fn"] for x in rows)
    n = len(rows)
    return {
        "by_case": by_case,
        "aggregate": {
            "case_reps": n,
            "passes": sum(x["pass"] for x in rows),
            "pass_rate": round(sum(x["pass"] for x in rows) / n, 4) if n else None,
            "chars_matches": sum(x["chars_match"] for x in rows),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "total_expected": sum(x["expected_count"] for x in rows),
            "fallbacks": sum(1 for x in rows if x.get("fallback")),
            "errors": sum(1 for x in rows if x["status"] in ("error", "unparseable")
                          or (x.get("mode") == "page"
                              and x.get("page_parse_ok") is False)),
            "cost_usd": round(sum(x["cost_usd"] or 0 for x in rows), 8),
            "latency_avg_s": round(sum(x["latency_s"] for x in rows) / n, 2) if n else None,
        },
    }


def render_markdown(summary: dict, cases: list[dict]) -> str:
    """Markdown tables (per-case pass counts + modal detections + aggregates)."""
    models = summary["models"]
    modes = summary.get("modes", ["panel"])
    reps = summary["reps"]
    per = summary.get("per_mode_model", {})
    # Backwards-compatible with the panel-only sweep summary format.
    if not per and "per_model" in summary:
        per = {f"panel|{m}": v for m, v in summary["per_model"].items()}
    cells = [(mode, model) for mode in modes for model in models]
    col_labels = [f"{mode}·{model.split('/')[-1]}" for mode, model in cells]

    lines: list[str] = []
    lines.append(f"### Detection sweep — modes x models "
                 f"({', '.join(modes)} x {', '.join(m.split('/')[-1] for m in models)}, "
                 f"{reps} reps x {len(cases)} cases, live OpenRouter)")
    lines.append("")

    # --- table 1: pass counts per case -----------------------------------
    lines.append(f"Per-case pass count over the {reps} reps (pass = exact "
                 f"character set + `unknown_present` semantics):")
    lines.append("")
    lines.append("| Case | expected | " + " | ".join(col_labels) + " |")
    lines.append("|" + "|".join(["---"] * (2 + len(cells))) + "|")
    total_pass = [0] * len(cells)
    total_case_reps = len(cases) * reps
    for case in cases:
        cid = case["id"]
        exp = ", ".join(case["expected"]["characters"]) or "—"
        row = [cid, exp]
        for i, (mode, model) in enumerate(cells):
            info = per[f"{mode}|{model}"]["by_case"][cid]
            passes = info["passes"]
            total_pass[i] += passes
            fbks = info.get("fallbacks", 0)
            cell = f"{passes}/{reps}"
            if fbks:
                cell += f" ({fbks} fbk)"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **Total** | | " + " | ".join(
        f"**{total_pass[i]}/{total_case_reps}**" for i in range(len(cells))) + " |")
    lines.append("")

    # --- table 2: modal detection per mode/model --------------------------
    lines.append("Modal (most frequent) detection per mode/model over the reps:")
    lines.append("")
    lines.append("| Case | expected | " + " | ".join(col_labels) + " |")
    lines.append("|" + "|".join(["---"] * (2 + len(cells))) + "|")
    for case in cases:
        cid = case["id"]
        exp = ", ".join(case["expected"]["characters"]) or "—"
        row = [cid, exp]
        for mode, model in cells:
            info = per[f"{mode}|{model}"]["by_case"][cid]
            modal = ", ".join(info["modal_detection"]) or "∅"
            row.append(f"{modal} ({info['modal_reps']}/{reps})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- table 3: aggregates ----------------------------------------------
    lines.append("Aggregates over all case-reps (TP/FP/FN on known characters; "
                 "fallbacks = reps resolved by a cropped-panel call; "
                 "parse-fail = unparseable/error calls or unparsed page answers):")
    lines.append("")
    lines.append("| mode · model | pass | precision | recall | fallbacks | "
                 "parse-fail | cost | avg latency |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for mode, model in cells:
        a = per[f"{mode}|{model}"]["aggregate"]
        lines.append(
            f"| {mode} · {model.split('/')[-1]} "
            f"| {a['passes']}/{a['case_reps']} "
            f"({a['pass_rate']:.0%}) "
            f"| {a['precision'] if a['precision'] is not None else '—'} "
            f"| {a['recall'] if a['recall'] is not None else '—'} "
            f"| {a.get('fallbacks', 0)} | {a.get('errors', 0)} "
            f"| ${a['cost_usd']:.6f} | {a['latency_avg_s']:.1f}s |"
        )
    lines.append("")
    total_cost = sum(per[f"{m}|{md}"]["aggregate"]["cost_usd"]
                     for m in modes for md in models)
    lines.append(f"Total OpenRouter cost for the sweep: ${total_cost:.6f}.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
