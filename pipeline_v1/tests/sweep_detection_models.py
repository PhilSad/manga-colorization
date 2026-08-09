#!/usr/bin/env python3
"""Detection model sweep over the integration-suite DET/OOV cases.

Runs the same stage-isolated panel detection the integration test performs
(committed pre-cropped panel -> one real OpenRouter call, V1 panel prompt)
across several OpenRouter vision models, each repeated N times to measure
run-to-run stability. Reports per-case pass counts, modal (most frequent)
detections, aggregate precision/recall, cost, and latency.

Output goes to `tests/output/YYYYMMDD-HHMMSS/` (gitignored, same convention
as integration sessions):
  - manifest.json  per-call records + totals
  - summary.json   per-model aggregation
  - results.md     markdown tables ready to paste into pipelines.md

Usage:
    .venv/bin/python pipeline_v1/tests/sweep_detection_models.py \
        [--models google/gemma-4-31b-it,openai/gpt-5.6-luna,xiaomi/mimo-v2.5] \
        [--reps 4] [--output-dir tests/output/YYYYMMDD-HHMMSS]
    .venv/bin/python pipeline_v1/tests/sweep_detection_models.py \
        --re-render tests/output/YYYYMMDD-HHMMSS/summary.json   # no API calls

Requires OPENROUTER_API_KEY in .env (paid calls: one per case per rep).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

from integration_support import (  # noqa: E402
    REFS_DIR,
    build_panel_detector,
    case_by_id,
    crop_path,
    load_fixture,
    write_json,
)
from test_integration_detection import DETECTION_CASES  # noqa: E402

DEFAULT_MODELS = [
    "google/gemma-4-31b-it",
    "openai/gpt-5.6-luna",
    "xiaomi/mimo-v2.5",
]
DEFAULT_REPS = 4
OUTPUT_ROOT = TESTS_DIR / "output"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated OpenRouter model ids")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--output-dir", default=None,
                        help="explicit output dir (default: timestamped)")
    parser.add_argument("--re-render", default=None,
                        help="regenerate results.md from an existing summary.json "
                             "(no API calls)")
    args = parser.parse_args()

    if args.re_render:
        summary = json.loads(Path(args.re_render).read_text(encoding="utf-8"))
        fixture = load_fixture()
        cases = [case_by_id(fixture, cid) for cid in summary["cases"]]
        out = Path(args.re_render).parent / "results.md"
        out.write_text(render_markdown(summary, cases), encoding="utf-8")
        print(f"re-rendered {out} from {args.re_render}")
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    fixture = load_fixture()
    cases = [case_by_id(fixture, cid) for cid in DETECTION_CASES]
    api_key = __import__("os").getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set in .env; aborting", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else \
        OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "detection-model-sweep",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "models": models,
            "reps": args.reps,
            "cases": [c["id"] for c in cases],
            "prompt": "panel-only (V1 panel prompt), same as test_integration_detection.py",
            "max_tokens": 1024,
            "temperature": 0.2,
        },
        "records": [],
        "totals": {},
    }
    print(f"output dir: {output_dir}")
    print(f"cases: {[c['id'] for c in cases]}")

    per_model: dict[str, dict] = {}
    for model in models:
        print(f"\n== model {model} ({args.reps} reps x {len(cases)} cases)", flush=True)
        detector = build_panel_detector(api_key, model=model)
        model_rows: list[dict] = []
        for rep in range(1, args.reps + 1):
            for case in cases:
                cid = case["id"]
                record = detector.detect(crop_path(cid), REFS_DIR)
                verdict = eval_case(case, record)
                row = {
                    "model": model,
                    "rep": rep,
                    "case": cid,
                    "expected": sorted(case["expected"]["characters"]),
                    "detected": sorted(record.characters),
                    "unknown_entries": record.unknown_entries,
                    "status": record.status,
                    **verdict,
                    "cost_usd": record.cost_usd,
                    "cost_source": record.cost_source,
                    "latency_s": round(record.latency_s, 3),
                    "model_returned": record.model_returned,
                    "error": record.error,
                }
                manifest["records"].append(row)
                model_rows.append(row)
                tag = "PASS" if verdict["pass"] else "fail"
                print(f"  {cid} rep{rep}: {sorted(record.characters)} {tag} "
                      f"(${record.cost_usd or 0:.6f}, {record.latency_s:.1f}s)")
            write_json(output_dir / "manifest.json", manifest)  # incremental

        by_case: dict[str, dict] = {}
        for case in cases:
            cid = case["id"]
            rows = [r for r in model_rows if r["case"] == cid]
            detections = Counter(tuple(r["detected"]) for r in rows)
            modal = max(detections.items(), key=lambda kv: (kv[1], kv[0]))[0]
            by_case[cid] = {
                "passes": sum(r["pass"] for r in rows),
                "chars_matches": sum(r["chars_match"] for r in rows),
                "modal_detection": list(modal),
                "modal_reps": detections[modal],
                "detection_histogram": {
                    ", ".join(d) or "(none)": n for d, n in
                    sorted(detections.items(), key=lambda kv: -kv[1])
                },
                "unknown_ok": sum(r["unknown_ok"] for r in rows),
                "errors": sum(1 for r in rows if r["status"] == "error"),
                "unparseable": sum(1 for r in rows if r["status"] == "unparseable"),
                "cost_usd": sum(r["cost_usd"] or 0 for r in rows),
            }
        tp = sum(r["tp"] for r in model_rows)
        fp = sum(r["fp"] for r in model_rows)
        fn = sum(r["fn"] for r in model_rows)
        total_expected = sum(r["expected_count"] for r in model_rows)
        passes = sum(r["pass"] for r in model_rows)
        per_model[model] = {
            "by_case": by_case,
            "aggregate": {
                "case_reps": len(model_rows),
                "passes": passes,
                "pass_rate": round(passes / len(model_rows), 4),
                "chars_matches": sum(r["chars_match"] for r in model_rows),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                "total_expected": total_expected,
                "errors": sum(1 for r in model_rows if r["status"] == "error"),
                "unparseable": sum(1 for r in model_rows if r["status"] == "unparseable"),
                "cost_usd": round(sum(r["cost_usd"] or 0 for r in model_rows), 8),
                "latency_avg_s": round(sum(r["latency_s"] for r in model_rows)
                                      / len(model_rows), 2),
            },
        }
        manifest["totals"][model] = per_model[model]["aggregate"]
        write_json(output_dir / "manifest.json", manifest)

    summary = {
        "models": models,
        "reps": args.reps,
        "cases": [c["id"] for c in cases],
        "per_model": per_model,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "results.md").write_text(
        render_markdown(summary, cases), encoding="utf-8"
    )
    print(f"\nwrote {output_dir / 'summary.json'} and {output_dir / 'results.md'}")
    return 0


def render_markdown(summary: dict, cases: list[dict]) -> str:
    """Markdown tables (per-case pass counts + modal detections + aggregates)."""
    models = summary["models"]
    reps = summary["reps"]
    per_model = summary["per_model"]

    lines: list[str] = []
    lines.append(f"### Detection model sweep — {', '.join(models)} "
                 f"({reps} reps x {len(cases)} cases, live OpenRouter, panel-only mode)")
    lines.append("")

    # --- table 1: pass counts per case -----------------------------------
    header = "| Case | expected | " + " | ".join(models) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (2 + len(models))) + "|")
    total_pass = [0] * len(models)
    total_case_reps = len(cases) * reps
    for case in cases:
        cid = case["id"]
        exp = ", ".join(case["expected"]["characters"]) or "—"
        row = [cid, exp]
        for m in models:
            info = per_model[m]["by_case"][cid]
            passes = info["passes"]
            total_pass[models.index(m)] += passes
            errs = info["errors"] + info["unparseable"]
            cell = f"{passes}/{reps}"
            if errs:
                cell += f" ({errs} parse-fail)"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **Total** | | " + " | ".join(
        f"**{total_pass[i]}/{total_case_reps}**" for i in range(len(models))) + " |")
    lines.append("")

    # --- table 2: modal detection per model -------------------------------
    lines.append("Modal (most frequent) detection per model over the reps:")
    lines.append("")
    header = "| Case | expected | " + " | ".join(
        f"{m.split('/')[-1]}" for m in models) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (2 + len(models))) + "|")
    for case in cases:
        cid = case["id"]
        exp = ", ".join(case["expected"]["characters"]) or "—"
        row = [cid, exp]
        for m in models:
            info = per_model[m]["by_case"][cid]
            modal = ", ".join(info["modal_detection"]) or "∅"
            row.append(f"{modal} ({info['modal_reps']}/{reps})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- table 3: aggregates ----------------------------------------------
    lines.append("Aggregates over all case-reps (TP/FP/FN on known characters):")
    lines.append("")
    lines.append("| model | pass | chars match | precision | recall | errors | cost | avg latency |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        a = per_model[m]["aggregate"]
        lines.append(
            f"| {m} | {a['passes']}/{a['case_reps']} "
            f"({a['pass_rate']:.0%}) | {a['chars_matches']}/{a['case_reps']} "
            f"| {a['precision'] if a['precision'] is not None else '—'} "
            f"| {a['recall'] if a['recall'] is not None else '—'} "
            f"| {a['errors']} | ${a['cost_usd']:.6f} "
            f"| {a['latency_avg_s']:.1f}s |"
        )
    lines.append("")
    total_cost = sum(per_model[m]["aggregate"]["cost_usd"] for m in models)
    lines.append(f"Total OpenRouter cost for the sweep: ${total_cost:.6f}.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
