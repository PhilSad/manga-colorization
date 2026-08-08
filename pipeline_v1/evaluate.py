#!/usr/bin/env python3
"""Evaluate a pipeline run against the V1.1 fixed failure set.

Writes two artifacts into `<run>/evaluation/`:

  report.json       machine-readable: per-case results + detection scoring
  color_review.md   human-review Markdown for the COL-* cases

Detection cases (DET-*, OOV-*) are scored automatically: character arrays are
compared as sets, with exact true positives / false positives / false
negatives and precision/recall. Color cases (COL-*) produce a review report
with the generated image and the fixture's expected output; the verdict is
always left to the user (`Pending user review`) — no code path in this tool
assigns a pass/fail. LAY-* and SIZE-* assert the recorded geometry and
request-size policy.

Usage:
    python pipeline_v1/evaluate.py --run <run_dir> [--cases <fixture.json>]
    python pipeline_v1/evaluate.py --validate-cases            # schema check only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from run_context import iso_now  # noqa: E402

DEFAULT_CASES = PIPELINE_DIR / "evaluation" / "v1_1_cases.json"
REPO_ROOT = PIPELINE_DIR.parent

PENDING = "Pending user review"
MISSING = "missing output"


# ---------------------------------------------------------------------------
# Fixture loading / validation

def validate_fixture(fixture: dict, repo_root: Path = REPO_ROOT,
                     check_inputs_exist: bool = False) -> list[str]:
    """Schema + resolvability checks. Returns a list of problems (empty when
    valid). `check_inputs_exist` additionally verifies non-generated input
    files exist (may be skipped when gitignored data dirs are absent)."""
    problems: list[str] = []
    if fixture.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    aliases = fixture.get("aliases")
    if not isinstance(aliases, dict) or not aliases:
        problems.append("missing non-empty 'aliases' map")
    else:
        for alias, relative in aliases.items():
            path = (repo_root / str(relative)).resolve()
            if not path.is_relative_to(repo_root):
                problems.append(f"alias {alias!r} escapes the repo")
            elif check_inputs_exist and not path.is_file():
                problems.append(f"alias {alias!r} -> {path} does not exist")

    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        return problems + ["missing non-empty 'cases' list"]
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        problems.append("case ids must be unique")
    for case in cases:
        case_id = case.get("id")
        stage = case.get("stage")
        if stage not in ("characters", "color", "layout", "size"):
            problems.append(f"{case_id}: unknown stage {stage!r}")
            continue
        entry = case.get("input")
        if not isinstance(entry, dict):
            problems.append(f"{case_id}: missing input")
            continue
        if "generate" in entry:
            gen = entry["generate"]
            if not isinstance(gen, dict) or gen.get("kind") not in ("white",):
                problems.append(f"{case_id}: unsupported generate recipe {gen!r}")
        else:
            source = entry.get("source_page")
            if source not in aliases:
                problems.append(f"{case_id}: unknown source_page {source!r}")
        if stage in ("characters", "color", "size") and not entry.get("panel"):
            problems.append(f"{case_id}: missing panel selector")
        expected = case.get("expected")
        if not isinstance(expected, dict) or not expected:
            problems.append(f"{case_id}: missing expected output")
        if stage == "characters":
            if not isinstance(expected.get("characters"), list):
                problems.append(f"{case_id}: expected.characters must be a list")
            if "unknown_present" not in expected:
                problems.append(f"{case_id}: expected.unknown_present missing")
            if not isinstance(case.get("baseline", {}).get("characters"), list):
                problems.append(f"{case_id}: baseline.characters must be a list")
        elif stage == "color":
            for key in ("characters", "required_colors", "forbidden_colors",
                        "preserve"):
                if not isinstance(expected.get(key), list) or not expected[key]:
                    problems.append(f"{case_id}: expected.{key} must be a non-empty list")
            if not entry.get("forced_characters"):
                problems.append(f"{case_id}: input.forced_characters missing")
        elif stage == "layout":
            if "panels" not in expected:
                problems.append(f"{case_id}: expected.panels missing")
        elif stage == "size":
            if "requested_size" not in expected:
                problems.append(f"{case_id}: expected.requested_size missing")
    return problems


def load_fixture(path: Path, check_inputs_exist: bool = False) -> dict:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_fixture(fixture, check_inputs_exist=check_inputs_exist)
    if problems:
        raise ValueError(f"invalid fixture {path}:\\n  " + "\\n  ".join(problems))
    return fixture


def resolve_alias(fixture: dict, alias: str) -> Path:
    return (REPO_ROOT / fixture["aliases"][alias]).resolve()


def page_stem_of(fixture: dict, case: dict) -> str:
    entry = case["input"]
    if "generate" in entry:
        return Path(entry["generate"]["name"]).stem
    return resolve_alias(fixture, entry["source_page"]).stem


# ---------------------------------------------------------------------------
# Per-case evaluation

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def evaluate_characters_case(run_dir: Path, fixture: dict, case: dict) -> dict:
    page = page_stem_of(fixture, case)
    panel = case["input"]["panel"]
    record_path = run_dir / "2_characters" / page / f"{panel}.json"
    doc = _read_json(record_path)
    expected = set(case["expected"]["characters"])
    baseline = set(case["baseline"].get("characters", []))

    if doc is None:
        return {
            "id": case["id"], "stage": "characters", "page": page, "panel": panel,
            "status": "missing record", "matches": None,
        }
    detected = set(doc.get("characters", []))
    unknown_entries = list(doc.get("unknown_entries", []))
    unknown_present = bool(unknown_entries)
    tp = len(detected & expected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not expected else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if not expected else 0.0)

    expected_unknown = set(case["expected"].get("expected_unknown_characters", []))
    unknown_handled = (
        unknown_present == bool(case["expected"]["unknown_present"])
        and (not expected_unknown or expected_unknown <= set(unknown_entries))
    )
    matches = detected == expected and unknown_handled

    return {
        "id": case["id"], "stage": "characters", "page": page, "panel": panel,
        "status": doc.get("status", "?"),
        "source": doc.get("source", "panel"),
        "detected": sorted(detected),
        "expected": sorted(expected),
        "baseline": sorted(baseline),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "unknown_present": unknown_present,
        "unknown_entries": unknown_entries,
        "expected_unknown": sorted(expected_unknown),
        "unknown_handled": unknown_handled,
        "matches": matches,
        "failure": case.get("failure"),
    }


def _find_generated(run_dir: Path, page: str, panel: str) -> Path | None:
    colorized_dir = run_dir / "3_colorized" / page
    if not colorized_dir.is_dir():
        return None
    for path in sorted(colorized_dir.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} \
                and path.stem == panel and "_atlas" not in path.stem:
            return path
    return None


def evaluate_color_case(run_dir: Path, fixture: dict, case: dict) -> dict:
    page = page_stem_of(fixture, case)
    panel = case["input"]["panel"]
    generated = _find_generated(run_dir, page, panel)
    input_crop = run_dir / "1_panels" / page / f"{panel}.png"
    atlas = run_dir / "3_colorized" / page / f"{panel}_atlas.jpg"
    return {
        "id": case["id"], "stage": "color", "page": page, "panel": panel,
        "forced_characters": case["input"].get("forced_characters", []),
        "review_status": PENDING if generated is not None else MISSING,
        "generated_image": _relative_report_link(run_dir, generated) if generated else None,
        "input_crop": _relative_report_link(run_dir, input_crop) if input_crop.is_file() else None,
        "atlas": _relative_report_link(run_dir, atlas) if atlas.is_file() else None,
        "expected": case["expected"],
        "failure": case.get("failure"),
    }


def evaluate_layout_case(run_dir: Path, fixture: dict, case: dict) -> dict:
    page = page_stem_of(fixture, case)
    panels_json = run_dir / "1_panels" / page / "panels.json"
    expected = case["expected"]
    doc = _read_json(panels_json)
    if doc is None:
        return {"id": case["id"], "stage": "layout", "page": page,
                "status": "missing panels.json", "matches": None}
    detections = doc.get("detections", [])
    blank_page = bool(doc.get("blank_page", False))
    skip_reason = doc.get("skip_reason")
    result = {
        "id": case["id"], "stage": "layout", "page": page,
        "panels": len(detections), "blank_page": blank_page,
        "skip_reason": skip_reason,
        "provenance": detections[0].get("provenance") if detections else None,
        "box": detections[0].get("box") if detections else None,
        "crop": detections[0].get("crop") if detections else None,
    }
    matches = None
    if "panels" in expected:
        matches = len(detections) == expected["panels"]
    if expected.get("box") is not None:
        matches = matches is not False and detections and \
            detections[0].get("box") == expected["box"]
    if expected.get("provenance"):
        matches = matches is not False and bool(detections) and \
            detections[0].get("provenance") == expected["provenance"]
    if "blank_page" in expected:
        matches = matches is not False and blank_page == expected["blank_page"]
        if expected.get("skip_reason"):
            matches = matches is not False and skip_reason == expected["skip_reason"]
    result["matches"] = matches
    return result


def evaluate_size_case(run_dir: Path, fixture: dict, case: dict) -> dict:
    page = page_stem_of(fixture, case)
    panel = case["input"]["panel"]
    expected = case["expected"]
    manifest = _read_json(run_dir / "manifest.json") or {}
    records = (manifest.get("steps", {}).get("colorize", {})
               .get("records", []))
    record = next(
        (r for r in records if r.get("page") == page and r.get("panel", "").startswith(panel)),
        None,
    )
    if record is None:
        return {"id": case["id"], "stage": "size", "page": page, "panel": panel,
                "status": "missing colorize record", "matches": None}
    requested = record.get("requested_size")
    requested_pixels = (requested["width"] * requested["height"]
                        if isinstance(requested, dict) else None)
    cap = record.get("max_megapixels") or case["input"].get("cap_megapixels")
    cap_pixels = (cap or 0) * 1_000_000
    exp_req = expected["requested_size"]
    matches = (
        isinstance(requested, dict)
        and requested == exp_req
        and record.get("cap_applied") is True
        and (requested_pixels or 0) <= cap_pixels
        and requested["width"] % 16 == 0 and requested["height"] % 16 == 0
    )
    return {
        "id": case["id"], "stage": "size", "page": page, "panel": panel,
        "original_size": record.get("original_size"),
        "requested_size": requested,
        "requested_pixels": requested_pixels,
        "scale": record.get("scale"),
        "cap_applied": record.get("cap_applied"),
        "max_megapixels": record.get("max_megapixels"),
        "expected_size": exp_req,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Markdown report

def _relative_report_link(run_dir: Path, path: Path) -> str:
    """Relative path from `<run>/evaluation/` to `path` (portable run dir)."""
    report_dir = run_dir / "evaluation"
    return os.path.relpath(path, report_dir)


def _expected_bullets(expected: dict) -> list[str]:
    lines = ["- " + "; ".join(expected["required_colors"])]
    lines.append("- Forbidden: " + "; ".join(expected["forbidden_colors"]))
    lines.append("- Preserve: "
                 + ", ".join(item.replace("_", " ") for item in expected["preserve"]))
    return lines


def _run_metadata(run_dir: Path) -> dict:
    manifest = _read_json(run_dir / "manifest.json") or {}
    configuration = manifest.get("configuration", {})
    return {
        "run_id": Path(manifest.get("run_directory") or run_dir).name,
        "run_directory": str(run_dir),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "model": configuration.get("vlm_model"),
        "seed": configuration.get("seed"),
        "detection_mode": configuration.get("detection_mode"),
        "flux_steps": configuration.get("flux_steps"),
        "guidance_scale": configuration.get("guidance_scale"),
        "lora_scale": configuration.get("lora_scale"),
        "max_megapixels": configuration.get("max_megapixels"),
        "colorizer_prompt_file": configuration.get("colorizer_prompt_file"),
        "prompt_hashes": manifest.get("prompt_hashes", {}),
    }


def _build_color_review(run_dir: Path, color_cases: list[dict], metadata: dict) -> str:
    lines = ["# V1.1 color review", ""]
    lines.append("- **Run:** `{}`".format(metadata["run_id"]))
    lines.append("- **Model/server:** {} (self-hosted FLUX, $0/call)"
                 .format(metadata.get("model")))
    lines.append("- **Detection mode:** {}".format(metadata.get("detection_mode")))
    lines.append("- **Seed:** {}".format(metadata.get("seed")))
    lines.append("- **FLUX settings:** {} steps, guidance {}, lora {}, "
                 "max {} MP".format(
                     metadata.get("flux_steps"), metadata.get("guidance_scale"),
                     metadata.get("lora_scale"), metadata.get("max_megapixels")))
    prompt_hash = (metadata.get("prompt_hashes", {}).get("colorizer_prompt_sha256") or "?")
    profiles_hash = (metadata.get("prompt_hashes", {}).get("profiles_sha256") or "?")
    lines.append("- **Colorizer prompt sha256:** {}".format(prompt_hash[:16]))
    lines.append("- **Profiles sha256:** {}".format(profiles_hash[:16]))
    lines.append("- **Generated:** {} — {}".format(
        metadata.get("started_at"), metadata.get("finished_at")))
    lines.append("")
    lines.append("Each section below is one generated variant of a COL-* case. "
                 "The evaluator never assigns a verdict: mark Pass/Fail yourself "
                 "and add notes.")
    lines.append("")

    for case in color_cases:
        lines.append("## {} — {}, {} panel {}".format(
            case["id"], ", ".join(case["forced_characters"]), case["page"],
            case["panel"]))
        lines.append("")
        lines.append("- **Input:** `{} / {}`".format(
            case["page"], case["panel"]))
        lines.append("- **Forced characters:** " + ", ".join(case["forced_characters"]))
        variant = "profiles-{}".format((metadata.get("prompt_hashes", {})
                                        .get("profiles_sha256") or "none")[:8])
        lines.append("- **Variant:** " + variant)
        lines.append("")
        if case["review_status"] == MISSING:
            lines.append("**Review status:** {} (no generated image in this run)"
                         .format(MISSING))
            lines.append("")
            continue
        lines.append("**Review status:** " + PENDING)
        lines.append("")
        lines.append("### Generated image")
        lines.append("")
        lines.append("![{} generated output]({})".format(case["id"],
                                                         case["generated_image"]))
        lines.append("")
        if case.get("input_crop"):
            lines.append("Monochrome input crop: ![input]({})".format(case["input_crop"]))
            lines.append("")
        if case.get("atlas"):
            lines.append("Atlas/reference used: ![atlas]({})".format(case["atlas"]))
            lines.append("")
        lines.append("### Expected output")
        lines.append("")
        lines.extend(_expected_bullets(case["expected"]))
        lines.append("")
        lines.append("### Review")
        lines.append("")
        lines.append("- [ ] Pass")
        lines.append("- [ ] Fail")
        lines.append("")
        lines.append("Notes:")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point

def run_evaluation(run_dir: Path, cases_path: Path = DEFAULT_CASES) -> dict:
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").is_file():
        raise ValueError(f"not a pipeline run directory: {run_dir}")
    fixture = load_fixture(cases_path, check_inputs_exist=False)

    detection_cases: list[dict] = []
    color_cases: list[dict] = []
    layout_cases: list[dict] = []
    size_cases: list[dict] = []
    for case in fixture["cases"]:
        stage = case["stage"]
        if stage == "characters":
            detection_cases.append(evaluate_characters_case(run_dir, fixture, case))
        elif stage == "color":
            color_cases.append(evaluate_color_case(run_dir, fixture, case))
        elif stage == "layout":
            layout_cases.append(evaluate_layout_case(run_dir, fixture, case))
        elif stage == "size":
            size_cases.append(evaluate_size_case(run_dir, fixture, case))

    scored = [c for c in detection_cases if c.get("matches") is not None]
    totals = {
        "tp": sum(c["tp"] for c in scored),
        "fp": sum(c["fp"] for c in scored),
        "fn": sum(c["fn"] for c in scored),
    }
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    totals["precision"] = round(tp / (tp + fp), 4) if (tp + fp) else None
    totals["recall"] = round(tp / (tp + fn), 4) if (tp + fn) else None
    totals["cases_scored"] = len(scored)

    metadata = _run_metadata(run_dir)
    report = {
        "run_directory": str(run_dir),
        "fixture": str(cases_path),
        "generated_at": iso_now(),
        "detection": {"cases": detection_cases, "totals": totals},
        "color": {
            "cases": color_cases,
            "verdict_mode": "human review only",
            "pending_count": sum(1 for c in color_cases
                                 if c["review_status"] == PENDING),
        },
        "layout": {"cases": layout_cases},
        "size": {"cases": size_cases},
        "metadata": metadata,
    }

    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (evaluation_dir / "color_review.md").write_text(
        _build_color_review(run_dir, color_cases, metadata), encoding="utf-8"
    )
    return report


def _print_report(report: dict) -> None:
    detection = report["detection"]
    totals = detection["totals"]
    print("Detection (auto-scored):")
    for case in detection["cases"]:
        if case.get("matches") is None:
            print(f"  {case['id']}: {case.get('status')}")
            continue
        print(f"  {case['id']}: tp={case['tp']} fp={case['fp']} fn={case['fn']} "
              f"matches={case['matches']}")
    print(f"  totals: tp={totals['tp']} fp={totals['fp']} fn={totals['fn']} "
          f"precision={totals['precision']} recall={totals['recall']}")
    color = report["color"]
    print(f"Color (human review only): {color['pending_count']} pending, "
          f"{len(color['cases']) - color['pending_count']} missing")
    for case in report["layout"]["cases"] + report["size"]["cases"]:
        print(f"  {case['id']}: matches={case.get('matches')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument("--run", type=Path, help="pipeline run directory to evaluate")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES,
                        help="fixture file (default: evaluation/v1_1_cases.json)")
    parser.add_argument("--validate-cases", action="store_true",
                        help="validate the fixture schema and exit")
    parser.add_argument("--check-inputs", action="store_true",
                        help="also require non-generated input files to exist")
    args = parser.parse_args(argv)

    if args.validate_cases:
        fixture = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        problems = validate_fixture(fixture, check_inputs_exist=args.check_inputs)
        if problems:
            print("fixture problems:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("fixture OK")
        return 0
    if args.run is None:
        parser.error("--run is required (or use --validate-cases)")
    report = run_evaluation(args.run, args.cases)
    _print_report(report)
    print(f"\nreport:        {args.run / 'evaluation' / 'report.json'}")
    print(f"color review:  {args.run / 'evaluation' / 'color_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
