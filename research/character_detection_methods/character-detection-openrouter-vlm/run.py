#!/usr/bin/env python3
"""Detect which reference characters appear in manga panels.

Sends ONE panel image per call to an OpenRouter chat model (vision-capable
where supported) and asks for a structured JSON output:
{"characters": ["CanonicalName", ...]} where names come from a reference list
derived from data/refs. Each model x panel result is saved as JSON under the
run directory; a flat CSV and a nested summary are written at the end.

Intended for paid OpenRouter chat models (user-funded). Per-call cost is read
from OpenRouter's usage.cost when present, otherwise computed from the model's
published per-token pricing. Failures are recorded per call rather than
aborting the run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from openai import APIError, APIConnectionError, BadRequestError, OpenAI, RateLimitError
from PIL import Image

METHOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = METHOD_DIR.parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "panels"
DEFAULT_REFS_DIR = REPO_ROOT / "data" / "refs"
API_BASE = "https://openrouter.ai/api/v1"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048
MAX_ATTEMPTS = 8
BASE_BACKOFF_S = 5.0

DEFAULT_MODELS = [
    "google/gemma-4-31b-it",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-4-26b-a4b-it",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

# Short neutral hints from the anime, used to help the VLM tell characters
# apart in black-and-white panels. Only names present in data/refs are used.
CHARACTER_HINTS: dict[str, str] = {
    "frieren": "elf, long silver hair, elven ears",
    "fern": "Frieren's apprentice, pale hair, mage staff",
    "stark": "warrior, spiky red hair, large build",
    "himmel": "hero, blue hair, confident",
    "heiter": "priest, light gray hair, cross pendant",
    "eisen": "dwarf, tall, white beard, heavy armor",
    "aura": "demon general, long black hair, dark dress",
    "denken": "elderly mage, black hair and beard",
    "flamme": "Frieren's master, tall elf, long white hair",
    "serie": "great mage, elf, long black hair, dark dress",
    "sein": "priest, black hair, relaxed",
    "kanne": "timid mage girl, dark hair",
    "laufen": "energetic girl, short dark hair, dark outfit",
    "lawine": "mage girl, light hair",
    "richter": "older city guard, short dark hair, mustache",
    "wirbel": "mage, light hair",
    "uebel": "shadow warrior, black hair, dark clothes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect reference characters in manga panels via OpenRouter chat "
            "models (one image per call, structured JSON list output)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing panel images (sorted by filename).",
    )
    parser.add_argument(
        "--refs-dir",
        type=Path,
        default=DEFAULT_REFS_DIR,
        help="Directory containing reference character images (source of the character list).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=METHOD_DIR / "output",
        help="Parent directory for fresh timestamped run directories.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="OpenRouter model ids to test (default: the four :free VLMs).",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=METHOD_DIR / "prompt.txt"
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between API calls (free tier rate limits).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N sorted panels (useful for a paid test run).",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N sorted panels before applying --limit (default 0).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.skip_first < 0:
        parser.error("--skip-first must be non-negative")
    return args


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image_format = (image.format or "").upper()
        dimensions = [image.width, image.height]
    mime = Image.MIME.get(image_format)
    if not mime:
        raise ValueError(f"Unsupported image format for {path}")
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "mime_type": mime,
        "dimensions": dimensions,
    }


def reference_label(path: Path) -> str:
    name = path.stem
    for suffix in ("_reference", "_anime_profile"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").strip().title()


def canonical_characters(refs_dir: Path) -> list[str]:
    seen: dict[str, str] = {}
    for path in sorted(refs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        label = reference_label(path)
        seen.setdefault(label.lower(), label)
    return [seen[key] for key in sorted(seen)]


def build_prompt(template: str, characters: list[str]) -> str:
    lines = []
    for name in characters:
        hint = CHARACTER_HINTS.get(name.lower())
        lines.append(f"- {name}: {hint}" if hint else f"- {name}")
    return template.replace("{characters}", "\n".join(lines))


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("openai", "Pillow", "python-dotenv", "requests"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def fetch_models_metadata(api_key: str, api_base: str) -> dict[str, dict[str, Any]]:
    response = requests.get(
        f"{api_base}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    models: dict[str, dict[str, Any]] = {}
    for entry in payload.get("data", []):
        model_id = entry.get("id")
        if not model_id:
            continue
        models[model_id] = {
            "id": model_id,
            "created": entry.get("created"),
            "context_length": entry.get("context_length"),
            "architecture": entry.get("architecture"),
            "pricing": entry.get("pricing"),
            "top_provider": entry.get("top_provider"),
            "description": entry.get("description"),
        }
    return models


def parse_characters(text: str) -> list[Any] | None:
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.S)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "characters" in data:
            if isinstance(data["characters"], list):
                return [str(item) for item in data["characters"]]
            return None
        if isinstance(data, list):
            return [str(item) for item in data]
    return None


def validate_characters(
    parsed: list[Any] | None, canonical: list[str]
) -> tuple[list[str], list[str]]:
    """Return (known characters in canonical casing, unknown entries)."""
    if parsed is None:
        return [], []
    lookup = {name.lower(): name for name in canonical}
    known: list[str] = []
    unknown: list[str] = []
    for item in parsed:
        key = str(item).strip()
        if not key:
            continue
        if key.lower() in lookup:
            name = lookup[key.lower()]
            if name not in known:
                known.append(name)
        elif key not in unknown:
            unknown.append(key)
    return known, unknown


def classify_panel(
    client: OpenAI,
    model: str,
    prompt: str,
    panel: dict[str, Any],
    canonical: list[str],
    max_tokens: int,
    temperature: float,
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{panel['mime_type']};base64,{panel['data_base64']}"
            },
        },
    ]
    messages = [{"role": "user", "content": content}]
    response_format: str | None = "json_object"
    attempts = 0
    while True:
        attempts += 1
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = {"type": response_format}
        started = time.monotonic()
        try:
            response = client.chat.completions.create(**kwargs)
        except RateLimitError as error:
            if attempts >= MAX_ATTEMPTS:
                return {"status": "error", "error": f"RateLimitError: {error}", "attempts": attempts}
            http_response = getattr(error, "response", None)
            headers = getattr(http_response, "headers", {}) or {}
            retry_after = headers.get("retry-after")
            delay = float(retry_after) if retry_after else BASE_BACKOFF_S * attempts
            print(
                f"    rate limited (attempt {attempts}), retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            continue
        except BadRequestError as error:
            message = str(error)
            unsupported_format = response_format == "json_object" and (
                "response_format" in message or "json_object" in message or "json" in message.lower()
            )
            if unsupported_format and attempts <= 2:
                print(
                    "    response_format=json_object unsupported, retrying without it",
                    flush=True,
                )
                response_format = None
                attempts = 0
                continue
            return {"status": "error", "error": f"BadRequestError: {error}", "attempts": attempts}
        except (APIConnectionError, APIError) as error:
            if attempts >= MAX_ATTEMPTS:
                return {"status": "error", "error": f"{type(error).__name__}: {error}", "attempts": attempts}
            delay = BASE_BACKOFF_S * attempts
            print(
                f"    transient error (attempt {attempts}): {type(error).__name__}, "
                f"retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            continue
        except Exception as error:  # noqa: BLE001 - record anything else
            return {"status": "error", "error": f"{type(error).__name__}: {error}", "attempts": attempts}

        latency = time.monotonic() - started
        usage = response.usage
        usage_record: dict[str, Any] = {}
        cost_usd: float | None = None
        cost_source = "unavailable"
        if usage is not None:
            usage_record = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            # OpenRouter includes an estimated cost in usage.cost when available.
            raw_usage = getattr(usage, "model_extra", None) or {}
            cost_value = getattr(usage, "cost", None)
            if cost_value is None:
                cost_value = raw_usage.get("cost")
            if cost_value is not None:
                try:
                    cost_usd = round(float(cost_value), 8)
                    cost_source = "usage.cost"
                except (TypeError, ValueError):
                    cost_usd = None
            if cost_usd is None and pricing is not None:
                try:
                    prompt_per = float(pricing.get("prompt") or 0)
                    completion_per = float(pricing.get("completion") or 0)
                except (TypeError, ValueError):
                    prompt_per = completion_per = 0.0
                cost_usd = round(
                    (usage.prompt_tokens or 0) * prompt_per
                    + (usage.completion_tokens or 0) * completion_per,
                    8,
                )
                cost_source = "computed_from_models_pricing"
        text = (response.choices[0].message.content or "") if response.choices else ""
        parsed = parse_characters(text)
        known, unknown = validate_characters(parsed, canonical)
        status = "ok"
        if parsed is None:
            status = "unparseable"
        elif unknown:
            status = "ok-with-unknown"
        return {
            "status": status,
            "attempts": attempts,
            "model_returned": response.model,
            "response_id": response.id,
            "response_format_used": response_format or "none",
            "response_text": text,
            "characters_parsed": parsed,
            "characters": known,
            "unknown_entries": unknown,
            "usage": usage_record,
            "cost_usd": cost_usd,
            "cost_source": cost_source,
            "latency_s": round(latency, 3),
            "error": None,
            "finished_at": iso_now(),
        }


def run() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key: set {args.api_key_env} or add it to {REPO_ROOT / '.env'}"
        )

    input_dir = args.input_dir.resolve()
    refs_dir = args.refs_dir.resolve()
    panels = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if args.skip_first:
        panels = panels[args.skip_first :]
    if args.limit:
        panels = panels[: args.limit]
    if not panels:
        raise SystemExit(f"No supported panel images found in {input_dir}")
    if not refs_dir.is_dir() or not any(
        p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES for p in refs_dir.iterdir()
    ):
        raise SystemExit(f"No supported reference images found in {refs_dir}")

    canonical = canonical_characters(refs_dir)
    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    prompt = build_prompt(prompt_template, canonical)

    run_dir = create_run_dir(args.output_root.resolve())
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "character-detection-openrouter-vlm",
        "status": "running",
        "started_at": iso_now(),
        "finished_at": None,
        "run_directory": str(run_dir),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "configuration": {
            "api_base": args.api_base,
            "models": args.models,
            "input_dir": str(input_dir),
            "refs_dir": str(refs_dir),
            "prompt_file": str(args.prompt_file.resolve()),
            "prompt": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "sleep_s": args.sleep,
            "skip_first": args.skip_first,
            "limit": args.limit,
        },
        "dependencies": {"python": sys.version, **package_versions()},
        "pricing_assumptions": {
            "currency": "USD",
            "date": datetime.now().astimezone().strftime("%Y-%m-%d"),
            "note": (
                "Mixed OpenRouter tier: google/gemma-4-31b-it and "
                "google/gemma-4-26b-a4b-it run on the paid tier (user-funded); "
                "nvidia/nemotron-nano-12b-v2-vl and "
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning have NO paid "
                "endpoint on OpenRouter (verified in /models), so their :free "
                "variants are used (rate-limited ~20 req/min, may queue). "
                "Per-call cost comes from usage.cost when OpenRouter reports it, "
                "otherwise computed from the model's published /models pricing "
                "(per-token) times the measured tokens. Image input is billed as "
                "tokens."
            )
            if args.api_base == API_BASE
            else (
                "Self-hosted inference: the endpoint is not OpenRouter, so no "
                "per-call billing applies (electricity only, reported as "
                "unpriced_calls). Do not compare with paid API pricing."
            ),
        },
        "reference_characters": canonical,
        "inputs": [image_info(path) for path in panels],
        "models_metadata": {},
        "calls": [],
        "totals": {
            "api_calls": 0,
            "successful_calls": 0,
            "error_calls": 0,
            "unpriced_calls": 0,
            "total_latency_s": 0.0,
            "cost_usd": 0.0,
        },
    }
    write_json(manifest_path, manifest)

    try:
        models_metadata = fetch_models_metadata(api_key, args.api_base)
        manifest["models_metadata"] = {
            model: models_metadata.get(model, {"note": "id not found in /models listing"})
            for model in args.models
        }
        write_json(manifest_path, manifest)

        client = OpenAI(api_key=api_key, base_url=args.api_base)
        for model in args.models:
            model_dir = run_dir / model_slug(model)
            model_dir.mkdir(exist_ok=True)
            print(f"== model: {model} ==", flush=True)
            for index, panel_path in enumerate(panels, start=1):
                info = image_info(panel_path)
                info["data_base64"] = base64.b64encode(panel_path.read_bytes()).decode()
                print(
                    f"  [{index}/{len(panels)}] {panel_path.name} ...",
                    flush=True,
                )
                record = classify_panel(
                    client,
                    model,
                    prompt,
                    info,
                    canonical,
                    args.max_tokens,
                    args.temperature,
                    (models_metadata.get(model) or {}).get("pricing"),
                )
                record["requested_model"] = model
                record["panel"] = panel_path.name
                record["panel_sha256"] = info["sha256"]
                record["characters"] = record.get("characters", [])
                record.pop("data_base64", None)
                output_path = model_dir / f"{panel_path.stem}.json"
                write_json(output_path, record)
                manifest["calls"].append(record)
                manifest["totals"]["api_calls"] += 1
                if record["status"] == "ok" or record["status"] == "ok-with-unknown":
                    manifest["totals"]["successful_calls"] += 1
                else:
                    manifest["totals"]["error_calls"] += 1
                manifest["totals"]["total_latency_s"] = round(
                    manifest["totals"]["total_latency_s"] + (record.get("latency_s") or 0.0),
                    3,
                )
                cost = record.get("cost_usd")
                if cost is not None:
                    manifest["totals"]["cost_usd"] = round(
                        manifest["totals"]["cost_usd"] + cost, 8
                    )
                else:
                    manifest["totals"]["unpriced_calls"] += 1
                write_json(manifest_path, manifest)
                print(
                    f"    -> status={record['status']} characters={record['characters']} "
                    f"({record.get('latency_s')}s)",
                    flush=True,
                )
                if args.sleep:
                    time.sleep(args.sleep)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = iso_now()
        write_json(manifest_path, manifest)
        # Aggregated views written on every exit path (best effort).
        try:
            summary: dict[str, Any] = {}
            rows = []
            for call in manifest["calls"]:
                summary.setdefault(call["requested_model"], {})[call["panel"]] = call["characters"]
                rows.append(
                    {
                        "panel": call["panel"],
                        "model": call["requested_model"],
                        "status": call["status"],
                        "characters": "; ".join(call["characters"]),
                        "unknown_entries": "; ".join(call.get("unknown_entries") or []),
                        "response_format_used": call.get("response_format_used"),
                        "latency_s": call.get("latency_s"),
                        "prompt_tokens": (call.get("usage") or {}).get("prompt_tokens"),
                        "completion_tokens": (call.get("usage") or {}).get("completion_tokens"),
                        "cost_usd": call.get("cost_usd"),
                        "cost_source": call.get("cost_source"),
                    }
                )
            write_json(run_dir / "results_by_model.json", summary)
            with (run_dir / "results_flat.csv").open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "panel", "model", "status", "characters", "unknown_entries",
                    "response_format_used", "latency_s", "prompt_tokens",
                    "completion_tokens", "cost_usd", "cost_source",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not write summary files: {exc}", file=sys.stderr)


if __name__ == "__main__":
    run()
