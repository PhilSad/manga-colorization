#!/usr/bin/env python3
"""Detect reference characters in manga panels by pairwise reference matching.

For every (panel, reference character) pair, send ONE request containing two
images (the panel + one reference image) to a self-hosted vLLM endpoint running
a vision model (default `qwen/qwen3.6-27b` at http://spark:8000/v1) and ask
whether that specific character appears in the panel. The answer is a small JSON
object {"present": bool, "confidence": float, "reason": str}.

All (panel x character) pairs are dispatched concurrently over a thread pool
(one request per thread; the vLLM server batches). Results are aggregated per
panel into the list of characters present, written as per-call records, per-panel
summaries, a flat CSV, and a manifest.

Self-hosted inference: $0 per call (electricity only). Do not compare with paid
API pricing.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import re
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

METHOD_DIR = Path(__file__).resolve().parent
REPO_ROOT = METHOD_DIR.parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "panels"
DEFAULT_REFS_DIR = REPO_ROOT / "data" / "refs"
DEFAULT_ENDPOINT = "http://spark:8000/v1"
DEFAULT_MODEL = "qwen/qwen3.6-27b"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_S = 300.0
MAX_ATTEMPTS = 5
BASE_BACKOFF_S = 2.0

# Endpoint advertises limit_mm_per_prompt image=2 -> panel + one ref fits.
MAX_IMAGES_PER_REQUEST = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect reference characters in manga panels: for each (panel, "
            "character) pair, ask a vision LLM (panel + reference image) whether "
            "the character is present, concurrently, then aggregate per panel."
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
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="OpenAI-compatible base URL of the self-hosted vLLM server.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Served model id on the endpoint."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent requests (threads).",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=METHOD_DIR / "prompt.txt"
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N sorted panels (useful for a quick test run).",
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
    if args.workers < 1:
        parser.error("--workers must be at least 1")
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
    (candidate / "calls").mkdir()
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


def character_ref_file(refs_dir: Path, character: str) -> Path:
    """First ref image whose canonical label matches the character name."""
    for path in sorted(refs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if reference_label(path).lower() == character.lower():
            return path
    raise FileNotFoundError(f"No reference image for character {character!r}")


def flatten_to_rgb_bytes(data: bytes, mime: str) -> tuple[bytes, str]:
    """Flatten RGBA onto white and re-encode; return (bytes, mime_type).

    VLMs handle RGBA inconsistently; a white background is a faithful rendition
    of a transparent manga/character image.
    """
    with Image.open(io.BytesIO(data)) as image:
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def build_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def build_prompt(template: str, character: str) -> str:
    return template.replace("{character}", character)


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("openai", "Pillow", "python-dotenv"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def parse_probe_response(text: str) -> tuple[bool | None, float | None, str | None]:
    """Return (present, confidence, reason) parsed from the model's answer.

    present may be None when the response cannot be parsed. Accepts a plain JSON
    object, fenced JSON, or a JSON object embedded in surrounding text/reasoning.
    """
    if not text:
        return None, None, None
    text = text.strip()

    def _try_load(candidate: str) -> dict[str, Any] | None:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    data = _try_load(text)
    if data is None:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fenced:
            data = _try_load(fenced.group(1).strip())
    if data is None:
        # Balanced-brace scan of the first top-level object (survives leading
        # reasoning tokens that may contain braces/JSON fragments).
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : index + 1]
                        data = _try_load(candidate)
                        if data is not None:
                            break
                        break
            if data is not None:
                break
            start = text.find("{", start + 1)

    if data is None:
        return None, None, None

    present = data.get("present")
    if isinstance(present, str):
        normalized = present.strip().lower()
        if normalized in ("true", "yes", "present"):
            present = True
        elif normalized in ("false", "no", "absent", "not present"):
            present = False
    if not isinstance(present, bool):
        present = None

    confidence = data.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        confidence = float(confidence)
    elif isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None
    else:
        confidence = None

    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = None
    return present, confidence, reason


def probe_character(
    client: OpenAI,
    model: str,
    prompt: str,
    panel_name: str,
    character: str,
    panel_data_url: str,
    ref_data_url: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": panel_data_url},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": ref_data_url},
                },
            ],
        }
    ]
    attempts = 0
    while True:
        attempts += 1
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
        except Exception as error:  # noqa: BLE001 - record anything, retry transient
            if attempts >= MAX_ATTEMPTS:
                return {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "attempts": attempts,
                }
            delay = BASE_BACKOFF_S * attempts
            print(
                f"    [{panel_name}/{character}] transient error (attempt {attempts}): "
                f"{type(error).__name__}, retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            continue

        latency = time.monotonic() - started
        usage = response.usage
        usage_record: dict[str, Any] = {}
        if usage is not None:
            usage_record = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        text = (response.choices[0].message.content or "") if response.choices else ""
        present, confidence, reason = parse_probe_response(text)
        if present is None:
            status = "unparseable"
        else:
            status = "ok"
        return {
            "status": status,
            "attempts": attempts,
            "model_returned": response.model,
            "response_id": response.id,
            "response_text": text,
            "present": present,
            "confidence": confidence,
            "reason": reason,
            "usage": usage_record,
            "latency_s": round(latency, 3),
            "error": None,
            "finished_at": iso_now(),
        }


def run() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")  # harmless if absent; endpoint needs no key

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

    run_dir = create_run_dir(args.output_root.resolve())
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "character-detection-qwen3-vllm-ref-pair",
        "status": "running",
        "started_at": iso_now(),
        "finished_at": None,
        "run_directory": str(run_dir),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "configuration": {
            "endpoint": args.endpoint,
            "model": args.model,
            "workers": args.workers,
            "input_dir": str(input_dir),
            "refs_dir": str(refs_dir),
            "prompt_file": str(args.prompt_file.resolve()),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout_s": args.timeout,
            "skip_first": args.skip_first,
            "limit": args.limit,
            "request_shape": (
                "one request per (panel, character) pair, two images per request "
                "(panel + reference image), JSON object response via "
                "response_format=json_object"
            ),
        },
        "dependencies": {"python": sys.version, **package_versions()},
        "pricing_assumptions": {
            "currency": "USD",
            "date": datetime.now().astimezone().strftime("%Y-%m-%d"),
            "note": (
                "Self-hosted vLLM inference on the DGX Spark (OpenAI-compatible "
                "endpoint). $0 per call; electricity only. Do not compare with "
                "paid API pricing."
            ),
        },
        "reference_characters": canonical,
        "inputs": [image_info(path) for path in panels],
        "references": {
            character: image_info(character_ref_file(refs_dir, character))
            for character in canonical
        },
        "calls": [],
        "totals": {
            "api_calls": 0,
            "successful_calls": 0,
            "unparseable_calls": 0,
            "error_calls": 0,
            "total_latency_s": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        },
    }
    write_json(manifest_path, manifest)

    # Pre-compute data URLs once per file (each image is reused for many calls).
    panel_urls: dict[str, str] = {}
    for panel in panels:
        info = image_info(panel)
        data, mime = flatten_to_rgb_bytes(panel.read_bytes(), info["mime_type"])
        panel_urls[panel.name] = build_data_url(data, mime)
    ref_urls: dict[str, str] = {}
    for character in canonical:
        ref = character_ref_file(refs_dir, character)
        info = image_info(ref)
        data, mime = flatten_to_rgb_bytes(ref.read_bytes(), info["mime_type"])
        ref_urls[character] = build_data_url(data, mime)

    client = OpenAI(base_url=args.endpoint, api_key="EMPTY")
    tasks = [
        (panel.name, character) for panel in panels for character in canonical
    ]
    print(
        f"== {len(tasks)} probes: {len(panels)} panels x {len(canonical)} characters "
        f"with {args.workers} workers ==",
        flush=True,
    )
    print(f"   endpoint: {args.endpoint}  model: {args.model}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for panel_name, character in tasks:
                prompt = build_prompt(prompt_template, character)
                future = executor.submit(
                    probe_character,
                    client,
                    args.model,
                    prompt,
                    panel_name,
                    character,
                    panel_urls[panel_name],
                    ref_urls[character],
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )
                futures[future] = (panel_name, character)

            completed = 0
            for future in as_completed(futures):
                panel_name, character = futures[future]
                try:
                    record = future.result()
                except Exception as error:  # noqa: BLE001 - unexpected thread failure
                    record = {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "attempts": 0,
                    }
                record["requested_model"] = args.model
                record["panel"] = panel_name
                record["character"] = character
                record["prompt"] = build_prompt(prompt_template, character)
                record.pop("response_text", None)
                call_path = run_dir / "calls" / f"{panel_name}__{model_slug(character)}.json"
                write_json(call_path, record)

                manifest["calls"].append(record)
                manifest["totals"]["api_calls"] += 1
                if record["status"] == "ok":
                    manifest["totals"]["successful_calls"] += 1
                elif record["status"] == "unparseable":
                    manifest["totals"]["unparseable_calls"] += 1
                else:
                    manifest["totals"]["error_calls"] += 1
                manifest["totals"]["total_latency_s"] = round(
                    manifest["totals"]["total_latency_s"] + (record.get("latency_s") or 0.0),
                    3,
                )
                usage = record.get("usage") or {}
                manifest["totals"]["total_prompt_tokens"] += usage.get("prompt_tokens") or 0
                manifest["totals"]["total_completion_tokens"] += (
                    usage.get("completion_tokens") or 0
                )
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    write_json(manifest_path, manifest)
                    print(
                        f"   [{completed}/{len(tasks)}] {panel_name} / {character}: "
                        f"status={record['status']} present={record.get('present')} "
                        f"({record.get('latency_s')}s)",
                        flush=True,
                    )
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
            # Per-panel summaries: list of characters present.
            for panel in panels:
                present: list[str] = []
                per_character: dict[str, dict[str, Any]] = {}
                for call in manifest["calls"]:
                    if call["panel"] != panel.name:
                        continue
                    entry = {
                        "status": call["status"],
                        "present": call.get("present"),
                        "confidence": call.get("confidence"),
                        "reason": call.get("reason"),
                        "latency_s": call.get("latency_s"),
                        "usage": call.get("usage"),
                        "error": call.get("error"),
                    }
                    per_character[call["character"]] = entry
                    if call.get("present") is True:
                        present.append(call["character"])
                write_json(
                    run_dir / f"{panel.stem}.json",
                    {
                        "panel": panel.name,
                        "panel_sha256": image_info(panel)["sha256"],
                        "characters": present,
                        "per_character": per_character,
                    },
                )
            # Flat CSV.
            with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "panel", "character", "status", "present", "confidence",
                    "latency_s", "prompt_tokens", "completion_tokens", "error",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for call in manifest["calls"]:
                    writer.writerow(
                        {
                            "panel": call["panel"],
                            "character": call["character"],
                            "status": call["status"],
                            "present": call.get("present"),
                            "confidence": call.get("confidence"),
                            "latency_s": call.get("latency_s"),
                            "prompt_tokens": (call.get("usage") or {}).get("prompt_tokens"),
                            "completion_tokens": (call.get("usage") or {}).get("completion_tokens"),
                            "error": call.get("error"),
                        }
                    )
            # Per-panel present-character CSV (the headline output).
            with (run_dir / "characters_per_panel.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["panel", "characters"])
                for panel in panels:
                    present = [
                        call["character"]
                        for call in manifest["calls"]
                        if call["panel"] == panel.name and call.get("present") is True
                    ]
                    writer.writerow([panel.name, "; ".join(present)])
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not write summary files: {exc}", file=sys.stderr)

    print(f"== done: {run_dir} ==", flush=True)
    for panel in panels:
        present = [
            call["character"]
            for call in manifest["calls"]
            if call["panel"] == panel.name and call.get("present") is True
        ]
        print(f"   {panel.name}: {present}", flush=True)


if __name__ == "__main__":
    run()
