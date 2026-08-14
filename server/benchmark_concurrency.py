#!/usr/bin/env python3
"""Benchmark the concurrency behavior of the self-hosted FLUX.2 Klein server.

Measures end-to-end `POST /edit` latency and throughput as a function of
client-side concurrency, plus (optionally) GPU utilization on the server host
sampled via ssh. The server's BentoML `traffic.concurrency` (service.py) is
the hard cap on in-flight requests: with concurrency=1 every extra client
connection just queues, so the sweep shows queueing behavior and whether the
single GB10 GPU is already saturated by one request.

Usage:
  .venv/bin/python server/benchmark_concurrency.py \\
      --endpoint http://spark:3000 \\
      --panels-dir pipeline_v1/output/<run>/1_panels/<page> \\
      --atlas data/refs/frieren_reference.webp \\
      --concurrency 1,2,4,8 --requests 8 --warmup 1 --gpu-host spark

  # serial /edit vs concurrency-2 /edit2 on the same panel set
  .venv/bin/python server/benchmark_concurrency.py --mode compare \\
      --endpoint http://spark:3000 \\
      --panels-dir pipeline_v1/output/<run>/1_panels/<page> \\
      --atlas data/refs/frieren_reference.webp \\
      --requests 10 --warmup 1 --gpu-host spark

Prints a summary table to stdout and writes the raw results JSON to
`--json-out` (default `concurrency_benchmark_<timestamp>.json` in the current
directory).

Dependencies: requests, Pillow (same as the pipeline client). GPU sampling
uses `ssh <gpu-host> nvidia-smi ...` and degrades to no sampling when ssh or
nvidia-smi is unavailable.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

DEFAULT_ENDPOINT = "http://spark:3000"
_ATLAS_INSTRUCTION = (
    "Use the labelled character reference atlas in #2 for canonical hair, eye, "
    "skin, clothing, and accessory colors whenever a referenced character appears."
)
_NO_ATLAS_INSTRUCTION = (
    "No reference atlas is provided: invent a coherent, restrained anime palette "
    "consistent with the series."
)
_NO_PROFILE_INSTRUCTION = (
    "No explicit character palette profiles are provided; derive canonical colors "
    "from the atlas and invent coherent colors consistent with the series."
)
PROMPT_TEMPLATE = (
    "Colorize the manga panel in #1 with a muted, painterly anime palette in the "
    "style of Frieren: Beyond Journey's End. Preserve the line art exactly; apply "
    "flat colors with soft shading, no outlines recolored. {atlas_instruction} "
    "{character_profiles} Canvas {width}x{height}."
)
MAX_MEGAPIXELS = 2.0
_TIMEOUT_SECONDS = 1800


def nearest_multiple_of(value: float) -> int:
    """Closest multiple of 16 (FLUX VAE requirement), never 0."""
    return max(16, int(round(value / 16)) * 16)


def requested_size(width: int, height: int) -> tuple[int, int]:
    """Mirror of pipeline_v1.config.bounded_requested_size (2 MP cap, /16)."""
    max_pixels = MAX_MEGAPIXELS * 1_000_000
    if width * height <= max_pixels:
        return nearest_multiple_of(width), nearest_multiple_of(height)
    import math

    scale = math.sqrt(max_pixels / (width * height))
    w = nearest_multiple_of(width * scale)
    h = nearest_multiple_of(height * scale)
    while w * h > max_pixels and w >= 16 and h >= 16:
        if w > h:
            w -= 16
        else:
            h -= 16
    return w, h


def make_request(
    endpoint: str,
    panel: Path,
    atlas: Path | None,
    seed: int,
    steps: int,
    guidance_scale: float,
    lora_scale: float | None,
    output_format: str,
) -> tuple[float, int | None, str | None]:
    """POST one /edit request; returns (latency_s, http_status, error)."""
    with Image.open(panel) as image:
        width, height = requested_size(image.width, image.height)
    fields = {
        "prompt": PROMPT_TEMPLATE.format(
            width=width,
            height=height,
            atlas_instruction=_ATLAS_INSTRUCTION if atlas else _NO_ATLAS_INSTRUCTION,
            character_profiles=_NO_PROFILE_INSTRUCTION,
        ),
        "width": str(width),
        "height": str(height),
        "num_inference_steps": str(steps),
        "guidance_scale": str(guidance_scale),
        "output_format": output_format,
    }
    if lora_scale is not None:
        fields["lora_scale"] = str(lora_scale)
    fields["seed"] = str(seed)

    files = [("images", (panel.name, open(panel, "rb"), "image/png"))]
    if atlas is not None:
        files.append(("images", (atlas.name, open(atlas, "rb"), "image/webp")))
    started = time.monotonic()
    try:
        response = requests.post(
            f"{endpoint}/edit", data=fields, files=files, timeout=_TIMEOUT_SECONDS
        )
        status = response.status_code
        error = None if status == 200 else response.text[:300]
    except Exception as exc:  # noqa: BLE001 - network errors
        status, error = None, f"{type(exc).__name__}: {exc}"
    finally:
        for _, (_, handle, _) in files:
            handle.close()
    return time.monotonic() - started, status, error


def make_request2(
    endpoint: str,
    panel_a: Path,
    panel_b: Path,
    atlas: Path | None,
    seed_a: int,
    seed_b: int,
    steps: int,
    guidance_scale: float,
    lora_scale: float | None,
    output_format: str,
) -> tuple[float, int | None, str | None, list[float] | None]:
    """POST one /edit2 request (two jobs, same shared params); returns
    (latency_s, http_status, error, server-reported job latencies)."""
    with Image.open(panel_a) as image:
        width, height = requested_size(image.width, image.height)
    prompt = PROMPT_TEMPLATE.format(
        width=width,
        height=height,
        atlas_instruction=_ATLAS_INSTRUCTION if atlas else _NO_ATLAS_INSTRUCTION,
        character_profiles=_NO_PROFILE_INSTRUCTION,
    )
    fields = {
        "prompt1": prompt,
        "prompt2": prompt,
        "width": str(width),
        "height": str(height),
        "num_inference_steps": str(steps),
        "guidance_scale": str(guidance_scale),
        "output_format": output_format,
        "seed1": str(seed_a),
        "seed2": str(seed_b),
    }
    if lora_scale is not None:
        fields["lora_scale"] = str(lora_scale)

    files = [
        ("images1", (panel_a.name, open(panel_a, "rb"), "image/png")),
        ("images2", (panel_b.name, open(panel_b, "rb"), "image/png")),
    ]
    if atlas is not None:
        files.append(("images1", (atlas.name, open(atlas, "rb"), "image/webp")))
        files.append(("images2", (atlas.name, open(atlas, "rb"), "image/webp")))
    started = time.monotonic()
    job_latencies = None
    try:
        response = requests.post(
            f"{endpoint}/edit2", data=fields, files=files, timeout=_TIMEOUT_SECONDS
        )
        status = response.status_code
        if status == 200:
            payload = response.json()
            images = payload.get("images", [])
            job_latencies = payload.get("job_latency_s")
            error = (
                None
                if len(images) == 2
                else f"expected 2 images in edit2 response, got {len(images)}"
            )
        else:
            error = response.text[:300]
    except Exception as exc:  # noqa: BLE001 - network errors
        status, error = None, f"{type(exc).__name__}: {exc}"
    finally:
        for _, (_, handle, _) in files:
            handle.close()
    return time.monotonic() - started, status, error, job_latencies


class GpuSampler:
    """Samples nvidia-smi on the server host (via ssh) while a level runs."""

    def __init__(self, gpu_host: str | None, interval_s: float = 0.5) -> None:
        self.gpu_host = gpu_host
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._mem_samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.gpu_host:
            return
        self._samples.clear()
        self._mem_samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _run(self) -> None:
        cmd = [
            "ssh",
            self.gpu_host,
            "nvidia-smi --query-gpu=utilization.gpu,memory.used "
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10
                ).stdout.strip()
                parts = out.split(",")
                if len(parts) == 2:
                    try:
                        self._samples.append(float(parts[0]))
                    except ValueError:
                        pass
                    # GB10 reports memory.used as [N/A]; skip it in that case.
                    try:
                        self._mem_samples.append(int(parts[1]))
                    except ValueError:
                        pass
            except Exception:  # noqa: BLE001 - sampling is best-effort
                pass
            self._stop.wait(self.interval_s)

    def stats(self) -> dict:
        if not self._samples:
            return {"samples": 0}
        stats = {
            "samples": len(self._samples),
            "gpu_util_mean_pct": round(statistics.mean(self._samples), 1),
            "gpu_util_max_pct": round(max(self._samples), 1),
            "gpu_util_busy_gt50_pct": round(
                100.0 * sum(s > 50 for s in self._samples) / len(self._samples), 1
            ),
        }
        # GB10 reports memory.used as [N/A]; omit mem stats in that case.
        if self._mem_samples:
            stats["mem_mean_gb"] = round(
                statistics.mean(self._mem_samples) / 1024, 1
            )
            stats["mem_max_gb"] = round(max(self._mem_samples) / 1024, 1)
        return stats


def run_level(
    endpoint: str,
    panels: list[Path],
    atlas: Path | None,
    concurrency: int,
    requests: int,
    seed_base: int,
    steps: int,
    guidance_scale: float,
    lora_scale: float | None,
    output_format: str,
    sampler: GpuSampler,
) -> dict:
    """Run `requests` requests with `concurrency` workers; return stats."""
    jobs = []
    for i in range(requests):
        panel = panels[i % len(panels)]
        jobs.append((panel, seed_base + i))

    latencies: list[float] = []
    statuses: list[int | None] = []
    errors: list[str] = []
    started = time.monotonic()
    sampler.start()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                make_request,
                endpoint,
                panel,
                atlas,
                seed,
                steps,
                guidance_scale,
                lora_scale,
                output_format,
            ): (panel.name, seed)
            for panel, seed in jobs
        }
        for future in as_completed(futures):
            latency, status, error = future.result()
            latencies.append(latency)
            statuses.append(status)
            errors.append(error or ("" if status == 200 else f"HTTP {status}"))
    wall = time.monotonic() - started
    sampler.stop()

    ok = [l for l, s in zip(latencies, statuses) if s == 200]
    latencies.sort()
    n = len(latencies)
    pct = lambda p: latencies[min(n - 1, int(p / 100 * n))]  # noqa: E731
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_s": round(wall, 3),
        "throughput_req_s": round(requests / wall, 3),
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_p50_s": round(pct(50), 3) if latencies else None,
        "latency_p95_s": round(pct(95), 3) if latencies else None,
        "latency_max_s": round(max(latencies), 3) if latencies else None,
        "ok": len(ok),
        "errors": sum(1 for s in statuses if s != 200),
        "error_samples": errors[:5],
        "gpu": sampler.stats(),
    }


def run_compare(
    endpoint: str,
    panels: list[Path],
    atlas: Path | None,
    requests: int,
    seed_base: int,
    steps: int,
    guidance_scale: float,
    lora_scale: float | None,
    output_format: str,
    sampler: GpuSampler,
) -> dict:
    """Serial /edit (N requests) vs /edit2 (ceil(N/2) requests x 2 jobs each)
    colorizing the same N panels; returns per-method stats + speedup."""
    n = requests
    pairs = [
        (panels[i % len(panels)], panels[(i + 1) % len(panels)])
        for i in range(0, n, 2)
    ]
    results: dict = {}

    # --- serial /edit ---
    print(f"\n--- serial /edit: {n} requests, 1 panel each ---", flush=True)
    latencies: list[float] = []
    ok = 0
    errors: list[str] = []
    sampler.start()
    started = time.monotonic()
    for i in range(n):
        lat, status, err = make_request(
            endpoint, panels[i % len(panels)], atlas, seed_base + i,
            steps, guidance_scale, lora_scale, output_format,
        )
        latencies.append(lat)
        ok += 1 if status == 200 else 0
        errors.append(err or ("" if status == 200 else f"HTTP {status}"))
    wall = time.monotonic() - started
    sampler.stop()
    serial = {
        "route": "/edit",
        "requests": n,
        "panels": n,
        "wall_s": round(wall, 3),
        "throughput_panels_s": round(n / wall, 3),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
        "ok": ok,
        "errors": n - ok,
        "error_samples": errors[:5],
        "gpu": sampler.stats(),
    }
    results["serial_edit"] = serial
    print(
        f"  wall={wall:.2f}s  {n / wall:.3f} panels/s  "
        f"latency mean/max = {statistics.mean(latencies):.2f}/"
        f"{max(latencies):.2f}s  ok={ok} errors={n - ok}"
    )
    if serial["gpu"].get("samples"):
        g = serial["gpu"]
        print(
            f"  gpu: util mean/max/busy>50% = {g['gpu_util_mean_pct']}/"
            f"{g['gpu_util_max_pct']}/{g['gpu_util_busy_gt50_pct']}% "
            f"({g['samples']} samples)"
        )

    # --- /edit2 ---
    n2 = len(pairs)
    print(f"\n--- /edit2: {n2} requests x 2 jobs = {n} panels ---", flush=True)
    latencies = []
    job_lat_all: list[float] = []
    ok = 0
    errors = []
    sampler.start()
    started = time.monotonic()
    for k, (pa, pb) in enumerate(pairs):
        lat, status, err, job_lat = make_request2(
            endpoint, pa, pb, atlas, seed_base + 2 * k, seed_base + 2 * k + 1,
            steps, guidance_scale, lora_scale, output_format,
        )
        latencies.append(lat)
        ok += 1 if status == 200 else 0
        errors.append(err or ("" if status == 200 else f"HTTP {status}"))
        if job_lat:
            job_lat_all.extend(job_lat)
    wall = time.monotonic() - started
    sampler.stop()
    batch = {
        "route": "/edit2",
        "requests": n2,
        "panels": n,
        "wall_s": round(wall, 3),
        "throughput_panels_s": round(n / wall, 3),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
        "job_latency_mean_s": (
            round(statistics.mean(job_lat_all), 3) if job_lat_all else None
        ),
        "ok": ok,
        "errors": n2 - ok,
        "error_samples": errors[:5],
        "gpu": sampler.stats(),
    }
    results["edit2"] = batch
    print(
        f"  wall={wall:.2f}s  {n / wall:.3f} panels/s  "
        f"request latency mean/max = {statistics.mean(latencies):.2f}/"
        f"{max(latencies):.2f}s  server job latency mean = "
        f"{batch['job_latency_mean_s']}s  ok={ok} errors={n2 - ok}"
    )
    if batch["gpu"].get("samples"):
        g = batch["gpu"]
        print(
            f"  gpu: util mean/max/busy>50% = {g['gpu_util_mean_pct']}/"
            f"{g['gpu_util_max_pct']}/{g['gpu_util_busy_gt50_pct']}% "
            f"({g['samples']} samples)"
        )

    results["speedup_wall_x"] = round(serial["wall_s"] / batch["wall_s"], 3)
    results["speedup_panels_s_x"] = round(
        batch["throughput_panels_s"] / serial["throughput_panels_s"], 3
    )
    print(
        f"\n  => /edit2 wall = {serial['wall_s']}s / {batch['wall_s']}s "
        f"= {results['speedup_wall_x']}x faster; "
        f"throughput = {results['speedup_panels_s_x']}x"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--mode",
        default="sweep",
        choices=["sweep", "compare"],
        help="sweep = client-concurrency sweep vs /edit; "
        "compare = serial /edit vs concurrency-2 /edit2 on the same panels",
    )
    parser.add_argument("--panels-dir", required=True, help="dir with panel PNGs")
    parser.add_argument("--atlas", default=None, help="optional reference image")
    parser.add_argument(
        "--concurrency", default="1,2,4,8", help="comma-separated levels"
    )
    parser.add_argument("--requests", type=int, default=8, help="requests per level")
    parser.add_argument("--warmup", type=int, default=1, help="warmup requests")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output-format", default="png")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--gpu-host", default=None, help="ssh host for nvidia-smi sampling (e.g. spark)"
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    panels = sorted(
        p for p in Path(args.panels_dir).iterdir()
        if p.suffix.lower() == ".png" and "overlay" not in p.name
    )
    if not panels:
        raise SystemExit(f"no panel PNGs found in {args.panels_dir}")
    atlas = Path(args.atlas) if args.atlas else None
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(
        f"endpoint={args.endpoint} panels={len(panels)} "
        f"atlas={atlas.name if atlas else 'none'} steps={args.steps} "
        f"levels={levels} requests/level={args.requests} warmup={args.warmup}"
    )
    print(f"panel sizes: {[(p.name, Image.open(p).size) for p in panels]}")

    # Warmup: model is loaded, but pay any one-off cost (Triton, etc.) first.
    if args.warmup:
        print(f"warmup x{args.warmup} ...", flush=True)
        for i in range(args.warmup):
            lat, status, err = make_request(
                args.endpoint, panels[0], atlas, args.seed + 1000 + i,
                args.steps, args.guidance_scale, args.lora_scale, args.output_format,
            )
            print(f"  warmup {i}: {lat:.2f}s status={status} {err or ''}")
        if args.mode == "compare" and len(panels) >= 2:
            lat, status, err, job_lat = make_request2(
                args.endpoint, panels[0], panels[1], atlas, args.seed + 1100,
                args.seed + 1101, args.steps, args.guidance_scale,
                args.lora_scale, args.output_format,
            )
            print(
                f"  warmup edit2: {lat:.2f}s status={status} {err or ''} "
                f"job_lat={job_lat}"
            )

    sampler = GpuSampler(args.gpu_host)

    if args.mode == "compare":
        results = run_compare(
            args.endpoint, panels, atlas, args.requests, args.seed,
            args.steps, args.guidance_scale, args.lora_scale,
            args.output_format, sampler,
        )
    else:
        results = []
        for level in levels:
            print(
                f"\n--- concurrency {level} (requests={args.requests}) ---",
                flush=True,
            )
            res = run_level(
                args.endpoint, panels, atlas, level, args.requests,
                args.seed + 2000, args.steps, args.guidance_scale,
                args.lora_scale, args.output_format, sampler,
            )
            results.append(res)
            print(
                f"  wall={res['wall_s']}s  throughput={res['throughput_req_s']} req/s  "
                f"latency mean/p50/p95/max = "
                f"{res['latency_mean_s']}/{res['latency_p50_s']}/"
                f"{res['latency_p95_s']}/{res['latency_max_s']}s  "
                f"ok={res['ok']} errors={res['errors']}"
            )
            if res["gpu"].get("samples"):
                g = res["gpu"]
                print(
                    f"  gpu: util mean/max/busy>50% = {g['gpu_util_mean_pct']}/"
                    f"{g['gpu_util_max_pct']}/{g['gpu_util_busy_gt50_pct']}%  "
                    f"mem mean/max = {g.get('mem_mean_gb', 'n/a')}/"
                    f"{g.get('mem_max_gb', 'n/a')} GB "
                    f"({g['samples']} samples)"
                )
            if res["errors"]:
                print(f"  errors: {res['error_samples']}")

    print("\n=== summary ===")
    if args.mode == "compare":
        print(
            f"{'route':>8} {'panels':>6} {'wall_s':>8} {'panels/s':>9} "
            f"{'mean_s':>8} {'max_s':>8} {'ok':>4} {'err':>4} {'gpu_mean%':>9}"
        )
        for key in ("serial_edit", "edit2"):
            r = results[key]
            g = r["gpu"]
            print(
                f"{r['route']:>8} {r['panels']:>6} {r['wall_s']:>8.2f} "
                f"{r['throughput_panels_s']:>9.3f} {r['latency_mean_s']:>8.2f} "
                f"{r['latency_max_s']:>8.2f} {r['ok']:>4} {r['errors']:>4} "
                f"{g.get('gpu_util_mean_pct', '-'):>9}"
            )
        print(
            f"speedup: wall {results['speedup_wall_x']}x, "
            f"throughput {results['speedup_panels_s_x']}x"
        )
    else:
        print(
            f"{'conc':>4} {'req':>4} {'wall_s':>8} {'req/s':>7} "
            f"{'mean_s':>7} {'p50_s':>7} {'p95_s':>7} {'max_s':>7} "
            f"{'ok':>4} {'err':>4} {'gpu_mean%':>9}"
        )
        for r in results:
            g = r["gpu"]
            print(
                f"{r['concurrency']:>4} {r['requests']:>4} {r['wall_s']:>8.2f} "
                f"{r['throughput_req_s']:>7.3f} "
                f"{r['latency_mean_s']:>7.2f} {r['latency_p50_s']:>7.2f} "
                f"{r['latency_p95_s']:>7.2f} {r['latency_max_s']:>7.2f} "
                f"{r['ok']:>4} {r['errors']:>4} "
                f"{g.get('gpu_util_mean_pct', '-'):>9}"
            )

    json_out = args.json_out or (
        f"concurrency_benchmark_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    Path(json_out).write_text(
        json.dumps(
            {
                "config": vars(args),
                "panels": [str(p) for p in panels],
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nresults -> {json_out}")


if __name__ == "__main__":
    main()
