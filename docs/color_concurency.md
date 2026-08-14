# Colorization server concurrency benchmark

Date: 2026-08-14. Runs: see `.output/20260814-2*` logs and
`.output/bench_{compare,sweep}_*.json` (gitignored, transient provenance).

## Question

The self-hosted FLUX.2 Klein server (`server/service.py`, BentoML) originally
processed **one request at a time** (`traffic={"concurrency": 1}`): the client
side (`pipeline_v1` colorize step) is strictly sequential, and a client-side
concurrency sweep showed flat throughput — extra connections only queued.
This experiment first adds server-side concurrency (a `/edit2` batch route +
a pipe pool) and benchmarks whether concurrency 2 improves throughput on the
DGX Spark's single GB10 GPU; it then reverts the default to concurrency 1 and
benchmarks `torch.compile` as the single-request speed lever. **Bottom line:
concurrency 2 is ~1.06× (small panels) / 0.98× (2 MP); `torch.compile` at
concurrency 1 is 1.26–1.40× — the bigger lever.**

## Environment

- Server host: DGX Spark — NVIDIA GB10, 120 GB unified memory, 1 GPU.
  Container `flux2-klein` (image `flux2-klein:latest`), BentoML 1.4.x,
  diffusers 0.39.0, torch 2.13.0+cu130, bf16.
- Deployment: step-distilled `black-forest-labs/FLUX.2-klein-9B` + thedeoxen
  manga-colorization LoRA (`mngclranm`), `FLUX2_STEPS=4` (guidance ignored by
  the distilled model), LoRA scale 1.0, PNG output.
- Client: `server/benchmark_concurrency.py` from this repo
  (`--mode compare` = serial `/edit` vs `/edit2` on the same panel set;
  `--mode sweep` = client-concurrency sweep). GPU utilization sampled on
  Spark via `ssh` + `nvidia-smi` at ~1 Hz.
- Payloads: 5 real panels from
  `pipeline_v1/output/20260808-221331/1_panels/0134-004/`
  (0.11–0.53 MP) + the same panels upscaled to ~2 MP
  (`.output/bench_payload_2mp/`, the pipeline's megapixel cap), each with the
  same reference atlas (`data/refs/frieren_reference.webp`), deterministic
  seeds. Warmup request(s) before every run (first call pays Triton compile).

## Baseline: server concurrency = 1 (before this work)

Sanity sweep (2 requests/level, small panels): throughput flat at
~0.13–0.14 req/s for client concurrency 1/2/4/8; per-request latency grew
with queueing (p50 10.1s → 14.6s). GPU util during a single request:
~84–96% (mean 84%). The server was the serialization point; the GPU was
already mostly busy per request.

## Method 1: `/edit2` batch route (2 pipeline instances, 2 threads)

Added `POST /edit2` (`server/service.py`): one request carries two edit jobs
(`images1/prompt1/seed1` + `images2/prompt2/seed2`, shared width/height/
steps/guidance/lora/format), run concurrently in a 2-thread pool, each job on
its **own** pipeline instance (thread safety: no shared mutable scheduler/
adapter state). Returns JSON `{"images": [base64, base64],
"job_latency_s": [...]}`. `/edit` unchanged.

Two bf16 pipes are ~67 GB resident (the 2nd pipe reuses shared text-encoder/
VAE tensors; single pipe was ~48 GB) — fits the 120 GB budget.

Results (10 panels, 4 steps, small panels):

| route | wall s | panels/s | req mean s | server job mean s | GPU busy mean |
|---|---|---|---|---|---|
| serial `/edit` ×10 | 87.17–87.29 | 0.115 | 8.72–8.73 | — | 82.8–86.2% |
| `/edit2` ×5 (2 jobs each) | 81.93–84.56 | 0.122 | 16.39–16.91 | 15.0–15.3 | 89.1–91.3% |

**Speedup: 1.03–1.07×** (two independent runs). Concurrent per-job latency
~15 s vs 8.7 s serial — jobs run almost completely back-to-back on the GPU;
only ~0.5 s of CPU-side gaps are overlapped per pair.

## Method 2: pipe pool + traffic concurrency 2 (concurrent HTTP requests)

Refactored the service to hand each in-flight request a pipe from a
`queue.Queue` pool (`traffic concurrency == FLUX2_NUM_PIPES`); `/edit` and
`/edit2` both acquire from the pool. Two concurrent `/edit` requests (the
shape a parallelized `pipeline_v1` client would produce) then run on distinct
pipes.

Client-concurrency sweep (10 requests, small panels):

| client conc | wall s | req/s | mean req s | GPU busy mean |
|---|---|---|---|---|
| 1 | 85.83 | 0.117 | 8.58 | 88.7% |
| 2 | 80.16 | 0.125 | 15.33 | 94.6% |
| 4 | 81.01 | 0.123 | 27.39 | 92.1% |

**Speedup at conc 2: 1.066×**; conc 4 gains nothing further (server caps at
2 in-flight; the rest queue).

## Method 3: true diffusers batching — infeasible as designed

Idea: batch two jobs into one `pipe(...)` call for better SM utilization.
Checked `Flux2KleinPipeline.__call__` / `prepare_image_latents` in diffusers
0.39: all conditioning images (edit target + references) are VAE-encoded into
**one** reference-token sequence that is `repeat(batch_size, ...)`ed — every
sample in a batch shares the same conditioning. Per-sample different
panel+atlas conditioning in one call is not supported without deep pipeline
surgery (the attention over concatenated tokens would mix jobs). Not pursued
further.

## Workload dependence: 2 MP panels (pipeline's real cap)

Same panels upscaled to ~2 MP (requested size capped to 2.0 MP, /16):

| route | wall s | panels/s | req mean s | server job mean s | GPU busy mean |
|---|---|---|---|---|---|
| serial `/edit` ×8 | 207.74 | 0.039 | 25.97 | — | 87.2% |
| `/edit2` ×4 (2 jobs each) | 211.31 | 0.038 | 52.82 | 47.73 | 86.8% |

**Speedup: 0.983× — slightly *slower*.** Per-job latency 47.7 s ≈ 1.84×
serial: pure SM contention, no measurable overlap. At the pipeline's realistic
resolution, concurrency 2 does not help.

## Summary

| workload | serial panels/s | concurrency-2 panels/s | speedup |
|---|---|---|---|
| small panels (0.11–0.53 MP), `/edit2` | 0.115 | 0.122 | **1.065×** |
| small panels, client conc 2 | 0.117 | 0.125 | **1.066×** |
| 2 MP panels (pipeline cap), `/edit2` | 0.039 | 0.038 | **0.983×** |

The GB10 is saturated by a single request (GPU busy 83–95% depending on
sampling window): one 4-step Klein edit already fills nearly all SMs, so
duplicated-pipe concurrency mostly *contends* rather than *overlaps*. The
only gain is filling the ~5–17% of CPU-side gaps (HTTP parse, PIL
encode/decode, prompt encode), which is worth ~6% at small panel sizes and
vanishes (even turns negative) at 2 MP.

## Revert to concurrency 1 + `torch.compile` (single-request speed)

Because concurrency 2 is not a real lever on this hardware, the server's
default went back to concurrency 1 (`FLUX2_NUM_PIPES` default 1; traffic
concurrency == pipe count). The remaining lever is making each request
faster with `torch.compile` (new in this service, `FLUX2_COMPILE` env):

- `FLUX2_COMPILE=1` compiles the transformer's `forward` — as a **method**,
  not the module, because replacing `pipe.transformer` wholesale hides the
  LoRA adapter registry (`set_adapters` → `get_list_adapters` finds nothing
  → HTTP 500).
- `FLUX2_COMPILE=2` additionally compiles the VAE encode/decode.
- `FLUX2_COMPILE_DYNAMIC=1` (default) uses dynamic shapes: different panel
  sizes reuse the compiled kernels instead of recompiling (static mode,
  `=0` + `reduce-overhead`, recompiles per new size — opt-in only).
- Compilation is lazy on the first inference (cold: ~73 s transformer,
  ~115 s with VAE); the Triton/inductor cache survives container recreates
  (observed 8.8 s first call after a recreate).

Results at concurrency 1, 4 steps (same payloads as above):

| workload | no compile | `=1` transformer | `=2` +VAE |
|---|---|---|---|
| small panels | 8.39 s / 0.119 req/s | 6.03 s / 0.166 (**1.39×**) | 6.00 s / 0.167 (**1.40×**) |
| 2 MP cap | 25.02 s / 0.040 | 19.81 s / 0.050 (**1.26×**) | 19.04 s / 0.053 (**1.31×**) |

`FLUX2_COMPILE=1` is the recommended deployment (default in
docker-compose.yml): VAE compilation adds only ~4% at 2 MP for +40 s of
first-call compile time. Different image sizes work without recompiles
(run 1 over 5 distinct panel sizes: 6.01 s mean vs 6.03 s steady state).

## Conclusions

1. **Concurrency 2** measurably improves small-panel throughput (~6%) and
   slightly hurts at 2 MP — the GPU, not request serialization, is the
   bottleneck on this GB10. The default is back to concurrency 1.
2. **`torch.compile` at concurrency 1 is the real win**: 1.26–1.40× per
   request (transformer-only, dynamic shapes), enabled by default
   (`FLUX2_COMPILE=1`), with a one-time cold-call compile cost.
3. The pipe-pool code remains for the opt-in concurrency-2 mode
   (`FLUX2_NUM_PIPES=2` enables `/edit2`), and is thread-safe (each request
   owns a pipe).

## Failure cases / caveats

- First call after container restart pays model load + Triton compile
  (27 s measured once); all steady-state numbers exclude warmup. With
  `FLUX2_COMPILE` on, the cold first call additionally pays inductor
  compilation (~73 s transformer, ~115 s +VAE); the cache survives container
  recreates.
- `/edit2` takes **both** pipes for its duration: a concurrent `/edit`
  request waits (no deadlock — traffic concurrency == pipe count), so don't
  mix parallel `/edit2` clients with `FLUX2_NUM_PIPES=2`.
- GB10 reports `memory.used` as `[N/A]` (driver "Not Supported"); GPU
  utilization sampling is ~1 Hz via ssh and noisy at short windows.
- nvidia-smi util on GB10 is a coarse SM-busy proxy; the conclusion rests on
  the consistent wall-time measurements across 5 runs, not on util alone.
- torch.compile: compile the transformer's `forward` method, not the module
  (module replacement hides the LoRA adapter registry and breaks
  `set_adapters` with HTTP 500).

## Cost / reproducibility

- $0 per call (self-hosted; electricity only) — same as `server/README.md`.
- Everything needed to reproduce: `server/service.py` + `server/benchmark_concurrency.py`
  (both committed), the payloads above, `FLUX2_STEPS=4`, seeds in the JSON
  results. Commands used, e.g.:

```bash
.venv/bin/python server/benchmark_concurrency.py --mode compare \
  --endpoint http://spark:3000 \
  --panels-dir pipeline_v1/output/20260808-221331/1_panels/0134-004 \
  --atlas data/refs/frieren_reference.webp \
  --requests 10 --warmup 1 --gpu-host spark \
  --json-out .output/bench_compare_n10_m3.json
```
