# Character detection in panels via OpenRouter VLMs

Companion experiment (not a colorizer): given one manga panel, detect which of the
reference characters (from `data/refs/`) appear in it, using OpenRouter
vision-language models. Paid tier for models that have one (user-funded); `:free`
variants for the two NVIDIA models that have no paid endpoint. This is the first
stage of a character-conditional colorization pipeline: knowing who is in a panel
lets a later stage apply the right reference colors.

## Method

One image per API call. The panel is sent as the only image alongside a prompt that:

1. lists the canonical reference characters (derived from `data/refs/`, deduped) with
   short distinguishing hints from the anime;
2. asks for a structured JSON output in exactly this shape:

```json
{"characters": ["Frieren", "Fern"]}
```

The response is parsed, validated against the reference list, and saved per
model × panel. Unknown/non-reference names the model emits are recorded as
`unknown_entries` (hallucination signal). Free-tier models can be overloaded or
rate-limited; per-call failures are recorded instead of aborting the run.

## Models tested (2026-08-08)

| Model | Tier | Result |
|---|---|---|
| `google/gemma-4-31b-it` | paid | see run manifest / `experience_classification_log.md` |
| `google/gemma-4-26b-a4b-it` | paid | [20260808-205113](output/20260808-205113/) — 4/4 ok, **$0.00032658 total**. More conservative than 4-31b-it: panel_0001 `[Fern]` (31b says `Frieren`), panel_0003 misses `Stark`, panel_0004 finds only `Fern, Stark` (31b adds `Frieren, Wirbel`). |
| `google/gemma-4-31b-it` (local re-run, Spark vLLM) | self-hosted | [20260808-204250](output/20260808-204250/) — the same single-shot method pointed at the Spark `qwen/qwen3.6-27b` vLLM endpoint via `--api-base` (see below). Panels 1–3 ok (`Fern` / `Fern, Frieren` / `Stark, Fern, Frieren`); the endpoint died mid-run before panel_0004, so that record is an error. |
| `nvidia/nemotron-nano-12b-v2-vl:free` | `:free` only (no paid endpoint) | |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | `:free` only (no paid endpoint) | |

Verified against OpenRouter `/models` (2026-08-08): `nemotron-nano-12b-v2-vl`
and `nemotron-3-nano-omni-30b-a3b-reasoning` have **no paid endpoint** — only
`:free` variants exist, so those are used. A request to the paid id returns
`404 No endpoints found`.

## Running against a local OpenAI-compatible endpoint (e.g. Spark vLLM)

The same script works against any OpenAI-compatible server: point `--api-base`
(and `--models`) at it, use a dummy `--api-key-env` value, and pass `--sleep 0`
since there are no rate limits. Example:

```bash
export DUMMY_API_KEY=local
.venv/bin/python character_detection_methods/character-detection-openrouter-vlm/run.py \
  --api-base http://spark:8000/v1 \
  --models qwen/qwen3.6-27b \
  --api-key-env DUMMY_API_KEY \
  --sleep 0 --max-tokens 4096
```

Notes:
- Reasoning models (Qwen3.x) can burn `max_tokens` on a thinking trace before
  emitting the JSON answer — raise `--max-tokens` (4096 worked; 2048 caused
  `unparseable` records at the token cap).
- The manifest `pricing_assumptions.note` switches automatically to a
  self-hosted note (electricity only, no per-call billing) when `--api-base` is
  not OpenRouter; such calls are reported as `unpriced_calls`.
- The endpoint must be up before the run; a request failure is recorded per call
  (with backoff retries) rather than aborting the run.

Earlier attempts: the original API key was stale (`401 User not found`); after
replacing it, the `:free` variants were upstream rate-limited (`429 Provider
returned error`). Failed attempts are preserved in `output/20260808-195921/`,
`output/20260808-200651/`, and `output/20260808-200712/`.

The `/models` metadata snapshot for the run is stored in the run directory
(`models_metadata` in `manifest.json`) and records each model's
`architecture.input_modalities` — i.e. whether the model advertises image input.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r character_detection_methods/character-detection-openrouter-vlm/requirements.txt
```

`run.py` loads `OPENROUTER_API_KEY` from the repository `.env` file.

## Run

```bash
.venv/bin/python character_detection_methods/character-detection-openrouter-vlm/run.py
```

Options: `--models` (space-separated OpenRouter ids, defaults to the four models
above — paid where available, `:free` where not), `--input-dir` (default
`data/panels`), `--refs-dir` (default `data/refs`), `--skip-first N`, `--limit N`,
`--max-tokens`, `--temperature`, `--sleep` (seconds between calls, default 2).

Every invocation creates a fresh `output/YYYYMMDD-HHMMSS/` directory containing:

- `manifest.json` — full configuration, prompt, models metadata, per-call records;
- `<model>/<panel>.json` — one file per model × panel with raw response and parsed result;
- `results_by_model.json` — nested `{model: {panel: [characters]}}` summary;
- `results_flat.csv` — panel, model, status, characters, tokens, latency, cost.

## Cost

Mixed tier: the two gemma models run paid (user added funds), the two NVIDIA
models are `:free` ($0, rate-limited ~20 req/min, may queue). Per-call cost is
taken from `usage.cost` when OpenRouter reports it, otherwise computed from the
model's published `/models` per-token pricing times the measured tokens. Image
input is billed as tokens. Costs are recorded per call in the run manifest and
summary CSV; totals are in the manifest.

## Notes

- Panels with alpha are sent as-is (webp/png).
- Some models may not support `response_format=json_object`; `run.py` records the
  failure or retries without the response format and marks
  `response_format_used: none` in the record.
