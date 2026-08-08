# Character detection in panels via pairwise reference matching (Qwen3.6-27B on Spark vLLM)

Companion experiment (not a colorizer): given one manga panel, detect which of the
reference characters (from `data/refs/`) appear in it. This method differs from
`character-detection-openrouter-vlm` in the fundamental setup: instead of one
request per panel listing all characters, it makes **one request per
(panel, character) pair**, each showing the panel **and the reference image** of
that one character, and asks a yes/no question. The reference image is the
discriminator — no memorized character hints are needed.

The model runs **self-hosted** on the DGX Spark: a vLLM (OpenAI-compatible)
server exposing `qwen/qwen3.6-27b` (`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`,
multimodal, NVFP4 quantized, DFlash speculative decoding) at `http://spark:8000/v1`.
The endpoint advertises `limit_mm_per_prompt: {image: 2}`, which is exactly the
two images (panel + reference) each request sends.

## Method

- Reference characters are derived from `data/refs/` with the same canonical-name
  logic as the OpenRouter method (`*_reference`/`*_anime_profile` suffixes
  stripped, deduped case-insensitively). Each canonical character maps to one
  reference image. Note: Frieren has two references, so the list contains both
  `Frieren` (manga lineart ref) and `Frieren Anime Profile` (colored anime ref) —
  each is probed separately.
- For every (panel, character) pair, one request with two images asks:

  > does the character from image #2 appear in the panel from image #1?

  with `response_format=json_object`, expecting
  `{"present": bool, "confidence": float, "reason": str}`.
- Requests are dispatched concurrently over a `ThreadPoolExecutor`
  (`--workers`, default 6; the server's `max_num_seqs=8` bounds useful
  concurrency). Responses are parsed defensively (fenced/embedded JSON survives
  leading reasoning tokens).
- RGBA reference/panel images are flattened onto white before encoding
  (VLMs handle alpha inconsistently).
- Per panel, characters with `present: true` are aggregated into the final list.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r character_detection_methods/character-detection-qwen3-vllm-ref-pair/requirements.txt
```

No API key needed (self-hosted endpoint; the script reads `.env` if present but
does not require it).

## Run

```bash
.venv/bin/python character_detection_methods/character-detection-qwen3-vllm-ref-pair/run.py
```

Options: `--endpoint` (default `http://spark:8000/v1`), `--model` (default
`qwen/qwen3.6-27b`), `--workers` (default 6), `--input-dir` (default
`data/panels`), `--refs-dir` (default `data/refs`), `--skip-first N`,
`--limit N`, `--max-tokens`, `--temperature`, `--timeout`.

Every invocation creates a fresh `output/YYYYMMDD-HHMMSS/` directory containing:

- `manifest.json` — configuration, prompt template, inputs, references,
  per-call records, totals;
- `calls/<panel>__<character>.json` — one file per probe (raw response text,
  parsed verdict, tokens, latency);
- `<panel>.json` — per-panel summary with `characters` (present list) and
  `per_character` details;
- `results.csv` — flat per-probe table;
- `characters_per_panel.csv` — the headline output: panel → present characters.

## Cost

**Measured: $0.00** per call — self-hosted vLLM inference on the DGX Spark
(electricity only; do not compare with paid API pricing). Token usage and
latency are recorded per call in the manifest and CSV.

## Notes

- First request pays model warm-up latency (the server was started before the
  run); subsequent requests are fast.
- `present` is a strict JSON boolean; `confidence` is the model's own
  self-reported value (0–1) and is not calibrated.
- The model may answer `unparseable` (non-JSON) or `error` (transient
  connection failures); these are recorded per call, not fatal, and count as
  "not present" in the per-panel list.
