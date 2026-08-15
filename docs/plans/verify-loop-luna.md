# Plan: Luna verification loop with fix prompts in pipeline_v1

> **Status: implemented 2026-08-15** (offline tests green:
> `pytest pipeline_v1/tests -q`). Deviations from this plan:
> - The loop also synthesizes a fix prompt from `analyse` when a mismatch
>   verdict carries an empty `fix_prompt` (planned; shipped as
>   `verify_loop._synthesize_fix`).
> - `steps/colorize.py` keeps the loop generic over the `Colorizer` protocol;
>   full-page mode is supported (one synthetic panel per page) but its extra
>   ~$0.05/page retry cost is only warned about in the README, not blocked.
> - New integration case VER-001 is not yet added to the suite; the loop is
>   covered offline end-to-end (mock backends). **Live run done**
>   (`output/20260815-165721`, full-page gpt-image-2 mode, 5 pages,
>   `--verify-attempts 3`): 6 Luna calls, **$0.00613 total ≈ $0.00102/call**
>   (paid OpenRouter tier); p003 was caught as a palette mismatch on
>   attempt 1, re-colorized with the fix prompt and verified on attempt 2;
>   the other 4 pages verified on attempt 1. One retry = +$0.05074
>   (6 gpt-image-2 calls $0.30056).
> - `run_verify_loop` prints `[verify] …` progress lines (including the fix
>   prompt block) — the "output the prompt to fix it" behaviour.
> - Manifest `totals.panels_colorized` counts distinct panels
>   (`successful_calls - colorization_retries`), not colorize calls.

## Goal

Add a **verification loop** to the pipeline's colorize stage: after a panel is
colorized, ask `openai/gpt-5.6-luna` (via OpenRouter, **strict structured
output**) whether every character in the panel has its canonical Frieren
palette. When the verdict is "not good", the loop **outputs a fix prompt**
(the corrective instruction, printed to the console and written to the run
dir) and — if attempts remain — re-colorizes the panel with that fix prompt
appended to the prompt, then verifies again. **Every Luna response and every
colorization attempt is recorded** (raw response text, verdict fields, usage,
cost, latency, model, and per-attempt colorize records with the exact prompt
used).

This is a *mode* of pipeline_v1 (documented in `pipelines.md`), not a new
method entry in `methods.md`.

## Context (what already exists)

- `pipeline_v1/verify_color.py` — `ColorVerifier` (default model
  `openai/gpt-5.6-luna`, OpenRouter `API_BASE`), one generic prompt
  (`verify_color_prompt.txt`), strict `json_schema` structured output with
  `provider.require_parameters: true`, no `temperature` (luna does not support
  it), retry/backoff + `usage.cost` accounting via `characters.call_vlm`.
  Verdict schema today: `analyse: str` + `good_color: bool`. Statuses:
  `verified | mismatch | unparseable | error`. Used by the COL-001..004
  evaluation suite (`test_integration_color.py`).
- `pipeline_v1/steps/colorize.py` — `_process_page` loops panels, builds the
  filtered atlas, renders the canonical-palette instruction via
  `profiles.palette_instruction`, calls `colorizer.colorize(panel, atlas,
  output, palette_instruction=palette)` and writes one record per panel into
  `3_colorized/` + `summary.json`. This is where the loop hooks in.
- The FLUX colorizer prompt (`colorizer_prompt.txt`) has a free-form
  `{character_profiles}` slot — the fix prompt can ride in that slot with
  **zero template changes**.

## Design overview

### 1. Structured output: add `fix_prompt` to the verdict

Extend the shared `COLOR_VERDICT_SCHEMA` (backwards-compatible superset — the
COL evaluation suite only reads `analyse`/`good_color`):

```jsonc
{
  "type": "object",
  "properties": {
    "analyse":    { "type": "string" },   // which palettes are right/wrong
    "good_color": { "type": "boolean" },
    "fix_prompt": { "type": "string" }    // NEW: concise corrective instruction
  },                                       //      to paste into the colorizer
  "required": ["analyse", "good_color", "fix_prompt"],
  "additionalProperties": false
}
```

- `fix_prompt` is empty when `good_color` is true; when false it names the
  character(s) and the exact canonical colors to apply (e.g. "Frieren: hair
  silver-white, eyes teal — current hair is lavender"). The model itself
  writes it, so it is grounded in what it actually saw.
- Update `verify_color_prompt.txt` to ask for the third field.
- `parse_color_verdict` gains `fix_prompt` (default `""`); `ColorVerifyRecord`
  gains a `fix_prompt` field serialized in `to_dict`.
- Fallback: if luna returns `good_color: false` but an empty `fix_prompt`,
  the loop synthesizes one from `analyse`
  ("Corrections from the verification pass: {analyse}") so the loop always has
  something to output.

### 2. New module `pipeline_v1/verify_loop.py`

Per-panel loop, unit-testable with mocks:

```
def run_verify_loop(*, colorizer, verifier, panel, atlas, output,
                    palette_instruction, max_attempts) -> VerifyLoopResult
```

Behaviour per attempt `n` (1..max_attempts):
1. **Colorize** → writes `<stem>.attempt_<n><ext>` (attempt 1 is written
   directly to `<stem><ext>` — the canonical name stitch expects; later
   attempts go to `attempt_<n>` files, and the final attempt is copied to
   `<stem><ext>` at the end, so the stitch step is untouched).
2. **Verify** (luna, structured) with the colorized image + monochrome crop +
   the same atlas the colorizer saw as context (`ColorVerifier.verify`).
3. Record the pair (colorize doc + full verdict doc) in the attempt log.
4. `good_color: true` → stop, outcome `verified`.
   `good_color: false` → print the fix prompt banner, write the fix prompt,
   and if `n < max_attempts` re-colorize with
   `palette_instruction = canonical + "\n\n" + fix block` (fix prompt wrapped
   as authoritative: "Correction from verification (overrides conflicting
   colors above): …"), same atlas; else stop, outcome `mismatch`.
5. Verifier `error`/`unparseable` → stop the loop for this panel (recorded),
   outcome `verifier_error`; the panel keeps its latest colorization. No
   retry burning.

`VerifyLoopResult`: final `ColorizeRecord`, `outcome`, `attempts` (list of
`{attempt, colorize, prompt_used, verify}` dicts), last `fix_prompt`.

### 3. Hook into `steps/colorize.py`

- When `config.verify_attempts > 0`, `_process_page` calls
  `run_verify_loop` instead of the single `colorize`, then:
  - writes `3_colorized/<page>/<panel>.verify.json` — the **complete** attempt
    log (every colorize record incl. prompt used + every luna raw
    `response_text`, verdict, usage, `cost_usd`, latency, model, timestamp);
  - writes `3_colorized/<page>/<panel>.fix_prompt.txt` — the last fix prompt
    (copy-paste convenience);
  - keeps the attempt images (`<panel>.attempt_2.png`, …) as provenance —
    "all colorization attempts recorded";
  - prints the fix prompt to stdout when a mismatch is judged
    ("output the prompt to fix it").
- Step `summary.json` gains a `verify` block with totals
  (`verify_calls`, `successful_verify_calls`, `verified_panels`,
  `mismatch_panels`, `verifier_error_panels`, `fix_prompts_output`,
  `colorization_retries`, `verify_cost_usd`).
- Full-page mode works unchanged: the loop is generic over the `Colorizer`
  protocol (one synthetic panel per page). Documented cost warning
  (gpt-image-2 retry ≈ $0.05/page).

### 4. Config (`config.py`) — new flags, all default off (no behaviour change)

| flag | default | meaning |
|---|---|---|
| `--verify-attempts` | `0` (off) | max total colorization attempts per panel. `0` = today's behaviour. `1` = verify each panel once, output fix prompt only (no re-colorize). `N` = verify + re-colorize up to N attempts |
| `--verify-model` | `openai/gpt-5.6-luna` | OpenRouter model for the verifier |
| `--verify-prompt-file` | `verify_color_prompt.txt` | prompt template (shared with the COL eval; now also asks for `fix_prompt`) |
| `--verify-api-key-env` | `OPENROUTER_API_KEY` | env var for the key |
| `--verify-max-tokens` | `1024` | completion cap |

Validation: `verify_attempts >= 0`; `--verify-attempts` requires the
OpenRouter key (already enforced for detection; same key reused).

### 5. Backends (`run.py`, `mock_backends.py`, `orchestrator.py`)

- `Backends` dataclass gains a `verifier` field.
- `build_backends`: when `verify_attempts > 0`, build the real
  `ColorVerifier(model=config.verify_model, api_key=…, max_tokens=…)`.
- Mock: `MockColorVerifier` — deterministic, records calls; default verdict
  `good_color: true` (loop stops after attempt 1); injectable
  per-page/per-panel verdicts (`good_color`, `fix_prompt`) for tests.
- Orchestrator manifest: new totals keys (`verify_calls`,
  `successful_verify_calls`, `verified_panels`, `mismatch_panels`,
  `verifier_error_panels`, `fix_prompts_output`, `colorization_retries`,
  `verify_cost_usd`), `verify_prompt_file` hash in `prompt_hashes`, and a
  `pricing_assumptions.verification` block (OpenRouter, measured
  `usage.cost`, model id, note).

### 6. Recording — everything lands in the run dir

- `3_colorized/<page>/<panel>.verify.json` — full attempt log (see above).
- `3_colorized/<page>/<panel>.fix_prompt.txt` — last fix prompt.
- `3_colorized/<page>/<panel>.attempt_<n>.<ext>` — every intermediate
  colorization (final attempt doubles as `<panel>.<ext>`).
- Step `summary.json` `verify` block + manifest `steps.colorize.verify` +
  `totals.*` — aggregate numbers incl. measured `verify_cost_usd`.
- Console: per-panel `[verify] … MISMATCH → fix prompt:` block (the "output
  the prompt to fix it" behaviour).

## Tests

Offline (`.venv/bin/pytest pipeline_v1/tests -q`, must stay green):
- `test_verify_color.py`: update the schema-pin test (3 properties, 3
  required, `fix_prompt` type), parser tests for `fix_prompt` default/empty,
  one request-shape test asserting the third field is requested.
- New `test_verify_loop.py` (mocks): verified on attempt 1 → 1 colorize +
  1 verify, outcome `verified`; mismatch → fix prompt printed/returned, second
  colorize receives canonical + fix block, verify called again; attempts
  exhausted → outcome `mismatch`, final output still on disk for stitch;
  verifier error → loop stops, outcome `verifier_error`, no extra colorize;
  attempt files + verify.json written with full raw `response_text`.
- Config validation tests; mock-verifier wiring test in `test_orchestrator.py`
  / end-to-end mock run (`--verify-attempts 2 --mock`).

Integration (real network, `-m integration`, paid): one new case `VER-001`
in `evaluation/v1_1_cases.json` — committed crop + forced characters, real
FLUX + real luna, `verify_attempts=2`; asserts the attempt log is complete
(every `response_text` present), the fix prompt exists on mismatch, and the
loop terminates. Uses the existing `integration_run` fixture (per-worker
timestamped dirs) and `record_color`-style recording. Cost note: 1-2 FLUX
calls + 1-2 luna calls per case.

## Docs

- `pipeline_v1/README.md`: verification-loop section (flags, semantics,
  recording layout, cost).
- `pipelines.md`: short write-up + run table entry when a real run exists;
  "pending next live run" note otherwise (no fabricated numbers).
- This plan file; `docs/plans/` convention.

## Cost

- Each verify call: paid OpenRouter `usage.cost` (measured per call,
  recorded; luna is a small VLM — expect ≈ $0.0005–0.002/call with 3 images,
  measured at run time, never estimated in the manifest).
- Each retry: one more FLUX colorize (self-hosted, $0 + electricity) or one
  more gpt-image-2 call in full-page mode (~$0.05/page) — the reason the
  default is off.

## Open questions for the user

1. **Auto-retry vs. output-only?** `--verify-attempts N` covers both (N=1 =
   verify + output fix prompt only; N≥2 = also re-colorize with it). Default
   `0` (off) so existing runs are byte-identical. OK?
2. **Shared schema change**: adding `fix_prompt` to the COL-eval verdict
   schema (superset, backwards compatible) vs. a separate loop-only schema.
   I recommend the shared superset.
3. **Attempt images kept** as `attempt_<n>` files (extra storage) vs.
   overwrite (only final kept, JSON still logs all attempts). I recommend
   keeping them.
