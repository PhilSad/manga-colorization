# Plan: bbox-guided region-edit in the verify loop (pipeline_v1)

> **Status: implemented 2026-08-16** (all four user decisions honored; code,
> offline tests, config, manifest accounting, docs updated. Live-run
> measurements still pending — see `pipelines.md` §"BBox probe" integration
> status; results will be recorded there when the paid run happens, never
> fabricated).
> Note: because the probes stay standalone (decision 3), `draw_boxes` /
> `region_instruction` / `parse_bbox_verdict` are *duplicated* into the
> library modules rather than imported by the probes — acceptable duplication
> chosen to keep the committed probe behavior untouched.
> Grounded in the two behavior probes of commits `99545a3` and `c483d3f`
> (`scripts/probe_luna_bboxes.py`, `scripts/probe_gpt_edit_bbox.py`), whose
> measured results are written up in `pipelines.md` §"BBox probe" (lines
> ~197–244).

## Goal

Wire the probed bbox experiment into `pipeline_v1`'s existing character-palette
verification loop as an **opt-in mode** (`--verify-mode bbox`): when Luna
judges a colorization mismatch, instead of re-colorizing the **whole panel**
with a fix prompt, have the verifier also emit **bounding boxes** of the
palette-wrong regions, draw them on the rejected image, and hand the boxed
image to `gpt-image-2` for a **region-scoped recolor** — then re-verify, and
iterate until the palette is verified or attempts run out.

This is a *mode* of the existing pipeline (documented in `pipelines.md`), not a
new method entry in `methods.md` — exactly like the fix-prompt loop it extends.

## Context (what already exists)

- `pipeline_v1/verify_loop.py` — `run_verify_loop(colorizer, verifier, panel,
  atlas, output, palette_instruction, max_attempts)`: colorize → verify →
  on mismatch re-colorize the **whole panel** with
  `canonical + "\n\n" + fix block` (fix prompt from the verdict or
  synthesized from `analyse`) → verify again. Attempts recorded in
  `<panel>.verify.json`, attempt images kept as `<stem>.attempt_<n><ext>`,
  final output copied to the canonical `<stem><ext>` the stitch step expects.
- `pipeline_v1/verify_color.py` — `ColorVerifier` (default
  `openai/gpt-5.6-luna`, OpenRouter, strict `json_schema`
  `COLOR_VERDICT_SCHEMA` = `analyse` + `good_color` + `fix_prompt`,
  `provider.require_parameters`, no `temperature`), `verify()` already
  accepts an `extra_body` (merged on top of the provider pin — tested), and
  the COL eval suite pins the shared schema. `ColorVerifier.verify(...)`
  returns `ColorVerifyRecord` with status `verified|mismatch|unparseable|error`.
- `pipeline_v1/gpt_colorizer.py` — `GptImage2Colorizer` (gpt-image-2
  `images.edit`/`generate`, minimal-size handling, atlas upload as
  `("atlas.jpg", buffer)` — the filename-carrying tuple is mandatory, see
  memory note b94e3c0), fixed `medium` quality, cost accounting.
- **Probes (committed, behavior-only)**:
  - `scripts/probe_luna_bboxes.py` — localization half: Luna with
    `reasoning: {effort: "high"}` + strict `BBOX_SCHEMA`
    (`analyse` / `good_color` / `regions[]` with `character`, `problem`,
    `fix_suggestion`, `bbox` [x1,y1,x2,y2] in normalized 0–1000) +
    `parse_bbox_verdict` + `draw_boxes` (Pillow overlay, cycling high-contrast
    colors). **Measured**: works — `$0.00176`, 30.0s, 9656 tokens (2588
    reasoning); `max_tokens=2048` truncates → **8192 required**.
  - `scripts/probe_gpt_edit_bbox.py` — edit half: `draw_boxes` output +
    labelled atlas → `GptImage2Colorizer` (`images.edit`, no mask, boxes are
    the only locator) with `region_instruction()` + edit prompt. **Measured**:
    edit landed exactly where told (both Eisen-beard regions fixed), `$0.04593`
    (medium, 672×1008, 38.5s); Luna re-probe `$0.00278` found only the missed
    Frieren-hair region.
  - **Key finding / bottleneck**: *region recall, not editing* — the one-shot
    localization pass missed Frieren's hair. Any integration must iterate and
    must handle "mismatch but empty regions" gracefully.

## Design overview

### 0. User decisions (2026-08-16)

1. **Merged single call** — one Luna call per retry carries the verdict,
   `fix_prompt`, and `regions` in a single `BBOX_VERDICT_SCHEMA` (no separate
   localization call; cheapest, 1 Luna call per retry).
2. **Full-page mode only** — `--verify-mode bbox` requires `--full-page`;
   panel-wise (FLUX) mode keeps the fix-prompt loop unchanged. Validation
   rejects bbox mode without full-page.
3. **Probes stay standalone** — no refactor of `probe_luna_bboxes.py` /
   `probe_gpt_edit_bbox.py`; the library modules carry their own copies of
   the shared helpers (duplication accepted).
4. **Empty-regions fallback = fix-prompt re-colorize** (recommendation
   accepted).

### 1. Verifier: bbox verdict schema (new, shared schema untouched)

Add to `verify_color.py` a second strict schema `BBOX_VERDICT_SCHEMA` — the
probe schema **plus `fix_prompt`** (so the empty-regions fallback and the
probe's exact recipe coexist):

```jsonc
{
  "type": "object",
  "properties": {
    "analyse":      { "type": "string" },   // which palettes are wrong & where
    "good_color":   { "type": "boolean" },
    "fix_prompt":   { "type": "string" },   // NEW vs probe: full-panel fallback
    "regions":      { "type": "array", "items": { ...probe region schema... } }
  },
  "required": ["analyse", "good_color", "fix_prompt", "regions"],
  "additionalProperties": false
}
```

- `ColorVerifier` gains a `schema`/`response_format` parameter (default =
  the existing `COLOR_VERDICT_SCHEMA`) and a `reasoning_effort` parameter
  (default `None`; bbox mode passes `"high"` via the existing `extra_body`
  plumbing). **Zero impact** on the COL eval suite and the fix-prompt mode.
- Move `parse_bbox_verdict` (from the probe) into `verify_color.py`; the
  record gains `regions: list` (default `[]`) serialized in `to_dict`.
- New prompt file `pipeline_v1/verify_bbox_prompt.txt` — the probe's
  `probe_luna_bbox_prompt.txt` content plus a line asking for `fix_prompt`
  ("if regions are empty, write a full-panel fix_prompt naming character +
  canonical colors") — this is the loop's safety net for recall misses.

### 2. New module `pipeline_v1/region_edit.py` (library, from the probe logic)

- `draw_boxes(image, regions, out, width=...)` — moved from the probe
  (normalized 0–1000 → pixel coords, cycling high-contrast colors, numbered
  regions so the prompt's "Region N" labels match the boxes).
- `region_instruction(regions)` — moved from the probe (numbered list for the
  edit prompt's slot).
- `GptImage2RegionEditor` — gpt-image-2 `images.edit` wrapper reusing
  `gpt_colorizer.py`'s request plumbing (atlas tuple upload, minimal-size,
  medium quality, cost recording): `edit(boxed_image, atlas, output,
  region_instruction, palette_instruction)`.
  - **Resolution rule**: boxes are drawn on the image at the resolution
    actually sent to gpt-image-2 (normalized coords scale exactly, so
    upscale-first-then-draw keeps Luna's boxes pixel-correct).
- New prompt file `pipeline_v1/gpt_region_edit_prompt.txt` — the probe's
  `probe_gpt_edit_prompt.txt` content (recolor ONLY inside the red
  rectangles; boxes are the only locator; remove the boxes from the output;
  use the atlas for canonical colors), with a `{region_instruction}` slot and
  an optional `{palette_instruction}` slot (canonical palette text, same as
  the colorize call).
- Per decision 3 the probe scripts stay standalone: they are **not**
  refactored to import the library helpers; the duplication is accepted.

### 3. Verify loop: `verify_mode` parameter (`verify_loop.py`)

`run_verify_loop` gains `verify_mode="fix-prompt" | "bbox"` and
`region_editor=None` (required for bbox mode). In **bbox mode**, the loop is
identical through the first verify call; on a mismatch the retry path
branches:

1. `verdict.regions` non-empty → `draw_boxes(attempt_output, regions)` →
   `<stem>.attempt_<n>.boxed.png` → `region_editor.edit(boxed, atlas,
   attempt_output_next, region_instruction, palette_instruction)` →
   next attempt is a **region edit** (record `retry_backend:
   "gpt-image-2-region-edit"` + `edit_cost_usd` + boxed-image path).
2. `verdict.regions` empty (recall miss — the probed failure mode) → fall
   back to the **current** fix-prompt full re-colorization
   (`_apply_fix(palette_instruction, fix_prompt)`), recorded with
   `retry_backend: "fix-prompt"`. The loop always has a retry path.
3. `good_color: true` → `verified` (stop). Verifier error/unparseable →
   stop, keep latest output (unchanged semantics). Attempts exhausted →
   `mismatch` (unchanged).

The iteration naturally handles the probe's recall bottleneck: after the edit
fixes the localized regions, the next verify call sees the *edited* image and
can box the remaining errors (exactly what the probe's re-probe showed for
Frieren's hair).

Everything else (attempt files, verify.json attempt docs, final-copy to the
canonical name, `_synthesize_fix` fallback) is untouched.

### 4. Config (`config.py`, `run.py` CLI) — all defaults preserve today's behavior

| flag | default | meaning |
|---|---|---|
| `--verify-mode` | `fix-prompt` | `bbox` enables region-guided retries (requires `--verify-attempts >= 1`) |
| `--verify-bbox-prompt-file` | `verify_bbox_prompt.txt` | bbox verdict prompt |
| `--verify-reasoning-effort` | `high` (bbox mode only) | passed as `reasoning.effort` |
| `--verify-max-tokens` | `1024` (8192 when `--verify-mode bbox`) | completion cap — the probe proved 8192 is required for high-effort bbox output |
| `--region-edit-prompt-file` | `gpt_region_edit_prompt.txt` | edit prompt template |
| `--region-edit-model` | `gpt-image-2` (reuses `--gpt-model` value when set) | editor model |
| reuse `--gpt-size`, `--gpt-atlas-scale`, `--openai-api-key-env` | | editor size/atlas/key |

Validation: `--verify-mode bbox` requires the OpenAI key (editor is paid
gpt-image-2) and `--verify-attempts >= 1`; `--verify-reasoning-effort` only
meaningful in bbox mode.

### 5. Backends (`run.py`, `mock_backends.py`, `orchestrator.py`)

- `Backends` gains `region_editor` (built only in bbox mode; real =
  `GptImage2RegionEditor`, mock = deterministic `MockRegionEditor` recording
  calls).
- `_build_verifier` builds the bbox variant (schema, prompt, 8192 tokens,
  high reasoning) when `--verify-mode bbox`.
- Orchestrator manifest: totals gain `region_edit_calls`,
  `region_edit_cost_usd`; prompt-hashes gain the bbox + edit prompt files;
  `pricing_assumptions` gains a `region_edit` block (OpenAI gpt-image-2,
  fixed price by size, measured `$0.04593` @ 672×1008 medium from the probe).

### 6. Recording

- `3_colorized/<page>/<panel>.verify.json`: each attempt doc gains `regions`,
  `boxed_image` (file record), `retry_backend`, `edit_prompt`,
  `edit_cost_usd` when a region edit happened.
- `<stem>.attempt_<n>.boxed.png` kept as provenance (in addition to the
  attempt images).
- Step `summary.json` + manifest `totals`: `region_edit_calls`,
  `region_edit_cost_usd`, plus existing verify totals.
- Console: existing `[verify]` progress lines gain the region count and
  retry backend (e.g. `MISMATCH (attempt 1): 2 regions → gpt-image-2 region
  edit`).

## Tests

Offline (`.venv/bin/pytest pipeline_v1/tests -q`, must stay green — no
network, no paid calls):
- `test_verify_color.py`: pin `BBOX_VERDICT_SCHEMA` (properties, required,
  strict, region item schema); `parse_bbox_verdict` (valid, missing bbox,
  out-of-range clamping, fenced JSON, malformed → None); verifier request
  shape with bbox schema + `reasoning: {effort: "high"}` in `extra_body`
  merged with `require_parameters`.
- New `test_region_edit.py`: `draw_boxes` 0–1000 → pixel mapping, box
  drawing + numbering, missing-bbox skip; `GptImage2RegionEditor.edit`
  request shape (images.edit endpoint, boxed image first, atlas uploaded as
  `("atlas.jpg", buffer)` — the mimetype regression, editor prompt slots
  filled, cost recorded).
- `test_verify_loop.py` (mock backends): bbox-mode flows — verified on
  attempt 1 → no editor call; mismatch + regions → boxed image written,
  editor called with boxed + atlas, verify called again, `retry_backend`
  recorded; mismatch + empty regions → fix-prompt fallback (no editor call);
  region edit fixes it → verified; attempts exhausted → mismatch; verifier
  error → stop without edit.
- Config validation tests (`--verify-mode bbox` without key/attempts);
  mock end-to-end `--verify-mode bbox --verify-attempts 2 --mock --limit 1`.
- Probe scripts still run with their CLI after the refactor (import check).

Integration (real network, paid — run after approval, results recorded, not
fabricated): one real full-page run on 1–2 pages with
`--verify-attempts 3 --verify-mode bbox` reusing a page with a known Eisen
beard / Frieren hair catch (e.g. vol 1 p003 or the probe's p010); assert the
attempt log contains regions, boxed images, edit cost, and the final page
verified. Optionally a fixture case (following the DET-005..010 procedure)
once the live run confirms stability.

## Docs

- `pipeline_v1/README.md`: bbox-mode section (flags, semantics, recording,
  cost warning for panel-wise mode).
- `pipelines.md`: extend the existing probe write-up with the integration
  status; live-run table entry with measured numbers after the run
  ("pending next live run" note otherwise — never fabricated).
- `pipeline_v1/ARCHITECTURE.md`: module map entries for `region_edit.py`,
  the verifier bbox variant.
- This plan file (`docs/plans/` convention); mark implemented with measured
  results when done.

## Cost (from the probe measurements)

Per bbox-mode retry in full-page mode:
- 1 Luna bbox call — `$0.00176` measured (8192 tokens, high effort; the
  2048-token variant wasted `$0.00208` on an unparseable response).
- 1 gpt-image-2 region edit — `$0.04593` measured (672×1008 medium, 38.5s).
- Total ≈ **$0.0477/retry**, vs ≈ $0.0518 for a fix-prompt full re-colorize
  (1 Luna call ≈ $0.001 + 1 gpt-image-2 call ≈ $0.0499) — slightly cheaper
  *and* targeted, at the price of possible extra iterations when localization
  recall misses (the probed bottleneck; the empty-regions fallback + loop
  iteration cover it).
- Panel-wise (FLUX) mode: the first colorize stays $0 (self-hosted), but each
  bbox retry introduces paid gpt-image-2 edits per panel — README warning
  like the full-page retry warning, not blocked.

## Open questions for the user

1. **Single merged call vs. separate localization call?** I recommend the
   merged `BBOX_VERDICT_SCHEMA` (verdict + regions + fix_prompt in one Luna
   call per retry): cheapest (1 Luna call), the fix_prompt fallback is free,
   and localization is grounded in the same reasoning pass that found the
   errors. The probed experiment used a *separate* localization call (2 Luna
   calls per retry); that variant is possible later via
   `--verify-bbox-prompt-file`, but I'd not default to it.
2. **Panel-wise mode allowed?** bbox retries switch the retry backend to paid
   gpt-image-2 per panel. Allow with a README cost warning (my
   recommendation), or restrict bbox mode to full-page for now?
3. **Probe scripts refactor**: move `draw_boxes`/`parse_bbox_verdict`/
   `region_instruction` into `verify_color.py`/`region_edit.py` and have the
   committed probes import them (one implementation, same CLI), or leave the
   probes standalone and copy? I recommend the refactor.
4. **Fallback when regions are empty** — fix-prompt full re-colorize (my
   recommendation) vs. treating it as a verifier error (stop) vs. one more
   localization attempt?
