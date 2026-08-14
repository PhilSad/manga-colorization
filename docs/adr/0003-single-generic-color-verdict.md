# Single generic color verdict with strict structured output

Status: accepted (2026-08-14)

The color evaluation cases (COL-*) were verified by two OpenRouter
vision-language verdicts: a palette-adherence verifier (the fixture's
required/forbidden colors rendered into the prompt, COL-001..003) and a
left-to-right geography verifier (an ordered per-character hair-color
expectation, COL-004). Both called `openai/gpt-5.6-luna` through OpenRouter
in legacy `json_object` mode and parsed the free-form JSON with tolerant
regex extraction.

They are now verified by one verifier with one generic prompt — "I tried to
automatically colorize this panel from the Frieren manga. Are all the
characters' color palettes correct?" — and a **strict structured output**
(`analyse: str`, `good_color: bool`) enforced through OpenRouter's
`json_schema` response_format with `provider.require_parameters: true`, so
the request only routes to endpoints that natively support structured
outputs and never silently degrades to loose JSON. The fixture's
required/forbidden/left-to-right expectations remain in
`v1_1_cases.json` as human-readable documentation but are no longer
rendered into the prompt.

Why: the two verifiers duplicated the call machinery, coupled the eval to
per-case prompt rendering, and kept two parallel verdict vocabularies. A
single holistic judgment relies on the model's own knowledge of the Frieren
canon — one code path, one schema, one prompt — and the strict schema
removes the unparseable-verdict class in practice.

**Considered options**

- *Keep two verifiers* — preserves the adherence/geography distinction in
  structured form, but keeps the duplicated machinery and per-case prompt
  coupling the simplification removes.
- *Single verifier, fixture hints still rendered* — keeps expectations in
  the prompt but retains the per-case rendering we wanted to delete.
- *Single generic verifier with strict structured output (chosen)* — one
  prompt, one schema, one code path; the cost is that palette-adherence vs
  palette-geography survives only as the fixture's per-case `failure` tag,
  and the structured per-position geography check (`per_position`) is gone —
  spatial nuance now lives only in the `analyse` prose.

**Consequences / notes**

- `characters.call_vlm` gained an optional `response_format` parameter:
  when set (json_schema structured outputs), the request also carries
  `provider.require_parameters: true` and a BadRequest rejection is recorded
  as `status=error` — never silently retried without the schema. Detection's
  legacy `json_object` path (with its retry-without fallback) is unchanged.
- `verify_color.py` is now one `ColorVerifier` + `parse_color_verdict` +
  `verify_color_prompt.txt`; `verify_l2r_prompt.txt` and
  `verify_palette_prompt.txt` were deleted.
- The integration suite asserts `status == "verified"` for COL-001..004
  (COL-004 stays a known-failing case until the geographic-atlas fix
  lands); the manifest records `good_color` + `analyse` per call.
- Cost per verification call is unchanged (one OpenRouter call per panel,
  `usage.cost` accounting kept).
- The generic prompt trusts the model's world knowledge of Frieren; if live
  runs show weak judgments (e.g. missing the COL-004 uniform blue wash),
  the knob is to add canonical-palette hints to the prompt — not to
  reintroduce per-case rendering.
