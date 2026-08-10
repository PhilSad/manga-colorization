# Integration suite: committed per-page inputs, one test per detection mode

Status: accepted (2026-08-10)

The pipeline's real-network integration suite (`pytest -m integration`)
previously re-ran panel extraction inside the detection tests (a module
fixture that grouped cases by page, ran YOLO + reading order, then batched
`detect_panels_with_page` per page and asserted the live crops matched the
committed ones). It was replaced with simple self-contained tests: four
parametrized functions, one per detection mode (`page`, `panel`, `panel-page`,
`panel-page-cast`), each calling the tested detector function directly with
**committed inputs** from `tests/data/` — pre-cropped panels for crop-only
modes, and committed full pages + complete per-page panel sets (all crops +
`panels.json` geometry, repo-relative `page_path`) for the page-context modes.

Why: the old shape's indirection (batching fixture, per-case artifact
ceremony, byte-identity check inside the test) made the tests hard to read
and coupled them to extraction; the new shape makes each test a readable
"call the function with committed data" statement, while the extraction
guarantee moved to a dedicated crop-stability tripwire in the layout stage
(real YOLO on the committed pages, asserting byte-identical crops), so the
eval cases' panel references cannot silently go stale.

**Considered options**

- *Panel-only mode only* (`detect()` on the committed crop) — simplest, but
  tests the fallback path, not the pipeline-default panel-page mode that
  produced the tracked DET-006..010 failures; rejected.
- *Full per-page committed sets vs. pages + geometry with crop-at-test-time*
  — committing the complete panel sets keeps the test body a pure
  copy-and-call and the annotation overlay identical to the real path; the
  data duplication vs. the per-case crops is accepted (~few MB).
- *One parametrize over `(mode, case_id)` vs. one parametrized function per
  mode* — per-mode functions chosen: each mode's call shape differs enough
  that a single body would need mode dispatch, and per-mode function names
  document the path under test.
- *Keeping the run-dir ceremony (per-case `input.png`, `record.json`,
  20-field manifest entries)* — slimmed to one shared record helper per
  stage (~8 fields in `manifest.json`); cost tracking, the repo's documented
  contract, is preserved while the redundant artifact copies are dropped.
