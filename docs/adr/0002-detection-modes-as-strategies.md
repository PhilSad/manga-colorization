# Character detection modes as strategies

Status: accepted (2026-08-10)

`OpenRouterCharacterDetector` previously implemented all detection modes as
methods on one class (`detect` for panel-only, `detect_page` for page-level,
`detect_panels_with_page` for panel+page), and the pipeline step dispatched to
them with `hasattr`/`getattr` on string method names. The modes are now one
strategy class each (`PanelStrategy`, `PageStrategy`, `PanelPageStrategy`,
`PanelPageCastStrategy`) behind a uniform per-page interface —
`DetectionStrategy.detect(page, panels_dir, expected_panels, refs_dir,
*, cast_key=None) -> PageCharacterRecord` — with a single unified entry point
`OpenRouterCharacterDetector.detect(mode, …)` and a `strategy_for(mode)`
factory. The step now selects the strategy once per run and calls it uniformly;
cast derivation moved out of the step into `PanelPageCastStrategy`.

Why: the string-method dispatch was fragile and untyped, the step carried
per-mode branches (page loop, panel loop, cast application) instead of one
path, and adding a mode meant touching the detector, the step, and the mocks
in three places. With strategies, each mode is an independently testable unit,
the step is one uniform path, and the integration suite tests the same
`detect(mode, …)` entry point in all four modes.

**Consequences / notes**

- The `panel` strategy also aggregates per-panel calls into a
  `PageCharacterRecord`; in the manifest its primary calls are counted under
  `page_calls` (previously panel-mode runs reported `page_calls: 0`). The
  totals' `page_calls` therefore means "the strategy's primary calls"
  (page-level calls for `page`/`panel-page` modes; one per panel for `panel`).
- The per-method call sites (unit tests, integration tests, sweep) were
  rewritten to the unified `detect(mode, …)`; the mock detectors gained
  matching `strategy_for(mode)` adapters so offline runs and the orchestrator
  tests are unchanged in behaviour.
