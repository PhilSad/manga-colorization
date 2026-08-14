# research-v2

New experiment area for pipeline-v2 work (sibling of `research/` and
`pipeline_v1/`, documented here rather than in `methods.md` until it becomes a
formal method or pipeline).

## Layout

- `data/pages/`: committed input pages (volume 1, pages p004–p005 … p010,
  original filenames, reading order). The shared input set for experiments.
- `output/`: timestamped run dirs (`YYYYMMDD-HHMMSS/`), gitignored.
- `split_panels.py`: panel extraction, pipeline_v1-style.

## split_panels.py

Splits pages into panels using the exact pipeline_v1 implementation imported
as a library (same YOLO26n detector `leoxs22/manga-panel-detector-yolo26n` and
weights, same Japanese reading order, same blank-ink check and full-page
fallback). Only the output layout is research-v2's own:

```text
output/<YYYYMMDD-HHMMSS>/
├── manifest.json              # command, config, backend, totals, per-page records
└── <page_stem>/
    ├── panel_0001.png ...     # crops numbered in reading order
    ├── panels.json            # geometry + order (pipeline_v1 schema)
    └── overlay.png            # page with numbered boxes
```

Run:

```bash
.venv/bin/python research-v2/split_panels.py                                  # data/pages, all pages
.venv/bin/python research-v2/split_panels.py --confidence 0.3 --panel-inset 2
.venv/bin/python research-v2/split_panels.py --skip-first 1 --limit 3
```

Flags: `--input-dir`, `--output-root`, `--confidence`, `--panel-inset`,
`--blank-ink-threshold`, `--no-full-page-fallback`, `--skip-first`, `--limit`.
Cost: $0 per call (local inference; first run downloads the weights to
`pipeline_v1/models/` if missing).

## Conventions

Same as the repo's methods: each run creates a fresh timestamped directory,
never overwrites a previous run, and records a manifest with input files,
configuration, and cost at run time.
