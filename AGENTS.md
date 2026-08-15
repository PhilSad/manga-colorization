# Repository instructions

## Project goal

This repository is a practical exploration of AI-assisted manga colorization. For every method, record:

- the colorization approach and model/service used;
- the input data and preprocessing steps;
- the visual output quality, including limitations and failure cases;
- the cost, including the pricing assumptions and the number of images/tokens/API calls used;
- enough configuration and provenance information to reproduce the result.

The purpose is to compare methods on quality and cost, not only to produce attractive images.

## Repository layout

- `data/`: source manga pages, reference images, and other input assets. Keep original inputs unchanged.
  - `data/volumes/`: raw manga volumes as `.cbz` archives (gitignored).
  - `data/page_per_volume/`: pages extracted from the volumes, one directory per volume, original filenames preserved (natural sort = reading order; gitignored).
- `research/colorization_methods/`: one self-contained directory per colorization method.
- `research/character_detection_methods/`: one self-contained directory per character-detection method. Companion experiments (not colorizers): given a panel or page, they list which reference characters appear in it, e.g. via vision-language models. They follow the same conventions as colorization methods (timestamped `output/` runs, manifests, `methods.md` entry).
- `methods.md`: the index and comparison table for all methods.
- `pipelines.md`: the index and comparison of full colorization pipelines (the pipeline-level counterpart of `methods.md`); contains the evaluation write-ups of `pipeline_v1`.
- `pipeline_v1/`: the panel-wise colorization pipeline (a pipeline, not a method — it composes the research methods as library modules). See the `# pipeline_v1` section below.
- `script/`: utility scripts shared across methods, currently volume tooling: `extract_pages.py` (unpack `data/volumes/*.cbz` into `data/page_per_volume/`) and `merge_to_cbz.py` (pack a page folder back into a `.cbz`). Both are Python 3 stdlib-only.
- `scrape_frieren_wiki.py` + `frieren_wiki_dataset/`: scraper for the Frieren wiki (MediaWiki API) and its output dataset (per-chapter page counts, summaries, characters in order of appearance). Rerun with `python3 scrape_frieren_wiki.py`; see the dataset's `README.md`.
- `associate_chapters_to_pages.py`: maps chapters to page-file ranges in `data/page_per_volume/` (filename chapter tags + padding rule, wiki-count fallback for the mislabeled v09, overrides for missing counts). Writes `frieren_wiki_dataset/chapter_page_map.json` + `chapter_pages.csv`.
- `build_chapter_casts.py`: generates `pipeline_v1/chapter_casts.json` (per-chapter `--cast-key` shortlists) from the wiki dataset's Characters in Order of Appearance intersected with the `data/refs/` roster, excluding mentioned-only and no-reference characters.
- `server/`: self-hosted inference server for the FLUX.2 Klein 9B method (BentoML, docker-packaged; weights are an external model dir mounted at runtime, never baked into the image). See `server/README.md`. The client side lives in the method directory (`local_fal_client.py` + `run.py --endpoint`).
- `AGENTS.md`: these repository-wide instructions.
- `.env`: contains the gemini api key, fal api key, openai api key, and openrouter api key

## Adding a method (colorization or character detection)

Use a descriptive, filesystem-safe method name and create a new directory under `colorization_methods/` (colorization) or `character_detection_methods/` (character detection). Each method must have its own persistent `output/` directory:

```text
colorization_methods/<method-name>/   # or character_detection_methods/<method-name>/
├── output/
├── README.md                 # method, setup, usage, quality, and cost notes
├── run.*                     # executable entry point, if applicable
└── ...                       # method-specific code/configuration
```

When a method is run, it must create a new timestamped subdirectory inside that method's `output/` directory. Use the local run start time in the format `YYYYMMDD-HHMMSS`, for example:

```text
colorization_methods/<method-name>/output/20260808-143015/
character_detection_methods/<method-name>/output/20260808-143015/
```

Never overwrite a previous run. If two runs could have the same second-level timestamp, append a short unique suffix (for example, `-01` or a run ID). The run directory should contain the generated images and a manifest with the input files, configuration/prompt, model or service version, timestamp, and cost data available at run time.

Do not commit generated outputs, credentials, downloaded model weights, or other large artifacts unless the repository explicitly needs them. Keep a small representative sample or an external artifact reference when useful for comparison.

## Updating `methods.md`

Every new method (colorization or character detection) must be added to `methods.md` in the same change that adds its implementation. At minimum, include:

- method name and link to its directory;
- model, service, or algorithm;
- input resolution and relevant settings;
- quality assessment and known failure cases;
- cost per image or per comparable test set, with currency and date;
- a link to a representative output run;
- status (experimental, usable, or retired).

Use measured values where possible. Clearly label estimates, free-tier usage, one-time setup costs, and recurring costs. Do not present results from different input sets or settings as directly comparable without noting the difference.

## Evaluation and reproducibility

Use the same reference inputs and evaluation criteria across methods whenever practical. Record both subjective observations (line preservation, color coherence, shading, artifacting, and faithfulness to references) and objective measurements when available. Preserve prompts, seeds, model versions, dependency versions, and relevant environment settings so a result can be reproduced.

Before considering a method complete, verify that its run entry point creates a fresh timestamped output directory and that the output and cost information are reflected in `methods.md`.

## General implementation conventions

- When searching the repository with `grep`/`rg`, exclude the `.venv/` directory by default (e.g. `rg -n pattern . --glob '!.venv/**'` or `grep -rn pattern --exclude-dir=.venv .`) so results aren't drowned in vendored dependencies. Only search inside `.venv/` when deliberately looking for something in it, and make such searches targeted (specific paths, not whole-repo scans).
- Prefer small, composable scripts with a clear entry point and documented dependencies.
- Keep method-specific dependencies and configuration inside that method's directory where practical.
- Never hard-code API keys or other secrets; load them from environment variables or an ignored local configuration file.
- Use paths relative to the repository or the script location so runs work from any current working directory.
- Avoid modifying files under `data/`; write generated artifacts only under the method's timestamped output directory. The exception is the volume tooling in `script/`: `extract_pages.py` populates `data/page_per_volume/` from `data/volumes/`, and `merge_to_cbz.py` writes a new `.cbz` when a colorized folder is packed back. These scripts never alter the original `.cbz` files in `data/volumes/`.
- Update the relevant method `README.md` when behavior, setup, quality, or cost assumptions change.
- Log long-running commands: whenever you run a command expected to take a while (method/pipeline runs, tests, downloads, data processing, model inference), run it through `tee` so the output is both shown and saved to the local `.output/` folder as a timestamped `.out` file (e.g. `.output/20260808-143015.out`). Create the folder with `mkdir -p .output` if needed. Note that `cmd | tee file` returns `tee`'s exit status — when the command's own exit code matters, capture it with `${PIPESTATUS[0]}` (or `set -o pipefail`). `.output/` is gitignored; these logs are transient provenance, not artifacts to commit.

# Spark inference server

The FLUX.2 Klein 9B model runs as a self-hosted BentoML inference server on the DGX Spark (`ssh spark`, 120 GB unified memory). The client code stays in this repository; only the inference happens on Spark.

Optional LoRA: the server can load `manga_colorization.safetensors` (thedeoxen's manga-colorization-by-reference LoRA, trigger word `mngclranm`) on top of the undistilled `FLUX.2-klein-base-9B` model. That deployment needs the base weights (gated) + the LoRA (public) downloaded into `server/models/` on Spark and the compose env `FLUX2_LORA_PATH`/`FLUX2_GUIDANCE_SCALE`/`FLUX2_STEPS` set; see `server/README.md` §2b. Clients send `guidance_scale` and `lora_scale` (run.py forwards them only with `--endpoint`).

## "Colorize a volume" means a small test run

When the user asks to "colorize a volume" (or similar), they usually mean a quick test run, not the whole book: skip the first 3 pages (typically title/credits) and colorize only the next 5 pages. In terms of the `run.py` flags, that is `--skip-first 3 --limit 5`. Flag names may vary slightly between methods; check the target method's `run.py` (most use `--skip-first` and `--limit`). If in doubt about the intended page range, ask instead of colorizing the entire volume.

## "Run on the spark" means use the inference server

When the user says to run something "on the spark", they mean **use the Spark inference server as the compute backend for inference** — i.e. invoke the method's `run.py` from this repository with `--endpoint http://spark:3000`. Do **not** ssh into Spark to copy the repo there or start `nohup`-style background jobs on the Spark machine. At most, the only actions taken on Spark itself are Docker operations to manage the inference server container (start, stop, rebuild, inspect).

## Checking if the server is already active

From any machine that can reach Spark (the repo machine resolves `spark`):

```bash
curl -s http://spark:3000/healthz   # -> {"status":"ok",...} when up
curl -s http://spark:3000/          # Swagger UI exposing the /edit schema
```

Or inspect the container directly on Spark:

```bash
ssh spark "docker ps --filter name=flux2-klein"   # STATUS should read "Up ... (healthy)"
```

## Endpoint

- Base URL: `http://spark:3000` (BentoML, port 3000; `POST /edit`, multipart form-data: `images` file parts `[current, atlas, previous?]`, `prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`, `lora_scale`, `seed`, `output_format`).
- No API key required; the method's manifest records self-hosted (zero per-call) pricing instead of fal pricing.
- From the client machine: `--endpoint http://spark:3000`. From Spark itself use `http://127.0.0.1:3000`.
- No auth on the endpoint; if needed, tunnel it: `ssh -N -L 3000:localhost:3000 spark` and point the client at `http://localhost:3000`.

## Running a method against it

Client-side (this repo), e.g. for the FLUX.2 Klein 9B edit method:

```bash
.venv/bin/python colorization_methods/fal-flux-2-klein-9b-edit-sequential/run.py \
  --model black-forest-labs/FLUX.2-klein-9B \
  --endpoint http://spark:3000 \
  --width 1216 --height 1824 \       # FLUX VAE needs multiples of 16; 1200x1800 gets floored to 1200x1792
  --num-inference-steps 4 \
  --output-format png \
  --input-dir data/chapter_134 \
  --refs-dir data/refs \
  --skip-first 3
```

The first request pays the model-loading cost (≈1–3 min); subsequent pages are fast. Cost of a run: $0 per call plus electricity (see `server/README.md`); do not compare these costs with fal pricing.

## Docker management (the only thing to do ON Spark)

The server is a docker container named `flux2-klein` (image `flux2-klein:latest`, weights mounted read-only from `models/FLUX.2-klein-9B` — never baked into the image). Sources and compose file live in `server/` in this repo.

```bash
ssh spark "cd /home/phil/agent_workspace/flux2-klein-server && docker compose up -d"          # start
ssh spark "docker ps --filter name=flux2-klein"                                                # status / health
ssh spark "docker exec flux2-klein nvidia-smi"                                                 # confirm GPU is used
ssh spark "cd /home/phil/agent_workspace/flux2-klein-server && docker compose down && docker compose up -d --build"   # rebuild after changes
```

Note: the server sources live on Spark under `/home/phil/agent_workspace/flux2-klein-server`; if you change `server/` in the repo, sync it there (only that folder) before rebuilding.

# pipeline_v1

`pipeline_v1/` is the full panel-wise manga colorization pipeline. It is a
pipeline, not a method: it composes the research methods as library modules
(character-detection via OpenRouter VLM, FLUX.2 Klein colorization), so it is
documented in `pipelines.md`, not `methods.md`. Per page it:

1. **Detect panels** — YOLO26n (`leoxs22/manga-panel-detector-yolo26n`, weights
   auto-downloaded to `pipeline_v1/models/`). Zero-detection pages get a
   blank-ink check; sparse full-page art gets one synthetic full-page box
   (`provenance: full-page-fallback`), effectively blank pages are skipped.
2. **Extract in Japanese reading order** (right-to-left, top-to-bottom) —
   crops `panel_0001.png …` + `panels.json` + `overlay.png` → `1_panels/<page>/`.
3. **Detect characters per page** — OpenRouter `google/gemma-4-31b-it`, one
   paid call per page mapping numbered panels to canonical characters; missing/
   invalid/`uncertain` panels get a cropped-panel fallback. `--detection-mode
   page|panel|panel-page|panel-page-cast|panel-page-prev2|panel-page-prev2-cast`
   selects page-level (V1.1), one-call-per-panel (V1), one-call-per-panel with
   the full page as context, that plus an **automatically derived per-chapter
   cast shortlist**, the full-page variant that also sends the two preceding
   pages as story context, and the prev2 variant with the per-chapter cast
   shortlist. `panel-page-prev2-cast` is the default. `panel-page-cast` and
   `panel-page-prev2-cast` derive the shortlist from the page's chapter via
   `frieren_wiki_dataset/chapter_page_map.json` —
   fixing mislabeled v09 tags; `--cast-key` overrides the derivation, so
   look-alike characters outside the chapter cast cannot be guessed (e.g.
   Flamme on p130 of ch. 5). An optional cached chapter cast shortlist
   (`--cast-key`) focuses the prompt → `2_characters/<page>/<panel>.json`.
4. **Colorize panel by panel** — step-distilled FLUX.2 Klein 9B + thedeoxen
   manga-colorization-by-reference LoRA (`mngclranm`, 4 steps) on the Spark
   server; the request is the panel + a labelled atlas of **only the detected
   characters** + an explicit canonical-palette instruction rendered from
   `character_profiles.json`. Panels with no detected characters are colorized
   panel-only. Oversized inputs are scaled to the megapixel cap
   (`--max-megapixels`, default 2.0 MP) → `3_colorized/<page>/`.
5. **Stitch** — each colorized panel resized back to its original box and
   pasted onto the page; everything outside the panels stays black & white →
   `4_stitched/<page>.png`.
6. **Debug annotation** — pure image processing (no backends): per stitched
   page, draw the detected panel boxes + a label per panel with the
   characters detected for it; B&W-fallback panels get an orange box and a
   `[B&W fallback]` tag → `5_debug/<page>.png` + `summary.json`.
7. **PDF export** — pure image processing (no backends): packs every
   stitched page (`4_stitched/`, filename order = reading order) into one
   multi-page PDF with Pillow's native PDF writer (no extra dependency) →
   `6_pdf/colorized.pdf` + `summary.json`; `--pdf-name` / `--pdf-dpi`
   (default 72) control the filename and the embedding resolution
   (page size in points = pixels × 72 / dpi).

- Entry point: `pipeline_v1/run.py`; full usage in `pipeline_v1/README.md`,
  module map and design decisions in `pipeline_v1/ARCHITECTURE.md`.
- Run conventions: same as methods — each invocation creates a fresh
  `pipeline_v1/output/YYYYMMDD-HHMMSS/` dir (never overwritten) with the six
  numbered intermediate directories and an incremental `manifest.json`
  (command, config, prompt/profile hashes, per-step records, measured costs).
  `output/` and `models/` are gitignored.
- A "colorize a volume" request also means `--skip-first 3 --limit 5` here
  (see above); run against Spark with `--endpoint http://spark:3000`.
- Useful flags: `--steps panels,characters`, `--from-step colorize`,
  `--resume <previous-run-dir>`, `--atlas-columns N`, `--num-inference-steps`
  (4 for the step-distilled model), `--lora-scale`, `--seed`, `--detection-mode`,
  `--cast-key`, `--only-panel PAGE:PANEL` (targeted rerun),
  `--force-characters PAGE:PANEL=Name` (ground-truth identities, no paid
  call), `--verify-attempts N` (per-panel character-palette verification
  loop with Luna `openai/gpt-5.6-luna` structured output: 1 = verify + write
  `<panel>.fix_prompt.txt` only; N≥2 = re-colorize with the fix prompt up to
  N−1 retries, keeping `<panel>.attempt_<n>.png`; every attempt recorded in
  `<panel>.verify.json` and `totals.verify_cost_usd`; `--verify-model` /
  `--verify-prompt-file` override the verifier), `--worker-detection N` (parallel page-level detection),
  `--worker-colorization N` (parallel page-level colorization — parallelizes
  the paid gpt-image-2 calls directly in full-page mode),
  `--stitch-bw-fallback` (a panel whose colorized output is missing — e.g. a
  failed FLUX call — is stitched from its original B&W crop instead of failing
  the stitch step; each fallback is logged to stderr and recorded per page and
  in `totals.panels_bw_fallback`), `--debug-font-size` / `--debug-bbox-width`
  (5_debug rendering knobs), `--pdf-name` / `--pdf-dpi` (6_pdf PDF filename,
  default `colorized.pdf`; embedding resolution, default 72 — page size in
  points = pixels × 72 / dpi).
- Full-page mode: `--full-page` skips panel extraction entirely and colorizes
  the whole page in one OpenAI `gpt-image-2` call (atlas + palette
  instruction, minimal aspect-preserving size). `--atlas-source {detected,cast}`
  picks where the atlas characters come from: `detected` (default) forces
  `--detection-mode page` (one VLM call per page), `cast` skips character
  detection entirely (zero VLM calls, `OPENROUTER_API_KEY` not needed — only
  `OPENAI_API_KEY`). `--atlas-source cast` requires `--full-page`. See
  `pipeline_v1/README.md` §Full-page gpt-image-2 mode.
- Debug annotation: the pipeline's final `debug` step (see step 6 above)
  writes `5_debug/` automatically at the end of every run. Its offline
  companion, `pipeline_v1/scripts/annotate_stitch.py --run-dir <run-dir>`,
  re-annotates any completed run with the same `steps.debug.run_debug_step`
  implementation (no pipeline rerun): a colored bounding box per panel and a
  label with the panel name + the characters detected for it (from
  `2_characters/<page>/<panel>.json`); panels stitched from their original
  B&W crop (`--stitch-bw-fallback`) get an orange box and a `[B&W fallback]`
  tag, read from the run's `manifest.json`. Writes `<run-dir>/5_debug/` +
  `summary.json`; options `--output-dir`, `--page SUBSTR` (repeatable
  filter), `--font-size`, `--bbox-width`. Offline, needs no
  backends/network, and never modifies the run's own outputs.
- Requirements: `OPENROUTER_API_KEY` in `.env` (paid detection) and the Spark
  FLUX server running (`curl http://spark:3000/healthz`); full-page mode
  (`--full-page`) instead needs `OPENAI_API_KEY` in `.env` (paid gpt-image-2)
  and no Spark server. Offline demo without
  any of that: `pipeline_v1/run.py --mock --limit 1` (mock backends). Tests:
  `.venv/bin/pytest pipeline_v1/tests -q` (fully offline; the real full-pipeline
  run with real backends is covered by the `integration`-marked
  `test_end_to_end_integration.py`, and `scripts/smoke_real.sh` remains handy
  for a quick manual run on any input folder).
- Cost: character detection ~$0.00008/panel (measured via `usage.cost`, paid
  OpenRouter tier; recorded in `2_characters/` and the manifest
  `totals.openrouter_cost_usd`); colorization $0 per call (self-hosted FLUX on
  Spark, electricity only). Full-page mode: gpt-image-2 is a paid API —
  projected ≈ $0.0499/page at the minimal 672×1008 size (research-v2
  measurement, standard tier), recorded in `totals.gpt_image_cost_usd`.
- Evaluation: the fixed failure set (`evaluation/v1_1_cases.json`) is run by
  the real-network integration suite — `.venv/bin/pytest pipeline_v1/tests -m integration -n 8` —
  (pytest-xdist, 8 parallel workers; each worker is its own pytest session and
  writes its own timestamped `tests/output/YYYYMMDD-HHMMSS-gwN/` dir, so the
  same-second dir-name collision that plain `mkdir(exist_ok=True)` would hide
  is avoided — see the `integration_run` fixture in `tests/conftest.py`)
  no mocks: real OpenRouter gemma panel detection (DET/OOV — one test per
  detection mode: `page`/`panel`/`panel-page`/`panel-page-cast`/`panel-page-prev2`/`panel-page-prev2-cast`), real FLUX on
  Spark + real gpt-5.6-luna validation (COL/SIZE), real YOLO (LAY), plus a
  full-pipeline end-to-end test (`test_end_to_end_integration.py`, E2E-P130)
  running all five stages with real backends on one real page (volume-1
  p130). Inputs
  are committed under `tests/data/` (per-case crops, plus full committed
  pages and per-page panel sets for the detection pages; regenerate with
  `tests/prepare_integration_data.py`); each worker session writes a timestamped
  `tests/output/YYYYMMDD-HHMMSS[-gwN]/` dir with per-case records and measured
  `usage.cost`. Known-failing cases fail loudly (tracked, not xfailed).
  Quality write-ups, run tables, and remaining known failures live in
  `pipelines.md`; debug views are in `docs/pipeline_v1/`.
- Adding an evaluation case (the DET-005..010 procedure):
  1. **Confirm the page layout first.** Panel IDs (`panel_0001`, …) refer to
     the crops produced by the real reading-order extraction (YOLO26n +
     `panel_ordering.reading_order`), not to positions on the page. Run the
     detector once (`.venv/bin/python -c …` with `YoloPanelDetector` +
     `reading_order`) to confirm the panel count/order before writing the
     fixture entry. A multi-panel page becomes one case per panel (e.g. six
     cases DET-005..010 for p130), each with its own expected set and crop.
  2. **Add the case to `pipeline_v1/evaluation/v1_1_cases.json`:** a durable
     source-page alias (`"P130": "data/page_per_volume/…"`) if the page is
     not aliased yet; then the case with `id` (sequential per stage, e.g.
     DET-005), `stage` (`characters` | `color` | `layout` | `size`), `failure`
     (reuse the taxonomy: `known-character-confusion`, `no-character-panel`,
     `out-of-vocabulary`, `palette-adherence`, `palette-geography`,
     `zero-panel-fallback`, `blank-page-skip`, `oversized-input-capping`, …),
     `input` (`source_page` + `panel`), `expected` (character set,
     `unknown_present`, and `expected_unknown_characters` for OOV), `baseline`
     (the **observed** prior-run detection, e.g. from
     `output/<run>/2_characters/<page>/<panel>.json` — never fabricate
     results), and a `note` citing the observed failure and run id. Update the
     fixture `description` when a new case class is added.
  3. **Regenerate the committed crops:**
     `.venv/bin/python pipeline_v1/tests/prepare_integration_data.py` creates
     the per-case crops `tests/data/panels/<case_id>.png`, plus the full
     per-page sets for detection pages (`tests/data/pages/<alias>.png` and
     `tests/data/panels/<alias>/` with all crops and `panels.json`), from the
     durable pages, and rewrites the `tests/data/README.md` provenance. Then
     check `git status` that all pre-existing crops stayed byte-identical —
     if any changed, detection is no longer deterministic on those pages and
     the fixture's panel IDs may no longer match the committed crops.
  4. **Wire the case into the integration suite:** add its id to the stage's
     case list in `pipeline_v1/tests/test_integration_<stage>.py` (e.g.
     `DETECTION_CASES` in `test_integration_detection.py`) and update the
     module docstring.
  5. **Document and sanity-check:** note the case and its observed failure in
     `pipelines.md` — a newly added case gets a "pending next live run" note,
     never fabricated numbers; keep the fixture valid JSON with all crops
     present; `.venv/bin/pytest pipeline_v1/tests -q` (offline) must stay
     green. Run the live stage test (`-m integration`) only when backends are
     available and the paid OpenRouter cost is acceptable.
  6. **Commit** with a meaningful title/description (see the contribution
     guide below).

# Contribution guide

When you finish implementing a feature or a bug fix or a meaningfull code change, you should commit it with a meaningfull commit title and description
