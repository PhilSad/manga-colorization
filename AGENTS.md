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
- `script/`: utility scripts shared across methods, currently volume tooling: `extract_pages.py` (unpack `data/volumes/*.cbz` into `data/page_per_volume/`) and `merge_to_cbz.py` (pack a page folder back into a `.cbz`). Both are Python 3 stdlib-only.
- `scrape_frieren_wiki.py` + `frieren_wiki_dataset/`: scraper for the Frieren wiki (MediaWiki API) and its output dataset (per-chapter page counts, summaries, characters in order of appearance). Rerun with `python3 scrape_frieren_wiki.py`; see the dataset's `README.md`.
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

- Prefer small, composable scripts with a clear entry point and documented dependencies.
- Keep method-specific dependencies and configuration inside that method's directory where practical.
- Never hard-code API keys or other secrets; load them from environment variables or an ignored local configuration file.
- Use paths relative to the repository or the script location so runs work from any current working directory.
- Avoid modifying files under `data/`; write generated artifacts only under the method's timestamped output directory. The exception is the volume tooling in `script/`: `extract_pages.py` populates `data/page_per_volume/` from `data/volumes/`, and `merge_to_cbz.py` writes a new `.cbz` when a colorized folder is packed back. These scripts never alter the original `.cbz` files in `data/volumes/`.
- Update the relevant method `README.md` when behavior, setup, quality, or cost assumptions change.

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


# Contribution guide

When you finish implementing a feature or a bug fix or a meaningfull code change, you should commit it with a meaningfull commit title and description