# Repository instructions

## Project goal

This repository is a practical exploration of AI-assisted manga colorization. For every method, record:

- the colorization approach and model/service used;
- the input data and preprocessing steps;
- the visual output quality, including limitations and failure cases;
- the cost, including the pricing assumptions and the number of images/tokens/API calls used;
- enough configuration and provenance information to reproduce the result.

The purpose is to compare methods on quality and cost, not only to produce attractive images.

## Running colorization methods

Never run colorization scripts or otherwise trigger a colorization job. Instead, provide the user with the full command needed to run it themselves, including any required options, input paths, and environment setup.

## Repository layout

- `data/`: source manga pages, reference images, and other input assets. Keep original inputs unchanged.
  - `data/volumes/`: raw manga volumes as `.cbz` archives (gitignored).
  - `data/page_per_volume/`: pages extracted from the volumes, one directory per volume, original filenames preserved (natural sort = reading order; gitignored).
- `colorization_methods/`: one self-contained directory per colorization method.
- `methods.md`: the index and comparison table for all methods.
- `script/`: utility scripts shared across methods, currently volume tooling: `extract_pages.py` (unpack `data/volumes/*.cbz` into `data/page_per_volume/`) and `merge_to_cbz.py` (pack a page folder back into a `.cbz`). Both are Python 3 stdlib-only.
- `server/`: self-hosted inference server for the FLUX.2 Klein 9B method (BentoML, docker-packaged; weights are an external model dir mounted at runtime, never baked into the image). See `server/README.md`. The client side lives in the method directory (`local_fal_client.py` + `run.py --endpoint`).
- `AGENTS.md`: these repository-wide instructions.
- `.env`: contains the gemini api key and fal api key

## Adding a colorization method

Use a descriptive, filesystem-safe method name and create a new directory under `colorization_methods/`. Each method must have its own persistent `output/` directory:

```text
colorization_methods/<method-name>/
├── output/
├── README.md                 # method, setup, usage, quality, and cost notes
├── run.*                     # executable entry point, if applicable
└── ...                       # method-specific code/configuration
```

When a method is run, it must create a new timestamped subdirectory inside that method's `output/` directory. Use the local run start time in the format `YYYYMMDD-HHMMSS`, for example:

```text
colorization_methods/<method-name>/output/20260808-143015/
```

Never overwrite a previous run. If two runs could have the same second-level timestamp, append a short unique suffix (for example, `-01` or a run ID). The run directory should contain the generated images and a manifest with the input files, configuration/prompt, model or service version, timestamp, and cost data available at run time.

Do not commit generated outputs, credentials, downloaded model weights, or other large artifacts unless the repository explicitly needs them. Keep a small representative sample or an external artifact reference when useful for comparison.

## Updating `methods.md`

Every new method must be added to `methods.md` in the same change that adds its implementation. At minimum, include:

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

# Available external server

You can run `ssh spark` to ssh to a DGX Spark server with 120GB of ram. You should only work in the folder `/home/phil/agent_workspace`


# Contribution guide

When you finish implementing a feature or a bug fix or a meaningfull code change, you should commit it with a meaningfull commit title and description