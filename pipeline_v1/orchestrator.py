"""Pipeline orchestrator: runs the four stages in order with fresh
timestamped run directories, aggregating everything into the manifest.

Step order and run directories:
  panels      -> 1_panels/    (detect + extract, reading order)
  characters  -> 2_characters/
  colorize    -> 3_colorized/
  stitch      -> 4_stitched/

`--steps` filters which steps run; `--from-step` skips earlier steps;
`--resume <dir>` pre-copies a previous run's step outputs so only the missing
steps re-run. A failed run keeps its artifacts and never overwrites anything.
"""

from __future__ import annotations

import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from config import STEP_DIRS, STEP_ORDER, PipelineConfig
from run_context import RunContext, iso_now, package_versions

PIPELINE_NAME = "panel-wise-flux9b-lora"

# Packages recorded in the manifest's dependency block.
DEPENDENCY_PACKAGES = [
    "pillow", "numpy", "requests", "python-dotenv", "openai", "tqdm", "ultralytics",
]


@dataclass
class Backends:
    detector: object
    character_detector: object
    colorizer: object


class PipelineRunner:
    def __init__(self, config: PipelineConfig, backends: Backends) -> None:
        self.config = config
        self.backends = backends

    # -- entry point --------------------------------------------------------

    def run(self) -> RunContext:
        ctx = RunContext.create(self.config.output_root, self._initial_manifest())
        started = time.monotonic()
        if self.config.resume:
            self._copy_resumed_outputs(ctx)
        try:
            for step in self._steps_to_run():
                if self._step_has_outputs(ctx, step):
                    print(f"== step {step}: outputs already present, skipping ==",
                          flush=True)
                    continue
                self._run_one(ctx, step)
            ctx.manifest["totals"]["wall_time_s"] = round(time.monotonic() - started, 1)
            ctx.set_status("completed")
        except KeyboardInterrupt:
            ctx.set_status("aborted", error="KeyboardInterrupt")
            raise
        except BaseException as exc:  # noqa: BLE001 - record and re-raise
            ctx.manifest["totals"]["wall_time_s"] = round(time.monotonic() - started, 1)
            ctx.set_status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        return ctx

    # -- manifest -----------------------------------------------------------

    def _initial_manifest(self) -> dict:
        return {
            "schema_version": 2,
            "pipeline": PIPELINE_NAME,
            "status": "running",
            "started_at": iso_now(),
            "finished_at": None,
            "run_directory": None,  # filled by RunContext.create
            "command": shlex.join([sys.executable, *sys.argv]),
            "configuration": self.config.to_dict(),
            "dependencies": {"python": sys.version,
                             **package_versions(DEPENDENCY_PACKAGES)},
            "prompt_hashes": self._prompt_hashes(),
            "pricing_assumptions": self._pricing_assumptions(),
            "steps": {},
            "totals": {
                "openrouter_cost_usd": 0.0,
                "character_calls": 0,
                "successful_character_calls": 0,
                "page_character_calls": 0,
                "fallback_character_calls": 0,
                "forced_character_panels": 0,
                "flux_calls": 0,
                "successful_flux_calls": 0,
                "panels_colorized": 0,
                "pages_stitched": 0,
                "wall_time_s": 0.0,
            },
        }

    def _prompt_hashes(self) -> dict:
        from util import sha256

        hashes = {}
        for name, path in (
            ("vlm_prompt", self.config.vlm_prompt_file),
            ("vlm_panel_prompt", self.config.vlm_panel_prompt_file),
            ("colorizer_prompt", self.config.colorizer_prompt_file),
            ("profiles", self.config.profiles_file),
        ):
            try:
                hashes[name + "_sha256"] = sha256(path)
            except (OSError, ValueError):
                hashes[name + "_sha256"] = None
        return hashes

    def _pricing_assumptions(self) -> dict:
        if self.config.mock:
            return {"note": "mock backends: no external calls, no cost."}
        return {
            "character_detection": {
                "provider": "OpenRouter",
                "model": self.config.vlm_model,
                "tier": "paid (user-funded)",
                "cost_source": "usage.cost per call (USD), measured",
                "note": "Per-call cost is recorded in the 2_characters records.",
            },
            "colorization": {
                "model": ("black-forest-labs/FLUX.2-klein-9B (step-distilled) + "
                          "thedeoxen manga-colorization-by-reference LoRA"),
                "hosting": "self-hosted BentoML server on the DGX Spark (see server/)",
                "steps": 4,
                "usd_per_call": 0.0,
                "note": ("No per-call fee (electricity only, ~350-400 W during "
                          "inference). Step-distilled model: guidance_scale is "
                          "ignored by diffusers (CFG off). Do not compare with "
                          "paid API pricing."),
            },
        }

    # -- step selection -----------------------------------------------------

    def _steps_to_run(self) -> list[str]:
        steps = list(self.config.steps)
        if self.config.from_step:
            start = STEP_ORDER.index(self.config.from_step)
            steps = [s for s in steps if STEP_ORDER.index(s) >= start]
        return steps

    def _copy_resumed_outputs(self, ctx: RunContext) -> None:
        """Copy step outputs from the resume run. With `--from-step STEP` only
        the outputs strictly before STEP are copied; later-stage outputs are
        regenerated in the fresh run (task 0001)."""
        resume_dir = Path(self.config.resume)
        if not resume_dir.is_dir():
            raise ValueError(f"--resume directory not found: {resume_dir}")
        steps = list(STEP_ORDER)
        if self.config.from_step:
            steps = list(STEP_ORDER[: STEP_ORDER.index(self.config.from_step)])
        for step in steps:
            source = resume_dir / STEP_DIRS[step]
            if source.is_dir():
                target = ctx.run_dir / STEP_DIRS[step]
                target.mkdir(parents=True, exist_ok=True)
                for item in source.iterdir():
                    _copytree(item, target / item.name)
        print(
            f"resumed outputs copied from {resume_dir} (steps before "
            f"{self.config.from_step or 'end'})",
            flush=True,
        )

    def _step_has_outputs(self, ctx: RunContext, step: str) -> bool:
        directory = ctx.run_dir / STEP_DIRS[step]
        if not directory.is_dir():
            return False
        if step == "panels":
            return any((d / "panels.json").is_file()
                       for d in directory.iterdir() if d.is_dir())
        if step == "characters":
            return (directory / "summary.json").is_file()
        if step == "colorize":
            return any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                       for p in directory.rglob("*"))
        if step == "stitch":
            return any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                       for p in directory.iterdir())
        return False

    # -- step dispatch ------------------------------------------------------

    def _run_one(self, ctx: RunContext, step: str) -> None:
        print(f"== step: {step} ==", flush=True)
        if step == "panels":
            from steps.panels import run_panels_step

            record = run_panels_step(ctx, self.config, self.backends.detector)
        elif step == "characters":
            from steps.characters import run_characters_step

            record = run_characters_step(
                ctx, self.config, self.backends.character_detector
            )
            totals = record["totals"]
            self.manifest_totals_update(ctx, {
                "openrouter_cost_usd": totals["cost_usd"],
                "character_calls": totals["api_calls"],
                "successful_character_calls": totals["successful_calls"],
                "page_character_calls": totals["page_calls"],
                "fallback_character_calls": totals["fallback_calls"],
                "forced_character_panels": totals["forced_panels"],
            })
        elif step == "colorize":
            from steps.colorize import run_colorize_step

            record = run_colorize_step(ctx, self.config, self.backends.colorizer)
            totals = record["totals"]
            self.manifest_totals_update(ctx, {
                "flux_calls": totals["api_calls"],
                "successful_flux_calls": totals["successful_calls"],
                "panels_colorized": totals["successful_calls"],
            })
        elif step == "stitch":
            from steps.stitch import run_stitch_step

            record = run_stitch_step(ctx, self.config)
            self.manifest_totals_update(ctx, {
                "pages_stitched": len(record["outputs"]),
            })
        else:
            raise ValueError(f"unknown step {step!r}")
        ctx.manifest.setdefault("steps", {})[step] = record
        ctx.write_manifest()

    def manifest_totals_update(self, ctx: RunContext, update: dict) -> None:
        for key, value in update.items():
            if key == "openrouter_cost_usd":
                ctx.manifest["totals"][key] = round(
                    ctx.manifest["totals"].get(key, 0.0) + value, 8
                )
            else:
                ctx.manifest["totals"][key] = (
                    ctx.manifest["totals"].get(key, 0) + value
                )


def _copytree(source: Path, destination: Path) -> None:
    """Copy a file or directory tree (shutil.copytree without the
    exists-ok quirk)."""
    import shutil

    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
