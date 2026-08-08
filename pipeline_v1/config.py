"""Pipeline configuration: dataclass + argparse CLI.

Every knob of every stage lives here so orchestrator, steps and tests share one
definition. Defaults follow the repo conventions (AGENTS.md) and the research
methods the pipeline ports.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent

# Pipeline stages in execution order; the run directory for each stage is
# prefixed with its 1-based index (1_panels/, 2_characters/, ...).
STEP_ORDER: tuple[str, ...] = ("panels", "characters", "colorize", "stitch")
STEP_DIRS: dict[str, str] = {
    "panels": "1_panels",
    "characters": "2_characters",
    "colorize": "3_colorized",
    "stitch": "4_stitched",
}

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_OUTPUT_FORMATS = ("png", "jpeg", "webp")

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "chapter_134"
DEFAULT_REFS_DIR = REPO_ROOT / "data" / "refs"
DEFAULT_OUTPUT_ROOT = PIPELINE_DIR / "output"
DEFAULT_ENDPOINT = "http://spark:3000"
DEFAULT_VLM_MODEL = "google/gemma-4-31b-it"
DEFAULT_VLM_PROMPT_FILE = PIPELINE_DIR / "prompt.txt"
DEFAULT_COLORIZER_PROMPT_FILE = PIPELINE_DIR / "colorizer_prompt.txt"

# FLUX VAE constraint: every requested dimension must be a multiple of 16.
FLUX_MULTIPLE = 16
# The smallest dimension we will ever request (rounding must not produce 0).
FLUX_MIN_SIDE = 16


@dataclass
class PipelineConfig:
    input_dir: Path = DEFAULT_INPUT_DIR
    refs_dir: Path = DEFAULT_REFS_DIR
    output_root: Path = DEFAULT_OUTPUT_ROOT

    # Stage 3 — character detection (OpenRouter)
    vlm_model: str = DEFAULT_VLM_MODEL
    vlm_prompt_file: Path = DEFAULT_VLM_PROMPT_FILE
    max_tokens: int = 2048
    temperature: float = 0.2
    sleep_s: float = 2.0
    api_key_env: str = "OPENROUTER_API_KEY"

    # Stage 4 — colorization (self-hosted FLUX.2 Klein 9B + LoRA)
    endpoint: str | None = DEFAULT_ENDPOINT
    colorizer_prompt_file: Path = DEFAULT_COLORIZER_PROMPT_FILE
    flux_steps: int = 4
    guidance_scale: float = 4.0
    lora_scale: float = 1.0
    seed: int | None = None
    output_format: str = "png"
    atlas_columns: int | None = None  # None -> ceil(sqrt(n)) square-ish grid

    # Geometry
    panel_inset: int = 0  # px trimmed from each panel crop side

    # Page selection (repo convention: --skip-first / --limit)
    skip_first: int = 0
    limit: int | None = None

    # Orchestration
    steps: tuple[str, ...] = STEP_ORDER
    mock: bool = False
    resume: Path | None = None   # previous run dir to reuse its step outputs
    from_step: str | None = None # start at this step (skip earlier ones)

    def step_dir(self, step: str) -> str:
        if step not in STEP_DIRS:
            raise ValueError(f"unknown step {step!r} (expected one of {STEP_ORDER})")
        return STEP_DIRS[step]

    def to_dict(self) -> dict:
        data = {
            "input_dir": str(self.input_dir),
            "refs_dir": str(self.refs_dir),
            "output_root": str(self.output_root),
            "vlm_model": self.vlm_model,
            "vlm_prompt_file": str(self.vlm_prompt_file),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "sleep_s": self.sleep_s,
            "api_key_env": self.api_key_env,
            "endpoint": self.endpoint,
            "colorizer_prompt_file": str(self.colorizer_prompt_file),
            "flux_steps": self.flux_steps,
            "guidance_scale": self.guidance_scale,
            "lora_scale": self.lora_scale,
            "seed": self.seed,
            "output_format": self.output_format,
            "atlas_columns": self.atlas_columns,
            "panel_inset": self.panel_inset,
            "skip_first": self.skip_first,
            "limit": self.limit,
            "steps": list(self.steps),
            "mock": self.mock,
            "resume": str(self.resume) if self.resume else None,
            "from_step": self.from_step,
        }
        return data


def parse_steps(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return STEP_ORDER
    parts = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [part for part in parts if part not in STEP_ORDER]
    if unknown:
        raise ValueError(
            f"unknown steps {unknown}; expected a comma-separated subset of {STEP_ORDER}"
        )
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return tuple(seen)


def _validate(config: PipelineConfig) -> None:
    if config.skip_first < 0:
        raise ValueError("--skip-first must be non-negative")
    if config.limit is not None and config.limit < 1:
        raise ValueError("--limit must be at least 1")
    if config.flux_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    if config.guidance_scale <= 0:
        raise ValueError("--guidance-scale must be positive")
    if config.lora_scale < 0:
        raise ValueError("--lora-scale must be non-negative")
    if config.output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"--output-format must be one of {SUPPORTED_OUTPUT_FORMATS}"
        )
    if config.panel_inset < 0:
        raise ValueError("--panel-inset must be non-negative")
    if config.from_step is not None and config.from_step not in STEP_ORDER:
        raise ValueError(
            f"--from-step must be one of {STEP_ORDER}, got {config.from_step!r}"
        )


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        prog="pipeline_v1",
        description=(
            "Panel-wise manga colorization: detect panels (YOLO26n), extract them "
            "in Japanese reading order, detect characters per panel (OpenRouter "
            "gemma-4-31b-it), colorize each panel with FLUX.2 Klein 9B base + LoRA "
            "(atlas filtered to the detected characters), and stitch the colorized "
            "panels back onto the original page."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory of manga pages (sorted by filename).")
    parser.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS_DIR,
                        help="Directory of reference character images.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Parent dir for fresh timestamped run directories.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="Self-hosted FLUX.2 Klein server base URL (e.g. http://spark:3000).")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL,
                        help="OpenRouter model id for per-panel character detection.")
    parser.add_argument("--vlm-prompt-file", type=Path, default=DEFAULT_VLM_PROMPT_FILE)
    parser.add_argument("--colorizer-prompt-file", type=Path,
                        default=DEFAULT_COLORIZER_PROMPT_FILE)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between OpenRouter calls.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--atlas-columns", type=int,
                        help="Atlas grid columns (default: ceil(sqrt(n))).")
    parser.add_argument("--num-inference-steps", type=int, default=4,
                        help=("FLUX inference steps (step-distilled 9B + LoRA: 4; "
                              "the undistilled base wants 20-50).")),
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-format", choices=SUPPORTED_OUTPUT_FORMATS, default="png")
    parser.add_argument("--panel-inset", type=int, default=0,
                        help="Px trimmed from each side of a panel crop (0 = none).")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="Skip the first N pages of the input folder.")
    parser.add_argument("--limit", type=int,
                        help="Process only the first N pages after skip.")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated subset of panels,characters,colorize,stitch.")
    parser.add_argument("--from-step", choices=STEP_ORDER,
                        help="Start at this step (skip earlier ones).")
    parser.add_argument("--resume", type=Path,
                        help="Reuse step outputs from a previous run directory.")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock backends (no YOLO/OpenRouter/FLUX calls).")
    args = parser.parse_args(argv)

    try:
        steps = parse_steps(args.steps)
        config = PipelineConfig(
            input_dir=args.input_dir,
            refs_dir=args.refs_dir,
            output_root=args.output_root,
            vlm_model=args.vlm_model,
            vlm_prompt_file=args.vlm_prompt_file,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            sleep_s=args.sleep,
            api_key_env=args.api_key_env,
            endpoint=args.endpoint,
            colorizer_prompt_file=args.colorizer_prompt_file,
            flux_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            lora_scale=args.lora_scale,
            seed=args.seed,
            output_format=args.output_format,
            atlas_columns=args.atlas_columns,
            panel_inset=args.panel_inset,
            skip_first=args.skip_first,
            limit=args.limit,
            steps=steps,
            mock=args.mock,
            resume=args.resume,
            from_step=args.from_step,
        )
        _validate(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def nearest_multiple_of(value: int, multiple: int = FLUX_MULTIPLE) -> int:
    """Closest multiple of `multiple` to `value`, clamped to a minimum of
    `multiple` (so rounding never yields 0)."""
    return max(multiple, round(value / multiple) * multiple)


def requested_panel_size(width: int, height: int) -> tuple[int, int]:
    """FLUX request size for a panel of `width`x`height`: the resolution
    closest to the original with both axes multiples of 16 (user-confirmed
    size policy)."""
    return (nearest_multiple_of(width), nearest_multiple_of(height))


if __name__ == "__main__":
    # Quick CLI sanity check: python pipeline_v1/config.py --steps panels,stitch
    config = parse_args()
    print(config.to_dict())
