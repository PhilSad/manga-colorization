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
STEP_ORDER: tuple[str, ...] = ("panels", "characters", "colorize", "stitch", "debug")
STEP_DIRS: dict[str, str] = {
    "panels": "1_panels",
    "characters": "2_characters",
    "colorize": "3_colorized",
    "stitch": "4_stitched",
    "debug": "5_debug",
}

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_OUTPUT_FORMATS = ("png", "jpeg", "webp")

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "chapter_134"
DEFAULT_REFS_DIR = REPO_ROOT / "data" / "refs"
DEFAULT_OUTPUT_ROOT = PIPELINE_DIR / "output"
DEFAULT_ENDPOINT = "http://spark:3000"
DEFAULT_VLM_MODEL = "google/gemma-4-31b-it"
DEFAULT_VLM_PROMPT_FILE = PIPELINE_DIR / "prompt.txt"
DEFAULT_VLM_PANEL_PROMPT_FILE = PIPELINE_DIR / "prompt_panel.txt"
DEFAULT_VLM_PANEL_PAGE_PROMPT_FILE = PIPELINE_DIR / "prompt_panel_page.txt"
DEFAULT_VLM_PANEL_PAGE_PREV2_PROMPT_FILE = PIPELINE_DIR / "prompt_panel_page_prev2.txt"
DEFAULT_COLORIZER_PROMPT_FILE = PIPELINE_DIR / "colorizer_prompt.txt"
DEFAULT_PROFILES_FILE = PIPELINE_DIR / "character_profiles.json"
DEFAULT_CHAPTER_CASTS_FILE = PIPELINE_DIR / "chapter_casts.json"
DEFAULT_CHAPTER_PAGE_MAP_FILE = (
    REPO_ROOT / "frieren_wiki_dataset" / "chapter_page_map.json"
)

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
    vlm_panel_prompt_file: Path = DEFAULT_VLM_PANEL_PROMPT_FILE
    max_tokens: int = 1024
    temperature: float = 0.0
    sleep_s: float = 1.0
    workers: int = 1               # parallel character-detection threads (1 = sequential)
    api_key_env: str = "OPENROUTER_API_KEY"
    # V1.1 (task 0003): one paid call per page with per-panel fallbacks; the
    # V1 per-panel behaviour; per-panel calls that send the full page as
    # context plus the target panel (panel-page); panel-page with an
    # automatically derived per-chapter cast shortlist (panel-page-cast);
    # panel-page sending the two preceding pages as extra story context
    # (panel-page-prev2); or that variant with the per-chapter cast
    # shortlist (panel-page-prev2-cast).
    detection_mode: str = "panel-page-prev2-cast"  # page | panel | panel-page | panel-page-cast | panel-page-prev2 | panel-page-prev2-cast
    vlm_panel_page_prompt_file: Path = DEFAULT_VLM_PANEL_PAGE_PROMPT_FILE
    vlm_panel_page_prev2_prompt_file: Path = DEFAULT_VLM_PANEL_PAGE_PREV2_PROMPT_FILE
    cast_key: str | None = None   # chapter_casts.json shortlist key (optional)
    chapter_casts_file: Path = DEFAULT_CHAPTER_CASTS_FILE
    # page -> chapter map for panel-page-cast / panel-page-prev2-cast (auto
    # shortlist derivation)
    chapter_page_map_file: Path = DEFAULT_CHAPTER_PAGE_MAP_FILE

    # Stage 4 — colorization (self-hosted FLUX.2 Klein 9B + LoRA)
    endpoint: str | None = DEFAULT_ENDPOINT
    colorizer_prompt_file: Path = DEFAULT_COLORIZER_PROMPT_FILE
    flux_steps: int = 4
    guidance_scale: float = 4.0
    lora_scale: float = 1.0
    seed: int | None = None
    output_format: str = "png"
    atlas_columns: int | None = None  # None -> ceil(sqrt(n)) square-ish grid

    # V1.1 (task 0002): shared canonical character profiles.
    profiles_file: Path = DEFAULT_PROFILES_FILE

    # Geometry
    panel_inset: int = 0  # px trimmed from each panel crop side

    # V1.1 (task 0004): full-page art + oversized-input policy.
    full_page_fallback: bool = True
    blank_ink_threshold: float = 0.005   # ink ratio below this -> blank page
    max_megapixels: float = 2.0          # FLUX request cap (area, MP)

    # Page selection (repo convention: --skip-first / --limit)
    skip_first: int = 0
    limit: int | None = None

    # Orchestration
    steps: tuple[str, ...] = STEP_ORDER
    mock: bool = False
    resume: Path | None = None   # previous run dir to reuse its step outputs
    from_step: str | None = None # start at this step (skip earlier ones)

    # Stitch robustness: a panel whose colorized output is missing (e.g. a
    # failed FLUX call) is stitched from the original black & white crop
    # instead of failing the whole step; each fallback is logged and recorded.
    stitch_bw_fallback: bool = False

    # Stage 5 (debug annotation): rendering knobs for 5_debug/. The step is
    # pure image processing (no backends); the standalone offline tool
    # scripts/annotate_stitch.py shares this implementation.
    debug_font_size: int = 42      # label font size in px
    debug_bbox_width: int = 5      # bounding-box stroke width in px

    # V1.1 (task 0001): targeted reruns.
    only_panels: tuple[str, ...] = ()  # "PAGE:PANEL" selectors (repeatable)
    forced_characters: dict[str, list[str]] = field(default_factory=dict)

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
            "vlm_panel_prompt_file": str(self.vlm_panel_prompt_file),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "sleep_s": self.sleep_s,
            "workers": self.workers,
            "api_key_env": self.api_key_env,
            "detection_mode": self.detection_mode,
            "vlm_panel_page_prompt_file": str(self.vlm_panel_page_prompt_file),
            "vlm_panel_page_prev2_prompt_file": str(self.vlm_panel_page_prev2_prompt_file),
            "cast_key": self.cast_key,
            "chapter_casts_file": str(self.chapter_casts_file),
            "chapter_page_map_file": str(self.chapter_page_map_file),
            "endpoint": self.endpoint,
            "colorizer_prompt_file": str(self.colorizer_prompt_file),
            "profiles_file": str(self.profiles_file),
            "flux_steps": self.flux_steps,
            "guidance_scale": self.guidance_scale,
            "lora_scale": self.lora_scale,
            "seed": self.seed,
            "output_format": self.output_format,
            "atlas_columns": self.atlas_columns,
            "panel_inset": self.panel_inset,
            "full_page_fallback": self.full_page_fallback,
            "blank_ink_threshold": self.blank_ink_threshold,
            "max_megapixels": self.max_megapixels,
            "skip_first": self.skip_first,
            "limit": self.limit,
            "steps": list(self.steps),
            "mock": self.mock,
            "resume": str(self.resume) if self.resume else None,
            "from_step": self.from_step,
            "stitch_bw_fallback": self.stitch_bw_fallback,
            "debug_font_size": self.debug_font_size,
            "debug_bbox_width": self.debug_bbox_width,
            "only_panels": list(self.only_panels),
            "forced_characters": self.forced_characters,
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
    if config.workers < 1:
        raise ValueError("--workers must be at least 1")
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
    if config.detection_mode not in (
        "page", "panel", "panel-page", "panel-page-cast",
        "panel-page-prev2", "panel-page-prev2-cast"
    ):
        raise ValueError(
            "--detection-mode must be 'page', 'panel', 'panel-page', "
            "'panel-page-cast', 'panel-page-prev2' or 'panel-page-prev2-cast'"
        )
    if config.blank_ink_threshold < 0 or config.blank_ink_threshold >= 1:
        raise ValueError("--blank-ink-threshold must be in [0, 1)")
    if config.max_megapixels <= 0:
        raise ValueError("--max-megapixels must be positive")
    for selector in config.only_panels:
        from selection import parse_only_panel

        parse_only_panel(selector)  # raises ValueError on bad format
    for key, names in config.forced_characters.items():
        from selection import parse_only_panel

        parse_only_panel(key)
        if not names:
            raise ValueError(f"--force-characters {key} has no names")


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        prog="pipeline_v1",
        description=(
            "Panel-wise manga colorization: detect panels (YOLO26n), extract them "
            "in Japanese reading order, detect characters per panel (OpenRouter "
            "gemma-4-31b-it), colorize each panel with FLUX.2 Klein 9B base + LoRA "
            "(atlas filtered to the detected characters), stitch the colorized "
            "panels back onto the original page, and annotate a debug copy of "
            "each stitched page (5_debug/)."
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
    parser.add_argument("--vlm-panel-prompt-file", type=Path,
                        default=DEFAULT_VLM_PANEL_PROMPT_FILE,
                        help="Per-panel fallback prompt (cropped-panel fallback calls).")
    parser.add_argument("--vlm-panel-page-prompt-file", type=Path,
                        default=DEFAULT_VLM_PANEL_PAGE_PROMPT_FILE,
                        help="Panel+page prompt (detection_mode='panel-page' calls).")
    parser.add_argument("--vlm-panel-page-prev2-prompt-file", type=Path,
                        default=DEFAULT_VLM_PANEL_PAGE_PREV2_PROMPT_FILE,
                        help="Panel+page+prev2 prompt (detection_mode="
                             "'panel-page-prev2' calls).")
    parser.add_argument("--detection-mode",
                        choices=("page", "panel", "panel-page", "panel-page-cast",
                                 "panel-page-prev2", "panel-page-prev2-cast"),
                        default="panel-page-prev2-cast",
                        help="page: one paid call per page with per-panel fallbacks "
                             "(V1.1); panel: V1 behaviour, one call per panel; "
                             "panel-page: one call per panel sending the full page as "
                             "context plus the target panel (per-panel fallback); "
                             "panel-page-cast: panel-page with an automatically "
                             "derived per-chapter cast shortlist (from the page's "
                             "chapter via chapter_page_map.json, --cast-key wins); "
                             "panel-page-prev2: panel-page that also sends the two "
                             "preceding pages in reading order as story context "
                             "(fewer when they do not exist); panel-page-prev2-cast: "
                             "panel-page-prev2 with the automatically derived "
                             "per-chapter cast shortlist (same resolution as "
                             "panel-page-cast).")
    parser.add_argument("--cast-key", default=None,
                        help="chapter_casts.json shortlist key (e.g. c001); with "
                             "panel-page-cast / panel-page-prev2-cast it overrides "
                             "the automatic per-page derivation.")
    parser.add_argument("--chapter-page-map", type=Path,
                        default=DEFAULT_CHAPTER_PAGE_MAP_FILE,
                        help="page->chapter map for panel-page-cast / "
                             "panel-page-prev2-cast auto cast "
                             "(default: frieren_wiki_dataset/chapter_page_map.json).")
    parser.add_argument("--colorizer-prompt-file", type=Path,
                        default=DEFAULT_COLORIZER_PROMPT_FILE)
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES_FILE,
                        help="Canonical character profiles (task 0002).")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="OpenRouter sampling temperature for detection (0 = mostly deterministic)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between OpenRouter calls (rate-limit backoff "
                             "is handled by the client; ignored when --workers > 1).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel character-detection worker threads "
                             "(1 = sequential; pages are processed concurrently).")
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
    parser.add_argument("--no-full-page-fallback", dest="full_page_fallback",
                        action="store_false",
                        help="Disable the synthetic full-page panel for pages with "
                             "zero YOLO detections (task 0004).")
    parser.add_argument("--blank-ink-threshold", type=float, default=0.005,
                        help="Ink-ratio below which a page is treated as blank.")
    parser.add_argument("--max-megapixels", type=float, default=2.0,
                        help="FLUX request size cap in megapixels (aspect preserved, "
                             "multiples of 16).")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="Skip the first N pages of the input folder.")
    parser.add_argument("--limit", type=int,
                        help="Process only the first N pages after skip.")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated subset of panels,characters,colorize,stitch,debug.")
    parser.add_argument("--debug-font-size", type=int, default=42,
                        help="5_debug label font size in px (default 42).")
    parser.add_argument("--debug-bbox-width", type=int, default=5,
                        help="5_debug bounding-box stroke width in px (default 5).")
    parser.add_argument("--from-step", choices=STEP_ORDER,
                        help="Start at this step (skip earlier ones).")
    parser.add_argument("--resume", type=Path,
                        help="Reuse step outputs from a previous run directory.")
    parser.add_argument("--stitch-bw-fallback", action="store_true",
                        help="Stitch missing colorized panels from the original "
                             "black & white crop (logged and recorded) instead of "
                             "failing the stitch step.")
    parser.add_argument("--only-panel", action="append", default=[], metavar="PAGE:PANEL",
                        help="Process only the selected panel(s) (repeatable; e.g. "
                             "P003:panel_0006). Pages are matched by exact stem, "
                             "fixture alias (P003), or substring.")
    parser.add_argument("--force-characters", action="append", default=[],
                        metavar="PAGE:PANEL=Name1,Name2",
                        help="Ground-truth identities for selected panels: no paid "
                             "detection call is made for them (repeatable).")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock backends (no YOLO/OpenRouter/FLUX calls).")
    args = parser.parse_args(argv)

    try:
        steps = parse_steps(args.steps)
        from selection import parse_force_characters

        forced_characters: dict[str, list[str]] = {}
        for value in args.force_characters:
            page_sel, panel_sel, names = parse_force_characters(value)
            forced_characters[f"{page_sel}:{panel_sel}"] = names
        config = PipelineConfig(
            input_dir=args.input_dir,
            refs_dir=args.refs_dir,
            output_root=args.output_root,
            vlm_model=args.vlm_model,
            vlm_prompt_file=args.vlm_prompt_file,
            vlm_panel_prompt_file=args.vlm_panel_prompt_file,
            vlm_panel_page_prompt_file=args.vlm_panel_page_prompt_file,
            vlm_panel_page_prev2_prompt_file=args.vlm_panel_page_prev2_prompt_file,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            sleep_s=args.sleep,
            workers=args.workers,
            api_key_env=args.api_key_env,
            detection_mode=args.detection_mode,
            cast_key=args.cast_key,
            chapter_page_map_file=args.chapter_page_map,
            endpoint=args.endpoint,
            colorizer_prompt_file=args.colorizer_prompt_file,
            profiles_file=args.profiles_file,
            flux_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            lora_scale=args.lora_scale,
            seed=args.seed,
            output_format=args.output_format,
            atlas_columns=args.atlas_columns,
            panel_inset=args.panel_inset,
            full_page_fallback=args.full_page_fallback,
            blank_ink_threshold=args.blank_ink_threshold,
            max_megapixels=args.max_megapixels,
            skip_first=args.skip_first,
            limit=args.limit,
            steps=steps,
            mock=args.mock,
            resume=args.resume,
            from_step=args.from_step,
            stitch_bw_fallback=args.stitch_bw_fallback,
            debug_font_size=args.debug_font_size,
            debug_bbox_width=args.debug_bbox_width,
            only_panels=tuple(args.only_panel),
            forced_characters=forced_characters,
        )
        _validate(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def nearest_multiple_of(value: int, multiple: int = FLUX_MULTIPLE) -> int:
    """Closest multiple of `multiple` to `value`, clamped to a minimum of
    `multiple` (so rounding never yields 0)."""
    return max(multiple, round(value / multiple) * multiple)


def bounded_requested_size(
    width: int,
    height: int,
    max_megapixels: float,
) -> tuple[int, int]:
    """FLUX request size for a panel of `width`x`height`, capped to at most
    `max_megapixels` (task 0004).

    Ordinary panels keep the V1 policy (closest resolution with both axes
    multiples of 16). Oversized panels are scaled down proportionally to fit
    the cap, then rounded to multiples of 16; if the rounded area still
    exceeds the cap, the axis with the larger rounding overshoot is reduced
    by one multiple until the area fits. Never upscales.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"panel size must be positive, got {width}x{height}")
    max_pixels = max_megapixels * 1_000_000
    if width * height <= max_pixels:
        return (nearest_multiple_of(width), nearest_multiple_of(height))
    import math

    scale = math.sqrt(max_pixels / (width * height))
    ideal_w = width * scale
    ideal_h = height * scale
    requested_w = nearest_multiple_of(ideal_w)
    requested_h = nearest_multiple_of(ideal_h)
    while requested_w * requested_h > max_pixels:
        if (requested_w - ideal_w) >= (requested_h - ideal_h):
            requested_w -= FLUX_MULTIPLE
        else:
            requested_h -= FLUX_MULTIPLE
    return (requested_w, requested_h)


def requested_panel_size(width: int, height: int) -> tuple[int, int]:
    """FLUX request size for a panel of `width`x`height`: the resolution
    closest to the original with both axes multiples of 16 (user-confirmed
    size policy)."""
    return (nearest_multiple_of(width), nearest_multiple_of(height))


if __name__ == "__main__":
    # Quick CLI sanity check: python pipeline_v1/config.py --steps panels,stitch
    config = parse_args()
    print(config.to_dict())
