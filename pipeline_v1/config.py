"""Pipeline configuration: dataclass + argparse CLI.

Every knob of every stage lives here so orchestrator, steps and tests share one
definition. Defaults follow the repo conventions (AGENTS.md) and the research
methods the pipeline ports.

Named CLI profiles (`--profile NAME`, `cli_profiles.json`) expand to default
flags: explicit command-line flags always win over profile values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent

# Pipeline stages in execution order; the run directory for each stage is
# prefixed with its 1-based index (1_panels/, 2_characters/, ...).
STEP_ORDER: tuple[str, ...] = (
    "panels", "characters", "colorize", "stitch", "debug", "pdf"
)
STEP_DIRS: dict[str, str] = {
    "panels": "1_panels",
    "characters": "2_characters",
    "colorize": "3_colorized",
    "stitch": "4_stitched",
    "debug": "5_debug",
    "pdf": "6_pdf",
}

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_OUTPUT_FORMATS = ("png", "jpeg", "webp")

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "chapter_134"
DEFAULT_REFS_DIR = REPO_ROOT / "data" / "refs"
DEFAULT_OUTPUT_ROOT = PIPELINE_DIR / "output"
DEFAULT_ENDPOINT = "http://spark:3000"
DEFAULT_VLM_MODEL = "google/gemma-4-31b-it"
DEFAULT_CLI_PROFILES_FILE = PIPELINE_DIR / "cli_profiles.json"
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

# Color verification (verify loop, --verify-attempts): OpenRouter vision model
# with strict json_schema structured output (analyse/good_color/fix_prompt).
DEFAULT_VERIFY_MODEL = "openai/gpt-5.6-luna"
DEFAULT_VERIFY_PROMPT_FILE = PIPELINE_DIR / "verify_color_prompt.txt"

# bbox mode (--verify-mode bbox, full-page only): the verifier's bbox verdict
# schema (analyse/good_color/fix_prompt/regions[]) with high reasoning effort
# and an 8192-token budget (the probe proved 2048 truncates and wastes the
# call); retries then region-edit the boxed areas with gpt-image-2
# (region_edit.py) instead of re-colorizing the whole page.
DEFAULT_VERIFY_BBOX_PROMPT_FILE = PIPELINE_DIR / "verify_bbox_prompt.txt"
DEFAULT_REGION_EDIT_PROMPT_FILE = PIPELINE_DIR / "gpt_region_edit_prompt.txt"

# FLUX VAE constraint: every requested dimension must be a multiple of 16.
FLUX_MULTIPLE = 16
# The smallest dimension we will ever request (rounding must not produce 0).
FLUX_MIN_SIDE = 16
# Hard floor enforced by the FLUX.2 Klein edit pipeline on Spark: any input
# image (panel or atlas) with an axis below this is rejected with "Image too
# small: WxH. Both dimensions must be at least 64px". Degenerate panels below
# the floor must be upscaled client-side or they cannot be colorized at all.
FLUX_MIN_AXIS = 64

# OpenAI gpt-image-2 (images/edit) size constraints (API docs, verified by
# research-v2): every edge a multiple of 16, area in [655,360, 8,294,400] px,
# max edge <= 3840, aspect ratio <= 3:1. Full-page mode colorizes the whole
# page at the *minimal* size that satisfies the API while preserving the
# page's exact aspect ratio (see minimal_gpt_image_size).
GPT_IMAGE_MULTIPLE = 16
GPT_IMAGE_MIN_PIXELS = 655_360
GPT_IMAGE_MAX_PIXELS = 8_294_400
GPT_IMAGE_MAX_EDGE = 3840
GPT_IMAGE_MAX_RATIO = 3.0
# Quality is user-confirmed fixed at medium (no --gpt-quality flag);
# research-v2 measured 672x1008 @ medium ~= $0.0499/page.
GPT_IMAGE_QUALITY = "medium"
# Default atlas prompt (generalized from research-v2/data/atlas/prompt.txt).
DEFAULT_GPT_IMAGE_PROMPT_FILE = PIPELINE_DIR / "gpt_image_prompt.txt"
DEFAULT_GPT_MODEL = "gpt-image-2"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


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
    worker_detection: int = 1      # parallel character-detection threads (1 = sequential)
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

    # Full-page gpt-image-2 atlas mode: no panel extraction, the whole page is
    # colorized in one call per page with a labelled reference atlas.
    full_page: bool = False
    # Where the atlas characters come from in full-page mode: "detected" =
    # one VLM call per page (detection_mode forced to "page"); "cast" = the
    # full chapter cast (auto-derived via cast_key_for_page / --cast-key),
    # zero VLM calls (characters step becomes a no-op).
    atlas_source: str = "detected"   # detected | cast
    gpt_model: str = DEFAULT_GPT_MODEL
    gpt_image_prompt_file: Path = DEFAULT_GPT_IMAGE_PROMPT_FILE
    gpt_size: str | None = None      # optional "WxH" override (default: minimal)
    gpt_atlas_scale: float = 1.0     # downscale the atlas before upload
    openai_api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV

    # Parallel colorization worker threads (1 = sequential, current behavior).
    worker_colorization: int = 1

    # Verification loop: 0 = disabled; 1 = verify each panel and output the
    # fix prompt without re-colorizing; N >= 2 = up to N colorization attempts
    # with at most N-1 fix-prompt retries (verify_loop.py).
    verify_attempts: int = 0
    verify_model: str = DEFAULT_VERIFY_MODEL
    verify_prompt_file: Path = DEFAULT_VERIFY_PROMPT_FILE
    verify_max_tokens: int = 1024
    verify_api_key_env: str = "OPENROUTER_API_KEY"
    # bbox mode (full-page only): retries region-edit the localized boxes via
    # gpt-image-2 instead of re-colorizing the whole page (region_edit.py).
    verify_mode: str = "fix-prompt"        # fix-prompt | bbox
    verify_bbox_prompt_file: Path = DEFAULT_VERIFY_BBOX_PROMPT_FILE
    verify_reasoning_effort: str = "high"  # bbox mode only (passed as reasoning.effort)
    region_edit_prompt_file: Path = DEFAULT_REGION_EDIT_PROMPT_FILE
    region_edit_model: str | None = None   # None -> reuse --gpt-model

    # Page selection (repo convention: --skip-first / --limit)
    skip_first: int = 0
    limit: int | None = None

    # Orchestration
    steps: tuple[str, ...] = STEP_ORDER
    mock: bool = False
    resume: Path | None = None   # previous run dir to reuse its step outputs
    from_step: str | None = None # start at this step (skip earlier ones)
    profile: str | None = None   # applied --profile (cli_profiles.json), if any

    # Stage 5 (debug annotation): rendering knobs for 5_debug/. The step is
    # pure image processing (no backends); the standalone offline tool
    # scripts/annotate_stitch.py shares this implementation.
    debug_font_size: int = 42      # label font size in px
    debug_bbox_width: int = 5      # bounding-box stroke width in px

    # Stage 6 (pdf, 6_pdf/): PDF export of the stitched pages via Pillow's
    # native PDF writer (pure image processing, no extra dependency).
    pdf_name: str = "colorized.pdf"  # output PDF filename in 6_pdf/
    pdf_dpi: int = 72   # embedding resolution; page size in pt = px * 72 / dpi

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
            "worker_detection": self.worker_detection,
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
            "full_page": self.full_page,
            "atlas_source": self.atlas_source,
            "gpt_model": self.gpt_model,
            "gpt_image_prompt_file": str(self.gpt_image_prompt_file),
            "gpt_size": self.gpt_size,
            "gpt_atlas_scale": self.gpt_atlas_scale,
            "openai_api_key_env": self.openai_api_key_env,
            "worker_colorization": self.worker_colorization,
            "verify_attempts": self.verify_attempts,
            "verify_model": self.verify_model,
            "verify_prompt_file": str(self.verify_prompt_file),
            "verify_max_tokens": self.verify_max_tokens,
            "verify_api_key_env": self.verify_api_key_env,
            "verify_mode": self.verify_mode,
            "verify_bbox_prompt_file": str(self.verify_bbox_prompt_file),
            "verify_reasoning_effort": self.verify_reasoning_effort,
            "region_edit_prompt_file": str(self.region_edit_prompt_file),
            "region_edit_model": self.region_edit_model,
            "skip_first": self.skip_first,
            "limit": self.limit,
            "steps": list(self.steps),
            "mock": self.mock,
            "resume": str(self.resume) if self.resume else None,
            "from_step": self.from_step,
            "profile": self.profile,
            "debug_font_size": self.debug_font_size,
            "debug_bbox_width": self.debug_bbox_width,
            "pdf_name": self.pdf_name,
            "pdf_dpi": self.pdf_dpi,
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
    if config.worker_detection < 1:
        raise ValueError("--worker-detection must be at least 1")
    if config.worker_colorization < 1:
        raise ValueError("--worker-colorization must be at least 1")
    if config.verify_attempts < 0:
        raise ValueError("--verify-attempts must be non-negative")
    if config.verify_max_tokens < 1:
        raise ValueError("--verify-max-tokens must be positive")
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
        "page", "page-cast", "panel", "panel-page", "panel-page-cast",
        "panel-page-prev2", "panel-page-prev2-cast"
    ):
        raise ValueError(
            "--detection-mode must be 'page', 'page-cast', 'panel', "
            "'panel-page', 'panel-page-cast', 'panel-page-prev2' or "
            "'panel-page-prev2-cast'"
        )
    if config.blank_ink_threshold < 0 or config.blank_ink_threshold >= 1:
        raise ValueError("--blank-ink-threshold must be in [0, 1)")
    if config.max_megapixels <= 0:
        raise ValueError("--max-megapixels must be positive")
    if not config.pdf_name.strip():
        raise ValueError("--pdf-name must not be empty")
    if not config.pdf_name.lower().endswith(".pdf"):
        raise ValueError("--pdf-name must end with '.pdf'")
    if config.pdf_dpi < 1:
        raise ValueError("--pdf-dpi must be at least 1")
    for selector in config.only_panels:
        from selection import parse_only_panel

        parse_only_panel(selector)  # raises ValueError on bad format
    for key, names in config.forced_characters.items():
        from selection import parse_only_panel

        parse_only_panel(key)
        if not names:
            raise ValueError(f"--force-characters {key} has no names")

    # Full-page gpt-image-2 mode: `--atlas-source detected` forces a
    # page-level detection mode (one VLM call per page) — either the plain
    # `page` or the cast-limited `page-cast`; `--atlas-source cast` is
    # full-page mode only (zero VLM calls).
    if config.full_page:
        if config.atlas_source == "detected" and config.detection_mode not in (
            "page", "page-cast"
        ):
            print(
                f"WARNING: --full-page --atlas-source detected forces "
                f"--detection-mode 'page' (got {config.detection_mode!r})",
                file=sys.stderr,
                flush=True,
            )
            config.detection_mode = "page"
    elif config.atlas_source == "cast":
        raise ValueError("--atlas-source cast requires --full-page")
    if config.atlas_source not in ("detected", "cast"):
        raise ValueError(
            f"--atlas-source must be 'detected' or 'cast', got {config.atlas_source!r}"
        )
    if not 0 < config.gpt_atlas_scale <= 1:
        raise ValueError("--gpt-atlas-scale must be in (0, 1]")
    if config.gpt_size is not None:
        parse_gpt_size(config.gpt_size)  # raises ValueError on bad format/size

    # bbox verify mode (user decisions, docs/plans/verify-bbox-region-edit.md):
    # full-page mode only, needs retries to be meaningful, and the region
    # editor is a paid gpt-image-2 call (OpenAI key check lives in run.py).
    if config.verify_mode not in ("fix-prompt", "bbox"):
        raise ValueError(
            f"--verify-mode must be 'fix-prompt' or 'bbox', got {config.verify_mode!r}"
        )
    if config.verify_mode == "bbox" and not config.full_page:
        raise ValueError("--verify-mode bbox requires --full-page")
    if config.verify_mode == "bbox" and config.verify_attempts < 1:
        raise ValueError("--verify-mode bbox requires --verify-attempts >= 1")
    if (
        config.verify_mode == "fix-prompt"
        and config.verify_reasoning_effort != "high"
    ):
        print(
            f"WARNING: --verify-reasoning-effort is only used in bbox mode "
            f"(got {config.verify_reasoning_effort!r} with --verify-mode "
            "fix-prompt)",
            file=sys.stderr,
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the pipeline CLI (built once per parse_args;
    also used to resolve --profile before the final parse)."""
    parser = argparse.ArgumentParser(
        prog="pipeline_v1",
        description=(
            "Panel-wise manga colorization: detect panels (YOLO26n), extract them "
            "in Japanese reading order, detect characters per panel (OpenRouter "
            "gemma-4-31b-it), colorize each panel with FLUX.2 Klein 9B base + LoRA "
            "(atlas filtered to the detected characters), stitch the colorized "
            "panels back onto the original page, annotate a debug copy of "
            "each stitched page (5_debug/), and export all stitched pages "
            "as a single multi-page PDF (6_pdf/)."
        ),
    )
    try:
        available = ", ".join(sorted(load_cli_profiles()))
    except ValueError:
        available = "(see pipeline_v1/cli_profiles.json)"
    parser.add_argument("--profile", default=None, metavar="NAME",
                        help="Apply a named default profile from "
                             "pipeline_v1/cli_profiles.json; explicit flags "
                             f"override profile values. Available: {available}")
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
                        choices=("page", "page-cast", "panel", "panel-page",
                                 "panel-page-cast", "panel-page-prev2",
                                 "panel-page-prev2-cast"),
                        default="panel-page-prev2-cast",
                        help="page: one paid call per page with per-panel fallbacks "
                             "(V1.1); page-cast: page with an automatically derived "
                             "per-chapter cast shortlist (same resolution order as "
                             "panel-page-cast, i.e. --cast-key wins); panel: V1 "
                             "behaviour, one call per panel; "
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
                             "is handled by the client; ignored when "
                             "--worker-detection > 1).")
    parser.add_argument("--worker-detection", type=int, default=1,
                        help="Parallel character-detection worker threads "
                             "(1 = sequential; pages are processed concurrently).")
    parser.add_argument("--worker-colorization", type=int, default=1,
                        help="Parallel colorization worker threads over pages "
                             "(1 = sequential; each page writes only its own "
                             "3_colorized/<page>/ dir, so workers never race).")
    parser.add_argument("--verify-attempts", type=int, default=0,
                        help="Verify each colorized panel with Luna (OpenRouter, "
                             "strict structured output) and re-colorize on palette "
                             "mismatch: 1 = verify + output the fix prompt only; "
                             "2+ = up to N total colorization attempts with up to "
                             "N-1 fix-prompt retries. 0 = disabled (default). "
                             "Every attempt and verdict is recorded in "
                             "<panel>.verify.json; every superseded attempt keeps "
                             "an attempt_<n> image (including attempt_1.png when a "
                             "retry wins) and the final attempt is copied to the "
                             "canonical name.")
    parser.add_argument("--verify-model", default=DEFAULT_VERIFY_MODEL,
                        help="OpenRouter vision model for color verification "
                             "(default: openai/gpt-5.6-luna).")
    parser.add_argument("--verify-prompt-file", type=Path,
                        default=DEFAULT_VERIFY_PROMPT_FILE,
                        help="Prompt for the color verification model "
                             "(default: verify_color_prompt.txt).")
    parser.add_argument("--verify-max-tokens", type=int,
                        help="Completion token cap for the verifier "
                             "(default: 1024; 8192 when --verify-mode bbox — "
                             "the probe proved 8192 is required for high-effort "
                             "bbox output).")
    parser.add_argument("--verify-api-key-env", default="OPENROUTER_API_KEY",
                        help="Env var holding the OpenRouter key used by the "
                             "verifier (default: OPENROUTER_API_KEY).")
    parser.add_argument("--verify-mode", choices=("fix-prompt", "bbox"),
                        default="fix-prompt",
                        help="Retry backend for the verify loop: 'fix-prompt' "
                             "(default) re-colorizes the whole page with the "
                             "verdict's fix prompt; 'bbox' (full-page mode only) "
                             "draws the verdict's bboxes on the rejected page and "
                             "gpt-image-2 recolors only those regions "
                             "(region_edit.py); a mismatch without regions falls "
                             "back to the fix-prompt re-colorization.")
    parser.add_argument("--verify-bbox-prompt-file", type=Path,
                        default=DEFAULT_VERIFY_BBOX_PROMPT_FILE,
                        help="Prompt for the bbox verdict (--verify-mode bbox; "
                             "default: verify_bbox_prompt.txt).")
    parser.add_argument("--verify-reasoning-effort", default="high",
                        help="OpenRouter reasoning.effort for the verifier in "
                             "bbox mode (default: high; ignored in fix-prompt "
                             "mode).")
    parser.add_argument("--region-edit-prompt-file", type=Path,
                        default=DEFAULT_REGION_EDIT_PROMPT_FILE,
                        help="Edit prompt template for bbox-mode region edits "
                             "(default: gpt_region_edit_prompt.txt).")
    parser.add_argument("--region-edit-model", default=None,
                        help="OpenAI image model for bbox-mode region edits "
                             "(default: reuses --gpt-model, i.e. gpt-image-2).")
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
                        help="Comma-separated subset of panels,characters,colorize,stitch,debug,pdf.")
    parser.add_argument("--debug-font-size", type=int, default=42,
                        help="5_debug label font size in px (default 42).")
    parser.add_argument("--debug-bbox-width", type=int, default=5,
                        help="5_debug bounding-box stroke width in px (default 5).")
    parser.add_argument("--pdf-name", default="colorized.pdf",
                        help="Output PDF filename in 6_pdf/ (default colorized.pdf).")
    parser.add_argument("--pdf-dpi", type=int, default=72,
                        help="PDF embedding resolution; page size in points = "
                             "pixel size * 72 / dpi (72 = 1 px : 1 pt, default).")
    parser.add_argument("--from-step", choices=STEP_ORDER,
                        help="Start at this step (skip earlier ones).")
    parser.add_argument("--resume", type=Path,
                        help="Reuse step outputs from a previous run directory.")
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
    parser.add_argument("--full-page", action="store_true",
                        help="Full-page gpt-image-2 atlas mode: no panel "
                             "extraction; the whole page is colorized in one call "
                             "per page (minimal aspect-preserving API size).")
    parser.add_argument("--atlas-source", choices=("detected", "cast"),
                        default="detected",
                        help="Where the atlas characters come from in full-page "
                             "mode: 'detected' = one VLM call per page (forces "
                             "--detection-mode page); 'cast' = the full chapter "
                             "cast (auto-derived via cast_key_for_page / "
                             "--cast-key), zero VLM calls. 'cast' requires "
                             "--full-page.")
    parser.add_argument("--gpt-model", default=DEFAULT_GPT_MODEL,
                        help="OpenAI image model (default gpt-image-2).")
    parser.add_argument("--gpt-image-prompt-file", type=Path,
                        default=DEFAULT_GPT_IMAGE_PROMPT_FILE,
                        help="Atlas prompt for full-page gpt-image-2 calls "
                             "(default: pipeline_v1/gpt_image_prompt.txt).")
    parser.add_argument("--gpt-size", default=None, metavar="WxH",
                        help="Optional gpt-image-2 output size override for "
                             "comparison runs; must satisfy the API constraints "
                             "(edges multiples of 16, area in "
                             "[655360, 8294400] px, max edge 3840, ratio <= 3:1). "
                             "Default: minimal size preserving the page's aspect "
                             "ratio.")
    parser.add_argument("--gpt-atlas-scale", type=float, default=1.0, metavar="F",
                        help="Downscale the built atlas by this factor before "
                             "upload (e.g. 0.5 = half the edge length = 1/4 the "
                             "pixels; gpt-image-2 bills input image tokens by "
                             "size).")
    parser.add_argument("--openai-api-key-env", default=DEFAULT_OPENAI_API_KEY_ENV,
                        help="Env var holding the OpenAI API key for full-page "
                             "gpt-image-2 calls (default OPENAI_API_KEY).")
    return parser


def load_cli_profiles(path: Path = DEFAULT_CLI_PROFILES_FILE) -> dict[str, dict]:
    """Load the named CLI profiles from `cli_profiles.json`: a mapping of
    profile name -> {"description": str, "args": {flag: value}}, where args
    keys are flag names without the leading dashes (e.g. "full-page")."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read cli profiles file {path}: {exc}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in cli profiles file {path}: {exc}")
    profiles = raw.get("profiles", raw) if isinstance(raw, dict) else raw
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            f"cli profiles file {path} must contain a non-empty 'profiles' mapping"
        )
    result: dict[str, dict] = {}
    for name, entry in profiles.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("args"), dict):
            raise ValueError(
                f"profile {name!r} in {path} must be an object with an 'args' "
                "mapping of flag -> value"
            )
        result[str(name)] = {
            "description": str(entry.get("description", "")),
            "args": dict(entry["args"]),
        }
    return result


def profile_to_argv(args: dict[str, object]) -> list[str]:
    """Expand a profile's {flag: value} mapping into CLI argv tokens:
    `--flag` for boolean True, `--flag <value>` otherwise; False and null
    values emit nothing (there is no bare flag that could express them)."""
    tokens: list[str] = []
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    for flag, value in args.items():
        if (
            not isinstance(flag, str) or not flag
            or flag.startswith("-")
            or any(ch not in allowed for ch in flag)
        ):
            raise ValueError(f"invalid profile flag {flag!r}")
        if value is None or value is False:
            continue
        tokens.append(f"--{flag}")
        if value is not True:
            tokens.append(str(value))
    return tokens


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # Resolve --profile before the final parse: profile values are injected as
    # argv tokens *before* the user's own flags, so argparse's last-wins
    # semantics give explicit command-line flags precedence over profile
    # defaults (store/append actions keep collecting, as documented).
    profile_name = None
    try:
        probe = parser.parse_args(argv)
        profile_name = getattr(probe, "profile", None)
        if profile_name:
            profiles = load_cli_profiles()
            if profile_name not in profiles:
                parser.error(
                    f"unknown profile {profile_name!r}; available: "
                    f"{sorted(profiles)}"
                )
            profile_args = profiles[profile_name]["args"]
            unknown = [
                f"--{flag}" for flag in profile_args
                if f"--{flag}" not in parser._option_string_actions
            ]
            if unknown:
                parser.error(
                    f"profile {profile_name!r} references unknown flag(s): "
                    f"{', '.join(sorted(unknown))}"
                )
            argv = profile_to_argv(profile_args) + argv
    except ValueError as exc:
        parser.error(str(exc))
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
            worker_detection=args.worker_detection,
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
            full_page=args.full_page,
            atlas_source=args.atlas_source,
            gpt_model=args.gpt_model,
            gpt_image_prompt_file=args.gpt_image_prompt_file,
            gpt_size=args.gpt_size,
            gpt_atlas_scale=args.gpt_atlas_scale,
            openai_api_key_env=args.openai_api_key_env,
            worker_colorization=args.worker_colorization,
            verify_attempts=args.verify_attempts,
            verify_model=args.verify_model,
            verify_prompt_file=args.verify_prompt_file,
            verify_max_tokens=(
                args.verify_max_tokens
                if args.verify_max_tokens is not None
                else 8192 if args.verify_mode == "bbox" else 1024
            ),
            verify_api_key_env=args.verify_api_key_env,
            verify_mode=args.verify_mode,
            verify_bbox_prompt_file=args.verify_bbox_prompt_file,
            verify_reasoning_effort=args.verify_reasoning_effort,
            region_edit_prompt_file=args.region_edit_prompt_file,
            region_edit_model=args.region_edit_model,
            skip_first=args.skip_first,
            limit=args.limit,
            steps=steps,
            mock=args.mock,
            resume=args.resume,
            from_step=args.from_step,
            profile=profile_name,
            debug_font_size=args.debug_font_size,
            debug_bbox_width=args.debug_bbox_width,
            pdf_name=args.pdf_name,
            pdf_dpi=args.pdf_dpi,
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
    min_axis: int = FLUX_MIN_AXIS,
) -> tuple[int, int]:
    """FLUX request size for a panel of `width`x`height`, capped to at most
    `max_megapixels` (task 0004).

    Ordinary panels keep the V1 policy (closest resolution with both axes
    multiples of 16). Oversized panels are scaled down proportionally to fit
    the cap, then rounded to multiples of 16; if the rounded area still
    exceeds the cap, the axis with the larger rounding overshoot is reduced
    by one multiple until the area fits.

    `min_axis` is the server-side floor (`FLUX_MIN_AXIS`): the Spark edit
    pipeline rejects any input image with an axis below 64 px ("Image too
    small"). Degenerate panels whose rounded size falls below the floor are
    upscaled proportionally to reach it (the only case where the size policy
    upscales — the alternative is a request the server always rejects); for
    such panels the requested size may exceed `max_megapixels` (a documented
    exception, noted in the colorizer record).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"panel size must be positive, got {width}x{height}")
    if min_axis < 1:
        raise ValueError(f"min_axis must be positive, got {min_axis}")
    max_pixels = max_megapixels * 1_000_000
    if width * height <= max_pixels:
        requested_w = nearest_multiple_of(width)
        requested_h = nearest_multiple_of(height)
    else:
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
    if requested_w < min_axis or requested_h < min_axis:
        scale = max(min_axis / requested_w, min_axis / requested_h)
        requested_w = max(min_axis, nearest_multiple_of(requested_w * scale))
        requested_h = max(min_axis, nearest_multiple_of(requested_h * scale))
    return (requested_w, requested_h)


def requested_panel_size(width: int, height: int) -> tuple[int, int]:
    """FLUX request size for a panel of `width`x`height`: the resolution
    closest to the original with both axes multiples of 16 (user-confirmed
    size policy)."""
    return (nearest_multiple_of(width), nearest_multiple_of(height))


def minimal_gpt_image_size(width: int, height: int) -> tuple[int, int]:
    """Smallest gpt-image-2 output size that keeps the page's exact aspect
    ratio while satisfying the API constraints (edges multiples of 16, area
    >= GPT_IMAGE_MIN_PIXELS, max edge <= GPT_IMAGE_MAX_EDGE, ratio <= 3:1).

    Exact-ratio, floor-driven algorithm (reproduces both research-v2 measured
    sizes):
      1. Reduce the page ratio to lowest terms: g = gcd(w, h), (w', h') = (w/g, h/g).
      2. Smallest integer multiplier k such that both edges are multiples of
         16: step = lcm(16/gcd(w',16), 16/gcd(h',16)).
      3. Area floor: k_min = ceil(sqrt(655_360 / (w' * h'))).
      4. k = smallest multiple of step >= k_min.
      5. Return (w' * k, h' * k).

    Raises ValueError when the page ratio is outside [1:3, 3:1] (the API
    rejects every size at that ratio — fail loudly rather than distort).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"page size must be positive, got {width}x{height}")
    ratio = width / height
    if not (1 / GPT_IMAGE_MAX_RATIO <= ratio <= GPT_IMAGE_MAX_RATIO):
        raise ValueError(
            f"page ratio {width}:{height} is outside [1:{int(GPT_IMAGE_MAX_RATIO)}, "
            f"{int(GPT_IMAGE_MAX_RATIO)}:1]; gpt-image-2 rejects every size at "
            "that ratio (no size can preserve the exact aspect ratio)"
        )
    g = math.gcd(width, height)
    w_prime, h_prime = width // g, height // g
    step = _gpt_ratio_step(w_prime, h_prime)
    k_min = math.ceil(math.sqrt(GPT_IMAGE_MIN_PIXELS / (w_prime * h_prime)))
    k = math.ceil(k_min / step) * step
    size = (w_prime * k, h_prime * k)
    # The API also caps the output (max edge 3840, max pixels 8.29 MP). The
    # minimal exact-ratio size only scales up with k, so if it already
    # violates the caps no larger k could satisfy them either — fail loudly
    # rather than request a size the API rejects (e.g. a 2480x3508 scan whose
    # reduced 620:877 ratio needs k=16 -> 9920x14032).
    if max(size) > GPT_IMAGE_MAX_EDGE or size[0] * size[1] > GPT_IMAGE_MAX_PIXELS:
        raise ValueError(
            f"page ratio {width}:{height} has no minimal size within the API "
            f"limits (max edge {GPT_IMAGE_MAX_EDGE}, max "
            f"{GPT_IMAGE_MAX_PIXELS} px): smallest exact-ratio size would be "
            f"{size[0]}x{size[1]}"
        )
    return size


def _gpt_ratio_step(w_prime: int, h_prime: int) -> int:
    """Smallest multiplier k such that both (w_prime * k) and (h_prime * k)
    are multiples of 16: lcm(16/gcd(w',16), 16/gcd(h',16))."""
    step_w = GPT_IMAGE_MULTIPLE // math.gcd(w_prime, GPT_IMAGE_MULTIPLE)
    step_h = GPT_IMAGE_MULTIPLE // math.gcd(h_prime, GPT_IMAGE_MULTIPLE)
    return step_w * step_h // math.gcd(step_w, step_h)


def parse_gpt_size(value: str) -> tuple[int, int]:
    """Parse and validate a --gpt-size "WxH" override against the API
    constraints (multiples of 16, area range, max edge, ratio <= 3:1).
    Raises ValueError on any violation."""
    try:
        w, h = (int(part) for part in value.lower().split("x"))
    except ValueError:
        raise ValueError(f"--gpt-size must be 'WxH' (e.g. 672x1008), got {value!r}")
    if w % GPT_IMAGE_MULTIPLE or h % GPT_IMAGE_MULTIPLE:
        raise ValueError(
            f"--gpt-size {w}x{h}: both edges must be multiples of "
            f"{GPT_IMAGE_MULTIPLE}"
        )
    if not (GPT_IMAGE_MIN_PIXELS <= w * h <= GPT_IMAGE_MAX_PIXELS):
        raise ValueError(
            f"--gpt-size {w}x{h}: area must be in "
            f"[{GPT_IMAGE_MIN_PIXELS}, {GPT_IMAGE_MAX_PIXELS}] px"
        )
    if max(w, h) > GPT_IMAGE_MAX_EDGE:
        raise ValueError(
            f"--gpt-size {w}x{h}: max edge must be <= {GPT_IMAGE_MAX_EDGE}"
        )
    if not (1 / GPT_IMAGE_MAX_RATIO <= w / h <= GPT_IMAGE_MAX_RATIO):
        raise ValueError(
            f"--gpt-size {w}x{h}: aspect ratio must be <= "
            f"{int(GPT_IMAGE_MAX_RATIO)}:1"
        )
    return (w, h)


if __name__ == "__main__":
    # Quick CLI sanity check: python pipeline_v1/config.py --steps panels,stitch
    config = parse_args()
    print(config.to_dict())
