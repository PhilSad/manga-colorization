#!/usr/bin/env python3
"""pipeline_v1 entry point.

Parses the pipeline configuration, builds the backends (real YOLO detector,
OpenRouter character detector, FLUX colorizer — or mocks with --mock), and runs
the orchestrator. Each run creates a fresh timestamped directory under
`--output-root` with the six numbered intermediate directories (1_panels/ …
6_pdf/) and a manifest.

Usage:
    .venv/bin/python pipeline_v1/run.py --help
    .venv/bin/python pipeline_v1/run.py \
        --input-dir data/chapter_134 --refs-dir data/refs \
        --endpoint http://spark:3000 --skip-first 3 --limit 1
    .venv/bin/python pipeline_v1/run.py --mock --limit 1   # offline demo
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from config import REPO_ROOT, parse_args  # noqa: E402


def build_backends(config):
    """Real backends (or mocks when --mock). Raises SystemExit on missing
    configuration (API key, endpoint, prompt files).

    `--full-page` selects the gpt-image-2 backend: no YOLO detector (each page
    gets one synthetic panel), and the character detector is only built for
    `--atlas-source detected` (one VLM call per page). `--atlas-source cast`
    needs no OpenRouter key at all — the characters step is a no-op and the
    colorize step derives the atlas names from the chapter cast.
    """
    if config.mock:
        from mock_backends import (
            MockCharacterDetector,
            MockColorizer,
            MockPageCharacterDetector,
            MockPanelDetector,
            MockVerifier,
        )
        from orchestrator import Backends

        print("[mock] mock backends enabled (no external calls)", file=sys.stderr)
        if config.detection_mode == "panel":
            character_detector = MockCharacterDetector()
        else:
            character_detector = MockPageCharacterDetector(
                cast_key=config.cast_key
            )
        return Backends(
            detector=MockPanelDetector(),
            character_detector=character_detector,
            colorizer=MockColorizer(
                backend="gpt-image-2" if config.full_page else "flux"
            ),
            verifier=MockVerifier(),
        )

    if config.full_page:
        return _build_full_page_backends(config)

    if not config.endpoint:
        raise SystemExit("--endpoint is required (e.g. http://spark:3000)")

    from orchestrator import Backends

    from colorizer import FluxColorizer
    from characters import OpenRouterCharacterDetector
    from detection import YoloPanelDetector

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key: set {config.api_key_env} or add it to {REPO_ROOT / '.env'}"
        )

    detector = YoloPanelDetector()
    character_detector = OpenRouterCharacterDetector(
        model=config.vlm_model,
        api_key=api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        profiles_file=config.profiles_file,
        chapter_casts_file=config.chapter_casts_file,
        chapter_page_map_file=config.chapter_page_map_file,
        cast_key=config.cast_key,
        worker_detection=config.worker_detection,
    )
    if not config.vlm_prompt_file.is_file():
        raise SystemExit(f"prompt file not found: {config.vlm_prompt_file}")
    if config.detection_mode in ("panel-page", "panel-page-cast") \
            and not config.vlm_panel_page_prompt_file.is_file():
        raise SystemExit(
            f"panel-page prompt file not found: {config.vlm_panel_page_prompt_file}"
        )
    if (
        config.detection_mode in ("panel-page-prev2", "panel-page-prev2-cast")
        and not config.vlm_panel_page_prev2_prompt_file.is_file()
    ):
        raise SystemExit(
            f"panel-page-prev2 prompt file not found: "
            f"{config.vlm_panel_page_prev2_prompt_file}"
        )
    character_detector.prepare(
        config.refs_dir,
        config.vlm_prompt_file,
        config.vlm_panel_prompt_file,
        config.vlm_panel_page_prompt_file,
        config.vlm_panel_page_prev2_prompt_file,
    )

    if not config.colorizer_prompt_file.is_file():
        raise SystemExit(f"prompt file not found: {config.colorizer_prompt_file}")
    prompt_template = config.colorizer_prompt_file.read_text(encoding="utf-8")
    colorizer = FluxColorizer(
        endpoint=config.endpoint,
        prompt_template=prompt_template,
        num_inference_steps=config.flux_steps,
        guidance_scale=config.guidance_scale,
        lora_scale=config.lora_scale,
        seed=config.seed,
        output_format=config.output_format,
        max_megapixels=config.max_megapixels,
    )
    return Backends(
        detector=detector,
        character_detector=character_detector,
        colorizer=colorizer,
        verifier=_build_verifier(config, api_key),
    )


def _build_full_page_backends(config):
    """Backends for `--full-page` (gpt-image-2 atlas mode).

    - Detector: None — the panels step writes one synthetic full-page panel
      per page without a detector (blank-page ink check still applies).
    - Character detector: only for `--atlas-source detected` (one VLM call per
      page, detection_mode forced to "page" by config validation); `cast` runs
      with zero VLM calls, so the detector stays None and no OpenRouter key is
      required.
    - Colorizer: `GptImage2Colorizer` (OpenAI `images/edit`), keyed by
      `--openai-api-key-env`.
    """
    from orchestrator import Backends

    from config import parse_gpt_size
    from gpt_colorizer import GptImage2Colorizer

    openai_key = os.getenv(config.openai_api_key_env)
    if not openai_key:
        raise SystemExit(
            f"Missing API key: set {config.openai_api_key_env} or add it to "
            f"{REPO_ROOT / '.env'}"
        )

    character_detector = None
    if config.atlas_source == "detected":
        from characters import OpenRouterCharacterDetector

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise SystemExit(
                f"Missing API key: set {config.api_key_env} or add it to "
                f"{REPO_ROOT / '.env'}"
            )
        character_detector = OpenRouterCharacterDetector(
            model=config.vlm_model,
            api_key=api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            profiles_file=config.profiles_file,
            chapter_casts_file=config.chapter_casts_file,
            chapter_page_map_file=config.chapter_page_map_file,
            cast_key=config.cast_key,
            worker_detection=config.worker_detection,
        )
        if not config.vlm_prompt_file.is_file():
            raise SystemExit(f"prompt file not found: {config.vlm_prompt_file}")
        character_detector.prepare(
            config.refs_dir,
            config.vlm_prompt_file,
            config.vlm_panel_prompt_file,
            config.vlm_panel_page_prompt_file,
            config.vlm_panel_page_prev2_prompt_file,
        )
    else:
        print(
            "[full-page] --atlas-source cast: characters step is a no-op "
            "(no OpenRouter key needed)",
            file=sys.stderr,
        )

    if not config.gpt_image_prompt_file.is_file():
        raise SystemExit(
            f"prompt file not found: {config.gpt_image_prompt_file}"
        )
    prompt_template = config.gpt_image_prompt_file.read_text(encoding="utf-8")
    colorizer = GptImage2Colorizer(
        prompt_template=prompt_template,
        model=config.gpt_model,
        size=parse_gpt_size(config.gpt_size) if config.gpt_size else None,
        atlas_scale=config.gpt_atlas_scale,
        api_key_env=config.openai_api_key_env,
        output_format=config.output_format,
    )
    return Backends(
        detector=None,
        character_detector=character_detector,
        colorizer=colorizer,
        verifier=_build_verifier(config, os.getenv(config.verify_api_key_env)),
    )


def _build_verifier(config, api_key):
    """ColorVerifier for the verify loop, or None when --verify-attempts is
    off (the default; zero extra OpenRouter calls). Raises SystemExit when
    enabled but the OpenRouter key or the prompt file is missing."""
    if config.verify_attempts <= 0:
        return None
    if not api_key:
        raise SystemExit(
            "--verify-attempts needs an OpenRouter key: set "
            f"{config.verify_api_key_env} or add it to {REPO_ROOT / '.env'}"
        )
    if not config.verify_prompt_file.is_file():
        raise SystemExit(
            f"verify prompt file not found: {config.verify_prompt_file}"
        )
    from verify_color import ColorVerifier

    return ColorVerifier(
        model=config.verify_model,
        api_key=api_key,
        prompt_template=config.verify_prompt_file.read_text(encoding="utf-8"),
        max_tokens=config.verify_max_tokens,
    )


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    config = parse_args()
    backends = build_backends(config)
    from orchestrator import PipelineRunner

    runner = PipelineRunner(config, backends)
    ctx = runner.run()
    print(f"\nrun directory: {ctx.run_dir}", flush=True)
    print(f"manifest:      {ctx.manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - friendly failure output
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
