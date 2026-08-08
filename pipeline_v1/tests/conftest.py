"""Shared test fixtures. Makes the pipeline package importable as flat modules
(`import config`, `import orchestrator`, ...) by inserting the package dir on
sys.path, so tests work regardless of the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


@pytest.fixture
def pipeline_dir() -> Path:
    return PIPELINE_DIR


@pytest.fixture
def tmp_run_context(tmp_path: Path):
    """A RunContext in a tmp dir, pre-seeded with a minimal manifest."""
    from run_context import RunContext

    ctx = RunContext(tmp_path / "run")
    ctx.run_dir.mkdir(parents=True)
    ctx.write_manifest()
    return ctx


@pytest.fixture
def minimal_config(tmp_path: Path):
    """A PipelineConfig pointing at tmp input/refs dirs and tmp output."""
    from config import PipelineConfig

    input_dir = tmp_path / "pages"
    refs_dir = tmp_path / "refs"
    input_dir.mkdir()
    refs_dir.mkdir()
    return PipelineConfig(
        input_dir=input_dir,
        refs_dir=refs_dir,
        output_root=tmp_path / "output",
        endpoint=None,
        mock=True,
    )
