"""Run-directory and manifest machinery (repo conventions).

Port of the established conventions from the research methods
(character_detection_methods/*/run.py): a fresh timestamped run directory
per invocation, atomic JSON writes, and an incremental manifest updated
throughout the run.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config import STEP_DIRS


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_run_dir(output_root: Path) -> Path:
    """Create a fresh timestamped run directory, never overwriting an
    existing one (collisions get a -NN suffix)."""
    output_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a JSON document (temp file + rename)."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class RunContext:
    """Owns the run directory and the manifest document for one pipeline run."""

    def __init__(self, run_dir: Path, manifest: dict[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest = manifest if manifest is not None else {
            "schema_version": 1,
            "status": "running",
            "started_at": iso_now(),
            "finished_at": None,
        }

    @classmethod
    def create(cls, output_root: Path, manifest: dict[str, Any]) -> "RunContext":
        run_dir = create_run_dir(output_root)
        ctx = cls(run_dir, manifest)
        ctx.write_manifest()
        return ctx

    @classmethod
    def load(cls, run_dir: Path) -> "RunContext":
        run_dir = Path(run_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest.json in {run_dir}")
        return cls(run_dir, read_json(manifest_path))

    # -- directories -------------------------------------------------------

    def step_dir(self, step: str) -> Path:
        """Path of the numbered intermediate directory for a step (e.g.
        1_panels/), created on first access."""
        path = self.run_dir / STEP_DIRS[step]
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- manifest ----------------------------------------------------------

    def write_manifest(self) -> None:
        write_json(self.manifest_path, self.manifest)

    def set_status(self, status: str, error: str | None = None) -> None:
        self.manifest["status"] = status
        self.manifest["finished_at"] = iso_now()
        if error is not None:
            self.manifest["error"] = error
        self.write_manifest()

    def __str__(self) -> str:
        return f"RunContext(run_dir={self.run_dir})"


def package_versions(packages: list[str]) -> dict[str, str]:
    """Best-effort installed version lookup for the manifest."""
    import importlib.metadata

    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def python_version() -> str:
    return sys.version
