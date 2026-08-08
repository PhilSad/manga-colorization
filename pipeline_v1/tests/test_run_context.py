"""Tests for run_context.py: fresh run dirs, atomic JSON, manifest lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from run_context import RunContext, create_run_dir, read_json, write_json


def test_create_run_dir_fresh(tmp_path):
    run_dir = create_run_dir(tmp_path)
    assert run_dir.is_dir()
    assert run_dir.name.startswith("20")  # YYYYMMDD-HHMMSS


def test_create_run_dir_never_overwrites(tmp_path):
    first = create_run_dir(tmp_path)
    second = create_run_dir(tmp_path)
    assert first != second
    # Force a collision by creating a dir with the same candidate name.
    third = create_run_dir(tmp_path)
    assert third != first and third != second


def test_write_json_atomic(tmp_path):
    target = tmp_path / "doc.json"
    write_json(target, {"a": 1})
    assert read_json(target) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_run_context_manifest_lifecycle(tmp_path):
    ctx = RunContext.create(
        tmp_path,
        {"schema_version": 1, "pipeline": "test", "status": "running"},
    )
    assert ctx.manifest_path.is_file()
    assert read_json(ctx.manifest_path)["status"] == "running"
    ctx.manifest["pages"] = [{"n": 1}]
    ctx.set_status("completed")
    assert read_json(ctx.manifest_path)["status"] == "completed"
    assert read_json(ctx.manifest_path)["finished_at"] is not None


def test_run_context_step_dirs(tmp_path):
    ctx = RunContext.create(tmp_path, {"status": "running"})
    panels = ctx.step_dir("panels")
    chars = ctx.step_dir("characters")
    assert panels == ctx.run_dir / "1_panels"
    assert chars == ctx.run_dir / "2_characters"
    assert panels.is_dir() and chars.is_dir()
    # step_dir is idempotent
    assert ctx.step_dir("panels") == panels


def test_run_context_load(tmp_path):
    ctx = RunContext.create(tmp_path, {"schema_version": 1, "status": "running"})
    loaded = RunContext.load(ctx.run_dir)
    assert loaded.manifest["schema_version"] == 1


def test_run_context_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunContext.load(tmp_path / "nope")


def test_run_context_error_status(tmp_path):
    ctx = RunContext.create(tmp_path, {"status": "running"})
    ctx.set_status("failed", error="boom")
    doc = read_json(ctx.manifest_path)
    assert doc["status"] == "failed"
    assert doc["error"] == "boom"
