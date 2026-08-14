"""Shared test fixtures. Makes the pipeline package importable as flat modules
(`import config`, `import orchestrator`, ...) by inserting the package dir on
sys.path, so tests work regardless of the current working directory.

Also hosts the real-network integration-suite scaffolding: the session-scoped
`integration_run` fixture (timestamped output dir + manifest, per the repo's
run conventions), plus the `openrouter_key` / `spark_endpoint` prerequisite
fixtures that skip their tests when the paid API key or the Spark FLUX
server is unavailable.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

REPO_ROOT = PIPELINE_DIR.parent
load_dotenv(REPO_ROOT / ".env")  # OPENROUTER_API_KEY for the integration suite

OUTPUT_ROOT = PIPELINE_DIR / "tests" / "output"


@pytest.fixture(scope="session")
def integration_run(request):
    """One timestamped output dir per `pytest -m integration` session
    (`tests/output/YYYYMMDD-HHMMSS[-gwN]/`) with an incremental
    manifest.json, the same convention as pipeline runs. Under pytest-xdist
    (`-n 8`) each worker is its own pytest session, so the run dir carries
    the xdist worker id (e.g. `20260810-120000-gw3`) to keep parallel
    workers that start in the same second from colliding on one dir (and
    one manifest). Yields a small namespace: `run_dir`, `manifest` (the
    manifest dict), `record(case_id, **fields)` which appends a per-case
    record and refreshes the cost totals.
    """
    from integration_support import iso_now, write_json

    worker_id = getattr(request.config, "workerinput", {}).get("workerid")
    suffix = f"-{worker_id}" if worker_id else ""
    run_dir = OUTPUT_ROOT / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "integration-test-run",
        "created_at": iso_now(),
        "cases": [],
        "totals": {"openrouter_cost_usd": 0.0},
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"[integration] output dir: {run_dir}", file=sys.stderr)

    class Run:
        def record(self, case_id: str, **fields) -> None:
            record = {"case_id": case_id, **fields}
            manifest["cases"].append(record)
            cost = fields.get("cost_usd")
            if isinstance(cost, (int, float)):
                manifest["totals"]["openrouter_cost_usd"] = round(
                    manifest["totals"]["openrouter_cost_usd"] + cost, 8
                )
            write_json(run_dir / "manifest.json", manifest)

    run = Run()
    run.run_dir = run_dir
    run.manifest = manifest
    return run


@pytest.fixture(scope="session")
def openrouter_key():
    """Real OpenRouter API key; skips the test when unavailable."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set; integration test skipped")
    return api_key


@pytest.fixture(scope="session")
def spark_endpoint():
    """Spark FLUX server base URL; skips the test when the server is down."""
    import urllib.request

    endpoint = os.getenv("FLUX_ENDPOINT", "http://spark:3000")
    try:
        with urllib.request.urlopen(f"{endpoint}/healthz", timeout=5) as response:
            if response.status != 200:
                raise OSError(f"healthz returned {response.status}")
    except Exception as error:  # noqa: BLE001 - any connectivity problem
        pytest.skip(f"Spark FLUX server not reachable at {endpoint} ({error})")
    return endpoint


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
