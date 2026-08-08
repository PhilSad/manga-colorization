#!/usr/bin/env python3
"""pipeline_v1 entry point.

Parses the pipeline configuration and (from task_0008 on) runs the orchestrator.
Until the orchestrator lands, this validates the CLI and prints the parsed
configuration as JSON so the scaffold can be smoke-tested.

Usage:
    .venv/bin/python pipeline_v1/run.py --help
    .venv/bin/python pipeline_v1/run.py --input-dir data/chapter_134 --refs-dir data/refs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import parse_args  # noqa: E402


def main() -> int:
    args = parse_args()
    if args.mock:
        print("[mock] mock backends enabled (no external calls)", file=sys.stderr)
    print(json.dumps(args.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
