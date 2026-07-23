#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "harvest_lamet.py"),
        "update",
    ]
    process = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
