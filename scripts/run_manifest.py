"""Run a LaMET workflow manifest through the package CLI.

Example usage:
    python scripts/run_manifest.py validate examples/demo_manifest.json
    python scripts/run_manifest.py workflow examples/demo_manifest.json
    python scripts/run_manifest.py run examples/demo_manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
