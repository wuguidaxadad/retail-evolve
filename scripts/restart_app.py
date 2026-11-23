"""Restart application processes as one operation, retaining MySQL and Redis."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
python = ROOT / ".venv/bin/python"
subprocess.run(
    [str(python), str(ROOT / "scripts/stop_local.py"), "--app-only"], cwd=ROOT, check=True
)
subprocess.run([str(python), str(ROOT / "scripts/start_local.py")], cwd=ROOT, check=True)
