"""Smoke test: each numbered CLI script responds to --help.

Pass condition: exit code 0 (clean help) or 2 (argparse usage error). Any
other exit indicates an import-time failure of the script.
"""
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

SCRIPT_NAMES = sorted(
    p.name for p in SCRIPTS_DIR.glob("0[1-7]*_*.py")
)


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_script_help(script_name):
    script = SCRIPTS_DIR / script_name
    assert script.exists(), f"missing script: {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        timeout=120,
    )
    assert result.returncode in (0, 2), (
        f"{script_name} exited {result.returncode}\n"
        f"stdout: {result.stdout.decode(errors='replace')[:500]}\n"
        f"stderr: {result.stderr.decode(errors='replace')[:500]}"
    )
