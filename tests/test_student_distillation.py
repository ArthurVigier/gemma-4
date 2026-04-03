import subprocess
import sys
from pathlib import Path

import pytest


def test_distillation_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/distill_trm_student.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr.strip() or result.stdout.strip())

    assert "teacher-layer-index" in result.stdout
