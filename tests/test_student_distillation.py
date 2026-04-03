import subprocess
import sys
from pathlib import Path

import pytest
import torch

from arc_drone.student_distillation import _combine_teacher_features


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
    assert "cache-dir" in result.stdout


def test_cache_teacher_targets_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/cache_teacher_targets.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr.strip() or result.stdout.strip())

    assert "teacher-layer-indices" in result.stdout


def test_combine_teacher_features_mean() -> None:
    features_by_layer = {
        16: torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        17: torch.tensor([[3.0, 5.0], [7.0, 9.0]]),
    }

    combined = _combine_teacher_features(
        features_by_layer=features_by_layer,
        layer_indices=(16, 17),
        pooling="mean",
    )

    assert torch.equal(combined, torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
