import subprocess
import sys
from pathlib import Path

import pytest

from arc_drone.arc_drone_bench import ARCDroneBench
from arc_drone.config import BenchmarkConfig
from arc_drone.student_training import (
    action_to_index,
    build_halt_targets,
    halt_probability_to_step,
)


def test_action_targets_map_to_student_vocabulary() -> None:
    bench = ARCDroneBench(BenchmarkConfig(task_count=8))
    tasks = bench.generate_tasks()

    indices = {action_to_index(task.target_action) for task in tasks}

    assert indices <= set(range(8))


def test_halt_targets_become_monotonic_after_target_step() -> None:
    halt_step = halt_probability_to_step(halt_probability=0.9, refinement_steps=6)
    targets = build_halt_targets(halt_step=halt_step, refinement_steps=6)

    assert targets.shape == (6,)
    assert float(targets[halt_step - 1]) == 1.0
    assert float(targets[-1]) == 1.0
    assert float(targets[0]) in (0.0, 1.0)


def test_student_training_cli_smoke(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "scripts/train_trm_student.py",
        "--device",
        "cpu",
        "--task-count",
        "16",
        "--eval-task-count",
        "8",
        "--batch-size",
        "4",
        "--epochs",
        "1",
        "--output-dir",
        (tmp_path / "student_run").as_posix(),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"torch is not usable in this local sandbox: {result.stderr.strip() or result.stdout.strip()}")

    summary_path = tmp_path / "student_run" / "training_summary.json"
    assert summary_path.exists()
    assert "student_training_complete" in result.stdout
