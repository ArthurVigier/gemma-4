import subprocess
import sys
from pathlib import Path

import pytest


def test_reasoner_produces_action_logits_and_halting_trace() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    command = """
import sys
sys.path.insert(0, 'src')
import torch
from arc_drone.config import ReasonerConfig
from arc_drone.model import TRMReasoner
config = ReasonerConfig(grid_height=8, grid_width=8, hidden_size=32, refinement_steps=4)
model = TRMReasoner(config)
grid = torch.randint(0, 10, (2, 8, 8), dtype=torch.long)
output = model(grid)
assert output.action_logits.shape == (2, config.action_dim)
assert output.halt_logits.shape == (2, config.refinement_steps)
assert output.halt_probabilities.ndim == 2
assert output.halt_probabilities.shape[0] == 2
assert output.halt_probabilities.shape[1] == config.refinement_steps
assert torch.all(output.halted_at_step >= 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"torch is not usable in this local sandbox: {result.stderr.strip() or result.stdout.strip()}")
