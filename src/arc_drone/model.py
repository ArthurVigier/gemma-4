"""TRM-like recursive reasoning core for ARC-drone tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ReasonerConfig


@dataclass(slots=True)
class TRMReasonerOutput:
    """Outputs produced by the recursive reasoner."""

    action_logits: Tensor
    halt_logits: Tensor
    halt_probabilities: Tensor
    hidden_states: list[Tensor]
    halted_at_step: Tensor


class SpatialEncoder(nn.Module):
    """Preserves 2D topology of the ARC grid using convolutional layers."""

    def __init__(self, config: ReasonerConfig) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input x is (batch, H, W, hidden_size)
        # Conv2d expects (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return self.cnn(x)


class TRMReasoner(nn.Module):
    """A compact recursive reasoner with adaptive halting.

    This version uses a SpatialEncoder to better capture ARC-like 2D relations
    from the symbolic grid.
    """

    def __init__(self, config: ReasonerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReasonerConfig()

        self.embedding = nn.Embedding(self.config.color_count, self.config.hidden_size)
        self.encoder = SpatialEncoder(self.config)
        self.refiner = nn.GRUCell(self.config.hidden_size, self.config.hidden_size)
        self.halting_head = nn.Linear(self.config.hidden_size, 1)
        self.action_head = nn.Linear(self.config.hidden_size, self.config.action_dim)

    def forward(self, grid: Tensor) -> TRMReasonerOutput:
        """Runs recursive self-refinement until the halting condition is met."""

        if grid.ndim != 3:
            raise ValueError("Expected grid tensor with shape (batch, height, width).")

        batch_size = grid.shape[0]
        embedded = self.embedding(grid.long())
        # embedded is (batch, H, W, hidden_size)
        state = self.encoder(embedded)

        hidden_states: list[Tensor] = [state]
        halt_logits: list[Tensor] = []
        halt_probabilities: list[Tensor] = []
        halted_at_step = torch.full(
            (batch_size,),
            fill_value=self.config.refinement_steps,
            dtype=torch.long,
            device=grid.device,
        )
        has_halted = torch.zeros(batch_size, dtype=torch.bool, device=grid.device)

        for step in range(1, self.config.refinement_steps + 1):
            state = self.refiner(state, hidden_states[-1])
            halt_logit = self.halting_head(state).squeeze(-1)
            halt_probability = torch.sigmoid(halt_logit)
            halt_logits.append(halt_logit)
            halt_probabilities.append(halt_probability)
            hidden_states.append(state)

            new_halts = (halt_probability >= self.config.halting_threshold) & (~has_halted)
            halted_at_step = torch.where(new_halts, torch.full_like(halted_at_step, step), halted_at_step)
            has_halted = has_halted | new_halts

        action_logits = self.action_head(hidden_states[-1])
        return TRMReasonerOutput(
            action_logits=action_logits,
            halt_logits=torch.stack(halt_logits, dim=1),
            halt_probabilities=torch.stack(halt_probabilities, dim=1),
            hidden_states=hidden_states,
            halted_at_step=halted_at_step,
        )
