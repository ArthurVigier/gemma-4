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


class GridAttentionEncoder(nn.Module):
    """Preserves 2D topology and captures global relations using self-attention."""

    def __init__(self, config: ReasonerConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Learned 2D Positional Embeddings
        self.row_embed = nn.Parameter(torch.randn(config.grid_height, self.hidden_size) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(config.grid_width, self.hidden_size) * 0.02)
        
        # Self-Attention Layer (Global)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            batch_first=True,
        )
        
        # Semantic projection
        self.norm = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input x: (batch, H, W, hidden_size)
        batch, h, w, d = x.shape
        
        # Add 2D Positional Embeddings
        x = x + self.row_embed.view(1, h, 1, d) + self.col_embed.view(1, 1, w, d)
        
        # Flatten grid to sequence: (batch, H*W, hidden_size)
        flat_x = x.reshape(batch, h * w, d)
        
        # Global Self-Attention
        attn_out, _ = self.mha(flat_x, flat_x, flat_x)
        out = self.norm(flat_x + attn_out)
        
        # Apply MLP and Global Mean Pool to get the final "thought" vector
        out = out + self.mlp(out)
        return out.mean(dim=1)


class TRMReasoner(nn.Module):
    """A compact recursive reasoner with adaptive halting.

    This version uses a GridAttentionEncoder to capture global spatial relations
    similar to how Gemma-4's transformer backbone operates.
    """

    def __init__(self, config: ReasonerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReasonerConfig()

        self.embedding = nn.Embedding(self.config.color_count, self.config.hidden_size)
        self.encoder = GridAttentionEncoder(self.config)
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
