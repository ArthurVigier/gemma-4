"""TRM-like recursive reasoning core for ARC-drone tasks with temporal video context."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ReasonerConfig


@dataclass(slots=True)
class TRMReasonerOutput:
    """Outputs produced by the recursive reasoner."""

    # (batch, chunk_size, action_dim) — full predicted action chunk.
    # Index 0 along dim=1 is the "current" action (backward-compatible use).
    action_chunk_logits: Tensor
    halt_logits: Tensor
    halt_probabilities: Tensor
    hidden_states: list[Tensor]
    halted_at_step: Tensor

    @property
    def action_logits(self) -> Tensor:
        """Convenience accessor: current-timestep action logits (batch, action_dim)."""
        return self.action_chunk_logits[:, 0, :]


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
        return out.mean(dim=1)  # (batch, hidden_size)


class TemporalContextEncoder(nn.Module):
    """Fuses a sequence of T per-frame embeddings into a single context vector.

    Runs a single-layer GRU over the temporal dimension.  The final hidden
    state captures object motion, velocity and appearance change across frames.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, frame_embeds: Tensor) -> Tensor:
        # frame_embeds: (batch, T, hidden_size)
        _, h_n = self.gru(frame_embeds)
        return h_n.squeeze(0)  # (batch, hidden_size)


class TRMReasoner(nn.Module):
    """A compact recursive reasoner with adaptive halting and temporal video context.

    Input:
        grids — (batch, T, H, W) integer grid sequence, or (batch, H, W) for
                single-frame use (automatically unsqueezed to T=1).

    Output:
        TRMReasonerOutput with action_chunk_logits of shape
        (batch, chunk_size, action_dim).
    """

    def __init__(self, config: ReasonerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReasonerConfig()

        self.embedding = nn.Embedding(self.config.color_count, self.config.hidden_size)
        self.encoder = GridAttentionEncoder(self.config)
        self.temporal_encoder = TemporalContextEncoder(self.config.hidden_size)
        self.refiner = nn.GRUCell(self.config.hidden_size, self.config.hidden_size)
        self.halting_head = nn.Linear(self.config.hidden_size, 1)
        # Predicts all chunk actions at once from the final refined hidden state.
        self.action_chunk_head = nn.Linear(
            self.config.hidden_size,
            self.config.action_chunk_size * self.config.action_dim,
        )

    def forward(self, grids: Tensor) -> TRMReasonerOutput:
        """Runs temporal context encoding then recursive self-refinement.

        Args:
            grids: (batch, T, H, W) or (batch, H, W).
        """
        if grids.ndim == 3:
            # Single-frame path — add temporal dimension for unified handling.
            grids = grids.unsqueeze(1)  # (batch, 1, H, W)

        if grids.ndim != 4:
            raise ValueError("Expected grids tensor with shape (batch, T, H, W) or (batch, H, W).")

        batch_size, T, H, W = grids.shape

        # --- Encode each frame independently ---
        # Flatten batch+time to process all frames in one pass through the encoder.
        flat_grids = grids.reshape(batch_size * T, H, W)
        embedded = self.embedding(flat_grids.long())       # (B*T, H, W, hidden)
        frame_embeds_flat = self.encoder(embedded)          # (B*T, hidden)
        frame_embeds = frame_embeds_flat.reshape(batch_size, T, self.config.hidden_size)

        # --- Fuse temporal context ---
        # GRU over T frames → single context vector summarising motion/appearance.
        state = self.temporal_encoder(frame_embeds)         # (batch, hidden)

        # --- Recursive refinement (reasoning) steps ---
        hidden_states: list[Tensor] = [state]
        halt_logits: list[Tensor] = []
        halt_probabilities: list[Tensor] = []
        halted_at_step = torch.full(
            (batch_size,),
            fill_value=self.config.refinement_steps,
            dtype=torch.long,
            device=grids.device,
        )
        has_halted = torch.zeros(batch_size, dtype=torch.bool, device=grids.device)

        for step in range(1, self.config.refinement_steps + 1):
            state = self.refiner(state, hidden_states[-1])
            halt_logit = self.halting_head(state).squeeze(-1)
            halt_probability = torch.sigmoid(halt_logit)
            halt_logits.append(halt_logit)
            halt_probabilities.append(halt_probability)
            hidden_states.append(state)

            new_halts = (halt_probability >= self.config.halting_threshold) & (~has_halted)
            halted_at_step = torch.where(
                new_halts, torch.full_like(halted_at_step, step), halted_at_step
            )
            has_halted = has_halted | new_halts

        # --- Action chunk prediction from final refined state ---
        chunk_flat = self.action_chunk_head(hidden_states[-1])          # (batch, C*A)
        action_chunk_logits = chunk_flat.reshape(
            batch_size, self.config.action_chunk_size, self.config.action_dim
        )

        return TRMReasonerOutput(
            action_chunk_logits=action_chunk_logits,
            halt_logits=torch.stack(halt_logits, dim=1),
            halt_probabilities=torch.stack(halt_probabilities, dim=1),
            hidden_states=hidden_states,
            halted_at_step=halted_at_step,
        )
