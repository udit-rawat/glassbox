"""The feed-forward network and the transformer block that wraps it around attention."""

import torch
import torch.nn as nn

from glassbox.model.attention import MultiHeadAttention
from glassbox.model.config import GPTConfig


class MLP(nn.Module):
    """Position-wise feed-forward: widen 4x, apply a nonlinearity, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # The 4x expansion is inherited from the original transformer and has
        # held across scales. Attention moves information between positions but
        # is linear in the values it mixes; this block is where per-position
        # nonlinear computation happens, and it holds roughly two thirds of the
        # parameters in the model.
        hidden = 4 * config.d_model
        self.up_proj = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act(self.up_proj(x))))


class TransformerBlock(nn.Module):
    """Pre-norm attention and MLP, each entered through a residual connection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalization sits inside the residual branch (pre-norm), not after
        # the addition (post-norm as originally published). The consequence is
        # an unnormalized path running from embeddings to output: gradients
        # reach early layers without passing through a LayerNorm at every step,
        # and deep stacks train without a warmup schedule to stay stable.
        attended, weights = self.attn(self.ln_1(x))
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x, weights
