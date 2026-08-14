"""The transformer block: normalization, attention, feed-forward, residual connections."""

import torch
import torch.nn as nn

from glassbox.model.attention import MultiHeadAttention
from glassbox.model.config import GPTConfig
from glassbox.model.feedforward import MLP, SwiGLU, build_feedforward
from glassbox.model.norm import build_norm

__all__ = ["MLP", "SwiGLU", "TransformerBlock"]


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward, each entered through a residual connection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = build_norm(config, config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = build_norm(config, config.d_model)
        self.mlp = build_feedforward(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalization sits inside the residual branch (pre-norm), not after
        # the addition (post-norm as originally published). The consequence is
        # an unnormalized path running from embeddings to output: gradients
        # reach early layers without passing through a norm at every step, and
        # deep stacks train without a warmup schedule to stay stable.
        attended, weights = self.attn(self.ln_1(x))
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x, weights
