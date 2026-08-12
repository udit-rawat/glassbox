"""Scaled dot-product attention, causal masking, and the multi-head wrapper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from glassbox.model.config import GPTConfig


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Lower-triangular True/False matrix; True marks a position a query may read."""
    # Diagonal is included, so every query attends to at least itself. That
    # matters: a row that was entirely False would softmax over all -inf and
    # produce NaN rather than an error, and the NaN would only surface later.
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention as a pure function: (..., T, d_head) tensors in, (values, weights) out."""
    d_k = q.size(-1)

    # q @ k^T pairs every query with every key. Each entry of the result is a
    # dot product over d_k terms, so its variance grows with d_k when q and k
    # are roughly unit-normal. Dividing by sqrt(d_k) pulls that variance back to
    # ~1. Skipping the division leaves large-magnitude scores, softmax saturates
    # into a near one-hot, and the gradient through it goes flat.
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # -inf rather than a large negative number: exp(-inf) is exactly 0, so
        # masked positions contribute an exact zero downstream rather than a
        # small residue that accumulates across layers.
        scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax runs along the key axis, so each query row is a distribution over
    # the positions it is permitted to read. Rows summing to 1 is the invariant
    # the tests pin down.
    weights = F.softmax(scores, dim=-1)

    # Dropout applies to the weights used for the value mixture, while the
    # returned weights stay clean. The visualizer reads the returned copy, and a
    # dropout-perforated attention map would show holes that the model does not
    # have at inference time.
    mixture = dropout(weights) if dropout is not None else weights
    return mixture @ v, weights


class MultiHeadAttention(nn.Module):
    """Runs h attention heads over disjoint slices of the embedding, in parallel."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # Separate q/k/v projections rather than one fused matrix. A fused qkv
        # is marginally faster, but grouped query attention in Phase 2 gives k
        # and v a smaller output width than q, which a single fused matrix
        # cannot express without slicing tricks.
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Built once at block_size and sliced per forward. persistent=False keeps
        # it out of the state dict — it is a constant derived from config, not a
        # learned parameter, and baking it into checkpoints would freeze the
        # context length.
        self.register_buffer(
            "causal", causal_mask(config.block_size), persistent=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        # (B, T, C) -> (B, n_heads, T, d_head). The transpose puts the head axis
        # next to the batch axis so the matmuls inside attention treat heads as
        # independent batch elements; nothing is shared between them until
        # out_proj mixes them back together.
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        attended, weights = scaled_dot_product_attention(
            q, k, v, mask=self.causal[:T, :T], dropout=self.attn_dropout
        )

        # Back to (B, T, C). contiguous() is required because transpose only
        # changes the stride pattern, and view refuses a non-contiguous tensor.
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)

        # Weights come back out at (B, n_heads, T, T), unreduced across heads.
        # Averaging them here would be cheaper to pass around and would destroy
        # the per-head detail the Phase 5 visualizer exists to show.
        return self.resid_dropout(self.out_proj(attended)), weights
