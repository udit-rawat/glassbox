"""Scaled dot-product attention, causal masking, and the multi-head wrapper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from glassbox.model.config import GPTConfig
from glassbox.model.rope import apply_rope, build_rope_cache


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


def repeat_kv(x: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Expand n_kv_heads up to n_heads by repeating each key/value head in place."""
    if n_groups == 1:
        return x
    # repeat_interleave rather than repeat: head i must land on the contiguous
    # block of query heads assigned to it. Tiling the whole tensor instead would
    # pair query heads with the wrong key/value heads while keeping every shape
    # correct, which is exactly the kind of bug that survives a test suite that
    # only checks shapes.
    return x.repeat_interleave(n_groups, dim=1)


class MultiHeadAttention(nn.Module):
    """Runs h attention heads over disjoint slices of the embedding, in parallel."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_head = config.d_head
        self.use_rope = config.pos_encoding == "rope"

        # Separate q/k/v projections rather than one fused matrix. Under grouped
        # query attention k and v produce fewer heads than q, so their output
        # width genuinely differs and a single fused matrix could not express it.
        self.q_proj = nn.Linear(
            config.d_model, config.n_heads * config.d_head, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.d_head, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.d_model, config.n_kv_heads * config.d_head, bias=config.bias
        )
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

        if self.use_rope:
            cos, sin = build_rope_cache(
                config.block_size, config.d_head, config.rope_theta
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        # (B, T, n*d_head) -> (B, n, T, d_head). The transpose puts the head axis
        # next to the batch axis so the matmuls inside attention treat heads as
        # independent batch elements; nothing is shared between them until
        # out_proj mixes them back together.
        def split(t: torch.Tensor, n: int) -> torch.Tensor:
            return t.view(B, T, n, self.d_head).transpose(1, 2)

        q = split(self.q_proj(x), self.n_heads)
        k = split(self.k_proj(x), self.n_kv_heads)
        v = split(self.v_proj(x), self.n_kv_heads)

        if self.use_rope:
            # Rotation is applied to queries and keys only, never to values.
            # Position enters through which positions match each other, not
            # through the content that gets passed along once they do.
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Grouped query attention stores fewer key/value heads than query heads
        # and expands them here. The saving is real at generation time, where
        # the KV cache shrinks by this factor; during training it is a wash.
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

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
