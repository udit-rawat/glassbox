"""Key/value cache for incremental decoding.

Without it, generating token t re-runs attention over the whole prefix: every
key and value from positions 0..t-1 is recomputed despite being identical to
last step. That makes a T-token sample cost O(T^2) forward passes worth of work.

The cache keeps what was already computed, so each new token contributes one
key and one value and the rest are read back. Generation becomes linear.
"""

import torch


class LayerCache:
    """Accumulated keys and values for one attention layer."""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """How many positions are stored."""
        return 0 if self.k is None else self.k.size(2)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add this step's keys and values, and return everything so far."""
        # Concatenated along the position axis of (B, n_kv_heads, T, d_head).
        # This runs *after* rotary embedding and *before* the key/value heads are
        # expanded to match the query heads: what is stored is the narrow,
        # already-rotated tensor. Storing pre-rotation would mean re-rotating the
        # whole history every step, and storing post-expansion would multiply the
        # memory by the grouping factor and throw away what GQA is for.
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v


class KVCache:
    """One LayerCache per transformer block."""

    def __init__(self, n_layers: int):
        self.layers = [LayerCache() for _ in range(n_layers)]

    def __getitem__(self, i: int) -> LayerCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def length(self) -> int:
        """Positions cached so far. Every layer holds the same number."""
        return self.layers[0].length

    def reset(self) -> None:
        for layer in self.layers:
            layer.k = layer.v = None
