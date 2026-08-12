"""Every shape the model depends on, declared in one place."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Char-level Tiny Shakespeare has a vocabulary in the mid-60s. The defaults
    # here describe a model small enough to forward on a laptop CPU in under a
    # second, which is the size the tests run against.
    vocab_size: int = 65
    block_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.0
    bias: bool = True

    # Phase 2 lands its architecture switches here — n_kv_heads for grouped
    # query attention, a norm/activation selector, a rope flag. Keeping shape
    # information in a config rather than in constructor arguments is what makes
    # that an edit to one file instead of a second model class.

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} does not divide into n_heads={self.n_heads}; "
                "heads partition the embedding, so the split must be exact"
            )

    @property
    def d_head(self) -> int:
        # Total width is partitioned across heads, not duplicated per head. Cost
        # is therefore flat in n_heads: more heads means more subspaces, each
        # narrower, for the same parameter count.
        return self.d_model // self.n_heads
