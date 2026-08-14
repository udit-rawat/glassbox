"""Every shape and architecture switch the model depends on, declared in one place."""

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

    # Phase 2 architecture switches. Defaults reproduce the Phase 1 model
    # exactly, so every test written before today still describes a real
    # configuration and the two architectures stay runnable side by side.
    norm: str = "layernorm"          # layernorm | rmsnorm
    activation: str = "gelu"         # gelu | swiglu
    pos_encoding: str = "learned"    # learned | rope
    n_kv_heads: int | None = None    # None means one key/value head per query head
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} does not divide into n_heads={self.n_heads}; "
                "heads partition the embedding, so the split must be exact"
            )
        if self.norm not in ("layernorm", "rmsnorm"):
            raise ValueError(f"unknown norm {self.norm!r}")
        if self.activation not in ("gelu", "swiglu"):
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.pos_encoding not in ("learned", "rope"):
            raise ValueError(f"unknown pos_encoding {self.pos_encoding!r}")

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads={self.n_kv_heads} exceeds n_heads={self.n_heads}; "
                "grouped query attention shares key/value heads across query heads, "
                "so there can never be more of them"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} is not a multiple of n_kv_heads="
                f"{self.n_kv_heads}; query heads are assigned to key/value heads in "
                "equal groups, so the division must be exact"
            )

        # RoPE rotates pairs of adjacent dimensions, so an odd head width has a
        # component with no partner to rotate against.
        if self.pos_encoding == "rope" and self.d_head % 2 != 0:
            raise ValueError(
                f"rope needs an even d_head, got {self.d_head}"
            )

    @property
    def d_head(self) -> int:
        # Total width is partitioned across heads, not duplicated per head. Cost
        # is therefore flat in n_heads: more heads means more subspaces, each
        # narrower, for the same parameter count.
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """How many query heads share each key/value head."""
        return self.n_heads // self.n_kv_heads
