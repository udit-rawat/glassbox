"""Derive a diagram of the model from its config.

The diagram is generated, never drawn. Every shape on every edge and every
parameter count comes from the same GPTConfig the model is built from, so the
picture cannot drift out of date with the code — and flipping a switch changes
the numbers because it would really change the model.

Nothing here imports torch or touches a checkpoint. Any config can be described,
including ones that were never trained, which is what lets the toggles respond
instantly and lets the page work with no model present at all.
"""

from dataclasses import dataclass, field

from glassbox.model.config import GPTConfig

# Colour is information, not decoration: one hue per kind of component, and the
# attention heatmaps downstream reuse the attention hue so the grid needs no
# legend. Norms are deliberately uncoloured — they are plumbing.
KINDS = ("embedding", "norm", "attention", "feedforward", "residual", "output")


@dataclass
class Block:
    """One node of the diagram."""

    id: str
    label: str
    kind: str
    detail: str = ""
    out_shape: tuple = ()
    params: int = 0
    source: str = ""
    # Blocks that vanish under a different configuration stay in the list with
    # present=False rather than being omitted, so the frontend can animate one
    # leaving instead of the whole diagram jumping.
    present: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "detail": self.detail,
            "out_shape": list(self.out_shape),
            "params": self.params,
            "source": self.source,
            "present": self.present,
            "note": self.note,
        }


@dataclass
class Diagram:
    config: GPTConfig
    blocks: list[Block] = field(default_factory=list)

    def as_dict(self) -> dict:
        present = [b for b in self.blocks if b.present]
        per_kind: dict[str, int] = {}
        for b in present:
            per_kind[b.kind] = per_kind.get(b.kind, 0) + b.params

        return {
            "config": {
                "vocab_size": self.config.vocab_size,
                "block_size": self.config.block_size,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "n_kv_heads": self.config.n_kv_heads,
                "d_head": self.config.d_head,
                "norm": self.config.norm,
                "activation": self.config.activation,
                "pos_encoding": self.config.pos_encoding,
                "bias": self.config.bias,
            },
            "blocks": [b.as_dict() for b in self.blocks],
            "totals": {
                "parameters": self.total_parameters,
                "by_kind": per_kind,
                # Per-block, so the "x6" badge in the diagram has a number.
                "per_layer": self.layer_parameters,
                "n_layers": self.config.n_layers,
            },
        }

    @property
    def layer_parameters(self) -> int:
        return sum(b.params for b in self.blocks if b.present and b.id.startswith("block."))

    @property
    def total_parameters(self) -> int:
        outside = sum(
            b.params for b in self.blocks if b.present and not b.id.startswith("block.")
        )
        return outside + self.layer_parameters * self.config.n_layers


def _norm_params(config: GPTConfig) -> int:
    # RMSNorm carries a gain and nothing else; LayerNorm adds a shift when
    # biases are on. The difference is two numbers per layer and it is the
    # smallest of the four switches by parameter count — which is worth seeing.
    if config.norm == "rmsnorm":
        return config.d_model
    return 2 * config.d_model if config.bias else config.d_model


def _linear(in_features: int, out_features: int, bias: bool) -> int:
    return in_features * out_features + (out_features if bias else 0)


def _ffn_hidden(config: GPTConfig) -> int:
    if config.activation == "swiglu":
        hidden = int(8 * config.d_model / 3)
        return 64 * ((hidden + 63) // 64)
    return 4 * config.d_model


def build_diagram(config: GPTConfig) -> Diagram:
    """Walk a config and describe every stage of its forward pass."""
    c = config
    d, B, T = c.d_model, "B", "T"
    q_width = c.n_heads * c.d_head
    kv_width = c.n_kv_heads * c.d_head
    hidden = _ffn_hidden(c)
    norm_label = "RMSNorm" if c.norm == "rmsnorm" else "LayerNorm"
    blocks: list[Block] = []

    blocks.append(Block(
        "tokens", "Token ids", "embedding",
        detail="Integers, one per character or BPE token.",
        out_shape=(B, T), source="glassbox/tokenizer/",
    ))
    blocks.append(Block(
        "token_embedding", "Token embedding", "embedding",
        detail=f"Lookup table, {c.vocab_size} rows of {d}. Learned, and reused "
               "transposed as the output head.",
        out_shape=(B, T, d), params=c.vocab_size * d,
        source="glassbox/model/gpt.py",
    ))
    blocks.append(Block(
        "position_embedding", "Position embedding", "embedding",
        detail=f"One learned vector per slot, {c.block_size} of them. Added to "
               "the token vector, and the reason context is capped.",
        out_shape=(B, T, d), params=c.block_size * d,
        present=c.pos_encoding == "learned",
        source="glassbox/model/gpt.py",
        note="Replaced by rotation inside attention when RoPE is on.",
    ))
    blocks.append(Block(
        "residual_start", "Residual stream", "residual",
        detail="The channel every block writes an increment into. Nothing "
               "replaces it; each layer only adds.",
        out_shape=(B, T, d), source="glassbox/model/gpt.py",
    ))

    blocks.append(Block(
        "block.ln_1", f"{norm_label}", "norm",
        detail=("Divides by the root mean square. No mean subtraction and no "
                "learned shift." if c.norm == "rmsnorm" else
                "Subtracts the mean, divides by the spread, then applies a "
                "learned gain and shift."),
        out_shape=(B, T, d), params=_norm_params(c),
        source="glassbox/model/norm.py",
    ))
    blocks.append(Block(
        "block.attn.q_proj", "Query projection", "attention",
        detail=f"{c.n_heads} heads of {c.d_head}.",
        out_shape=(B, c.n_heads, T, c.d_head), params=_linear(d, q_width, c.bias),
        source="glassbox/model/attention.py",
    ))
    blocks.append(Block(
        "block.attn.kv_proj", "Key and value projections", "attention",
        detail=(f"{c.n_kv_heads} head{'s' if c.n_kv_heads != 1 else ''} each, shared "
                f"across {c.n_kv_groups} query head{'s' if c.n_kv_groups != 1 else ''}."
                if c.n_kv_heads != c.n_heads else
                f"{c.n_kv_heads} heads each, one per query head."),
        out_shape=(B, c.n_kv_heads, T, c.d_head),
        params=2 * _linear(d, kv_width, c.bias),
        source="glassbox/model/attention.py",
        note=("Narrower than the query side — this is the tensor the KV cache "
              "stores, so the saving lands at generation time."
              if c.n_kv_heads != c.n_heads else ""),
    ))
    blocks.append(Block(
        "block.attn.rope", "Rotary embedding", "attention",
        detail="Rotates queries and keys by an angle set by position. Never "
               "applied to values: position decides what matches, not what is "
               "passed along.",
        out_shape=(B, c.n_heads, T, c.d_head),
        present=c.pos_encoding == "rope",
        source="glassbox/model/rope.py",
        note="Scores then depend only on the gap between two positions.",
    ))
    blocks.append(Block(
        "block.attn.scores", "Scores, mask, softmax", "attention",
        detail=f"Every query against every key, scaled by 1/sqrt({c.d_head}), "
               "future positions set to -inf, then softmaxed into a "
               "distribution over readable positions.",
        out_shape=(B, c.n_heads, T, T),
        source="glassbox/model/attention.py",
        note="This tensor is what the attention grid renders.",
    ))
    blocks.append(Block(
        "block.attn.out_proj", "Output projection", "attention",
        detail="Recombines the heads, which until now never saw each other.",
        out_shape=(B, T, d), params=_linear(d, d, c.bias),
        source="glassbox/model/attention.py",
    ))
    blocks.append(Block(
        "block.residual_1", "Residual add", "residual",
        detail="Write one of two per block.",
        out_shape=(B, T, d), source="glassbox/model/blocks.py",
    ))

    blocks.append(Block(
        "block.ln_2", f"{norm_label}", "norm",
        detail="Same normalization, before the feed-forward branch.",
        out_shape=(B, T, d), params=_norm_params(c),
        source="glassbox/model/norm.py",
    ))
    if c.activation == "swiglu":
        blocks.append(Block(
            "block.ffn", f"SwiGLU  {d} → {hidden} → {d}", "feedforward",
            detail=f"Two branches to {hidden}: one squashed by SiLU gates the "
                   "other, then a projection back. The gate makes the "
                   "nonlinearity depend on the input.",
            out_shape=(B, T, d),
            params=2 * _linear(d, hidden, c.bias) + _linear(hidden, d, c.bias),
            source="glassbox/model/feedforward.py",
            note=f"Three matrices, so the width is 8/3 rather than 4x — which "
                 f"keeps the parameter count level with GELU.",
        ))
    else:
        blocks.append(Block(
            "block.ffn", f"GELU MLP  {d} → {hidden} → {d}", "feedforward",
            detail=f"Widen to {hidden}, apply one fixed curve to every unit, "
                   "project back. Where per-position computation happens.",
            out_shape=(B, T, d),
            params=_linear(d, hidden, c.bias) + _linear(hidden, d, c.bias),
            source="glassbox/model/feedforward.py",
        ))
    blocks.append(Block(
        "block.residual_2", "Residual add", "residual",
        detail="Write two of two. Variance grows with depth, which is why the "
               "projections writing here start smaller in deeper models.",
        out_shape=(B, T, d), source="glassbox/model/blocks.py",
    ))

    blocks.append(Block(
        "ln_f", f"Final {norm_label}", "norm",
        detail="One last normalization before the stream is scored.",
        out_shape=(B, T, d), params=_norm_params(c),
        source="glassbox/model/gpt.py",
    ))
    blocks.append(Block(
        "lm_head", "Output head (tied)", "output",
        detail=f"Scores the stream against all {c.vocab_size} tokens using the "
               "embedding matrix transposed. Free in parameters, and the reason "
               "an untrained model predicts the token already present.",
        out_shape=(B, T, c.vocab_size), params=0,
        source="glassbox/model/gpt.py",
        note="Weights shared with the token embedding.",
    ))
    blocks.append(Block(
        "logits", "Logits", "output",
        detail="One score per token at every position.",
        out_shape=(B, T, c.vocab_size), source="glassbox/model/gpt.py",
    ))

    return Diagram(config=c, blocks=blocks)
