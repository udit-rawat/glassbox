"""The decoder stack: embeddings in, next-token logits out."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from glassbox.model.blocks import TransformerBlock
from glassbox.model.cache import KVCache
from glassbox.model.config import GPTConfig
from glassbox.model.norm import build_norm


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Attention is permutation-invariant: shuffle the input positions and
        # the same set of outputs comes back shuffled. Order has to be injected,
        # and a learned per-position vector is the cheapest way to do it. RoPE
        # injects it inside attention instead, rotating queries and keys, so
        # this table is not built at all in that configuration.
        self.position_embedding = (
            nn.Embedding(config.block_size, config.d_model)
            if config.pos_encoding == "learned"
            else None
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.ln_f = build_norm(config, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: the matrix that maps a token id to a vector is reused,
        # transposed, to score vectors against tokens. Both directions describe
        # the same token-vector relationship, and at this vocabulary size the
        # embedding is a large fraction of the parameter budget.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Residual streams accumulate one contribution per branch, so variance
        # at the output grows with depth. Shrinking the projections that write
        # into the stream by 1/sqrt(2 * n_layers) — two writers per block —
        # holds the scale roughly constant however deep the stack goes.
        residual_writers = ("out_proj.weight", "down_proj.weight")
        for name, param in self.named_parameters():
            if name.endswith(residual_writers):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # std=0.02 is the GPT-2 value, small enough that initial logits sit near
        # zero and the starting loss lands at ln(vocab_size) — the loss of a
        # uniform guess, and the cheapest end-to-end wiring check available.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
        return_hidden: bool = False,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """(B, T) token ids -> (B, T, vocab_size) logits, optional loss, optional attention."""
        B, T = idx.shape

        # With a cache, idx holds only the new tokens; everything before them is
        # already stored. Position therefore continues from the cache rather
        # than restarting at zero.
        start = cache.length if cache is not None else 0
        if start + T > self.config.block_size:
            raise ValueError(
                f"position {start + T} exceeds block_size {self.config.block_size}; "
                "position information is only defined up to block_size"
            )

        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            pos = torch.arange(start, start + T, device=idx.device)
            x = x + self.position_embedding(pos)
        x = self.dropout(x)

        # Attention maps are collected per layer only when asked for. Holding
        # every layer's (B, n_heads, T, T) tensor is quadratic in sequence
        # length, which is affordable for a visualizer call and wasteful for
        # every step of a training run.
        attentions: list[torch.Tensor] | None = [] if return_attention else None

        # The residual stream after every block, plus the embedding that starts
        # it. Collected only on request: holding n_layers copies of (B, T, C) is
        # wasted memory on every training step, and the visualizer is the only
        # thing that reads them.
        self._hidden = [x] if return_hidden else None

        for i, block in enumerate(self.blocks):
            x, weights = block(x, cache=cache[i] if cache is not None else None)
            if attentions is not None:
                attentions.append(weights)
            if self._hidden is not None:
                self._hidden.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flattened to (B*T, vocab) against (B*T,): position t predicts the
            # token at t+1, and the shift lives in how the batch is cut, not
            # here. Every position contributes a prediction, which is what makes
            # a decoder-only model train on T targets per sequence instead of 1.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss, attentions

    def hidden_states(self) -> list[torch.Tensor]:
        """The residual stream captured by the last forward(return_hidden=True)."""
        if getattr(self, "_hidden", None) is None:
            raise RuntimeError(
                "no hidden states recorded; call forward(..., return_hidden=True) first"
            )
        return self._hidden

    @torch.no_grad()
    def logit_lens(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """What the model would predict if it stopped at each layer.

        The residual stream is a channel each block writes an increment into, and
        the output head reads that channel. So the head can be pointed at it
        early — normalize the stream as it stands after layer L, score it against
        the vocabulary, and read off the prediction the model has formed so far.
        Run over every depth it shows a guess sharpening as the layers refine it,
        which is the clearest picture of what the extra depth is buying.

        Returns n_layers + 1 logit tensors: the bare embedding first, then one
        after each block.
        """
        was_training = self.training
        self.eval()
        try:
            self(idx, return_hidden=True)
            return [self.lm_head(self.ln_f(h)) for h in self.hidden_states()]
        finally:
            self.train(was_training)

    def num_parameters(self, include_embeddings: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not include_embeddings:
            # lm_head is tied to token_embedding and so is not double counted by
            # parameters(); only the position table is a separate tensor, and
            # under RoPE it does not exist at all.
            if self.position_embedding is not None:
                total -= self.position_embedding.weight.numel()
            total -= self.token_embedding.weight.numel()
        return total
