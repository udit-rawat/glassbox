"""Autoregressive generation, with the three knobs that shape the output distribution."""

import torch
import torch.nn.functional as F

from glassbox.model.cache import KVCache


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest-scoring tokens, discard the rest."""
    k = min(k, logits.size(-1))
    # The k-th largest value becomes the threshold. Anything below it is set to
    # -inf, which softmax turns into exactly zero probability.
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probabilities sum to at least p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)

    # Subtracting the token's own probability before comparing keeps the first
    # token that crosses the threshold. Comparing the raw cumulative sum would
    # discard everything when one token already exceeds p on its own, leaving
    # nothing to sample from.
    remove_sorted = (cumulative - probs) > p

    # The mask was built in sorted order; scatter puts each flag back onto the
    # vocabulary position it came from before it is applied to the real logits.
    remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Extend each sequence in idx by max_new_tokens, sampling one token at a time."""
    # Dropout must be off while generating, but silently leaving the model in
    # eval mode afterwards would disable it for the rest of training. The
    # previous mode is restored on the way out.
    was_training = model.training
    model.eval()

    cache = KVCache(model.config.n_layers) if use_cache else None

    try:
        for _ in range(max_new_tokens):
            if cache is not None and idx.size(1) < model.config.block_size:
                # First pass feeds the whole prompt and fills the cache; every
                # pass after feeds a single token, because the keys and values
                # for everything before it are already stored.
                idx_cond = idx if cache.length == 0 else idx[:, -1:]
                logits, _, _ = model(idx_cond, cache=cache)
            else:
                # Past the context window the cache cannot help. The window has
                # to slide, and sliding means renumbering positions — but RoPE
                # baked absolute positions into the stored keys when they were
                # written, so they cannot be reinterpreted at a different offset.
                # Falling back to the cropped full pass keeps the output
                # identical to uncached generation, which matters more than the
                # speed of the tail.
                if cache is not None:
                    cache.reset()
                idx_cond = idx[:, -model.config.block_size :]
                logits, _, _ = model(idx_cond)
            # Only the final position predicts the next token; the rest were
            # already used during training and are discarded here.
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # Not a limit that can be taken numerically — dividing by zero
                # gives NaN — so greedy decoding is handled as its own branch.
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature rescales the gap between logits before softmax.
                # Below 1 the gaps widen and sampling concentrates on the top
                # candidates; above 1 they narrow and the distribution flattens.
                logits = logits / temperature

                if top_k is not None:
                    logits = _top_k_filter(logits, top_k)
                if top_p is not None:
                    logits = _top_p_filter(logits, top_p)

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
    finally:
        model.train(was_training)

    return idx
