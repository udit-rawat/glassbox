"""Autoregressive generation, with the three knobs that shape the output distribution."""

import torch
import torch.nn.functional as F


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
) -> torch.Tensor:
    """Extend each sequence in idx by max_new_tokens, sampling one token at a time."""
    # Dropout must be off while generating, but silently leaving the model in
    # eval mode afterwards would disable it for the rest of training. The
    # previous mode is restored on the way out.
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            # Learned position embeddings only exist up to block_size, so the
            # context is cropped to the most recent window. This is also why
            # generation is quadratic today: every step re-reads the whole
            # window from scratch. Phase 2's KV cache is what fixes it.
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
