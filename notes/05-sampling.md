# Sampling

The model outputs a score for every character at every position. Turning that
into text is a separate decision from the model itself, and it changes the
output more than most people expect.

## Greedy

Always take the highest-scoring token. Deterministic, and reliably bad over any
length: it falls into loops, because once a repeated phrase becomes locally
most likely there is nothing to break out of it.

Handled as its own branch rather than as `temperature = 0`, because dividing
the logits by zero produces NaN rather than approaching argmax in the limit.

## Temperature

Divide the logits before softmax. Below 1 the gaps between scores widen and
sampling concentrates on the top candidates; above 1 the gaps narrow and the
distribution flattens toward uniform.

It is a single knob on the whole distribution, which is also its weakness: it
cannot distinguish "the top few are plausible and the rest are garbage" from
"everything is roughly equally plausible". Both get rescaled identically.

## Top-k

Keep the `k` highest-scoring tokens, set everything else to `-inf`, renormalize.

Fixes the failure temperature cannot: with `k = 40` the long tail of characters
that are individually unlikely but collectively significant can never be drawn.
The weakness is that `k` is fixed regardless of how confident the model is —
the same 40 candidates whether one token holds 99% of the mass or the top forty
are evenly matched.

## Top-p (nucleus)

Keep the smallest set of tokens whose probabilities sum to at least `p`. The
size of that set adapts: when the model is confident it may contain one token,
when uncertain it may contain thirty.

Two implementation details, both tested:

**Keep the token that crosses the threshold.** The comparison is
`(cumulative - own_probability) > p`, not `cumulative > p`. With the naive
version, a single token holding 95% of the mass exceeds any `p` below 0.95 on
its own and gets masked along with everything else, leaving nothing to sample
from.

**Scatter the mask back before applying it.** The filter sorts to compute
cumulative probabilities, so the mask is in sorted order and must be returned
to vocabulary order before it touches the real logits. Skipping the scatter
masks the wrong tokens while producing output that still looks superficially
plausible — the kind of bug that survives a long time.

## Context cropping

Learned position embeddings exist only up to `block_size`, so the context is
cropped to the most recent window at every step. This is also why generation is
quadratic today: each new token re-reads the entire window from scratch. Phase
2's KV cache is what removes that.

## Restoring training mode

`generate` records `model.training`, switches to eval, and restores the previous
mode in a `finally` block. Sampling mid-training and leaving the model in eval
mode would silently disable dropout for every remaining step — no error, no
crash, just a run that quietly stops regularizing.
