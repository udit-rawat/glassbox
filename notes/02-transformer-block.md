# The transformer block

## The two halves

Attention moves information **between** positions but mixes values linearly.
The MLP applies nonlinear computation **at** each position independently. Both
are needed: a stack of attention with no MLP collapses toward a linear map, and
a stack of MLPs with no attention is a per-token feedforward net that never
looks at context.

The MLP widens 4x, applies GELU, and projects back. That expansion holds roughly
two thirds of the parameters in the model — most of a transformer's capacity is
in the feed-forward layers, not in attention.

## Pre-norm, and why it matters here

Normalization sits **inside** the residual branch:

```
x = x + attn(ln_1(x))
x = x + mlp(ln_2(x))
```

not after the addition, as originally published. The difference is that an
unnormalized path now runs unbroken from the embeddings to the final norm.
Gradients reach layer 0 without passing through a LayerNorm at every step, so
deep stacks train without the warmup schedule that post-norm needs to stay
stable. Every model built after roughly 2020 uses this arrangement.

## The residual stream

The identity path is best read as a channel that each block writes an increment
into, rather than a transformation applied in sequence. Two consequences fall
out of that reading:

**Initialization scales with depth.** Every block writes twice, so variance
accumulates as `O(n_layers)` at the output. Projections that write into the
stream (`out_proj`, `down_proj`) are initialized at
`0.02 / sqrt(2 * n_layers)`, which holds the output scale roughly constant
regardless of depth.

**What enters, persists.** A token's embedding is still present in the stream
at the final layer unless a block actively cancels it. That is the mechanism
behind the copying bias recorded in `03-copying-bias.md`.
