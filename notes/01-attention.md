# Attention

## What it computes

Every position emits three vectors. A **query** describing what it is looking
for, a **key** advertising what it offers, and a **value** carrying what it
actually passes on. Scoring one position's query against every key gives a
relevance row; softmaxing that row turns it into weights; the weighted sum of
values is the output. Attention is a content-addressed read from a memory whose
addresses are computed rather than fixed.

## Why the sqrt(d_k)

Each score is a dot product of `d_k` terms. With unit-normal inputs the sum has
variance `d_k`, so scores at `d_head = 64` land in the ±8 range before anything
is trained. Softmax over inputs that wide is effectively a hard max: one weight
near 1, the rest near 0, and a Jacobian near zero in every direction. Dividing
by `sqrt(d_k)` restores unit variance and keeps the distribution soft enough to
carry gradient. It is a variance correction, not a temperature knob.

## Why -inf for the mask

`masked_fill(~mask, -inf)` makes `exp(-inf) = 0` exactly. A large negative
constant like -1e9 gets close but leaves a residue on the order of 1e-9 per
masked position, and with a 128-token context and several layers those residues
are a real leak of future information. Exact zero is what allows the causality
test to assert bitwise equality rather than approximate agreement.

The consequence to watch: a row that is entirely masked softmaxes to NaN, not
an error. The causal mask includes the diagonal, so every query sees at least
itself and no row is ever empty. `test_causal_mask_always_admits_self` pins
that down, because the failure mode is silent.

## Why heads

One attention head produces one distribution per position — one relationship at
a time. Splitting `d_model` into `h` slices runs `h` of them over different
learned subspaces simultaneously, so a position can attend to its syntactic
governor in one head and the previous line break in another. The split is a
partition, not a duplication: parameter count is flat in `n_heads`, and more
heads means more, narrower subspaces.

## Why the weights come back out

`scaled_dot_product_attention` returns `(output, weights)` and every caller
propagates the second element upward. Nothing in Phase 1 reads it. Phase 5 is
the whole reason: an attention map is `(B, n_heads, T, T)`, and reconstructing
it after the fact would mean re-running the model with hooks. Returning it from
the first commit makes the visualizer a rendering problem.

They are returned **unaveraged across heads** and **before dropout**. Averaging
would collapse exactly the per-head structure worth looking at; returning the
dropped-out copy would show holes at inference time that the model does not
have.

## Verified

- Weights are non-negative and sum to 1 along the key axis.
- Masked positions receive exactly 0, not approximately 0.
- With `q = k = 0` the output equals the mean of the values — the closed-form
  uniform case.
- Rewriting a future value leaves every earlier output bitwise identical.
