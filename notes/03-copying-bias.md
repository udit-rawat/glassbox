# An untrained model is already a copier

## What happened

The plan was to check wiring by asserting that initial cross-entropy equals
`ln(vocab_size)` — the loss of a uniform guess. Written as
`model(idx, targets=idx)`, it failed:

```
measured   3.4765
expected   4.1744  (ln 65)
```

Too low by 0.7 nats, and consistently so across seeds. An untrained model
beating chance by that margin means either the test or the model is wrong.

## Isolating it

| setup | loss |
|---|---|
| tied head, targets = input | 3.4765 |
| tied head, targets independent of input | 4.1498 |
| untied head, targets = input | 4.2158 |

Both controls land at `ln(vocab_size)`. The anomaly appears only when weight
tying and self-prediction are combined, which locates the cause precisely.

## The mechanism

Weight tying reuses the embedding matrix, transposed, as the output head. So
the logit for vocabulary entry `v` at position `t` is a dot product between the
residual stream at `t` and `emb[v]`.

The residual stream at `t` still contains `emb[tok_t]` — that is what the
embedding layer wrote into it, and nothing in an untrained block has learned to
remove it. The dot product is therefore largest when `v == tok_t`, because a
vector's dot product with itself is its squared norm.

The model predicts the token already present. It is an identity function
before it has seen a single training example.

## Consequences

**The model is fine; the test was wrong.** `test_initial_loss_matches_a_uniform_guess`
now draws targets independently of the inputs and lands at 4.21 against an
expected 4.17.

**Self-scored loss is not a wiring check.** Any diagnostic that feeds the input
in as its own target measures this artifact rather than initialization
correctness.

**Worth keeping as a behavioural test.** The copying bias is pinned by
`test_tied_weights_bias_an_untrained_model_toward_copying`. If a future change
to tying or to the initialization removes it, that should be a deliberate
decision rather than a silent drift.

**A plausible reason tying helps.** Language is heavily repetitive — recent
tokens recur. Starting at "predict the current token" is a better prior than
starting at uniform, which may be part of why tied embeddings train faster at
small scale beyond the parameter saving usually cited.
