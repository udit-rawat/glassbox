# Tokenization and batching

## Why character level first

The vocabulary is whatever distinct characters appear in the corpus — 65 for
Tiny Shakespeare. That is the entire tokenizer: a sorted list, a dict from
character to id, and its inverse.

The cost is that context is expensive. A 128-token window is roughly one
sentence, and the model spends capacity learning that `t` follows `h` inside
`the` rather than learning anything about language. The benefit is that there
is nothing between the raw text and the model to be wrong about, and a model
this small produces visible results in minutes. Phase 2's BPE replaces it.

## Sorting is load-bearing

`from_text` builds the vocabulary from `set(text)`, which has no defined order.
Without the sort in `__init__`, two runs over the same corpus could assign
different ids to the same characters, and a checkpoint trained under one
ordering would decode into noise under the other. It would not crash — the
shapes all still match — which is what makes it worth a test.

## No unknown token

`encode` raises `KeyError` on an unseen character rather than substituting a
placeholder. The vocabulary is derived from the corpus being trained on, so
every character is known by construction. An unseen one means the wrong
tokenizer was paired with the text, and that should surface immediately rather
than becoming a silent stream of placeholder ids.

## The split is contiguous, not shuffled

Validation is the last 10% of the corpus, taken as one unbroken block. The
tempting alternative — shuffle characters, then split — puts fragments of the
same speech on both sides, so the model has effectively seen the validation set
during training and the reported number means nothing.

## Batching: random windows, no epochs

The corpus is one long id tensor. A batch is `batch_size` windows cut from
uniformly random offsets:

```
ix = torch.randint(len(data) - block_size, (batch_size,))
x  = data[i : i + block_size]
y  = data[i + 1 : i + 1 + block_size]
```

There is no epoch and no shuffling machinery. Independent random draws make
consecutive batches uncorrelated by construction, and `len(data) - block_size`
as the upper bound guarantees every window plus its one-position lookahead
stays inside the split.

## The shift is the whole supervision signal

`y` is `x` moved one position later. Position `t` of `x` is asked to predict
position `t` of `y`, which is character `t+1` of the corpus.

That single offset is the entire training objective, and it lives in exactly
one place in the codebase. Combined with the causal mask, it means one forward
pass over a 128-character window yields 128 separate supervised predictions —
the efficiency that makes this style of training viable.

`test_targets_are_inputs_shifted_by_one` pins it down by asserting
`x[:, 1:] == y[:, :-1]`: everything in `y` except its last column is already
present in `x`, so the model is only ever asked for the one character it has
not been shown.
