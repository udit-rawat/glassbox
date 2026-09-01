# glassbox

**A transformer built from first principles, to understand it.**

Attention, normalisation, positional encoding, the tokeniser, the training
loop, the sampler, the KV cache — written from scratch in pure PyTorch so that
every line can be explained, tested and watched while it runs. Then an
interactive visualiser that shows what each attention head is actually doing.

**[Open the live demo →](https://udit-rawat.github.io/glassbox/)**

![The attention grid](docs/images/activations.png)

*Thirty-six attention heads from a model trained from scratch, each one a real
map of where it looked. Sorted by behaviour, labelled by what they do.*

## What this is, and what it isn't

**It is a rebuild, not a scale result.** The models are small on purpose: 2.7M
parameters on Tiny Shakespeare, 11M on TinyStories, 49 minutes on a free T4.
The architecture is Llama-*shaped* — RMSNorm, RoPE, SwiGLU, grouped-query
attention — but nowhere near Llama-*sized*, and nothing here should be read as
a competitive number. The goal was to own every line, not to win on scale.

**The dataset path is the well-trodden one.** Tiny Shakespeare then TinyStories
is the standard route for this genre of project, close to the nanoGPT path. The
problem selection here is not original; the execution is where the work went —
the controlled ablation, 194 behavioural tests, and a diagram generated from
the config rather than drawn.

**Half the commercially useful work is still missing.** Phase 4 — LoRA and
adapting a real pretrained model — is not built yet. Training a small model
from scratch demonstrates that the internals are understood. Adapting a large
existing one is closer to what the work actually looks like in practice, and
that half is open.

**No HuggingFace, until Phase 4.** That is a learning constraint, not a claim
about good practice. Nobody should write their own BPE tokeniser at work. The
point was to be unable to hide behind an abstraction while learning what it
does; Phase 4 is where the real libraries come in.

## Progress

- [x] **Phase 1 — Attention & the Transformer.** Scaled dot-product
  attention, multi-head attention, causal masking, a full GPT-style
  decoder trained char-level on Tiny Shakespeare.
  *2.7M parameters, validation loss 4.17 → 1.5039, about 4 minutes on a
  free T4.*
- [x] **Phase 2 — Modern LLM internals.** RMSNorm, RoPE, SwiGLU, grouped
  query attention, a KV cache, temperature/top-k/top-p sampling, and a BPE
  tokenizer from scratch.
  *Same data, same seed, one variable at a time: 1.4700 against Phase 1's
  1.5039, with 12% fewer parameters. See the ablation below.*
- [x] **Phase 3 — A real training run.** TinyStories with mixed precision,
  gradient accumulation, cosine LR schedule, and crash-safe checkpointing.
  *11M parameters, 133M tokens, validation loss 1.5787 — perplexity 4.8,
  against 39.5 for a bigram on the same data. 49 minutes on a free T4.*
- [ ] **Phase 4 — Fine-tuning a real model.** LoRA implemented from
  scratch (~80 lines), then instruction-tuning a real pretrained model.
  *Not started. The most directly applicable phase, and the one still open.*
- [x] **Phase 5 — The visualizer.** Enter a prompt, watch attention heads
  light up per layer, inspect token-by-token next-token probabilities.
  *Plus an architecture diagram generated from the config, with the Phase 2
  switches as live toggles. Arbitrary prompts need the local server; the
  hosted demo answers from recordings.*

## What each modern component is actually worth

Six runs on Tiny Shakespeare, identical in every respect except one switch —
same seed, same data, same iterations, float32 throughout so that differences
of a few hundredths are not fp16 rounding. Biases stay on everywhere, including
the combined run, so exactly four variables move.

| variant | val loss | vs baseline | parameters |
|---|---|---|---|
| baseline — LayerNorm, GELU, learned, 6 kv | 1.5039 | — | 2,706,624 |
| RMSNorm only | 1.5004 | −0.0035 | 2,704,128 |
| SwiGLU only | 1.5085 | **+0.0045** | 2,708,160 |
| RoPE only | 1.4749 | **−0.0290** | 2,682,048 |
| 2 key/value heads only | 1.5055 | +0.0016 | 2,410,176 |
| all four | **1.4700** | **−0.0339** | **2,384,640** |

**RoPE does almost all of the work** — 0.0290 of a 0.0339 total. **SwiGLU alone
is slightly worse** at this scale, which is a real negative result and stays in.
**Grouped-query attention is free**: 0.0016 of loss, which is noise, for 11%
fewer parameters. And the combination beats the sum of the parts by about
0.008, so the switches are not independent.

At this size these are small differences on a small corpus, and none of it
transfers automatically to a model a thousand times larger. What it does show is
the discipline: one variable at a time, held-out loss, negative results kept.

```bash
# notebooks/ablation_colab.ipynb — six runs, about 25 minutes on a free T4
```

## Reading the numbers

A validation loss means nothing on its own. `scripts/baselines.py` measures what
the same data costs under models that require no learning at all, using the same
BPE tokeniser and the same held-out split, so the trained model can be read
against something rather than against nothing.

| | loss | perplexity |
|---|---|---|
| uniform over the 4,096-token vocabulary | 8.3178 | 4096.0 |
| unigram — token frequency alone | 6.0260 | 414.0 |
| bigram — the previous token only | 3.6774 | 39.5 |
| **glassbox, 11M parameters** | **1.5787** | **4.8** |

So the model is **8.2× lower perplexity than a bigram**, which is the honest
statement of what it learned. What it is *not* is calibrated against published
TinyStories models of comparable size — that comparison would be the next
useful anchor and has not been done.

```bash
python scripts/baselines.py
```

## How the training actually runs

Nothing trains locally. Every run in this repo happened on a free Colab T4
through the notebooks in `notebooks/`, which clone the repo, install it, mount
Drive for checkpoints, and call the same scripts you would run by hand:

| notebook | what it did |
|---|---|
| `train_colab.ipynb` | reproduced Phase 1 on Tiny Shakespeare |
| `ablation_colab.ipynb` | the six-run ablation above |
| `tinystories_colab.ipynb` | Phase 3, 20,000 iterations in 49 minutes |

They are committed with their outputs, so the loss curves are readable without
running anything. The scripts work locally too — the test suite runs on CPU in
five seconds — but a laptop is for building and debugging here, not training.

## Phase 5 — the visualizer

A single page with three views, no framework and no build step. Run it against
your own checkpoints, or open the [hosted
demo](https://udit-rawat.github.io/glassbox/).

```bash
python -m glassbox.viz.server --checkpoints checkpoints/ablation
```

### Architecture — generated, never drawn

![The architecture view](docs/images/architecture.png)

Every tensor shape and every parameter count comes from the same `GPTConfig`
the model is built from, so the diagram cannot drift out of step with the code.
A test asserts the totals equal `GPT.num_parameters()` across eight
configurations, and they match all seven trained checkpoints exactly.

The four Phase 2 switches are live. Flip **RoPE off** and the position table
reappears in the diagram and the total climbs by exactly `block_size × d_model`.
Open any stage to branch its arithmetic out sideways, or press **walkthrough**
to have the whole model unfold one stage at a time and fold back up.

### Activations — one prompt, one forward pass

Thirty-six attention heads as heatmaps, sortable by **mean attention distance,
previous-token score or entropy**, so heads of a kind cluster instead of
scattering by index. Click one for the enlarged map with token labels; hovering
reports the actual weight, and above the diagonal it tells you the query cannot
see that key because it comes later.

The **logit lens** reads the model's own output head against every layer's
residual stream, so you watch a prediction sharpen with depth. On the trained
model, prompted with `ROMEO:\nWhat is th`:

```
embedding  'h' 1.000     <- the copying bias: the token already present
layer 1    'e' 0.510     <- one block of attention is enough to see "th"
layer 2    'e' 0.534
layer 3    'e' 0.638
layer 4    'e' 0.733
layer 5    'e' 0.778
layer 6    'e' 0.746
```

That first row is the finding from day one, visible: weight tying plus the
residual stream make an untrained model a copier, and it shows up here as the
embedding predicting the character that is already there.

### Generate — the ablation, side by side

![The generate view](docs/images/generate.png)

The same prompt and the same seed through any of the six ablation checkpoints,
with each one's validation loss and parameter count beside the output. Switching
from `baseline` to `rope` is a real before-and-after of one architecture change.

### Hosting

`scripts/export_static.py` bakes the page into one self-contained HTML file.
The architecture view stays fully interactive there — all 32 switch
combinations are precomputed, because describing a configuration is arithmetic
rather than inference. Activations and generations are **recordings of real
forward passes**, since a static host has no Python behind it, and the page
says so rather than pretending otherwise.

## Phase 3 result — it writes English

An 11M-parameter model, trained from scratch on 133 million tokens of
TinyStories with a from-scratch BPE tokenizer, in 49 minutes on a free T4.
Validation loss 1.5787, which is a perplexity of 4.8 against a vocabulary of
4,096 — the model has narrowed four thousand options down to about five.

Prompted with `The dog was very sad because`, sampled at temperature 0.8 and
top-p 0.9:

```
The dog was very sad because he could not be friends with a cat. The cat
wanted to be friends with the dog, but the dog was too big and strong. So,
the cat tried to play with the dog. But the dog was too big and the cat
could not lift it.

The cat found a big tree to sit under. The cat was very happy. The cat and
the dog became best friends. They played and laughed together every day.
```

And from `Once upon a time, there was a little`, it produced a complete story
with two named characters, dialogue, an argument over a toy box, a consequence,
and a closing line: *"The moral of the story is to not fight and be kind to each
other."*

Nothing taught it story structure, dialogue punctuation, or that stories end
with a moral. All of it is a consequence of predicting the next token.

```
iter    500  train 2.8910  val 2.8883
iter  5000  train 1.8613  val 1.8735
iter 10000  train 1.7052  val 1.7308
iter 15000  train 1.6060  val 1.6222
iter 20000  train 1.5555  val 1.5787
```

The train/validation gap is 0.023, so this is undertrained rather than overfit —
loss was still falling when the schedule ran out.

```bash
python scripts/prepare_tinystories.py --train-mb 500 --vocab-size 4096
python scripts/train_tinystories.py --max-iters 20000 --schedule cosine
```

## Phase 1 result

A 2.7M-parameter decoder trained from scratch on Tiny Shakespeare, character
level. Validation loss falls from 4.17 — the loss of a uniform guess over 65
characters — to **1.5039**, in about four minutes on a free T4.

That number has some history worth stating. The first run reached 1.4990, but a
bug wrote its checkpoint over the wrong directory and destroyed it; the fix and
a regression test are in the history. Reproducing it on a T4 gave 1.4949, and
the figure above, 1.5039, comes from the ablation baseline — same architecture,
current training loop, float32. **1.5039 is the one you can reproduce today**,
and it is what Phase 2 is measured against.

The loss curve and sample below are from the original run and predate a change
to Adam's β₂ default, so the exact digits will differ slightly if you rerun it.

```
iter   500  train 2.0469  val 2.1112
iter  1500  train 1.5135  val 1.7014
iter  2500  train 1.3899  val 1.5871
iter  3500  train 1.3198  val 1.5430
iter  5000  train 1.2703  val 1.4990
```

Sampled at temperature 0.8, top-k 40, from the prompt `ROMEO:`:

```
ROMEO:
O, my lord, that I will fear:
Seal I hard not sent daughter, I know not what I live you mock:
I know not into your own city, and you be so passed
But, no sir, and no man as not resolved
Might have menDed man's study flesh eyes.

POMPEY:
Men in the maid of hope may be gone of her with her
my husband
```

Nothing here was given to the model: not that speakers are capitalised and
followed by a colon, not that lines break where they do, not a single English
word. It learned all of it from raw characters and next-character prediction.
The grammar is local and the meaning does not survive a full sentence, which is
exactly what 2.7M parameters at character level buys.

```bash
python scripts/train_shakespeare.py --max-iters 5000 --lr 1e-3
python scripts/sample.py --prompt "ROMEO:" --temperature 0.8 --top-k 40
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/ -q                # 194 behavioural tests, ~5s on CPU
python scripts/sanity_check.py  # untrained forward pass, shapes and loss
python -m glassbox.viz.server   # the visualizer on localhost:8000
```

Torch and numpy are the only dependencies. The visualizer is standard-library
HTTP with a hand-written frontend — no framework and no build step.

## Layout

```
glassbox/
├── model/       # attention, rope, norms, feed-forward, the KV cache, the GPT
├── tokenizer/   # char-level and BPE, both from scratch
├── training/    # corpus, batching, memory-mapped tokens, the loop,
│                #   precision selection, the LR schedule
├── sampling/    # greedy, temperature, top-k, nucleus
└── viz/         # diagram generator, head stats, inspection, model registry,
                 #   stdlib server, and the frontend under static/
scripts/         # prepare, train, sample, export, baselines, sanity check
notebooks/       # the Colab runs, committed with their outputs
docs/            # the built static page GitHub Pages serves
tests/           # 194 behavioural tests
notes/           # write-ups of the parts that were hardest to get right
```

## Design principles

- **Every attention call returns its weights.** From the first commit —
  the visualizer is a rendering problem, not a refactor.
- **Tests verify behavior, not execution.** Perturbing a future token
  must not change past outputs; that's causality, proven.
- **Each phase ships on its own.** Phases 1 and 2 are tagged `v1-phase1` and
  `v2-phase2`; 3 and 5 are on `main` and were never tagged.
- **The hard parts are written up.** `notes/` covers attention, the block, the
  copying bias, tokenisation and sampling — five write-ups, not one per module.
  The rest of the reasoning lives in comments beside the code.

Device handling is automatic: CUDA, then MPS, then CPU. Mixed precision is
CUDA-only by design — Apple Silicon has no hardware bfloat16 and its fp16
kernels are not worth the risk, so MPS deliberately falls back to float32.
