# glassbox

**A language model you can see inside.**

A GPT built from first principles in pure PyTorch — no HuggingFace, no
abstractions until every line underneath is owned — grown phase by phase
into a modern Llama-style architecture, trained for real, fine-tuned with
LoRA written from scratch, and finished with an interactive visualizer
that shows you exactly what every attention head is doing.

> Visualizer screenshot lands here — Phase 5.

## Why "glassbox"

Most language models are black boxes. This one was built so that every
component — attention, normalization, positional encoding, the tokenizer,
the training loop, the fine-tuning adapters — can be opened, read, tested,
and watched while it runs. The opposite of a black box.

## Progress

- [ ] **Phase 1 — Attention & the Transformer.** Scaled dot-product
  attention, multi-head attention, causal masking, a full GPT-style
  decoder trained char-level on Tiny Shakespeare.
- [ ] **Phase 2 — Modern LLM internals.** RMSNorm, RoPE, SwiGLU, grouped
  query attention, KV cache, temperature/top-k/top-p sampling, and a BPE
  tokenizer from scratch.
- [ ] **Phase 3 — A real training run.** TinyStories with mixed precision,
  gradient accumulation, cosine LR schedule, and crash-safe checkpointing.
- [ ] **Phase 4 — Fine-tuning a real model.** LoRA implemented from
  scratch (~80 lines), then instruction-tuning a real pretrained model.
- [ ] **Phase 5 — The visualizer.** Enter a prompt, watch attention heads
  light up per layer, inspect token-by-token next-token probabilities.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/ -q              # 18 behavioural tests
python scripts/sanity_check.py  # untrained forward pass, shapes and loss
```

## Layout

```
glassbox/
├── model/       # attention, blocks, the GPT — evolves Phase 1 → 2
├── tokenizer/   # char-level, then BPE from scratch
├── training/    # loop, schedule, checkpointing
├── sampling/    # greedy → temperature / top-k / top-p
└── viz/         # the attention visualizer
tests/           # behavioral tests: causality, invariants, shapes
notes/           # per-module notes: what it does and why it exists
```

## Design principles

- **Every attention call returns its weights.** From the first commit —
  the visualizer is a rendering problem, not a refactor.
- **Tests verify behavior, not execution.** Perturbing a future token
  must not change past outputs; that's causality, proven.
- **Each phase ships on its own.** The repo is presentable at every tag.
- **Every module carries a note** on what it does and why it exists
  that way.

Runs on Apple Silicon (MPS) with CPU fallback; heavier training runs on a
free cloud GPU. Device-agnostic throughout.
