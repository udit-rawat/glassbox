"""Everything the visualizer needs from one forward pass, as plain data.

No torch types cross this boundary and nothing here renders anything. The point
is that the whole payload survives json.dumps, so the frontend is a rendering
problem and this file is testable on its own.
"""

import torch
import torch.nn.functional as F

from glassbox.viz.heads import describe, head_stats

# Whitespace has to be visible when it labels an axis. A heatmap row labelled
# with an actual newline is a blank row.
_GLYPHS = {" ": "␣", "\n": "⏎", "\t": "⇥", "\r": "␍"}

# Four decimals is well past what a heatmap can show, and it keeps the payload
# from growing faster than it needs to — attention is layers x heads x T x T.
_PLACES = 4


def display_text(text: str) -> str:
    for raw, glyph in _GLYPHS.items():
        text = text.replace(raw, glyph)
    return text


def _token_entries(tokenizer, ids: list[int]) -> list[dict]:
    out = []
    for i, tid in enumerate(ids):
        # Decoded one at a time, because a BPE token can hold part of a
        # multi-byte character and decode() already replaces what it cannot form.
        text = tokenizer.decode([tid])
        out.append({"position": i, "id": tid, "text": text,
                    "display": display_text(text) or "·"})
    return out


def _top_k(logits: torch.Tensor, tokenizer, k: int) -> list[dict]:
    """Top k tokens for one position's logits."""
    probs = F.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, min(k, probs.numel()))
    entries = []
    for prob, idx in zip(values.tolist(), indices.tolist()):
        text = tokenizer.decode([idx])
        entries.append({"id": idx, "text": text,
                        "display": display_text(text) or "·",
                        "prob": round(prob, 5)})
    return entries


@torch.no_grad()
def inspect(model, tokenizer, prompt: str, top_k: int = 8) -> dict:
    """Run one prompt through the model and describe what happened."""
    if not prompt:
        raise ValueError("prompt is empty")

    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt encoded to no tokens")
    if len(ids) > model.config.block_size:
        raise ValueError(
            f"prompt is {len(ids)} tokens but block_size is "
            f"{model.config.block_size}"
        )

    was_training = model.training
    model.eval()
    try:
        idx = torch.tensor([ids], dtype=torch.long)
        # Both flags in one pass. Asking twice would double the work and, with
        # dropout off, produce identical numbers anyway.
        logits, _, attentions = model(idx, return_attention=True, return_hidden=True)
        hidden = model.hidden_states()

        # The lens: the model's own head pointed at each layer's output in turn.
        # The final entry is the real prediction, which is what the frontend
        # renders as the next-token bars — it is not stored twice.
        lens = []
        for depth, h in enumerate(hidden):
            layer_logits = model.lm_head(model.ln_f(h))[0]
            lens.append({
                "depth": depth,
                "label": "embedding" if depth == 0 else f"layer {depth}",
                "positions": [_top_k(layer_logits[t], tokenizer, top_k)
                              for t in range(layer_logits.size(0))],
            })
    finally:
        model.train(was_training)

    attention, heads = [], []
    for layer, weights in enumerate(attentions):
        per_layer = []
        for head in range(weights.size(1)):
            w = weights[0, head]
            per_layer.append([[round(v, _PLACES) for v in row] for row in w.tolist()])
            stats = head_stats(w)
            heads.append({"layer": layer, "head": head,
                          "kind": describe(stats), **stats})
        attention.append(per_layer)

    config = model.config
    return {
        "prompt": prompt,
        "tokens": _token_entries(tokenizer, ids),
        "n_tokens": len(ids),
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "block_size": config.block_size,
        "vocab_size": config.vocab_size,
        "attention": attention,
        "lens": lens,
        "heads": heads,
    }
