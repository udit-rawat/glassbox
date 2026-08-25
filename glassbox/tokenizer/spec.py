"""Turning a tokenizer into something a checkpoint can carry, and back again.

A checkpoint that cannot rebuild its own tokenizer is only half a checkpoint: the
ids decode to nothing without it. Keeping the conversion here rather than inside
either tokenizer means neither of them has to know what a checkpoint is.
"""

from glassbox.tokenizer.bpe import BPETokenizer
from glassbox.tokenizer.char import CharTokenizer


def tokenizer_spec(tokenizer) -> dict:
    """Describe a tokenizer as plain data."""
    if isinstance(tokenizer, BPETokenizer):
        return {"kind": "bpe", "merges": [list(m) for m in tokenizer.merge_list]}
    if isinstance(tokenizer, CharTokenizer):
        return {"kind": "char", "chars": list(tokenizer.chars)}
    raise TypeError(
        f"no checkpoint representation for {type(tokenizer).__name__}; "
        "add one here rather than letting the checkpoint save without a tokenizer"
    )


def load_tokenizer(spec: dict):
    """Rebuild a tokenizer from the description stored in a checkpoint."""
    kind = spec.get("kind")
    if kind == "bpe":
        # Merge order is the tokenizer, so the list is rebuilt in sequence.
        return BPETokenizer([tuple(m) for m in spec["merges"]])
    if kind == "char":
        return CharTokenizer(spec["chars"])
    raise ValueError(f"unknown tokenizer kind {kind!r} in checkpoint")


def load_tokenizer_from_checkpoint(ckpt: dict):
    """Read whichever tokenizer record a checkpoint happens to carry."""
    spec = ckpt.get("tokenizer")
    if spec is not None:
        return load_tokenizer(spec)
    # Checkpoints written before the spec existed carried a bare character list.
    if ckpt.get("chars"):
        return CharTokenizer(ckpt["chars"])
    raise ValueError(
        "checkpoint carries no tokenizer; its ids cannot be decoded back to text"
    )
