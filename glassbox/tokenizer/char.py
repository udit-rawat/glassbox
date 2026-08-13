"""Character-level tokenizer: the vocabulary is whatever characters the corpus contains."""

import json
from pathlib import Path


class CharTokenizer:
    """Maps each distinct character in a corpus to an integer id."""

    def __init__(self, chars: list[str]):
        # Sorted, so the same corpus always produces the same ids. Without this
        # the vocabulary would depend on set iteration order and a checkpoint
        # would decode into gibberish against a re-derived tokenizer.
        self.chars = sorted(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # No unknown-token handling and none wanted. The vocabulary is derived
        # from the corpus being trained on, so every character is known by
        # construction, and a KeyError here means the wrong tokenizer was
        # loaded rather than a case worth silently absorbing.
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])
