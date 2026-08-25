"""Byte-pair encoding, trained from scratch.

The character tokenizer spends one token per character, so a 128-token window is
about one sentence. BPE learns which sequences of bytes occur together often
enough to deserve a token of their own, and buys back several times the context
for the same sequence length.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Text is split on this before any merging, and merges never cross a chunk
# boundary. Without it the most frequent pair in English is "e "+something, and
# the tokenizer happily learns tokens that straddle word ends — which wrecks
# generalisation, because "the" then encodes differently depending on what
# follows it. The leading-space alternatives are how a word keeps its space
# attached, so " the" is one token and decoding needs no join rules.
SPLIT_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def _pair_counts(symbols: tuple[int, ...], counts: Counter, weight: int = 1) -> None:
    for pair in zip(symbols, symbols[1:]):
        counts[pair] += weight


def _merge(symbols: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Byte-level BPE: 256 byte values, plus one id per learned merge."""

    def __init__(self, merges: list[tuple[int, int, int]]):
        # Ordered, and the order is the tokenizer. Merge 300 may only apply
        # after merge 299 has created the symbol it consumes, so a set or an
        # unordered dict would silently produce a different encoding.
        self.merges: dict[tuple[int, int], int] = {(a, b): idx for a, b, idx in merges}
        self.merge_list = list(merges)

        # Every id resolves to a literal byte string, which is what makes decode
        # exact rather than approximate.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for a, b, idx in merges:
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------- training

    @classmethod
    def train(
        cls, text: str, vocab_size: int, verbose: bool = False
    ) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size={vocab_size} is below the 256 byte values that form "
                "the base alphabet; there would be no way to represent raw bytes"
            )

        # Counting pairs across the whole corpus on every merge would be
        # hopeless — thousands of merges times a million characters. Identical
        # words are collapsed into one entry with a frequency instead, so the
        # work scales with the vocabulary of the corpus rather than its length.
        word_freqs = Counter(SPLIT_PATTERN.findall(text))
        words = {word: tuple(word.encode("utf-8")) for word in word_freqs}

        # Counts are maintained incrementally rather than recomputed. Rebuilding
        # them each merge means walking every symbol of every word thousands of
        # times: on a corpus with 100k distinct words and 4k merges that is
        # billions of operations, and the difference between minutes and hours.
        # `where` maps each pair to the words containing it, so a merge only has
        # to touch the words it actually appears in.
        counts: Counter = Counter()
        where: dict[tuple[int, int], set[str]] = defaultdict(set)
        for word, symbols in words.items():
            freq = word_freqs[word]
            for pair in zip(symbols, symbols[1:]):
                counts[pair] += freq
                where[pair].add(word)

        merges: list[tuple[int, int, int]] = []
        for i in range(vocab_size - 256):
            if not counts:
                # The corpus ran out of adjacent pairs before the target size.
                break

            # The pair itself breaks ties, so the outcome never depends on dict
            # ordering. Without that, two runs over the same corpus could choose
            # differently among equally frequent pairs.
            pair = max(counts, key=lambda p: (counts[p], p))
            freq = counts[pair]
            if freq < 2:
                # Merging a pair that occurs once trades a vocabulary slot for
                # nothing; every later merge would be worth even less.
                break

            new_id = 256 + i
            for word in list(where[pair]):
                symbols = words[word]
                weight = word_freqs[word]

                # Withdraw this word's old pairs, apply the merge, then add its
                # new ones back. Everything it contributed changes, because a
                # merge shifts the neighbours on both sides of the pair too.
                for p in zip(symbols, symbols[1:]):
                    counts[p] -= weight
                    if counts[p] <= 0:
                        del counts[p]
                    where[p].discard(word)

                merged = _merge(symbols, pair, new_id)
                words[word] = merged

                for p in zip(merged, merged[1:]):
                    counts[p] += weight
                    where[p].add(word)

            merges.append((pair[0], pair[1], new_id))

            if verbose and (i + 1) % 500 == 0:
                print(f"  merge {i + 1:>5}  freq {freq:>7,}", flush=True)

        return cls(merges)

    # ------------------------------------------------------------- encoding

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merge_list)

    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        symbols = tuple(chunk.encode("utf-8"))
        while len(symbols) >= 2:
            # Apply the *earliest-learned* applicable merge, not the most
            # frequent one present. Merges were learned in an order where each
            # depends on the symbols the previous ones created, so applying them
            # out of order produces ids that decode to the right text but do not
            # match what the model was trained on.
            candidates = {
                pair: self.merges[pair]
                for pair in zip(symbols, symbols[1:])
                if pair in self.merges
            }
            if not candidates:
                break
            best = min(candidates, key=candidates.get)
            symbols = _merge(symbols, best, self.merges[best])

        self._cache[chunk] = symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        # A single token can hold part of a multi-byte character, so decoding a
        # truncated stream must not raise — mid-generation output is routinely
        # cut between the bytes of one character.
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- storage

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"merges": [list(m) for m in self.merge_list]}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])
