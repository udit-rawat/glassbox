"""Cross-entropy of trivial models on the same data, as a yardstick.

A validation loss is only meaningful next to something. 1.5787 nats on
TinyStories is internally consistent but says nothing on its own — a reader has
no way to tell whether it is good, mediocre, or merely self-consistent.

This measures what you get for free: guessing uniformly, guessing by token
frequency, and guessing from the previous token alone. The trained model has to
be read against those, not against nothing.
"""

import argparse
import math
import urllib.request
from collections import Counter
from pathlib import Path

from glassbox.tokenizer.bpe import BPETokenizer
from glassbox.training.data import TINYSTORIES_TRAIN_URL, TINYSTORIES_VALID_URL, download


def fetch_prefix(url: str, path: Path, megabytes: float) -> Path:
    """Download only the first N megabytes, using a ranged request."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{int(megabytes * 1e6)}"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def cross_entropy_unigram(train_ids, val_ids, vocab_size, alpha=1.0) -> float:
    counts = Counter(train_ids)
    total = len(train_ids) + alpha * vocab_size
    logp = {t: math.log((counts[t] + alpha) / total) for t in set(val_ids)}
    return -sum(logp[t] for t in val_ids) / len(val_ids)


def cross_entropy_bigram(train_ids, val_ids, vocab_size, alpha=0.1) -> float:
    """Predict each token from the one before it, with add-alpha smoothing."""
    following: dict[int, Counter] = {}
    for a, b in zip(train_ids, train_ids[1:]):
        following.setdefault(a, Counter())[b] += 1
    totals = {a: sum(c.values()) for a, c in following.items()}

    denom_unseen = math.log(alpha * vocab_size)
    total = 0.0
    for a, b in zip(val_ids, val_ids[1:]):
        c = following.get(a)
        if c is None:
            # Nothing ever followed this token in training: fall back to uniform.
            total += math.log(vocab_size)
            continue
        total += -math.log((c[b] + alpha) / (totals[a] + alpha * vocab_size))
    return total / (len(val_ids) - 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=str, default="checkpoints/tinystories/tokenizer.json")
    p.add_argument("--data-dir", type=str, default="data/tinystories")
    p.add_argument("--fit-mb", type=float, default=60,
                   help="megabytes of training text the n-gram models are fitted on")
    p.add_argument("--eval-mb", type=float, default=6,
                   help="megabytes of held-out text they are scored on")
    p.add_argument("--model-loss", type=float, default=1.5787,
                   help="the trained model's validation loss, for the comparison")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer = BPETokenizer.load(args.tokenizer)
    V = tokenizer.vocab_size

    print(f"tokenizer      {V} tokens")
    print("fetching       train prefix and the held-out validation split")
    train_path = fetch_prefix(TINYSTORIES_TRAIN_URL, data_dir / "train_prefix.txt",
                              args.fit_mb)
    valid_path = download(TINYSTORIES_VALID_URL, data_dir / "valid.txt")

    fit_text = train_path.read_text(encoding="utf-8", errors="replace")
    val_text = valid_path.read_text(encoding="utf-8", errors="replace")[
        : int(args.eval_mb * 1e6)]

    print("encoding       ", end="", flush=True)
    train_ids = tokenizer.encode(fit_text)
    val_ids = tokenizer.encode(val_text)
    print(f"{len(train_ids):,} fit / {len(val_ids):,} held out")

    uniform = math.log(V)
    unigram = cross_entropy_unigram(train_ids, val_ids, V)
    bigram = cross_entropy_bigram(train_ids, val_ids, V)

    print(f"\n{'model':<26}{'loss':>9}{'perplexity':>13}")
    print("-" * 48)
    rows = [
        ("uniform over the vocabulary", uniform),
        ("unigram (token frequency)", unigram),
        ("bigram (previous token)", bigram),
        ("glassbox 11M", args.model_loss),
    ]
    for label, loss in rows:
        print(f"{label:<26}{loss:>9.4f}{math.exp(loss):>13.1f}")

    print(f"\nagainst bigram: {bigram - args.model_loss:+.4f} nats "
          f"({math.exp(bigram) / math.exp(args.model_loss):.1f}x lower perplexity)")


if __name__ == "__main__":
    main()
