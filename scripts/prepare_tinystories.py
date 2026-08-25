"""Download TinyStories, train a BPE tokenizer on it, and write token files.

    python scripts/prepare_tinystories.py --train-mb 500 --vocab-size 4096

Runs once. Produces train.bin, val.bin and tokenizer.json, which the training
script memory-maps rather than tokenising anything at run time.
"""

import argparse
import time
from pathlib import Path

from glassbox.tokenizer.bpe import BPETokenizer
from glassbox.training.data import download_tinystories
from glassbox.training.tokens import MAX_VOCAB, write_tokens

# Encoding is done in blocks rather than on the whole corpus at once. findall
# over half a gigabyte of text would build a list of every chunk in it before
# returning, which costs several times the file size in memory for no benefit.
BLOCK_CHARS = 8 * 1024 * 1024


def read_prefix(path: Path, megabytes: float | None) -> str:
    if megabytes is None:
        return path.read_text(encoding="utf-8", errors="replace")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(int(megabytes * 1024 * 1024))


def encode_in_blocks(tokenizer: BPETokenizer, text: str, label: str) -> list[int]:
    ids: list[int] = []
    total = len(text)
    start = time.time()
    pos = 0
    while pos < total:
        end = min(pos + BLOCK_CHARS, total)
        # Nudge the boundary to whitespace so a word is never split across two
        # blocks, which would encode its halves as separate chunks and produce
        # tokens the model never sees anywhere else.
        if end < total:
            nl = text.rfind("\n", pos, end)
            if nl > pos:
                end = nl + 1
        ids.extend(tokenizer.encode(text[pos:end]))
        pos = end
        done = pos / total
        print(
            f"  {label}: {done:5.1%}  {len(ids):>12,} tokens  "
            f"{time.time() - start:6.1f}s",
            end="\r",
            flush=True,
        )
    print()
    return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/tinystories")
    p.add_argument("--out-dir", type=str, default="data/tinystories")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument(
        "--bpe-mb", type=float, default=40,
        help="megabytes of text used to learn the merges",
    )
    p.add_argument(
        "--train-mb", type=float, default=500,
        help="megabytes of the training corpus to tokenise; None for all of it",
    )
    args = p.parse_args()

    if args.vocab_size > MAX_VOCAB:
        raise SystemExit(f"--vocab-size must be at most {MAX_VOCAB} to fit in uint16")

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    print("downloading (1.9 GB on the first run, cached after)")
    train_path, valid_path = download_tinystories(data_dir)
    print(f"  train  {train_path.stat().st_size / 1e9:.2f} GB")
    print(f"  valid  {valid_path.stat().st_size / 1e6:.1f} MB\n")

    # Merges are learned from a sample, not the whole corpus. TinyStories uses a
    # deliberately small vocabulary, so forty megabytes already contains almost
    # every word in it; more text would multiply the training time for merges
    # that come out nearly identical.
    print(f"learning {args.vocab_size - 256} merges from {args.bpe_mb:.0f} MB")
    t0 = time.time()
    sample = read_prefix(train_path, args.bpe_mb)
    tokenizer = BPETokenizer.train(sample, vocab_size=args.vocab_size, verbose=True)
    print(f"  vocab {tokenizer.vocab_size} in {time.time() - t0:.1f}s\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")

    print("encoding")
    train_text = read_prefix(train_path, args.train_mb)
    train_ids = encode_in_blocks(tokenizer, train_text, "train")
    del train_text

    valid_text = read_prefix(valid_path, None)
    val_ids = encode_in_blocks(tokenizer, valid_text, "  val")
    valid_chars = len(valid_text)
    del valid_text

    write_tokens(train_ids, out_dir / "train.bin")
    write_tokens(val_ids, out_dir / "val.bin")

    ratio = (args.train_mb * 1024 * 1024) / len(train_ids) if args.train_mb else 0
    print(f"\ntrain    {len(train_ids):>12,} tokens   {out_dir / 'train.bin'}")
    print(f"val      {len(val_ids):>12,} tokens   {out_dir / 'val.bin'}")
    print(f"vocab    {tokenizer.vocab_size}")
    print(f"chars per token  {ratio:.2f}")
    print(f"val chars        {valid_chars:,}")


if __name__ == "__main__":
    main()
