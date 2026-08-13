"""Train the Phase 1 GPT on Tiny Shakespeare, then sample from it.

    python scripts/train_shakespeare.py --max-iters 3000
"""

import argparse

import torch

from glassbox.device import get_device
from glassbox.model import GPT, GPTConfig
from glassbox.sampling.generate import generate
from glassbox.tokenizer.char import CharTokenizer
from glassbox.training.data import CharDataset, download_tiny_shakespeare
from glassbox.training.loop import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sample-tokens", type=int, default=500)
    args = p.parse_args()

    device = get_device(args.device)
    path = download_tiny_shakespeare()
    text = path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.from_text(text)
    dataset = CharDataset(text, tokenizer)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model = GPT(config)

    print(f"device      {device}")
    print(f"corpus      {len(text):,} chars, vocab {tokenizer.vocab_size}")
    print(f"split       {len(dataset.train):,} train / {len(dataset.val):,} val")
    print(f"parameters  {model.num_parameters():,}")
    print()

    train_cfg = TrainConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
    )
    train(model, dataset, train_cfg, device)
    tokenizer.save("checkpoints/tokenizer.json")

    # Sampling starts from a single newline rather than an empty tensor: the
    # model needs at least one token of context, and a newline is what precedes
    # a speaker name everywhere in the corpus.
    print("\n--- sample (temperature 0.8, top-k 40) ---")
    start = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
    out = generate(model, start, args.sample_tokens, temperature=0.8, top_k=40)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
