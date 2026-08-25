"""Train a GPT on Tiny Shakespeare, then sample from it.

    python scripts/train_shakespeare.py --max-iters 5000 --lr 1e-3

Architecture defaults reproduce Phase 1, so the same command with and without
the switches below is a controlled comparison:

    --norm rmsnorm --activation swiglu --pos-encoding rope --n-kv-heads 2
"""

import argparse
from pathlib import Path

import torch

from glassbox.device import get_device
from glassbox.model import GPT, GPTConfig
from glassbox.sampling.generate import generate
from glassbox.tokenizer.char import CharTokenizer
from glassbox.training.data import CharDataset, download_tiny_shakespeare
from glassbox.training.loop import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    # architecture switches
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--activation", choices=["gelu", "swiglu"], default="gelu")
    p.add_argument("--pos-encoding", choices=["learned", "rope"], default="learned")
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--no-bias", action="store_true")
    # training
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    # plumbing
    p.add_argument("--out-dir", type=str, default="checkpoints")
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
        norm=args.norm,
        activation=args.activation,
        pos_encoding=args.pos_encoding,
        n_kv_heads=args.n_kv_heads,
        bias=not args.no_bias,
    )
    model = GPT(config)

    out_dir = Path(args.out_dir)
    print(f"device      {device}")
    print(f"corpus      {len(text):,} chars, vocab {tokenizer.vocab_size}")
    print(f"split       {len(dataset.train):,} train / {len(dataset.val):,} val")
    print(
        f"arch        {config.norm} / {config.activation} / {config.pos_encoding} "
        f"/ kv_heads {config.n_kv_heads}"
    )
    print(f"parameters  {model.num_parameters():,}")
    print(f"out_dir     {out_dir}")
    print(flush=True)

    train_cfg = TrainConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        schedule=args.schedule,
        warmup_iters=args.warmup_iters,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        betas=(0.9, args.beta2),
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
        amp=not args.no_amp,
        resume=args.resume,
        out_dir=str(out_dir),
    )
    train(model, dataset, train_cfg, device)

    # Beside the checkpoints it belongs to, not in a fixed location — two runs
    # into different directories must not overwrite each other's tokenizer.
    tokenizer.save(out_dir / "tokenizer.json")

    if args.sample_tokens > 0:
        # Sampling starts from a single newline rather than an empty tensor: the
        # model needs at least one token of context, and a newline is what
        # precedes a speaker name everywhere in the corpus.
        print("\n--- sample (temperature 0.8, top-k 40) ---")
        start = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
        out = generate(model, start, args.sample_tokens, temperature=0.8, top_k=40)
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
