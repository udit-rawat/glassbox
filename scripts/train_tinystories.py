"""Train the Phase 2 architecture on TinyStories.

    python scripts/prepare_tinystories.py          # once
    python scripts/train_tinystories.py --max-iters 20000 --schedule cosine

Defaults are the modern architecture — RMSNorm, SwiGLU, rotary positions,
grouped query attention — because Phase 2 measured it ahead of the Phase 1 stack
on identical data.
"""

import argparse
from pathlib import Path

import torch

from glassbox.device import get_device
from glassbox.model import GPT, GPTConfig
from glassbox.sampling.generate import generate
from glassbox.tokenizer.bpe import BPETokenizer
from glassbox.training.loop import TrainConfig, train
from glassbox.training.tokens import TokenDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/tinystories")
    p.add_argument("--out-dir", type=str, default="checkpoints/tinystories")
    # model — roughly 11M parameters at these defaults, which fits a T4 with
    # room to spare and sits in the range the TinyStories paper found produces
    # coherent English.
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    p.add_argument("--activation", choices=["gelu", "swiglu"], default="swiglu")
    p.add_argument("--pos-encoding", choices=["learned", "rope"], default="rope")
    p.add_argument("--bias", action="store_true", help="off by default, Llama style")
    # training
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sample-tokens", type=int, default=300)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    for name in ("train.bin", "val.bin", "tokenizer.json"):
        if not (data_dir / name).exists():
            raise SystemExit(
                f"{data_dir / name} is missing — run scripts/prepare_tinystories.py first"
            )

    device = get_device(args.device)
    tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")
    dataset = TokenDataset(data_dir / "train.bin", data_dir / "val.bin", tokenizer)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        dropout=args.dropout,
        norm=args.norm,
        activation=args.activation,
        pos_encoding=args.pos_encoding,
        bias=args.bias,
    )
    model = GPT(config)

    tokens_seen = args.max_iters * args.batch_size * args.grad_accum * args.block_size
    print(f"device      {device}")
    print(f"data        {dataset}")
    print(f"vocab       {tokenizer.vocab_size}")
    print(
        f"arch        {config.norm} / {config.activation} / {config.pos_encoding} "
        f"/ kv_heads {config.n_kv_heads}"
    )
    print(f"parameters  {model.num_parameters():,}")
    # Worth seeing before committing hours: how many tokens the run will consume
    # against how many exist. Above 1.0 the model is repeating the corpus.
    print(
        f"tokens      {tokens_seen:,} to be seen "
        f"({tokens_seen / len(dataset.train):.2f} epochs)"
    )
    print(f"out_dir     {args.out_dir}")
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
        out_dir=args.out_dir,
    )
    train(model, dataset, train_cfg, device)

    if args.sample_tokens > 0:
        print("\n--- sample (temperature 0.8, top-p 0.9) ---")
        prompt = "Once upon a time, there was a little"
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = generate(model, idx, args.sample_tokens, temperature=0.8, top_p=0.9)
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
