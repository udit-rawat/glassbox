"""Load a trained checkpoint and sample from it.

    python scripts/sample.py --prompt "ROMEO:" --temperature 0.8 --top-k 40
"""

import argparse

import torch

from glassbox.device import get_device
from glassbox.model import GPT
from glassbox.sampling.generate import generate
from glassbox.tokenizer.spec import load_tokenizer_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device(args.device)
    # weights_only=False because the checkpoint carries the GPTConfig dataclass
    # alongside the tensors. Safe here since these files are produced locally by
    # scripts/train_shakespeare.py and never downloaded.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Read whichever tokenizer the checkpoint was trained with rather than
    # assuming characters — a BPE checkpoint carries merges and no character
    # list, and hardcoding one kind here breaks silently on the other.
    tokenizer = load_tokenizer_from_checkpoint(ckpt)

    model = GPT(ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    cfg = ckpt["model_config"]
    print(f"checkpoint  iter {ckpt['iter']}, val loss {ckpt['val_loss']:.4f}")
    print(
        f"arch        {cfg.norm} / {cfg.activation} / {cfg.pos_encoding} "
        f"/ kv_heads {cfg.n_kv_heads}"
    )
    print(f"tokenizer   {type(tokenizer).__name__}, vocab {tokenizer.vocab_size}")
    print(f"sampling    temp {args.temperature}, top_k {args.top_k}, top_p {args.top_p}\n")

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(
        model,
        idx,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
