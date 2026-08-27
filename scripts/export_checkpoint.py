"""Strip a training checkpoint down to what inference actually reads.

    python scripts/export_checkpoint.py checkpoints/tinystories/best.pt

A training checkpoint carries AdamW's two moment tensors for every parameter so
a dropped run can resume exactly. Inference never touches them, and on an 11M
parameter model they are two thirds of the file — 132 MB becomes 44 MB by
deleting nothing that matters.
"""

import argparse
from pathlib import Path

import torch

from glassbox.model import GPT
from glassbox.tokenizer.spec import load_tokenizer_from_checkpoint

# Everything needed to rebuild a working model and decode its output. The rule
# from Phase 1 still holds: a checkpoint that cannot reconstruct itself is a trap
# you set for yourself later, so the config and tokenizer stay however small the
# file needs to be.
KEEP = ("model", "model_config", "tokenizer", "chars", "iter", "val_loss")

# Present only to resume training. Dropping them is the whole point.
DROP = ("optimizer", "scaler", "history")


def export(src: Path, dst: Path, half: bool = False) -> tuple[int, int]:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    slim = {k: ckpt[k] for k in KEEP if k in ckpt}
    slim["exported_from"] = str(src)

    if half:
        # Halves the file again, and costs a little precision. Fine for a demo
        # or a visualizer; not what you want if this model will be fine-tuned,
        # because the optimizer would inherit the rounding.
        slim["model"] = {k: v.half() if v.is_floating_point() else v
                         for k, v in slim["model"].items()}

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    return src.stat().st_size, dst.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--half", action="store_true", help="store weights as float16")
    args = p.parse_args()

    src = Path(args.checkpoint)
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_slim.pt")

    before, after = export(src, dst, half=args.half)

    # Loading it back is the actual check. A file that saved without error but
    # cannot rebuild the model is worse than no file at all.
    ckpt = torch.load(dst, map_location="cpu", weights_only=False)
    model = GPT(ckpt["model_config"])
    model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
    tokenizer = load_tokenizer_from_checkpoint(ckpt)

    dropped = [k for k in DROP if k in torch.load(src, map_location="cpu",
                                                  weights_only=False)]
    print(f"in          {src}  {before / 1e6:.1f} MB")
    print(f"out         {dst}  {after / 1e6:.1f} MB")
    print(f"saved       {(before - after) / 1e6:.1f} MB  ({1 - after / before:.0%})")
    print(f"dropped     {', '.join(dropped) or 'nothing'}")
    print(f"verified    {model.num_parameters():,} params, "
          f"{type(tokenizer).__name__} vocab {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
