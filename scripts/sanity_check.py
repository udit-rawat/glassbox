"""Forward the untrained model once and report what the numbers should be.

    python scripts/sanity_check.py
"""

import math

import torch

from glassbox.device import get_device
from glassbox.model import GPT, GPTConfig


def main() -> None:
    torch.manual_seed(0)
    device = get_device()
    config = GPTConfig()
    model = GPT(config).to(device).eval()

    shape = (4, config.block_size)
    idx = torch.randint(0, config.vocab_size, shape, device=device)
    targets = torch.randint(0, config.vocab_size, shape, device=device)
    with torch.no_grad():
        logits, loss, attentions = model(idx, targets=targets, return_attention=True)
        _, self_loss, _ = model(idx, targets=idx)

    expected = math.log(config.vocab_size)
    print(f"device                {device}")
    print(f"parameters            {model.num_parameters():,}")
    print(f"  non-embedding       {model.num_parameters(include_embeddings=False):,}")
    print(f"logits                {tuple(logits.shape)}")
    print(f"attention maps        {len(attentions)} layers x {tuple(attentions[0].shape)}")
    print()
    # These two agreeing is the whole point of the script: an untrained model
    # predicts uniformly, and any wiring mistake in embeddings or the loss
    # reduction moves the number off ln(vocab_size).
    print(f"loss, random targets  {loss.item():.4f}")
    print(f"ln(vocab_size)        {expected:.4f}")
    print(f"delta                 {abs(loss.item() - expected):.4f}")
    print()
    # Scored against its own input the same model does measurably better than
    # chance before seeing any data. Tied weights plus the residual stream make
    # the token already present the highest-scoring prediction.
    print(f"loss, input as target {self_loss.item():.4f}  (copying bias, expected)")


if __name__ == "__main__":
    main()
