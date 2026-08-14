"""Learning-rate schedules."""

import math


def cosine_with_warmup(
    it: int,
    peak_lr: float,
    warmup_iters: int,
    max_iters: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear ramp up, then a cosine decay down to a floor."""
    min_lr = peak_lr * min_lr_ratio

    # Warmup exists because Adam's per-parameter step sizes are estimated from
    # gradient statistics it has barely any of yet. The first few updates are
    # therefore large and badly aimed, and at full learning rate they can move
    # the model somewhere it never recovers from.
    if it < warmup_iters:
        return peak_lr * (it + 1) / warmup_iters

    if it >= max_iters:
        return min_lr

    # Cosine rather than a step or linear decay: it spends a long time near the
    # peak while progress is fast, then decays smoothly. The end of training is
    # fine-tuning what has already been learned, and a large step there mostly
    # undoes it.
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


def constant(it: int, peak_lr: float, *_args, **_kwargs) -> float:
    """No schedule — what Phase 1 used."""
    return peak_lr
