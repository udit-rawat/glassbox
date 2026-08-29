"""Four numbers that describe what an attention head does.

A grid of thirty-six heatmaps is thirty-six identical squares until you can tell
them apart. These statistics separate a head that always looks one token back
from one that reads the whole context, so the grid can be sorted and read rather
than only stared at.
"""

import torch


def head_stats(weights: torch.Tensor) -> dict[str, float]:
    """Describe one head's (T, T) attention matrix."""
    T = weights.size(-1)
    if T == 0:
        raise ValueError("empty attention matrix")

    idx = torch.arange(T, device=weights.device)
    # How far back each key sits from its query. Clamped because the upper
    # triangle is masked to zero weight anyway, and a negative distance there
    # would drag the mean the wrong way if any residue survived.
    distance = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0).float()

    # Weighted mean of that distance: near 1 is a head reading the previous
    # token, large is a head reaching far back.
    mean_distance = (weights * distance).sum(-1).mean().item()

    # How much a head attends to the token itself. High usually means a head
    # that passes its position through rather than gathering from elsewhere.
    diagonal = weights.diagonal(0, -2, -1).mean().item()

    # The single most common learned head type gets its own number.
    previous = weights.diagonal(-1, -2, -1).mean().item() if T > 1 else 0.0

    # Entropy in nats, averaged over queries. Low means decisive, high means the
    # head spreads its attention broadly. Clamped before the log because masked
    # positions are exactly zero and log(0) is -inf.
    p = weights.clamp_min(1e-12)
    entropy = (-(p * p.log()).sum(-1)).mean().item()

    return {
        "mean_distance": round(mean_distance, 4),
        "diagonal": round(diagonal, 4),
        "previous_token": round(previous, 4),
        "entropy": round(entropy, 4),
    }


def describe(stats: dict[str, float]) -> str:
    """A short label for the head, from its statistics."""
    # Deliberately coarse. These are descriptions of behaviour, not claims about
    # what a head has learned to represent — that would need far more evidence
    # than one prompt's attention map.
    if stats["previous_token"] > 0.5:
        return "previous token"
    if stats["diagonal"] > 0.5:
        return "self"
    if stats["mean_distance"] > 8 and stats["entropy"] > 2.0:
        return "broad context"
    if stats["entropy"] < 1.0:
        return "focused"
    return "mixed"
