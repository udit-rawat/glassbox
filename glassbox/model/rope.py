"""Rotary position embeddings — position as a rotation of queries and keys."""

import torch


def build_rope_cache(
    seq_len: int, d_head: int, theta: float = 10000.0, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cosine and sine tables, shaped (seq_len, d_head)."""
    # Each adjacent pair of dimensions is treated as a 2-D plane and rotated by
    # an angle proportional to the position. Different pairs rotate at
    # geometrically spaced rates: the fastest completes a turn every few
    # positions, the slowest barely moves across the whole context. Together
    # they encode position the way the hands of a clock encode time — several
    # dials at different speeds, unambiguous when read as a set.
    half = d_head // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))

    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)  # (seq_len, half)

    # Each angle serves both halves of the vector, so the table is duplicated to
    # full width and lines up with the rotate_half convention below.
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair each dimension with its partner half a width away and swap them, negating one."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x by the angle stored for each position. x is (B, n_heads, T, d_head)."""
    # The standard planar rotation, written without trigonometry: a vector times
    # the cosine, plus its perpendicular partner times the sine.
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin
