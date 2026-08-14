"""RMSNorm — LayerNorm with the centering step removed."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Scales each position by its own root-mean-square, then applies a learned gain."""

    def __init__(self, d_model: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        # No learned shift. LayerNorm's beta pairs with the mean subtraction it
        # performs; with no centering there is nothing for a shift to re-centre,
        # and every published RMSNorm leaves it out.
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm subtracts the mean, then divides by the standard deviation.
        # RMSNorm skips the subtraction and divides by the root mean square.
        # Two reductions become one, and the empirical finding is that the
        # centering contributed almost nothing — the rescaling was doing the work.
        dtype = x.dtype

        # Squaring under fp16 autocast is where this would silently overflow, so
        # the statistic is always computed in float32 and cast back afterwards.
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (x_f * rms).to(dtype) * self.weight

        return out + self.bias if self.bias is not None else out


def build_norm(config, d_model: int) -> nn.Module:
    """Pick the normalization named by the config."""
    if config.norm == "rmsnorm":
        return RMSNorm(d_model, bias=config.bias)
    return nn.LayerNorm(d_model, bias=config.bias)
