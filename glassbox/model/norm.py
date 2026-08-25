"""RMSNorm — LayerNorm with the centering step removed."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Scales each position by its own root-mean-square, then applies a learned gain."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        # There is deliberately no learned shift, and no option to add one.
        # LayerNorm's beta exists to undo its mean subtraction; RMSNorm performs
        # no subtraction, so a shift would be re-centring something that was
        # never centred. Every published RMSNorm omits it, and making it
        # configurable would only invite a nonstandard module by accident.

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
        return (x_f * rms).to(dtype) * self.weight


def build_norm(config, d_model: int) -> nn.Module:
    """Pick the normalization named by the config."""
    if config.norm == "rmsnorm":
        # config.bias governs the linear projections, not this. RMSNorm having
        # no shift is part of what RMSNorm is, so the flag does not reach here.
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model, bias=config.bias)
