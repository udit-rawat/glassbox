"""The two feed-forward variants: a plain GELU network and a gated SwiGLU one."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from glassbox.model.config import GPTConfig


class MLP(nn.Module):
    """Position-wise feed-forward: widen 4x, apply a nonlinearity, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # The 4x expansion is inherited from the original transformer and has
        # held across scales. Attention moves information between positions but
        # is linear in the values it mixes; this block is where per-position
        # nonlinear computation happens, and it holds roughly two thirds of the
        # parameters in the model.
        hidden = 4 * config.d_model
        self.up_proj = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act(self.up_proj(x))))


class SwiGLU(nn.Module):
    """Gated feed-forward: one branch decides what passes, the other supplies it."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Three matrices instead of two, so the hidden width is cut to 8/3 of
        # the model width rather than 4x. That lands the parameter count within
        # a percent of the GELU MLP, which is what makes a swap comparison
        # honest — otherwise a win could just be the extra capacity talking.
        hidden = int(8 * config.d_model / 3)
        hidden = 64 * ((hidden + 63) // 64)  # round up for kernel-friendly shapes

        self.gate_proj = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The gate branch is squashed by SiLU and multiplied into the up branch,
        # so each hidden unit can be suppressed or passed through depending on
        # the input. GELU applies a fixed curve to every unit alike; this makes
        # the nonlinearity itself input-dependent, which is the whole idea.
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def build_feedforward(config: GPTConfig) -> nn.Module:
    """Pick the feed-forward variant named by the config."""
    return SwiGLU(config) if config.activation == "swiglu" else MLP(config)
