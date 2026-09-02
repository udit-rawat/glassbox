"""LoRA: adapting a frozen model by learning a low-rank correction.

Fine-tuning normally means updating every weight, which costs a full optimizer
state per parameter and produces a copy of the whole model per task. LoRA
leaves the original weights frozen and learns a small correction beside them:
instead of W, the layer computes W + BA, where B and A are thin enough that
their product has rank r. A 4096x4096 projection has 16.8M parameters; at rank
8 the correction has 65,536 — four tenths of a percent.

Two properties make it usable, and both are asserted in the tests:

  B starts at zero, so BA is zero and the adapted model is bit-identical to the
  original before any training. Attaching an adapter can never break a model.

  BA can be folded into W afterwards, giving back a plain nn.Linear with no
  extra parameters and no inference cost. Adaptation is free at serving time.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Where LoRA is normally attached. Adapting the attention projections is enough
# for most tasks; the original paper adapts queries and values only, and that
# remains the common default.
DEFAULT_TARGETS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """A frozen nn.Linear with a trainable low-rank correction beside it."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank must be positive, got {r}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        # The correction is scaled by alpha/r so that changing the rank does not
        # change how much the adapter can move the output. Without it, raising r
        # would silently raise the effective learning rate too.
        self.scaling = alpha / r

        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        self.lora_dropout = nn.Dropout(dropout)

        # A is initialised the way any weight would be; B is left at zero, so
        # their product is zero and the layer starts out exactly equal to the
        # layer it wraps. It is also why the pair is not symmetric: initialising
        # both randomly would inject noise into a trained model on attachment.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = F.linear(F.linear(self.lora_dropout(x), self.lora_a), self.lora_b)
        return self.base(x) + correction * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Fold the correction into the frozen weight and hand back a plain Linear."""
        out = nn.Linear(self.in_features, self.out_features,
                        bias=self.base.bias is not None)
        # (out, r) @ (r, in) -> (out, in), the same shape as the weight it joins.
        delta = (self.lora_b @ self.lora_a) * self.scaling
        out.weight.copy_(self.base.weight + delta)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out.to(self.base.weight.device)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}")


def _walk(module: nn.Module):
    for name, child in module.named_children():
        yield module, name, child
        yield from _walk(child)


def apply_lora(model: nn.Module, targets=DEFAULT_TARGETS, r: int = 8,
               alpha: int = 16, dropout: float = 0.0) -> nn.Module:
    """Freeze the model and wrap every matching nn.Linear with a LoRA correction.

    Modified in place and returned. Matching is by attribute name, so passing
    ("q_proj", "v_proj") adapts those projections in every block at once.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for parent, name, child in list(_walk(model)):
        if name in targets and isinstance(child, nn.Linear):
            setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1

    if replaced == 0:
        # Silence here would mean training a model with nothing trainable in it,
        # which looks exactly like a run that simply refuses to learn.
        raise ValueError(
            f"no nn.Linear named any of {tuple(targets)} was found; "
            "nothing would be trainable"
        )
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear with the equivalent plain Linear."""
    for parent, name, child in list(_walk(model)):
        if isinstance(child, LoRALinear):
            setattr(parent, name, child.merged())
    return model


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def trainable_report(model: nn.Module) -> dict:
    """How much of the model is actually being trained."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "fraction": trainable / total if total else 0.0,
        "adapters": sum(1 for _, _, c in _walk(model) if isinstance(c, LoRALinear)),
    }


def adapter_state(model: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}


def save_adapter(model: nn.Module, path: str | Path, **meta) -> Path:
    """Write only the correction. The base model is not ours to ship."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = [c for _, _, c in _walk(model) if isinstance(c, LoRALinear)]
    torch.save({
        "lora": adapter_state(model),
        "config": {
            "r": layers[0].r if layers else None,
            "alpha": layers[0].alpha if layers else None,
            "targets": sorted({n for p, n, c in _walk(model)
                               if isinstance(c, LoRALinear)}),
        },
        **meta,
    }, path)
    return path


def load_adapter(model: nn.Module, path: str | Path) -> nn.Module:
    """Attach a saved adapter to a matching base model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if not any(isinstance(c, LoRALinear) for _, _, c in _walk(model)):
        apply_lora(model, targets=cfg["targets"], r=cfg["r"], alpha=cfg["alpha"])
    missing = model.load_state_dict(ckpt["lora"], strict=False)
    unexpected = getattr(missing, "unexpected_keys", [])
    if unexpected:
        raise ValueError(f"adapter does not fit this model: {unexpected[:4]}")
    return model
