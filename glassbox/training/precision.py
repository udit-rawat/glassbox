"""Choosing a mixed-precision mode for whatever hardware is actually present."""

from dataclasses import dataclass

import torch


@dataclass
class Precision:
    device_type: str | None  # None disables autocast entirely
    dtype: torch.dtype | None
    use_scaler: bool

    @property
    def enabled(self) -> bool:
        return self.device_type is not None

    def describe(self) -> str:
        if not self.enabled:
            return "fp32"
        name = str(self.dtype).replace("torch.", "")
        return f"{name}{' + scaler' if self.use_scaler else ''}"


def select_precision(device: torch.device, enabled: bool = True) -> Precision:
    """Pick the best precision this device supports, rather than assuming one."""
    if not enabled:
        return Precision(None, None, False)

    if device.type == "cuda":
        # bfloat16 carries the same exponent range as float32, so values never
        # overflow and no loss scaling is needed. Only Ampere and later have it;
        # a T4 is Turing, and there fp16 is the option — which does overflow, so
        # a gradient scaler has to multiply the loss up before the backward pass
        # and divide it back out before the step.
        if torch.cuda.is_bf16_supported():
            return Precision("cuda", torch.bfloat16, False)
        return Precision("cuda", torch.float16, True)

    # Apple Silicon has autocast, but no hardware bfloat16 and a history of
    # correctness gaps in fp16 kernels. The local machine is for building and
    # debugging, not for the long run, so fp32 is the honest default here.
    return Precision(None, None, False)
