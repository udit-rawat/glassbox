"""Device selection, resolved once and shared by every entry point."""

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick cuda, then mps, then cpu — or honour an explicit request."""
    # The same code runs on an M1 laptop and a rented CUDA box; only this
    # function knows the difference. cuda is checked first because when both
    # exist the machine is a cloud GPU box and mps is not present anyway.
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
