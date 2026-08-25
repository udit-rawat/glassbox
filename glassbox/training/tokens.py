"""A memory-mapped token stream, for corpora too large to hold in memory.

Tiny Shakespeare fits in a tensor. TinyStories does not, and tokenising in
Python during training would leave the GPU waiting on the CPU for most of every
step. So the corpus is tokenised once into a flat array of 16-bit integers on
disk, and batches become slices of a memory-mapped file: no decode in the hot
path, and the operating system pages in only what is read.
"""

from pathlib import Path

import numpy as np
import torch

# uint16 holds ids up to 65,535. Every BPE vocabulary this project trains is far
# below that, and halving the file size halves the read cost of every batch.
TOKEN_DTYPE = np.uint16
MAX_VOCAB = np.iinfo(TOKEN_DTYPE).max + 1


def write_tokens(ids, path: str | Path) -> Path:
    """Write a sequence of token ids to a flat binary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and arr.max() >= MAX_VOCAB:
        raise ValueError(
            f"token id {arr.max()} does not fit in uint16; a vocabulary this "
            f"large needs a wider dtype than {TOKEN_DTYPE.__name__}"
        )
    arr.astype(TOKEN_DTYPE).tofile(path)
    return path


class TokenDataset:
    """Random windows over two memory-mapped token files."""

    def __init__(self, train_path, val_path, tokenizer):
        # mode="r" maps the file rather than reading it. A 300 MB corpus costs
        # no resident memory until batches actually touch those pages.
        self.train = np.memmap(train_path, dtype=TOKEN_DTYPE, mode="r")
        self.val = np.memmap(val_path, dtype=TOKEN_DTYPE, mode="r")
        # Carried so the training loop can record it in the checkpoint; a
        # checkpoint whose ids cannot be decoded is only half a checkpoint.
        self.tokenizer = tokenizer

    def __repr__(self) -> str:
        return (
            f"TokenDataset(train={len(self.train):,} tokens, "
            f"val={len(self.val):,} tokens)"
        )

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val

        if len(data) <= block_size:
            raise ValueError(
                f"the {split} split holds {len(data)} tokens but block_size is "
                f"{block_size}; no window of that length can be cut from it"
            )

        # torch.randint rather than numpy's, so a single torch.manual_seed makes
        # a run reproducible. Two independent generators would mean the seed
        # controlled the model but not the data it saw.
        ix = torch.randint(len(data) - block_size, (batch_size,))

        # Sliced out of the map and copied into one array before becoming a
        # tensor. int64 because the embedding lookup requires it, and the copy
        # is what detaches the batch from the read-only mapping.
        x = np.stack([data[i : i + block_size] for i in ix.tolist()]).astype(np.int64)
        y = np.stack(
            [data[i + 1 : i + 1 + block_size] for i in ix.tolist()]
        ).astype(np.int64)

        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
