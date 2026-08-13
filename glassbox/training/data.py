"""Corpus download, the train/validation split, and random batch sampling."""

import urllib.request
from pathlib import Path

import torch

from glassbox.tokenizer.char import CharTokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str | Path = "data/tinyshakespeare.txt") -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


class CharDataset:
    """Holds the whole corpus as one long id tensor and cuts random windows from it."""

    def __init__(self, text: str, tokenizer: CharTokenizer, val_fraction: float = 0.1):
        self.tokenizer = tokenizer
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # Split by position rather than at random, so validation is a contiguous
        # held-out tail. Shuffling characters first would put the same sentence
        # on both sides of the split and make validation loss meaningless.
        split = int(len(data) * (1 - val_fraction))
        self.train = data[:split]
        self.val = data[split:]

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val

        # Windows are drawn at uniformly random offsets rather than marched
        # through in order. One "epoch" has no meaning here: the corpus is a
        # single stream, and sampling independently means consecutive batches
        # are uncorrelated without any shuffling machinery.
        ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in ix])
        # This one-position offset is the entire supervision signal. Position t
        # of x is asked to predict position t of y, which is character t+1 of
        # the corpus. The shift lives here and nowhere else in the codebase.
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

        return x.to(device), y.to(device)
