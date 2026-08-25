"""Corpus download, the train/validation split, and random batch sampling."""

import urllib.request
from pathlib import Path

import torch

from glassbox.tokenizer.char import CharTokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# TinyStories is fetched as two plain text files over HTTP. The dataset happens
# to be hosted on the HuggingFace CDN, but nothing here imports their libraries —
# this is the same urlretrieve used for Shakespeare, and the "no HuggingFace
# until Phase 4" rule is about not leaning on their abstractions, not about
# refusing to download a file.
TINYSTORIES_TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
)
TINYSTORIES_VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
)


def download(url: str, path: str | Path) -> Path:
    """Fetch a URL to a path, skipping the download if it is already there."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written to a temporary name first, so an interrupted download cannot
        # leave a truncated file that later runs mistake for a complete one.
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def download_tiny_shakespeare(path: str | Path = "data/tinyshakespeare.txt") -> Path:
    return download(TINY_SHAKESPEARE_URL, path)


def download_tinystories(data_dir: str | Path = "data/tinystories") -> tuple[Path, Path]:
    """Fetch the TinyStories train and validation text. Train is roughly 1.9 GB."""
    data_dir = Path(data_dir)
    return (
        download(TINYSTORIES_TRAIN_URL, data_dir / "train.txt"),
        download(TINYSTORIES_VALID_URL, data_dir / "valid.txt"),
    )


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

        if len(data) <= block_size:
            # torch.randint raises an opaque error when its upper bound is not
            # positive, and the real cause — a corpus too small for the context
            # length — is nowhere in that message.
            raise ValueError(
                f"the {split} split holds {len(data)} tokens but block_size is "
                f"{block_size}; no window of that length can be cut from it"
            )

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
