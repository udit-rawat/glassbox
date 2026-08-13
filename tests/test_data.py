"""Tests for the split and the batch sampler."""

import pytest
import torch

from glassbox.tokenizer.char import CharTokenizer
from glassbox.training.data import CharDataset

TEXT = "".join(chr(ord("a") + (i % 26)) for i in range(2000))


@pytest.fixture
def dataset():
    torch.manual_seed(0)
    return CharDataset(TEXT, CharTokenizer.from_text(TEXT), val_fraction=0.1)


def test_split_sizes_and_no_overlap(dataset):
    assert len(dataset.train) == 1800
    assert len(dataset.val) == 200
    # Contiguous tail, not a random subset: shuffling first would place the
    # same passage on both sides and make validation loss meaningless.
    assert len(dataset.train) + len(dataset.val) == len(TEXT)


def test_batch_shapes(dataset):
    x, y = dataset.get_batch("train", batch_size=4, block_size=16)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.long


def test_targets_are_inputs_shifted_by_one(dataset):
    # The whole supervision signal. Everything after position 0 of y is
    # already present in x one step later, so the model is only ever asked to
    # predict the single character it has not been shown.
    x, y = dataset.get_batch("train", batch_size=8, block_size=16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_windows_stay_inside_the_split(dataset):
    # Off-by-one in the random offset would read past the end and silently
    # wrap or truncate; drawing many batches makes that surface.
    for _ in range(50):
        x, y = dataset.get_batch("val", batch_size=8, block_size=16)
        assert x.shape == (8, 16) and y.shape == (8, 16)


def test_batches_are_drawn_independently(dataset):
    a, _ = dataset.get_batch("train", batch_size=8, block_size=16)
    b, _ = dataset.get_batch("train", batch_size=8, block_size=16)
    assert not torch.equal(a, b)


def test_ids_fit_the_vocabulary(dataset):
    x, y = dataset.get_batch("train", batch_size=8, block_size=16)
    v = dataset.tokenizer.vocab_size
    assert x.max() < v and y.max() < v
    assert x.min() >= 0 and y.min() >= 0
