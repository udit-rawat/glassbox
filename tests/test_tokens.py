"""Tests for the memory-mapped token dataset."""

import numpy as np
import pytest
import torch

from glassbox.tokenizer.char import CharTokenizer
from glassbox.training.tokens import MAX_VOCAB, TokenDataset, write_tokens

TEXT = "the cat sat on the mat. " * 200


@pytest.fixture
def paths(tmp_path):
    tok = CharTokenizer.from_text(TEXT)
    ids = tok.encode(TEXT)
    write_tokens(ids, tmp_path / "train.bin")
    write_tokens(ids[: len(ids) // 4], tmp_path / "val.bin")
    return tmp_path / "train.bin", tmp_path / "val.bin", tok


@pytest.fixture
def dataset(paths):
    torch.manual_seed(0)
    return TokenDataset(*paths)


def test_tokens_round_trip_through_the_file(paths):
    train_path, _, tok = paths
    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    # The file is the corpus. If this drifts, every batch is silently wrong.
    assert tok.decode(np.asarray(data).tolist()) == TEXT


def test_file_is_two_bytes_per_token(paths):
    train_path, _, tok = paths
    assert train_path.stat().st_size == 2 * len(tok.encode(TEXT))


def test_ids_too_large_for_uint16_are_refused(tmp_path):
    # Silently wrapping at 65,536 would map distinct tokens onto each other and
    # produce a corpus that trains without ever erroring.
    with pytest.raises(ValueError, match="uint16"):
        write_tokens([MAX_VOCAB], tmp_path / "bad.bin")


def test_batch_shapes_and_dtype(dataset):
    x, y = dataset.get_batch("train", batch_size=4, block_size=16)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    # int64 because the embedding lookup requires it, whatever the file holds.
    assert x.dtype == torch.long


def test_targets_are_inputs_shifted_by_one(dataset):
    x, y = dataset.get_batch("train", batch_size=8, block_size=16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batches_stay_inside_the_split(dataset):
    for _ in range(50):
        x, y = dataset.get_batch("val", batch_size=8, block_size=16)
        assert x.shape == (8, 16) and y.shape == (8, 16)


def test_a_split_shorter_than_the_context_fails_clearly(dataset):
    with pytest.raises(ValueError, match="block_size"):
        dataset.get_batch("val", batch_size=2, block_size=10_000)


def test_batches_are_drawn_independently(dataset):
    a, _ = dataset.get_batch("train", batch_size=8, block_size=16)
    b, _ = dataset.get_batch("train", batch_size=8, block_size=16)
    assert not torch.equal(a, b)


def test_one_seed_controls_both_model_and_data(paths):
    # The batch sampler uses torch's generator, not numpy's. Two generators
    # would mean a seeded run reproduced the weights but not the data they saw.
    torch.manual_seed(7)
    first, _ = TokenDataset(*paths).get_batch("train", 4, 16)
    torch.manual_seed(7)
    second, _ = TokenDataset(*paths).get_batch("train", 4, 16)
    assert torch.equal(first, second)


def test_dataset_carries_its_tokenizer(dataset, paths):
    # The training loop writes this into every checkpoint; without it the ids
    # decode to nothing.
    assert dataset.tokenizer is paths[2]


def test_memmap_does_not_load_the_file(dataset):
    # np.memmap keeps the data on disk until touched, which is the entire
    # reason this class exists rather than a tensor.
    assert isinstance(dataset.train, np.memmap)
    assert isinstance(dataset.val, np.memmap)
