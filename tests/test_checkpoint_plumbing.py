"""Tests for the bugs that let a run silently write to the wrong place."""

import pytest
import torch

from glassbox.model import GPT, GPTConfig
from glassbox.model.norm import RMSNorm, build_norm
from glassbox.tokenizer.bpe import BPETokenizer
from glassbox.tokenizer.char import CharTokenizer
from glassbox.tokenizer.spec import (
    load_tokenizer,
    load_tokenizer_from_checkpoint,
    tokenizer_spec,
)
from glassbox.training.data import CharDataset
from glassbox.training.loop import TrainConfig, train

TEXT = "abcabcabcabc" * 200


@pytest.fixture
def setup():
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(TEXT)
    dataset = CharDataset(TEXT, tokenizer, val_fraction=0.1)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=16, d_model=32,
        n_layers=2, n_heads=4, dropout=0.0,
    )
    return GPT(config), dataset


def _cfg(out_dir, **over):
    base = dict(
        max_iters=10, batch_size=8, eval_interval=10, eval_iters=3,
        amp=False, out_dir=str(out_dir),
    )
    base.update(over)
    return TrainConfig(**base)


# ------------------------------------------------- out_dir is honoured


def test_training_writes_only_to_the_requested_directory(setup, tmp_path):
    """The regression that cost a trained model.

    --out-dir was parsed and then dropped, so a run announced one destination
    and wrote to another — overwriting whatever was already sitting in the
    default directory. Asserting the requested path exists is only half a test;
    the half that would have caught it is asserting the default stays empty.
    """
    model, dataset = setup
    requested = tmp_path / "run_a"
    default = tmp_path / "checkpoints"
    default.mkdir()

    train(model, dataset, _cfg(requested), torch.device("cpu"))

    assert (requested / "best.pt").exists()
    assert (requested / "last.pt").exists()
    assert list(default.iterdir()) == []


def test_two_runs_into_different_directories_do_not_collide(setup, tmp_path):
    model, dataset = setup
    train(model, dataset, _cfg(tmp_path / "one"), torch.device("cpu"))
    train(model, dataset, _cfg(tmp_path / "two", seed=99), torch.device("cpu"))

    a = torch.load(tmp_path / "one" / "best.pt", weights_only=False)
    b = torch.load(tmp_path / "two" / "best.pt", weights_only=False)
    assert a["val_loss"] != b["val_loss"]


# ------------------------------------------------- tokenizer round trip


def test_char_tokenizer_survives_a_checkpoint():
    tok = CharTokenizer.from_text(TEXT)
    restored = load_tokenizer(tokenizer_spec(tok))
    assert isinstance(restored, CharTokenizer)
    assert restored.encode("abc") == tok.encode("abc")


def test_bpe_tokenizer_survives_a_checkpoint():
    tok = BPETokenizer.train("the cat sat on the mat. " * 40, vocab_size=300)
    restored = load_tokenizer(tokenizer_spec(tok))
    assert isinstance(restored, BPETokenizer)
    # Merge order is the tokenizer; a reordered list would still decode to the
    # right text while producing ids the model was never trained on.
    assert restored.merge_list == tok.merge_list
    assert restored.encode("the cat") == tok.encode("the cat")


def test_checkpoint_tokenizer_is_read_not_assumed(setup, tmp_path):
    # sample.py used to construct a CharTokenizer unconditionally, which throws
    # away the merges on any BPE checkpoint.
    model, dataset = setup
    train(model, dataset, _cfg(tmp_path), torch.device("cpu"))
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)

    restored = load_tokenizer_from_checkpoint(ckpt)
    assert restored.encode("abc") == dataset.tokenizer.encode("abc")


def test_legacy_checkpoint_with_only_chars_still_loads():
    restored = load_tokenizer_from_checkpoint({"chars": list("abc")})
    assert isinstance(restored, CharTokenizer)
    assert restored.vocab_size == 3


def test_checkpoint_without_a_tokenizer_is_rejected():
    with pytest.raises(ValueError, match="no tokenizer"):
        load_tokenizer_from_checkpoint({"model": {}})


def test_unknown_tokenizer_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown tokenizer kind"):
        load_tokenizer({"kind": "sentencepiece"})


def test_unsupported_tokenizer_cannot_be_saved_silently():
    class Homemade:
        pass

    # Better to refuse than to write a checkpoint whose ids decode to nothing.
    with pytest.raises(TypeError, match="no checkpoint representation"):
        tokenizer_spec(Homemade())


# ------------------------------------------------- small guards


def test_a_split_shorter_than_the_context_fails_clearly():
    tokenizer = CharTokenizer.from_text("abc")
    dataset = CharDataset("abc" * 10, tokenizer, val_fraction=0.1)
    with pytest.raises(ValueError, match="block_size"):
        dataset.get_batch("val", batch_size=2, block_size=64)


def test_rmsnorm_has_no_shift_regardless_of_the_bias_flag():
    # config.bias governs the linear projections. RMSNorm having no shift is
    # part of what RMSNorm is, so the flag must not reach it.
    norm = build_norm(GPTConfig(norm="rmsnorm", bias=True), 32)
    assert isinstance(norm, RMSNorm)
    assert not any("bias" in name for name, _ in norm.named_parameters())


def test_layernorm_still_follows_the_bias_flag():
    assert build_norm(GPTConfig(norm="layernorm", bias=True), 32).bias is not None
    assert build_norm(GPTConfig(norm="layernorm", bias=False), 32).bias is None
