"""Tests for stripping a training checkpoint down to an inference one."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_checkpoint import export  # noqa: E402

from glassbox.model import GPT, GPTConfig
from glassbox.tokenizer.char import CharTokenizer
from glassbox.training.data import CharDataset
from glassbox.training.loop import TrainConfig, train
from glassbox.tokenizer.spec import load_tokenizer_from_checkpoint

TEXT = "abcabcabcabc" * 200


@pytest.fixture
def trained(tmp_path):
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(TEXT)
    dataset = CharDataset(TEXT, tokenizer, val_fraction=0.1)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=16, d_model=32,
        n_layers=2, n_heads=4, dropout=0.0,
    )
    model = GPT(config)
    cfg = TrainConfig(max_iters=10, batch_size=8, eval_interval=10, eval_iters=3,
                      amp=False, out_dir=str(tmp_path))
    train(model, dataset, cfg, torch.device("cpu"))
    return tmp_path / "best.pt"


def test_optimizer_state_is_dropped(trained, tmp_path):
    dst = tmp_path / "slim.pt"
    before, after = export(trained, dst)

    slim = torch.load(dst, map_location="cpu", weights_only=False)
    assert "optimizer" not in slim
    assert "scaler" not in slim
    assert after < before


def test_exported_model_produces_identical_logits(trained, tmp_path):
    """The assertion that matters: stripping must change nothing observable."""
    full = torch.load(trained, map_location="cpu", weights_only=False)
    dst = tmp_path / "slim.pt"
    export(trained, dst)
    slim = torch.load(dst, map_location="cpu", weights_only=False)

    idx = torch.randint(0, full["model_config"].vocab_size, (2, 8))

    a = GPT(full["model_config"])
    a.load_state_dict(full["model"])
    b = GPT(slim["model_config"])
    b.load_state_dict(slim["model"])

    with torch.no_grad():
        assert torch.equal(a.eval()(idx)[0], b.eval()(idx)[0])


def test_everything_needed_to_rebuild_survives(trained, tmp_path):
    dst = tmp_path / "slim.pt"
    export(trained, dst)
    slim = torch.load(dst, map_location="cpu", weights_only=False)

    # Config and tokenizer stay however small the file needs to be — a
    # checkpoint that cannot reconstruct itself is worse than a larger one.
    model = GPT(slim["model_config"])
    model.load_state_dict(slim["model"])
    tokenizer = load_tokenizer_from_checkpoint(slim)
    assert tokenizer.decode(tokenizer.encode("abc")) == "abc"
    assert slim["iter"] == 10


def test_half_precision_export_is_smaller_and_still_loads(trained, tmp_path):
    plain, half = tmp_path / "plain.pt", tmp_path / "half.pt"
    _, plain_size = export(trained, plain)
    _, half_size = export(trained, half, half=True)
    assert half_size < plain_size

    slim = torch.load(half, map_location="cpu", weights_only=False)
    assert slim["model"]["token_embedding.weight"].dtype is torch.float16
    # Widened back on load, so the model itself is unchanged in structure.
    model = GPT(slim["model_config"])
    model.load_state_dict({k: v.float() for k, v in slim["model"].items()})


def test_export_records_its_source(trained, tmp_path):
    dst = tmp_path / "slim.pt"
    export(trained, dst)
    slim = torch.load(dst, map_location="cpu", weights_only=False)
    assert str(trained) in slim["exported_from"]
