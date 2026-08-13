"""Tests for the training loop — that it actually learns, and leaves state intact."""

import pytest
import torch

from glassbox.model import GPT, GPTConfig
from glassbox.tokenizer.char import CharTokenizer
from glassbox.training.data import CharDataset
from glassbox.training.loop import TrainConfig, estimate_loss, train

# A short repeating cycle: learnable within a few dozen iterations, so the test
# runs in seconds while still requiring the optimizer to genuinely work.
TEXT = "abcabcabcabc" * 200


@pytest.fixture
def setup():
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(TEXT)
    dataset = CharDataset(TEXT, tokenizer, val_fraction=0.1)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )
    return GPT(config), dataset


def test_training_reduces_loss(setup, tmp_path):
    model, dataset = setup
    cfg = TrainConfig(
        max_iters=120,
        batch_size=16,
        learning_rate=1e-2,
        eval_interval=60,
        eval_iters=10,
        out_dir=str(tmp_path),
    )

    # Baseline is taken before the first step. Comparing two mid-training
    # evaluations does not work here: a three-character cycle is learned within
    # a few dozen iterations, so both points sit at convergence and their
    # difference is sampling noise rather than progress.
    before = estimate_loss(model, dataset, cfg, torch.device("cpu"))["val"]
    train(model, dataset, cfg, torch.device("cpu"))
    after = cfg.history[-1][2]

    assert after < before
    # An untrained model sits near ln(vocab_size); a cycle this regular should
    # end far below it. A loop with the optimizer disconnected would not move.
    assert after < 0.5


def test_best_checkpoint_is_written_and_reloadable(setup, tmp_path):
    model, dataset = setup
    cfg = TrainConfig(
        max_iters=60,
        batch_size=16,
        learning_rate=1e-2,
        eval_interval=30,
        eval_iters=5,
        out_dir=str(tmp_path),
    )
    train(model, dataset, cfg, torch.device("cpu"))

    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    # Everything needed to reconstruct the model must travel with the weights;
    # a checkpoint that needs the training script to interpret it is a trap.
    assert {"model", "model_config", "chars", "iter", "val_loss"} <= ckpt.keys()

    restored = GPT(ckpt["model_config"])
    restored.load_state_dict(ckpt["model"])


def test_evaluation_restores_training_mode(setup):
    model, dataset = setup
    model.train()
    cfg = TrainConfig(eval_iters=2, batch_size=4)
    estimate_loss(model, dataset, cfg, torch.device("cpu"))
    # Evaluation disables dropout; failing to restore would silently switch it
    # off for the remainder of the run.
    assert model.training


def test_weight_decay_skips_biases_and_norms(setup):
    model, _ = setup
    decayed = sum(1 for p in model.parameters() if p.dim() >= 2)
    one_dim = sum(1 for p in model.parameters() if p.dim() < 2)
    # Decaying a LayerNorm gain fights the normalization it performs, and
    # decaying a bias drags the layer's whole output toward zero.
    assert decayed > 0 and one_dim > 0
