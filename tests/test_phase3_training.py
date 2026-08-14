"""Tests for the Phase 3 training engineering: schedule, accumulation, precision, resume."""

import pytest
import torch

from glassbox.model import GPT, GPTConfig
from glassbox.tokenizer.char import CharTokenizer
from glassbox.training.data import CharDataset
from glassbox.training.loop import TrainConfig, train
from glassbox.training.precision import select_precision
from glassbox.training.schedule import cosine_with_warmup

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


# ------------------------------------------------------------- schedule


def test_warmup_ramps_from_near_zero_to_peak():
    lrs = [cosine_with_warmup(i, 1e-3, warmup_iters=100, max_iters=1000) for i in range(100)]
    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[-1] == pytest.approx(1e-3)
    # Strictly increasing: Adam's per-parameter steps are estimated from
    # statistics it barely has yet, so the first updates must stay small.
    assert all(b > a for a, b in zip(lrs, lrs[1:]))


def test_cosine_decays_to_the_floor():
    peak, ratio = 1e-3, 0.1
    end = cosine_with_warmup(1000, peak, 100, 1000, ratio)
    assert end == pytest.approx(peak * ratio)
    # Past the end it stays at the floor rather than turning back upward.
    assert cosine_with_warmup(5000, peak, 100, 1000, ratio) == pytest.approx(peak * ratio)


def test_cosine_holds_near_peak_before_decaying():
    peak = 1e-3
    quarter = cosine_with_warmup(325, peak, 100, 1000)
    half = cosine_with_warmup(550, peak, 100, 1000)
    # A quarter of the way through decay it should still be above the midpoint —
    # the curve is flat near the peak, which is the reason to prefer it.
    assert quarter > half > peak * 0.1
    assert quarter > peak * 0.75


def test_schedule_is_monotonic_after_warmup():
    lrs = [cosine_with_warmup(i, 1e-3, 100, 1000) for i in range(100, 1001)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))


# ------------------------------------------------------------- precision


def test_precision_disabled_yields_plain_fp32():
    p = select_precision(torch.device("cpu"), enabled=False)
    assert not p.enabled and p.describe() == "fp32"


def test_cpu_and_mps_fall_back_to_fp32():
    # Apple Silicon has autocast but no hardware bfloat16, and the long run
    # happens on a rented CUDA card anyway.
    for dev in ("cpu", "mps"):
        assert not select_precision(torch.device(dev), enabled=True).enabled


def test_scaler_is_only_used_with_float16():
    # bfloat16 carries float32's exponent range so it cannot overflow; fp16 can,
    # and needs the loss scaled up before the backward pass.
    from glassbox.training.precision import Precision

    assert not Precision("cuda", torch.bfloat16, False).use_scaler
    assert Precision("cuda", torch.float16, True).use_scaler


# ------------------------------------------------------- accumulation


def test_accumulation_matches_a_single_large_batch(setup):
    """Four micro-batches of 4 must produce the same gradient as one batch of 16."""
    model, dataset = setup
    torch.manual_seed(3)
    x, y = dataset.get_batch("train", 16, model.config.block_size)

    # One big batch.
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(x, targets=y)
    loss.backward()
    big = model.blocks[0].attn.q_proj.weight.grad.clone()

    # The same 16 rows, split into four accumulated micro-batches.
    model.zero_grad(set_to_none=True)
    for i in range(4):
        _, loss, _ = model(x[i * 4 : (i + 1) * 4], targets=y[i * 4 : (i + 1) * 4])
        (loss / 4).backward()
    accumulated = model.blocks[0].attn.q_proj.weight.grad.clone()

    # Without the division by grad_accum_steps this would be four times larger,
    # silently multiplying the effective learning rate.
    assert torch.allclose(big, accumulated, atol=1e-5)


def test_accumulation_runs_end_to_end(setup, tmp_path):
    model, dataset = setup
    cfg = TrainConfig(
        max_iters=20, batch_size=4, grad_accum_steps=4, learning_rate=1e-2,
        eval_interval=10, eval_iters=3, out_dir=str(tmp_path), amp=False,
    )
    train(model, dataset, cfg, torch.device("cpu"))
    assert cfg.history[-1][2] < cfg.history[0][2] + 1.0


# ------------------------------------------------------------- resume


def test_resume_continues_from_the_recorded_iteration(setup, tmp_path):
    model, dataset = setup
    common = dict(
        batch_size=8, learning_rate=1e-2, eval_interval=10, eval_iters=3,
        out_dir=str(tmp_path), amp=False, schedule="cosine", warmup_iters=5,
    )

    first = TrainConfig(max_iters=20, **common)
    train(model, dataset, first, torch.device("cpu"))

    ckpt = torch.load(tmp_path / "last.pt", weights_only=False)
    assert ckpt["iter"] == 20
    # Optimizer moments have to travel too: resuming with a fresh AdamW would
    # discard every per-parameter statistic and jolt the model.
    assert "optimizer" in ckpt and ckpt["optimizer"]["state"]

    second = TrainConfig(max_iters=40, resume=True, **common)
    train(model, dataset, second, torch.device("cpu"))
    # History carries across the restart rather than starting over.
    assert [h[0] for h in second.history] == [10, 20, 30, 40]


def test_resume_without_a_checkpoint_starts_clean(setup, tmp_path):
    model, dataset = setup
    cfg = TrainConfig(
        max_iters=10, batch_size=8, eval_interval=10, eval_iters=3,
        out_dir=str(tmp_path), amp=False, resume=True,
    )
    train(model, dataset, cfg, torch.device("cpu"))
    assert cfg.history[0][0] == 10


def test_checkpoint_describes_its_own_tokenizer(setup, tmp_path):
    model, dataset = setup
    cfg = TrainConfig(
        max_iters=10, batch_size=8, eval_interval=10, eval_iters=3,
        out_dir=str(tmp_path), amp=False,
    )
    train(model, dataset, cfg, torch.device("cpu"))
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ckpt["tokenizer"]["kind"] == "char"
    assert ckpt["tokenizer"]["chars"] == dataset.tokenizer.chars


def test_last_and_best_are_separate_files(setup, tmp_path):
    # best.pt exists for quality, last.pt for resumption. Conflating them means
    # a disconnect either loses progress or overwrites the best weights.
    model, dataset = setup
    cfg = TrainConfig(
        max_iters=10, batch_size=8, eval_interval=10, eval_iters=3,
        out_dir=str(tmp_path), amp=False,
    )
    train(model, dataset, cfg, torch.device("cpu"))
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
