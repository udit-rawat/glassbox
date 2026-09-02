"""Tests for LoRA: the two properties that make it usable, and the guards."""

import pytest
import torch
import torch.nn as nn

from glassbox.finetune import (
    LoRALinear,
    apply_lora,
    load_adapter,
    lora_parameters,
    merge_lora,
    save_adapter,
    trainable_report,
)
from glassbox.model import GPT, GPTConfig


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def model():
    return GPT(GPTConfig(
        vocab_size=41, block_size=32, d_model=64, n_layers=3, n_heads=4,
        n_kv_heads=2, dropout=0.0, norm="rmsnorm", activation="swiglu",
        pos_encoding="rope", bias=False)).eval()


@pytest.fixture
def idx():
    return torch.randint(0, 41, (2, 12))


# ----------------------------------------------------------- the two properties


def test_attaching_an_adapter_changes_nothing(model, idx):
    """B starts at zero, so the wrapped model is bit-identical to the original.

    This is what makes an adapter safe to attach: before any training it cannot
    have altered the model it wraps. Initialising both matrices randomly would
    inject noise into a trained model the moment you touched it.
    """
    before, _, _ = model(idx)
    apply_lora(model, r=8, alpha=16)
    after, _, _ = model(idx)
    assert torch.equal(before, after)


def test_merged_equals_unmerged(model, idx):
    """Folding BA into W must change nothing observable.

    A merge that is subtly wrong still generates fluent text — the model simply
    behaves like a slightly different one. Nothing crashes, so only exact
    equality catches it.
    """
    apply_lora(model, r=4, alpha=8)
    with torch.no_grad():                       # give the adapter something to fold
        for p in lora_parameters(model):
            p.normal_(0, 0.05)

    unmerged, _, _ = model(idx)
    merged, _, _ = merge_lora(model)(idx)
    assert torch.allclose(unmerged, merged, atol=1e-5)


# ---------------------------------------------------------------- what trains


def test_only_the_adapter_is_trainable(model):
    apply_lora(model, r=8)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n for n in trainable)


def test_the_base_weights_stay_frozen(model, idx):
    apply_lora(model, r=8)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.normal_(0, 0.05)

    base = model.blocks[0].attn.q_proj.base.weight.clone()
    _, loss, _ = model(idx, targets=idx)
    loss.backward()
    assert model.blocks[0].attn.q_proj.base.weight.grad is None
    assert torch.equal(model.blocks[0].attn.q_proj.base.weight, base)
    assert model.blocks[0].attn.q_proj.lora_a.grad is not None


def test_the_trainable_fraction_is_small(model):
    apply_lora(model, r=8)
    report = trainable_report(model)
    # The whole point: a correction thin enough to be a rounding error against
    # the model it adapts.
    assert report["fraction"] < 0.05
    assert report["adapters"] == model.config.n_layers * 2   # q_proj and v_proj


def test_lora_can_still_learn_with_the_base_frozen(model):
    """Frozen weights and a falling loss — otherwise the adapter is decoration.

    Averaged over several steps at each end rather than compared first to last:
    a single step is noisy enough to fail on a bad draw even when the run is
    clearly learning.
    """
    apply_lora(model, r=8, alpha=16)
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=5e-3)

    torch.manual_seed(3)
    x = torch.randint(0, 41, (4, 12))
    y = torch.randint(0, 41, (4, 12))

    losses = []
    for _ in range(80):
        _, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


# ------------------------------------------------------------------- scaling


def _correction_size(r, alpha, x):
    base = nn.Linear(16, 16, bias=False)
    layer = LoRALinear(base, r=r, alpha=alpha)
    with torch.no_grad():
        layer.lora_a.fill_(0.1)
        layer.lora_b.fill_(0.1)
    assert layer.scaling == alpha / r
    return (layer(x) - base(x)).abs().mean().item()


def test_the_scaling_makes_the_correction_independent_of_rank():
    """Why LoRA divides by r at all.

    B @ A sums r terms, so the product grows with the rank. Dividing by r
    cancels that exactly, which is what lets you raise the rank to give the
    adapter more capacity without also raising how hard it pushes — otherwise
    changing r would silently change the effective learning rate too.
    """
    torch.manual_seed(1)
    x = torch.randn(3, 16)
    assert _correction_size(8, 8, x) == pytest.approx(_correction_size(4, 8, x), rel=1e-4)


def test_alpha_sets_how_hard_the_adapter_pushes():
    torch.manual_seed(1)
    x = torch.randn(3, 16)
    doubled = _correction_size(4, 16, x)
    assert doubled == pytest.approx(2 * _correction_size(4, 8, x), rel=1e-4)


def test_a_nonzero_adapter_actually_changes_the_output():
    base = nn.Linear(16, 16, bias=False)
    layer = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.equal(layer(x), base(x))
    with torch.no_grad():
        layer.lora_b.normal_(0, 0.1)
    assert not torch.equal(layer(x), base(x))


def test_bias_survives_the_merge():
    base = nn.Linear(12, 12, bias=True)
    with torch.no_grad():
        base.bias.normal_()
    layer = LoRALinear(base, r=4)
    merged = layer.merged()
    assert merged.bias is not None
    assert torch.equal(merged.bias, base.bias)


# -------------------------------------------------------------- saving state


def test_the_adapter_file_holds_only_the_correction(model, tmp_path):
    apply_lora(model, r=8)
    path = save_adapter(model, tmp_path / "adapter.pt", note="test")
    saved = torch.load(path, weights_only=False)

    assert all("lora_" in k for k in saved["lora"])
    assert saved["config"]["r"] == 8
    assert saved["note"] == "test"
    # Small enough to attach to an email, against a model that is not.
    full = sum(p.numel() for p in model.parameters()) * 4
    assert path.stat().st_size < full / 20


def test_an_adapter_round_trips_onto_a_fresh_model(tmp_path, idx):
    def build():
        torch.manual_seed(0)
        return GPT(GPTConfig(
            vocab_size=41, block_size=32, d_model=64, n_layers=3, n_heads=4,
            n_kv_heads=2, dropout=0.0, norm="rmsnorm", activation="swiglu",
            pos_encoding="rope", bias=False)).eval()

    trained = build()
    apply_lora(trained, r=8)
    with torch.no_grad():
        for p in lora_parameters(trained):
            p.normal_(0, 0.05)
    expected, _, _ = trained(idx)
    save_adapter(trained, tmp_path / "adapter.pt")

    restored = load_adapter(build(), tmp_path / "adapter.pt")
    actual, _, _ = restored(idx)
    assert torch.allclose(expected, actual, atol=1e-6)


# -------------------------------------------------------------------- guards


def test_matching_nothing_is_an_error(model):
    # Otherwise the model trains with nothing trainable in it, which looks
    # exactly like a run that simply refuses to learn.
    with pytest.raises(ValueError, match="nothing would be trainable"):
        apply_lora(model, targets=("does_not_exist",))


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(8, 8), r=0)


def test_merging_leaves_no_adapters_behind(model):
    apply_lora(model, r=4)
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert isinstance(model.blocks[0].attn.q_proj, nn.Linear)


def test_targets_can_be_chosen(model):
    apply_lora(model, targets=("q_proj",), r=4)
    assert trainable_report(model)["adapters"] == model.config.n_layers
