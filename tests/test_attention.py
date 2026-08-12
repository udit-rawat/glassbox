"""Behavioural tests for attention: the invariants, then causality."""

import pytest
import torch

from glassbox.model.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from glassbox.model.config import GPTConfig

B, H, T, D = 2, 3, 6, 8


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


def qkv():
    return (torch.randn(B, H, T, D) for _ in range(3))


def test_output_shape_follows_values():
    q, k, v = qkv()
    out, weights = scaled_dot_product_attention(q, k, v)
    # Attention returns a re-mixture of v, so it inherits v's shape regardless
    # of how many keys were scored.
    assert out.shape == (B, H, T, D)
    assert weights.shape == (B, H, T, T)


def test_weights_form_a_distribution_per_query():
    q, k, v = qkv()
    _, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)
    assert (weights >= 0).all()


def test_masked_positions_receive_exactly_zero_weight():
    q, k, v = qkv()
    mask = causal_mask(T)
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    # Exactly zero, not merely small: exp(-inf) is 0, so the upper triangle
    # contributes nothing that could accumulate across layers.
    assert (weights[..., ~mask] == 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)


def test_future_values_do_not_reach_earlier_outputs():
    q, k, v = qkv()
    mask = causal_mask(T)
    out, _ = scaled_dot_product_attention(q, k, v, mask=mask)

    # Rewriting v at the last position changes what a query would read there.
    # No earlier query is allowed to have read it.
    v_perturbed = v.clone()
    v_perturbed[:, :, -1, :] = torch.randn(B, H, D)
    out_perturbed, _ = scaled_dot_product_attention(q, k, v_perturbed, mask=mask)

    assert torch.equal(out[:, :, :-1, :], out_perturbed[:, :, :-1, :])
    assert not torch.equal(out[:, :, -1, :], out_perturbed[:, :, -1, :])


def test_uniform_attention_averages_values():
    # Zeroed queries and keys make every score 0, so softmax is exactly uniform
    # and the output must equal the mean of the visible values. A closed-form
    # case the implementation cannot fake.
    q = torch.zeros(1, 1, T, D)
    k = torch.zeros(1, 1, T, D)
    v = torch.randn(1, 1, T, D)
    out, _ = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, v.mean(dim=-2, keepdim=True).expand_as(out), atol=1e-6)


def test_multihead_preserves_shape_and_exposes_every_head():
    config = GPTConfig(d_model=32, n_heads=4, block_size=T, dropout=0.0)
    attn = MultiHeadAttention(config).eval()
    x = torch.randn(B, T, config.d_model)

    out, weights = attn(x)
    assert out.shape == x.shape
    # Per head, not averaged — the visualizer reads this tensor directly.
    assert weights.shape == (B, config.n_heads, T, T)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, config.n_heads, T), atol=1e-6)


def test_multihead_is_causal_under_perturbation():
    config = GPTConfig(d_model=32, n_heads=4, block_size=T, dropout=0.0)
    attn = MultiHeadAttention(config).eval()
    x = torch.randn(B, T, config.d_model)
    t = 3

    baseline, _ = attn(x)
    perturbed_input = x.clone()
    perturbed_input[:, t + 1, :] = torch.randn(B, config.d_model)
    perturbed, _ = attn(perturbed_input)

    # Bitwise identity is the claim, not approximate agreement: masked weights
    # are exact zeros, and adding 0.0 into the value mixture is exact in
    # floating point. Any drift here means information leaked backwards.
    assert torch.equal(baseline[:, : t + 1, :], perturbed[:, : t + 1, :])
    assert not torch.equal(baseline[:, t + 1, :], perturbed[:, t + 1, :])


def test_causal_mask_always_admits_self():
    mask = causal_mask(T)
    # A fully masked row would softmax to NaN rather than raising, so the
    # diagonal being set is load-bearing rather than incidental.
    assert mask.diagonal().all()
    assert mask.sum() == T * (T + 1) // 2


def test_head_count_must_divide_width():
    with pytest.raises(ValueError):
        GPTConfig(d_model=30, n_heads=4)
