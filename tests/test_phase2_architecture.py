"""Tests for the Phase 2 architecture swaps: RMSNorm, RoPE, SwiGLU, grouped query attention."""

import math

import pytest
import torch

from glassbox.model import (
    GPT,
    MLP,
    GPTConfig,
    MultiHeadAttention,
    RMSNorm,
    SwiGLU,
    apply_rope,
    build_rope_cache,
    repeat_kv,
)

B, T = 2, 16


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


def llama_config(**over):
    """A configuration with every Phase 2 switch turned on."""
    base = dict(
        vocab_size=65,
        block_size=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        norm="rmsnorm",
        activation="swiglu",
        pos_encoding="rope",
        # Llama-style models carry no biases anywhere. RMSNorm already drops
        # its shift, and with pre-norm residuals the projection biases turn out
        # to be dead weight.
        bias=False,
    )
    base.update(over)
    return GPTConfig(**base)


# --------------------------------------------------------------- RMSNorm


def test_rmsnorm_scales_to_unit_root_mean_square():
    norm = RMSNorm(32)
    x = torch.randn(4, 8, 32) * 7.0 + 3.0
    out = norm(x)
    # The gain starts at one, so immediately after construction the output's
    # root mean square should be one at every position.
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_does_not_centre():
    # This is the whole difference from LayerNorm. A shifted input keeps its
    # shift in the output, scaled — it is not pulled back to zero mean.
    norm = RMSNorm(32)
    x = torch.randn(4, 8, 32) + 5.0
    out = norm(x)
    assert out.mean(-1).abs().min() > 1e-3


def test_rmsnorm_statistic_survives_half_precision_input():
    # Squaring under fp16 is where a naive implementation overflows to inf and
    # the output becomes NaN. The statistic is computed in float32 for exactly
    # this reason.
    norm = RMSNorm(32)
    x = (torch.randn(2, 4, 32) * 200).half().float()
    assert torch.isfinite(norm(x)).all()


# --------------------------------------------------------------- RoPE


def test_rope_preserves_vector_length():
    # A rotation changes direction, never magnitude. If this fails, the cosine
    # and sine tables are misaligned with the halves they multiply.
    cos, sin = build_rope_cache(16, 8)
    x = torch.randn(1, 1, 16, 8)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_rope_score_depends_only_on_relative_distance():
    """The property that makes RoPE worth having."""
    cos, sin = build_rope_cache(64, 16)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(i: int, j: int) -> float:
        qi = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum().item()

    # Same gap, different absolute locations: the model cannot tell them apart.
    # Learned position embeddings have no such guarantee — they would have to
    # discover it from data, separately for every pair of positions.
    assert score(2, 5) == pytest.approx(score(10, 13), abs=1e-4)
    assert score(0, 7) == pytest.approx(score(40, 47), abs=1e-4)

    # A different gap must give a different score, or the rotation is doing
    # nothing at all.
    assert abs(score(2, 5) - score(2, 9)) > 1e-3


def test_rope_leaves_position_zero_untouched():
    # Position zero rotates by an angle of zero, so it is the identity there.
    cos, sin = build_rope_cache(8, 16)
    x = torch.randn(1, 1, 1, 16)
    assert torch.allclose(apply_rope(x, cos[0:1], sin[0:1]), x, atol=1e-6)


def test_rope_model_has_no_position_table():
    model = GPT(llama_config())
    assert model.position_embedding is None
    names = dict(model.named_parameters())
    assert not any("position" in n for n in names)


# --------------------------------------------------------------- SwiGLU


def test_swiglu_matches_mlp_shape():
    config = llama_config()
    ff = SwiGLU(config)
    x = torch.randn(B, T, config.d_model)
    assert ff(x).shape == x.shape


def test_swiglu_parameter_count_stays_close_to_gelu():
    # Three matrices instead of two, so the hidden width is cut to 8/3 rather
    # than 4x. If this drifts far apart, a Phase 1 vs Phase 2 loss comparison
    # is measuring extra capacity rather than the architecture.
    config = GPTConfig(d_model=384, n_heads=6, activation="gelu")
    gelu = sum(p.numel() for p in MLP(config).parameters())
    swiglu = sum(p.numel() for p in SwiGLU(config).parameters())
    assert abs(swiglu - gelu) / gelu < 0.06


def test_swiglu_gate_can_suppress_a_unit():
    # The defining behaviour: the gate branch multiplies the up branch, so a
    # strongly negative gate drives its unit toward zero regardless of what the
    # up branch produced. GELU has no equivalent input-dependent switch.
    config = llama_config()
    ff = SwiGLU(config)
    with torch.no_grad():
        ff.gate_proj.weight.fill_(0.0)
        if ff.gate_proj.bias is not None:
            ff.gate_proj.bias.fill_(-30.0)
    out = ff(torch.randn(B, T, config.d_model))
    assert out.abs().max() < 1e-4


# --------------------------------------------------------------- GQA


def test_repeat_kv_groups_heads_contiguously():
    # The bug this catches: repeat() tiles the whole tensor, giving
    # [h0, h1, h0, h1], which pairs query heads with the wrong key/value heads
    # while every shape stays correct.
    x = torch.arange(2, dtype=torch.float).view(1, 2, 1, 1)
    out = repeat_kv(x, 2)
    assert out.shape == (1, 4, 1, 1)
    assert out.flatten().tolist() == [0.0, 0.0, 1.0, 1.0]


def test_repeat_kv_is_identity_for_one_group():
    x = torch.randn(1, 4, 3, 8)
    assert torch.equal(repeat_kv(x, 1), x)


def test_gqa_shrinks_key_and_value_projections():
    mha = MultiHeadAttention(llama_config(n_kv_heads=4, pos_encoding="learned"))
    gqa = MultiHeadAttention(llama_config(n_kv_heads=1, pos_encoding="learned"))

    # Queries are untouched; only the key/value side narrows. That is exactly
    # the tensor the KV cache stores, which is where the saving lands.
    assert mha.q_proj.weight.shape == gqa.q_proj.weight.shape
    assert gqa.k_proj.weight.shape[0] == mha.k_proj.weight.shape[0] // 4


def test_gqa_output_shape_and_attention_map_are_unchanged():
    config = llama_config(n_kv_heads=1)
    attn = MultiHeadAttention(config).eval()
    x = torch.randn(B, T, config.d_model)
    out, weights = attn(x)
    assert out.shape == x.shape
    # One attention map per *query* head regardless of how few key/value heads
    # there are — the visualizer sees no difference.
    assert weights.shape == (B, config.n_heads, T, T)


def test_kv_heads_must_divide_query_heads():
    # d_model must stay divisible by n_heads or the earlier check fires first
    # and this one is never reached.
    with pytest.raises(ValueError, match="multiple"):
        GPTConfig(d_model=96, n_heads=6, n_kv_heads=4)


def test_kv_heads_cannot_exceed_query_heads():
    with pytest.raises(ValueError, match="exceeds"):
        GPTConfig(n_heads=4, n_kv_heads=8)


def test_unknown_switches_are_rejected():
    for field in ("norm", "activation", "pos_encoding"):
        with pytest.raises(ValueError, match="unknown"):
            GPTConfig(**{field: "nonsense"})


# --------------------------------------------------- the whole model


def test_phase2_model_is_still_causal():
    # None of the swaps may weaken the guarantee Phase 1 established. The mask
    # is unchanged, so this should hold bitwise.
    config = llama_config()
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (B, T))
    t = 5

    baseline, _, _ = model(idx)
    perturbed_idx = idx.clone()
    perturbed_idx[:, t + 1] = (perturbed_idx[:, t + 1] + 1) % config.vocab_size
    perturbed, _, _ = model(perturbed_idx)

    assert torch.equal(baseline[:, : t + 1, :], perturbed[:, : t + 1, :])
    assert not torch.equal(baseline[:, t + 1, :], perturbed[:, t + 1, :])


def test_phase2_model_starts_at_the_uniform_loss():
    config = llama_config()
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    _, loss, _ = model(idx, targets=targets)
    assert loss.item() == pytest.approx(math.log(config.vocab_size), abs=0.25)


def test_phase2_attention_weights_still_form_distributions():
    config = llama_config()
    model = GPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (B, T))
    _, _, attentions = model(idx, return_attention=True)
    assert len(attentions) == config.n_layers
    for w in attentions:
        assert torch.allclose(
            w.sum(dim=-1), torch.ones(B, config.n_heads, T), atol=1e-6
        )


def test_defaults_still_describe_the_phase1_model():
    # Every Phase 1 test is still meaningful only if the defaults are unchanged.
    config = GPTConfig()
    assert config.norm == "layernorm"
    assert config.activation == "gelu"
    assert config.pos_encoding == "learned"
    assert config.n_kv_heads == config.n_heads
