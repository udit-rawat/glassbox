"""End-to-end tests for the assembled decoder."""

import math

import pytest
import torch

from glassbox.model import GPT, GPTConfig

B, T = 2, 16


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def config():
    return GPTConfig(
        vocab_size=65, block_size=32, d_model=64, n_layers=3, n_heads=4, dropout=0.0
    )


@pytest.fixture
def model(config):
    return GPT(config).eval()


def tokens(config, seq_len=T):
    return torch.randint(0, config.vocab_size, (B, seq_len))


def test_forward_produces_logits_per_position(model, config):
    logits, loss, attentions = model(tokens(config))
    # One distribution over the vocabulary at every position, not just the last:
    # a decoder-only model trains on T predictions per sequence.
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None
    assert attentions is None


def test_initial_loss_matches_a_uniform_guess(config):
    # An untrained model has no reason to prefer any token, so cross-entropy
    # should sit at ln(vocab_size) — 4.17 for a vocabulary of 65. Landing far
    # off means initialisation scale or the loss reduction is wrong, and
    # catching it here costs a second instead of a training run.
    #
    # Targets are drawn independently of the inputs. Scoring the input against
    # itself measures something else entirely; see the test below.
    torch.manual_seed(0)
    model = GPT(config).eval()
    idx, targets = tokens(config), tokens(config)
    _, loss, _ = model(idx, targets=targets)
    assert loss.item() == pytest.approx(math.log(config.vocab_size), abs=0.25)


def test_tied_weights_bias_an_untrained_model_toward_copying(config):
    # Measured, not assumed: asking the untrained model to predict its own
    # input scores 3.48 against a uniform baseline of 4.17. The embedding a
    # token enters with survives along the residual stream, and the tied head
    # scores that stream against the same embedding matrix — so the largest
    # logit is the token already present. The model starts as a copier.
    #
    # Harmless, and arguably a good starting point, but it makes self-scored
    # loss useless as a wiring check, which is why the test above uses
    # independent targets.
    torch.manual_seed(0)
    model = GPT(config).eval()
    idx = tokens(config)
    _, self_loss, _ = model(idx, targets=idx)
    assert self_loss.item() < math.log(config.vocab_size) - 0.3


def test_attention_is_returned_for_every_layer(model, config):
    _, _, attentions = model(tokens(config), return_attention=True)
    assert len(attentions) == config.n_layers
    for weights in attentions:
        assert weights.shape == (B, config.n_heads, T, T)
        assert torch.allclose(
            weights.sum(dim=-1), torch.ones(B, config.n_heads, T), atol=1e-6
        )


def test_model_is_causal_end_to_end(model, config):
    idx = tokens(config)
    t = 5

    baseline, _, _ = model(idx)
    perturbed_idx = idx.clone()
    # Swap the token at t+1 for a different one. Everything at or before t was
    # computed without access to it and must be untouched.
    perturbed_idx[:, t + 1] = (perturbed_idx[:, t + 1] + 1) % config.vocab_size
    perturbed, _, _ = model(perturbed_idx)

    assert torch.equal(baseline[:, : t + 1, :], perturbed[:, : t + 1, :])
    assert not torch.equal(baseline[:, t + 1, :], perturbed[:, t + 1, :])


def test_sequences_beyond_block_size_are_refused(model, config):
    # Learned position embeddings exist only up to block_size; indexing past
    # the table should fail loudly rather than silently truncating.
    with pytest.raises(ValueError, match="block_size"):
        model(tokens(config, seq_len=config.block_size + 1))


def test_lm_head_is_tied_to_the_embedding(model):
    assert model.lm_head.weight is model.token_embedding.weight


def test_loss_gradient_reaches_the_first_block(config):
    model = GPT(config)
    idx = tokens(config)
    _, loss, _ = model(idx, targets=idx)
    loss.backward()
    # A residual path that is broken anywhere shows up as a None or all-zero
    # gradient at the earliest parameter in the stack.
    grad = model.blocks[0].attn.q_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert (grad != 0).any()


def test_shorter_sequences_are_accepted(model, config):
    # The causal mask is built once at block_size and sliced per call, so a
    # partial context must work without rebuilding anything.
    logits, _, _ = model(tokens(config, seq_len=1))
    assert logits.shape == (B, 1, config.vocab_size)
