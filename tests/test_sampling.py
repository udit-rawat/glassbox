"""Tests for generation and the sampling filters."""

import pytest
import torch

from glassbox.model import GPT, GPTConfig
from glassbox.sampling.generate import _top_k_filter, _top_p_filter, generate


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=17, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0
    )
    return GPT(config).eval()


@pytest.fixture
def prompt():
    return torch.zeros((2, 3), dtype=torch.long)


def test_generation_appends_exactly_the_requested_tokens(model, prompt):
    out = generate(model, prompt, max_new_tokens=7)
    assert out.shape == (2, 10)
    # The prompt is carried through unchanged rather than regenerated.
    assert torch.equal(out[:, :3], prompt)


def test_greedy_is_deterministic(model, prompt):
    a = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    b = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    assert torch.equal(a, b)


def test_top_k_of_one_matches_greedy(model, prompt):
    # Restricting to a single candidate leaves multinomial no choice, so it
    # must agree with argmax. A mismatch means the filter or the scatter that
    # restores vocabulary order is wrong.
    greedy = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    top1 = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=1)
    assert torch.equal(greedy, top1)


def test_generation_runs_past_the_context_window(model, prompt):
    # Learned positions stop at block_size, so the context has to be cropped
    # while generating. Asking for more tokens than the window is the case
    # that catches a missing crop.
    out = generate(model, prompt, max_new_tokens=model.config.block_size + 5)
    assert out.shape == (2, 3 + model.config.block_size + 5)


def test_generate_restores_training_mode(model, prompt):
    model.train()
    generate(model, prompt, max_new_tokens=3)
    # Leaving the model in eval mode would silently disable dropout for the
    # remainder of a training run.
    assert model.training


def test_top_k_keeps_exactly_k_candidates():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = _top_k_filter(logits, 2)
    assert torch.isfinite(filtered).sum() == 2
    # The survivors must be the two largest, in their original positions.
    assert torch.isfinite(filtered[0, 1]) and torch.isfinite(filtered[0, 4])


def test_top_p_keeps_the_smallest_sufficient_set():
    # Probabilities are 0.6, 0.3, 0.1 after softmax of these log values.
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    filtered = _top_p_filter(logits, 0.8)
    # 0.6 alone is under 0.8, so the second token is needed; the third is not.
    assert torch.isfinite(filtered[0, 0]) and torch.isfinite(filtered[0, 1])
    assert filtered[0, 2] == float("-inf")


def test_top_p_always_keeps_at_least_one_token():
    # One token already exceeds any threshold below its own mass. Comparing the
    # raw cumulative sum here would mask everything and leave nothing to sample.
    logits = torch.log(torch.tensor([[0.95, 0.03, 0.02]]))
    filtered = _top_p_filter(logits, 0.5)
    assert torch.isfinite(filtered).sum() >= 1
    assert torch.isfinite(filtered[0, 0])


def test_top_p_of_one_keeps_everything():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    filtered = _top_p_filter(logits, 1.0)
    assert torch.isfinite(filtered).all()


def test_filters_preserve_vocabulary_positions():
    # Both filters sort internally; this checks the values land back on the
    # tokens they came from rather than in sorted order.
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    assert torch.equal(_top_k_filter(logits, 3), logits)
    assert torch.equal(_top_p_filter(logits, 1.0), logits)
