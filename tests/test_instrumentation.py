"""Tests for the Phase 5 instrumentation: KV cache, hidden states, logit lens."""

import pytest
import torch

from glassbox.model import GPT, GPTConfig
from glassbox.model.cache import KVCache
from glassbox.sampling.generate import generate

B, T = 2, 12


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


def make(**over):
    base = dict(
        vocab_size=41, block_size=32, d_model=64, n_layers=3, n_heads=4,
        n_kv_heads=2, dropout=0.0, norm="rmsnorm", activation="swiglu",
        pos_encoding="rope", bias=False,
    )
    base.update(over)
    return GPT(GPTConfig(**base)).eval()


# ------------------------------------------------------------------ cache


@pytest.mark.parametrize("pos_encoding", ["rope", "learned"])
def test_cached_generation_is_identical_to_uncached(pos_encoding):
    """The test the whole cache rests on.

    A cache that is subtly wrong still produces fluent-looking text — the model
    simply attends to slightly the wrong history. Nothing crashes and nothing
    looks obviously off, which is how this bug survives for months. Demanding
    token-for-token equality with the uncached path is what makes it visible.
    """
    model = make(pos_encoding=pos_encoding)
    prompt = torch.randint(0, model.config.vocab_size, (B, 4))

    torch.manual_seed(11)
    cached = generate(model, prompt, 15, temperature=0.9, top_k=10, use_cache=True)
    torch.manual_seed(11)
    plain = generate(model, prompt, 15, temperature=0.9, top_k=10, use_cache=False)

    assert torch.equal(cached, plain)


def test_cached_greedy_matches_uncached_exactly():
    # Greedy removes sampling from the comparison, so any difference is the
    # cache rather than the random draw.
    model = make()
    prompt = torch.randint(0, model.config.vocab_size, (B, 5))
    a = generate(model, prompt, 12, temperature=0.0, use_cache=True)
    b = generate(model, prompt, 12, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_cache_grows_one_position_per_token():
    model = make()
    cache = KVCache(model.config.n_layers)
    idx = torch.randint(0, model.config.vocab_size, (1, 5))

    model(idx, cache=cache)
    assert cache.length == 5

    for expected in (6, 7, 8):
        model(idx[:, -1:], cache=cache)
        assert cache.length == expected


def test_cache_stores_the_narrow_key_value_heads():
    # The whole point of grouped query attention at generation time. Caching
    # after the expansion would store n_heads copies and throw the saving away.
    model = make(n_heads=4, n_kv_heads=1)
    cache = KVCache(model.config.n_layers)
    model(torch.randint(0, model.config.vocab_size, (1, 6)), cache=cache)
    assert cache[0].k.shape[1] == model.config.n_kv_heads == 1


def test_incremental_forward_matches_a_single_pass():
    # Feeding tokens one at a time through the cache must reproduce the logits
    # of one full forward over the same sequence.
    model = make()
    idx = torch.randint(0, model.config.vocab_size, (1, 8))

    full, _, _ = model(idx)

    cache = KVCache(model.config.n_layers)
    for t in range(idx.size(1)):
        step, _, _ = model(idx[:, t : t + 1], cache=cache)

    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-5)


def test_cache_survives_running_past_the_context_window():
    # RoPE bakes absolute positions into stored keys, so the window cannot
    # slide. Generation must fall back rather than produce nonsense.
    model = make(block_size=16)
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    out = generate(model, prompt, 20, temperature=0.0, use_cache=True)
    plain = generate(model, prompt, 20, temperature=0.0, use_cache=False)
    assert out.shape == (1, 24)
    assert torch.equal(out, plain)


def test_cache_reset_clears_every_layer():
    model = make()
    cache = KVCache(model.config.n_layers)
    model(torch.randint(0, model.config.vocab_size, (1, 6)), cache=cache)
    cache.reset()
    assert cache.length == 0
    assert all(layer.k is None for layer in cache.layers)


def test_cache_processes_one_token_per_step():
    """Measures the work rather than the clock.

    A timing assertion would be flaky at test sizes, where Python overhead
    swamps the saving. Counting the positions actually pushed through the
    embedding is deterministic and measures the thing that makes it faster.
    """
    model = make(block_size=64)
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))

    def count(use_cache):
        seen = []
        handle = model.token_embedding.register_forward_hook(
            lambda mod, inp, out: seen.append(inp[0].shape[1])
        )
        generate(model, prompt, 8, temperature=0.0, use_cache=use_cache)
        handle.remove()
        return seen

    cached = count(True)
    plain = count(False)

    # Prompt once, then a single token per step.
    assert cached == [4] + [1] * 7
    # Without the cache the whole prefix is re-read every step.
    assert plain == [4, 5, 6, 7, 8, 9, 10, 11]
    assert sum(cached) < sum(plain) / 5


# ---------------------------------------------------------- hidden states


def test_hidden_states_cover_the_embedding_and_every_block():
    model = make()
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    model(idx, return_hidden=True)
    hidden = model.hidden_states()

    # One per block, plus the embedding the stream starts from.
    assert len(hidden) == model.config.n_layers + 1
    for h in hidden:
        assert h.shape == (B, T, model.config.d_model)


def test_hidden_states_are_not_collected_unless_asked():
    model = make()
    model(torch.randint(0, model.config.vocab_size, (B, T)))
    with pytest.raises(RuntimeError, match="return_hidden"):
        model.hidden_states()


def test_the_last_hidden_state_produces_the_returned_logits():
    model = make()
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    logits, _, _ = model(idx, return_hidden=True)
    final = model.hidden_states()[-1]
    assert torch.allclose(model.lm_head(model.ln_f(final)), logits, atol=1e-5)


# ------------------------------------------------------------- logit lens


def test_logit_lens_reads_out_at_every_depth():
    model = make()
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    lens = model.logit_lens(idx)

    assert len(lens) == model.config.n_layers + 1
    for layer in lens:
        assert layer.shape == (B, T, model.config.vocab_size)


def test_the_deepest_lens_readout_equals_the_real_output():
    # The lens is the model's own head pointed at an earlier layer, so at the
    # last layer it must agree with the model exactly.
    model = make()
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    logits, _, _ = model(idx)
    assert torch.allclose(model.logit_lens(idx)[-1], logits, atol=1e-5)


def test_logit_lens_restores_training_mode():
    model = make()
    model.train()
    model.logit_lens(torch.randint(0, model.config.vocab_size, (B, T)))
    assert model.training


def test_lens_readouts_differ_between_layers():
    """Each block must change what the readout says, or depth buys nothing.

    Note this asserts the logits move, not that the argmax moves. On an
    untrained model the argmax is the *same* at every depth, and for a reason
    worth knowing: the blocks start with deliberately tiny contributions, so the
    residual stream is still dominated by the token embedding, and the tied head
    scores that embedding against itself. It is the copying bias from Phase 1,
    visible once more. On the trained checkpoint the argmax does move — the
    embedding predicts the current character, and layer 1 onward predicts the
    next one.
    """
    model = make()
    idx = torch.randint(0, model.config.vocab_size, (1, T))
    lens = model.logit_lens(idx)

    for shallow, deep in zip(lens, lens[1:]):
        assert not torch.allclose(shallow, deep, atol=1e-4)
