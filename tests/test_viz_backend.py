"""Tests for the visualizer backend: head stats, inspection, registry, routing."""

import json

import pytest
import torch

from glassbox.model import GPT, GPTConfig
from glassbox.tokenizer.char import CharTokenizer
from glassbox.viz.heads import describe, head_stats
from glassbox.viz.inspect import display_text, inspect
from glassbox.viz.registry import ModelRegistry, discover
from glassbox.viz.server import Router

TEXT = "hello world\nthis is a tiny corpus for tests"


def _config(**over):
    base = dict(vocab_size=0, block_size=32, d_model=32, n_layers=2, n_heads=4,
                n_kv_heads=2, dropout=0.0, norm="rmsnorm", activation="swiglu",
                pos_encoding="rope", bias=False)
    base.update(over)
    return base


@pytest.fixture
def model_and_tokenizer():
    torch.manual_seed(0)
    tok = CharTokenizer.from_text(TEXT)
    return GPT(GPTConfig(**_config(vocab_size=tok.vocab_size))).eval(), tok


@pytest.fixture
def checkpoints(tmp_path):
    """Two tiny checkpoints on disk, so nothing depends on trained artifacts."""
    torch.manual_seed(0)
    tok = CharTokenizer.from_text(TEXT)
    root = tmp_path / "models"
    root.mkdir()
    names = []
    for name, over in [("alpha", {}), ("beta", dict(pos_encoding="learned"))]:
        cfg = GPTConfig(**_config(vocab_size=tok.vocab_size, **over))
        torch.save({"model": GPT(cfg).state_dict(), "model_config": cfg,
                    "tokenizer": {"kind": "char", "chars": tok.chars},
                    "iter": 10, "val_loss": 1.5, "history": [[10, 2.0, 2.1]]},
                   root / f"{name}.pt")
        names.append(name)
    (root / "results.json").write_text(json.dumps(
        {"variants": [{"name": n, "label": f"{n} label"} for n in names]}))
    return root


# ----------------------------------------------------------------- heads


def test_previous_token_head_is_recognised():
    T = 8
    w = torch.zeros(T, T); w[0, 0] = 1
    for i in range(1, T):
        w[i, i - 1] = 1
    stats = head_stats(w)
    assert stats["previous_token"] == 1.0
    assert stats["entropy"] == 0.0
    assert describe(stats) == "previous token"


def test_self_attending_head_is_recognised():
    stats = head_stats(torch.eye(8))
    assert stats["diagonal"] == 1.0
    assert stats["mean_distance"] == 0.0
    assert describe(stats) == "self"


def test_uniform_head_has_the_highest_entropy():
    T = 8
    uniform = torch.tril(torch.ones(T, T))
    uniform = uniform / uniform.sum(-1, keepdim=True)
    focused = torch.eye(T)
    assert head_stats(uniform)["entropy"] > head_stats(focused)["entropy"]


def test_distance_ignores_the_masked_upper_triangle():
    # Masked positions carry exactly zero weight, but a negative distance there
    # would still drag the mean if it were not clamped.
    T = 6
    w = torch.zeros(T, T); w[:, 0] = 1
    assert head_stats(w)["mean_distance"] == pytest.approx((0 + 1 + 2 + 3 + 4 + 5) / T)


def test_empty_attention_is_rejected():
    with pytest.raises(ValueError):
        head_stats(torch.zeros(0, 0))


# --------------------------------------------------------------- inspect


def test_whitespace_is_given_visible_glyphs():
    # A heatmap axis labelled with a real newline is a blank row.
    assert display_text(" ") == "␣"
    assert display_text("\n") == "⏎"
    assert display_text(" the") == "␣the"


def test_inspection_shapes_match_the_config(model_and_tokenizer):
    model, tok = model_and_tokenizer
    d = inspect(model, tok, "hello", top_k=5)

    assert d["n_tokens"] == len(tok.encode("hello"))
    assert len(d["attention"]) == model.config.n_layers
    assert len(d["attention"][0]) == model.config.n_heads
    assert len(d["attention"][0][0]) == d["n_tokens"]
    assert len(d["heads"]) == model.config.n_layers * model.config.n_heads
    # One readout per block, plus the embedding the stream starts from.
    assert len(d["lens"]) == model.config.n_layers + 1


def test_attention_rows_still_sum_to_one_after_rounding(model_and_tokenizer):
    model, tok = model_and_tokenizer
    d = inspect(model, tok, "hello world")
    for layer in d["attention"]:
        for head in layer:
            for row in head:
                assert sum(row) == pytest.approx(1.0, abs=1e-3)


def test_deepest_lens_readout_is_the_models_real_prediction(model_and_tokenizer):
    model, tok = model_and_tokenizer
    d = inspect(model, tok, "hello", top_k=3)
    logits, _, _ = model(torch.tensor([tok.encode("hello")]))
    expected = logits[0, -1].argmax().item()
    assert d["lens"][-1]["positions"][-1][0]["id"] == expected


def test_inspection_is_json_serialisable(model_and_tokenizer):
    # It crosses an HTTP boundary, so a stray tensor would only surface at
    # request time.
    model, tok = model_and_tokenizer
    payload = inspect(model, tok, "hello world")
    assert json.loads(json.dumps(payload)) == payload


def test_inspection_restores_training_mode(model_and_tokenizer):
    model, tok = model_and_tokenizer
    model.train()
    inspect(model, tok, "hello")
    assert model.training


def test_empty_and_oversized_prompts_are_rejected(model_and_tokenizer):
    model, tok = model_and_tokenizer
    with pytest.raises(ValueError, match="empty"):
        inspect(model, tok, "")
    with pytest.raises(ValueError, match="block_size"):
        inspect(model, tok, "h" * (model.config.block_size + 1))


# -------------------------------------------------------------- registry


def test_discovery_reads_labels_and_order_from_results(checkpoints):
    reg = discover(checkpoints)
    assert reg.names() == ["alpha", "beta"]
    assert reg.describe()[0]["label"] == "alpha label"


def test_registry_loads_lazily_and_caches(checkpoints):
    reg = discover(checkpoints)
    assert reg._loaded == {}          # startup read descriptions, not weights
    first = reg.get("alpha")
    assert reg.get("alpha") is first  # second call reuses the loaded model


def test_unknown_model_names_the_alternatives(checkpoints):
    with pytest.raises(KeyError, match="alpha"):
        discover(checkpoints).get("nonexistent")


def test_missing_directories_are_skipped(tmp_path):
    assert len(discover(tmp_path / "nope")) == 0


# ---------------------------------------------------------------- router


@pytest.fixture
def router(checkpoints):
    return Router(discover(checkpoints), default="alpha")


def call(router, method, path, body=None):
    status, kind, data = router.handle(
        method, path, json.dumps(body).encode() if body is not None else b"")
    return status, json.loads(data) if kind == "application/json" else data


def test_meta_lists_every_model(router):
    status, payload = call(router, "GET", "/meta")
    assert status == 200
    assert [m["name"] for m in payload["models"]] == ["alpha", "beta"]
    assert payload["default"] == "alpha"


def test_architecture_needs_no_model(checkpoints):
    # The toggles must respond before anything is loaded, and the static export
    # has to work with no checkpoints at all.
    empty = Router(ModelRegistry([]))
    status, payload = call(empty, "POST", "/architecture",
                           {"config": {"d_model": 64, "n_heads": 4, "n_layers": 2}})
    assert status == 200 and payload["totals"]["parameters"] > 0


def test_toggling_rope_removes_the_position_table(router):
    _, with_rope = call(router, "POST", "/architecture", {"model": "alpha"})
    _, learned = call(router, "POST", "/architecture",
                      {"model": "alpha", "config": {"pos_encoding": "learned"}})
    cfg = with_rope["config"]
    assert learned["totals"]["parameters"] - with_rope["totals"]["parameters"] == (
        cfg["block_size"] * cfg["d_model"])


def test_unknown_config_fields_are_refused_not_ignored(router):
    status, payload = call(router, "POST", "/architecture", {"config": {"bogus": 1}})
    assert status == 400 and "bogus" in payload["error"]


def test_model_is_a_selector_not_a_config_field(router):
    # Regression: with the fields at the top level, "model" was being validated
    # as a config key and rejected.
    assert call(router, "POST", "/architecture", {"model": "beta"})[0] == 200


def test_inspect_and_generate_round_trip(router):
    status, payload = call(router, "POST", "/inspect",
                           {"model": "alpha", "prompt": "hello"})
    assert status == 200 and payload["model"] == "alpha"

    status, payload = call(router, "POST", "/generate",
                           {"model": "alpha", "prompt": "hello", "max_tokens": 5})
    assert status == 200
    assert payload["text"].startswith("hello")
    assert payload["completion"] == payload["text"][len("hello"):]


def test_generation_is_reproducible_from_a_seed(router):
    body = {"model": "alpha", "prompt": "hello", "max_tokens": 8, "seed": 3}
    assert call(router, "POST", "/generate", body)[1]["text"] == \
           call(router, "POST", "/generate", body)[1]["text"]


def test_missing_prompt_is_a_client_error(router):
    assert call(router, "POST", "/inspect", {})[0] == 400
    assert call(router, "POST", "/generate", {})[0] == 400


def test_unknown_routes_and_methods(router):
    assert call(router, "GET", "/nope")[0] == 404
    assert call(router, "POST", "/nope", {})[0] == 404
    assert call(router, "PUT", "/meta")[0] == 405


def test_malformed_json_is_reported_not_raised(router):
    status, _, _ = router.handle("POST", "/inspect", b"{not json")
    assert status == 400


def test_static_paths_cannot_escape_the_static_directory(router):
    # Localhost-only is not a reason to leave a traversal hole open.
    status, _ = call(router, "GET", "/static/../../../../etc/passwd")
    assert status in (403, 404)


def test_query_strings_do_not_break_routing(router):
    # Deep links carry state for the page in the query string; the router must
    # look past it rather than treating it as part of the path.
    assert call(router, "GET", "/meta?anything=1")[0] == 200
    status, _, _ = router.handle("GET", "/?view=activations")
    assert status in (200, 404)      # 404 only when no frontend is built
    assert router.handle("GET", "/?view=activations")[0] == \
           router.handle("GET", "/")[0]
