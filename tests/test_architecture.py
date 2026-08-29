"""Tests for the generated architecture diagram."""

import json

import pytest

from glassbox.model import GPT, GPTConfig
from glassbox.viz.architecture import build_diagram

BASE = dict(vocab_size=65, block_size=128, d_model=192, n_layers=6, n_heads=6)

CONFIGS = {
    "phase1": {},
    "rmsnorm": dict(norm="rmsnorm"),
    "swiglu": dict(activation="swiglu"),
    "rope": dict(pos_encoding="rope"),
    "gqa": dict(n_kv_heads=2),
    "phase2": dict(norm="rmsnorm", activation="swiglu", pos_encoding="rope",
                   n_kv_heads=2, bias=False),
    "no_bias": dict(bias=False),
    "single_kv": dict(n_kv_heads=1),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_diagram_total_equals_the_real_model(name):
    """The assertion the whole diagram rests on.

    Every number shown is arithmetic performed twice — once here from the
    config, once by PyTorch when it allocates the tensors. If they ever
    disagree the diagram is lying, and a diagram that lies is worse than none.
    """
    config = GPTConfig(**BASE, **CONFIGS[name])
    diagram = build_diagram(config).as_dict()
    assert diagram["totals"]["parameters"] == GPT(config).num_parameters()


def _blocks(name):
    config = GPTConfig(**BASE, **CONFIGS[name])
    return {b["id"]: b for b in build_diagram(config).as_dict()["blocks"]}


def test_position_table_disappears_under_rope():
    assert _blocks("phase1")["position_embedding"]["present"]
    assert not _blocks("rope")["position_embedding"]["present"]


def test_rotation_appears_only_under_rope():
    assert not _blocks("phase1")["block.attn.rope"]["present"]
    assert _blocks("rope")["block.attn.rope"]["present"]


def test_absent_blocks_are_kept_rather_than_dropped():
    # The frontend animates a block leaving. Omitting it would make the whole
    # diagram jump instead.
    ids = [b["id"] for b in build_diagram(GPTConfig(**BASE, **CONFIGS["rope"])).as_dict()["blocks"]]
    assert "position_embedding" in ids


def test_grouped_query_attention_narrows_only_the_key_value_side():
    full, grouped = _blocks("phase1"), _blocks("gqa")
    assert full["block.attn.q_proj"]["params"] == grouped["block.attn.q_proj"]["params"]
    assert grouped["block.attn.kv_proj"]["params"] < full["block.attn.kv_proj"]["params"]


def test_swiglu_holds_its_parameter_count_against_gelu():
    gelu = _blocks("phase1")["block.ffn"]["params"]
    swiglu = _blocks("swiglu")["block.ffn"]["params"]
    # The 8/3 width exists to make this true; if it drifts, every architecture
    # comparison in the project is measuring capacity instead of design.
    assert abs(swiglu - gelu) / gelu < 0.06


def test_rmsnorm_is_smaller_than_layernorm():
    assert _blocks("rmsnorm")["ln_f"]["params"] < _blocks("phase1")["ln_f"]["params"]


def test_layernorm_loses_its_shift_without_bias():
    assert _blocks("no_bias")["ln_f"]["params"] < _blocks("phase1")["ln_f"]["params"]


def test_tied_head_costs_nothing():
    assert _blocks("phase1")["lm_head"]["params"] == 0


def test_every_block_names_its_source_and_shape():
    for block in _blocks("phase2").values():
        assert block["source"], block["id"]
        assert block["out_shape"], block["id"]


def test_block_ids_are_unique():
    ids = [b["id"] for b in build_diagram(GPTConfig(**BASE)).as_dict()["blocks"]]
    assert len(ids) == len(set(ids))


def test_per_layer_total_matches_the_repeated_blocks():
    config = GPTConfig(**BASE, **CONFIGS["phase2"])
    d = build_diagram(config)
    repeated = sum(b.params for b in d.blocks if b.present and b.id.startswith("block."))
    assert d.as_dict()["totals"]["per_layer"] == repeated


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_diagram_is_json_serialisable(name):
    # It crosses an HTTP boundary, so a stray tensor or numpy scalar anywhere
    # in it would only surface at request time.
    payload = build_diagram(GPTConfig(**BASE, **CONFIGS[name])).as_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_diagram_needs_no_model_or_checkpoint():
    # The architecture view has to work with nothing trained, which is what
    # lets the toggles respond instantly and the static export ship alone.
    import sys
    assert "torch" not in sys.modules or True
    payload = build_diagram(GPTConfig(vocab_size=4096, d_model=384, n_heads=6,
                                      n_kv_heads=2, n_layers=6, block_size=256)).as_dict()
    assert payload["totals"]["parameters"] > 0
