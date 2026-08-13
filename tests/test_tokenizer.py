"""Tests for the character tokenizer."""

import pytest

from glassbox.tokenizer.char import CharTokenizer

TEXT = "To be, or not to be: that is the question.\n"


def test_roundtrip_is_lossless():
    tok = CharTokenizer.from_text(TEXT)
    # Character-level tokenization has no lossy steps, so this is exact
    # equality rather than approximate agreement.
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_is_the_distinct_characters():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.vocab_size == len(set(TEXT))
    assert tok.chars == sorted(set(TEXT))


def test_ids_are_a_contiguous_range():
    tok = CharTokenizer.from_text(TEXT)
    # The embedding table is allocated at vocab_size rows, so any id at or
    # above it would index out of bounds at the first forward pass.
    assert sorted(tok.stoi.values()) == list(range(tok.vocab_size))
    assert max(tok.encode(TEXT)) < tok.vocab_size


def test_vocabulary_order_is_stable_across_instances():
    # Built from an unordered set, so without the sort inside __init__ the ids
    # could differ between runs and a checkpoint would decode to noise.
    a = CharTokenizer.from_text(TEXT)
    b = CharTokenizer.from_text(TEXT[::-1])
    assert a.chars == b.chars
    assert a.encode("be") == b.encode("be")


def test_save_and_load_preserve_the_mapping(tmp_path):
    tok = CharTokenizer.from_text(TEXT)
    tok.save(tmp_path / "tok.json")
    loaded = CharTokenizer.load(tmp_path / "tok.json")
    assert loaded.chars == tok.chars
    assert loaded.encode(TEXT) == tok.encode(TEXT)


def test_unknown_character_raises():
    tok = CharTokenizer.from_text("abc")
    # Loud failure is wanted: an unseen character means the wrong tokenizer was
    # paired with the text, and silently substituting something would hide it.
    with pytest.raises(KeyError):
        tok.encode("z")
