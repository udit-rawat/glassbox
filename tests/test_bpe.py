"""Tests for the from-scratch byte-pair encoder."""

import pytest

from glassbox.tokenizer.bpe import SPLIT_PATTERN, BPETokenizer

CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "a dog sat on a log. the dog ate the frog. "
) * 40


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=320)


def test_roundtrip_is_lossless(tok):
    for text in [CORPUS, "the cat", "unseen words entirely", "", "!!!", "123 456"]:
        assert tok.decode(tok.encode(text)) == text


def test_unseen_text_still_encodes():
    # Byte level means there is no unknown token and no way to fail: anything
    # not covered by a merge falls back to its raw bytes.
    tok = BPETokenizer.train("aaaa bbbb", vocab_size=260)
    weird = "zqx 日本語 \t\n"
    assert tok.decode(tok.encode(weird)) == weird


def test_merges_shorten_the_sequence(tok):
    raw_bytes = len(CORPUS.encode("utf-8"))
    tokens = len(tok.encode(CORPUS))
    # The whole point. With a repetitive corpus and 64 merges this should be a
    # large factor; the assertion stays loose so it measures direction, not luck.
    assert tokens < raw_bytes / 2


def test_vocab_size_accounts_for_every_id(tok):
    assert tok.vocab_size == 256 + len(tok.merge_list)
    assert max(tok.encode(CORPUS)) < tok.vocab_size


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    # Same corpus, same merges, in the same order — otherwise a checkpoint and
    # a re-derived tokenizer would disagree.
    assert a.merge_list == b.merge_list


def test_merge_ids_are_assigned_in_learning_order(tok):
    ids = [idx for _, _, idx in tok.merge_list]
    assert ids == list(range(256, 256 + len(ids)))


def test_merges_never_cross_a_word_boundary(tok):
    # Every learned token must decode to text that lies inside a single chunk.
    # A token spanning a boundary would make a word encode differently
    # depending on what followed it.
    for _, _, idx in tok.merge_list:
        piece = tok.vocab[idx].decode("utf-8", errors="replace")
        assert len(SPLIT_PATTERN.findall(piece)) == 1


def test_leading_space_is_part_of_the_token(tok):
    # " the" and "the" are different tokens. This is what removes the need for
    # any join rule at decode time.
    with_space = tok.encode(" the")
    without = tok.encode("the")
    assert with_space != without
    assert tok.decode(with_space) == " the"


def test_encoding_is_position_independent(tok):
    # A word must encode identically wherever it appears, which is what the
    # chunk split guarantees. The needle carries its leading space, because
    # " the" and "the" are deliberately different tokens — see the test above.
    a = tok.encode(" the cat sat")
    b = tok.encode("a dog and the cat sat down")
    assert _sublist(a, b)


def test_vocab_below_256_is_rejected():
    with pytest.raises(ValueError, match="256"):
        BPETokenizer.train("abc", vocab_size=100)


def test_training_stops_when_no_pair_repeats():
    # Asking for more merges than the corpus can justify must terminate rather
    # than emit slots that each fire once.
    tok = BPETokenizer.train("abcdefgh", vocab_size=1000)
    assert tok.vocab_size < 1000


def test_save_and_load_preserve_encoding(tmp_path, tok):
    tok.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    assert loaded.merge_list == tok.merge_list
    assert loaded.encode(CORPUS) == tok.encode(CORPUS)


def test_truncated_multibyte_output_does_not_raise():
    tok = BPETokenizer.train("日本語のテキスト " * 20, vocab_size=280)
    ids = tok.encode("日本語")
    # Generation routinely stops between the bytes of one character.
    assert isinstance(tok.decode(ids[:1]), str)


def _sublist(needle, haystack) -> bool:
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))
