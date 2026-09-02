"""Tests for the three things instruction tuning needs: stop tokens, a loss
mask, and data shaped as prompt/response pairs rather than one long stream."""

import pytest
import torch
import torch.nn.functional as F

from glassbox.finetune import (
    ALPACA,
    IGNORE_INDEX,
    Example,
    InstructionDataset,
    load_jsonl,
)
from glassbox.model import GPT, GPTConfig
from glassbox.sampling.generate import generate
from glassbox.tokenizer.char import CharTokenizer

EXAMPLES = [
    Example("Say hello", "Hello there."),
    Example("Add two numbers", "4", input="2 and 2"),
    Example("Name a colour", "Blue."),
    Example("Count to three", "One two three."),
]


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def tokenizer():
    text = "".join(e.prompt() + e.output + e.input for e in EXAMPLES)
    return CharTokenizer.from_text(text + "0123456789")


@pytest.fixture
def dataset(tokenizer):
    return InstructionDataset(EXAMPLES, tokenizer, eos_id=0, block_size=256,
                              val_fraction=0.25)


@pytest.fixture
def model(tokenizer):
    return GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64,
                         d_model=32, n_layers=2, n_heads=4, dropout=0.0)).eval()


# ------------------------------------------------------------- stop tokens


def test_generation_stops_at_the_stop_token(model):
    """A base model has no reason to ever stop; an instruction model must.

    Rather than forcing an argmax by editing weights — which cannot work here,
    since lm_head is tied to the embedding and zeroing one zeroes both — the
    token the model would emit anyway is taken and declared the stop token.
    """
    prompt = torch.zeros((1, 3), dtype=torch.long)
    would_emit = generate(model, prompt, max_new_tokens=1,
                          temperature=0.0)[0, -1].item()

    out = generate(model, prompt, max_new_tokens=20, temperature=0.0,
                   eos_id=would_emit)
    assert out.shape == (1, 4)          # the prompt, plus the stop token
    assert out[0, -1].item() == would_emit


def test_without_a_stop_token_nothing_changes(model):
    prompt = torch.zeros((1, 3), dtype=torch.long)
    a = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    b = generate(model, prompt, max_new_tokens=6, temperature=0.0, eos_id=None)
    assert torch.equal(a, b)
    assert a.shape == (1, 9)


def test_a_finished_row_keeps_the_batch_rectangular(model):
    # Rows finish at different times but the tensor has to stay rectangular, so
    # a finished row keeps emitting its stop token rather than being removed.
    prompt = torch.zeros((3, 2), dtype=torch.long)
    would_emit = generate(model, prompt, max_new_tokens=1,
                          temperature=0.0)[0, -1].item()

    out = generate(model, prompt, max_new_tokens=8, temperature=0.0,
                   eos_id=would_emit)
    assert out.shape == (3, 3)
    assert (out[:, -1] == would_emit).all()


# ------------------------------------------------------------- loss masking


def test_masked_positions_contribute_nothing(model):
    """The loss must equal the loss over the unmasked positions alone."""
    idx = torch.randint(0, model.config.vocab_size, (2, 10))
    targets = torch.randint(0, model.config.vocab_size, (2, 10))

    masked = targets.clone()
    masked[:, :6] = IGNORE_INDEX

    logits, loss, _ = model(idx, targets=masked)

    kept_logits = logits[:, 6:].reshape(-1, logits.size(-1))
    kept_targets = targets[:, 6:].reshape(-1)
    assert loss.item() == pytest.approx(
        F.cross_entropy(kept_logits, kept_targets).item(), abs=1e-5)


def test_masking_changes_the_loss(model):
    idx = torch.randint(0, model.config.vocab_size, (2, 10))
    targets = torch.randint(0, model.config.vocab_size, (2, 10))
    _, full, _ = model(idx, targets=targets)

    masked = targets.clone()
    masked[:, :6] = IGNORE_INDEX
    _, partial, _ = model(idx, targets=masked)
    assert full.item() != pytest.approx(partial.item(), abs=1e-6)


# ---------------------------------------------------------------- the data


def test_scoring_starts_exactly_at_the_response(tokenizer):
    example = Example("Say hello", "Hello there.")
    dataset = InstructionDataset([example], tokenizer, eos_id=0, block_size=256,
                                 val_fraction=0.0)
    x, y = dataset.train[0]

    first = next(i for i, t in enumerate(y) if t != IGNORE_INDEX)
    # Everything before the response is masked, and the first thing scored is
    # the response's own first character.
    assert tokenizer.decode([y[first]]) == "H"
    assert all(t == IGNORE_INDEX for t in y[:first])
    assert tokenizer.decode(x[:first + 1]).endswith("### Response:\n")


def test_the_response_ends_with_the_stop_token(tokenizer):
    dataset = InstructionDataset([Example("hi", "yes")], tokenizer, eos_id=0,
                                 block_size=256, val_fraction=0.0)
    _, y = dataset.train[0]
    assert y[-1] == 0


def test_nothing_is_scored_after_a_row_ends(dataset):
    """Padding must not be trained on.

    Checking `x == eos_id` would be wrong here: the stop id is a real character
    in this vocabulary, so it matches genuine newlines too. The invariant that
    actually holds is positional — once a row stops being scored, it is never
    scored again, so padding can never contribute.
    """
    x, y = dataset.get_batch("train", 6, 256)
    assert x.shape == y.shape

    for row in y:
        scored = (row != IGNORE_INDEX).nonzero().flatten().tolist()
        if not scored:
            continue
        assert scored == list(range(scored[0], scored[-1] + 1))
        assert bool((row[scored[-1] + 1:] == IGNORE_INDEX).all())


def test_batch_interface_matches_the_other_datasets(dataset, model):
    # The training loop calls this signature and reads .tokenizer; keeping the
    # shape means instruction tuning needs no changes to the loop.
    x, y = dataset.get_batch("train", 2, 64, "cpu")
    assert x.dtype == torch.long and y.dtype == torch.long
    assert dataset.tokenizer is not None


def test_the_input_field_switches_template(tokenizer):
    plain = Example("Add", "4").prompt()
    with_input = Example("Add", "4", input="2 and 2").prompt()
    assert "### Input:" not in plain
    assert "2 and 2" in with_input


def test_examples_that_cannot_fit_are_dropped_and_counted(tokenizer):
    long_one = Example("x" * 400, "y")
    dataset = InstructionDataset(EXAMPLES + [long_one], tokenizer, eos_id=0,
                                 block_size=128, val_fraction=0.25)
    # Truncating would leave a prompt asking half a question, so it is dropped
    # and the count is reported rather than silently absorbed.
    assert dataset.skipped >= 1


def test_everything_too_long_is_an_error(tokenizer):
    with pytest.raises(ValueError, match="block_size"):
        InstructionDataset([Example("x" * 500, "y")], tokenizer, eos_id=0,
                           block_size=32)


def test_train_and_validation_are_disjoint(dataset):
    train = {tuple(x) for x, _ in dataset.train}
    val = {tuple(x) for x, _ in dataset.val}
    assert train and val and not (train & val)


def test_jsonl_accepts_the_usual_field_names(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"instruction": "a", "output": "b"}\n'
        '{"prompt": "c", "response": "d"}\n'
        '\n'
        '{"instruction": "e", "input": "f", "output": "g"}\n')
    rows = load_jsonl(path)
    assert [r.instruction for r in rows] == ["a", "c", "e"]
    assert [r.output for r in rows] == ["b", "d", "g"]
    assert rows[2].input == "f"
