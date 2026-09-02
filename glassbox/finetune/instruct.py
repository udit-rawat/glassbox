"""Instruction data: prompt/response pairs, a template, and a loss mask.

Pretraining and instruction tuning need differently shaped data. Pretraining
cuts random windows from one continuous stream, and every position is a target.
Instruction tuning has discrete examples, and only the *response* is a target —
scoring the prompt as well would train the model to generate the questions.

So each example is rendered through a template, tokenised, and paired with a
target row in which every prompt position is replaced by IGNORE_INDEX. The
batch interface matches the other datasets, so the training loop is unchanged.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

# What cross-entropy skips. -100 is the PyTorch default, kept rather than
# invented so a target row is readable by anything that expects the convention.
IGNORE_INDEX = -100

ALPACA = (
    "Below is an instruction describing a task. "
    "Write a response that completes it.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
ALPACA_WITH_INPUT = (
    "Below is an instruction describing a task, paired with further context. "
    "Write a response that completes it.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


@dataclass
class Example:
    instruction: str
    output: str
    input: str = ""

    def prompt(self, template: str = ALPACA, with_input: str = ALPACA_WITH_INPUT) -> str:
        if self.input:
            return with_input.format(instruction=self.instruction, input=self.input)
        return template.format(instruction=self.instruction)


def load_jsonl(path: str | Path) -> list[Example]:
    """Read the usual instruction format: one JSON object per line."""
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        examples.append(Example(
            instruction=row.get("instruction", row.get("prompt", "")),
            output=row.get("output", row.get("response", row.get("completion", ""))),
            input=row.get("input", ""),
        ))
    return examples


class InstructionDataset:
    """Prompt/response pairs, tokenised once, with the prompt masked out."""

    # eos_id should be a token reserved for the purpose. A tokenizer built from
    # a corpus has no spare ids, so passing an ordinary one — id 0 is a newline
    # in the character tokenizer — makes padding indistinguishable from real
    # text by value. The targets stay correct either way, because padding is
    # masked positionally rather than by token, but anything that tries to find
    # padding by comparing ids will be wrong. Real tokenizers ship a dedicated
    # end-of-sequence token; use it.
    def __init__(self, examples, tokenizer, eos_id: int, block_size: int,
                 template: str = ALPACA, val_fraction: float = 0.05,
                 seed: int = 0):
        if not examples:
            raise ValueError("no examples")
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.block_size = block_size

        encoded, self.skipped = [], 0
        for ex in examples:
            row = self._encode(ex, template)
            if row is None:
                # Longer than the context even before the response: there is no
                # honest way to keep it, so it is dropped and counted rather
                # than truncated into a prompt that asks half a question.
                self.skipped += 1
            else:
                encoded.append(row)
        if not encoded:
            raise ValueError(
                f"every example was longer than block_size={block_size}")

        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(encoded), generator=generator).tolist()
        cut = max(1, int(len(order) * (1 - val_fraction)))
        self.train = [encoded[i] for i in order[:cut]]
        self.val = [encoded[i] for i in order[cut:]] or self.train[:1]

    def _encode(self, example: Example, template: str):
        prompt_ids = self.tokenizer.encode(example.prompt(template))
        response_ids = self.tokenizer.encode(example.output) + [self.eos_id]

        if len(prompt_ids) + 1 >= self.block_size:
            return None
        ids = (prompt_ids + response_ids)[: self.block_size + 1]

        # x predicts y one step ahead, so the first response token is the target
        # at index len(prompt) - 1. Everything before that is prompt predicting
        # prompt, and is masked.
        x = ids[:-1]
        y = list(ids[1:])
        for i in range(min(len(prompt_ids) - 1, len(y))):
            y[i] = IGNORE_INDEX
        return x, y

    def __len__(self) -> int:
        return len(self.train) + len(self.val)

    def __repr__(self) -> str:
        return (f"InstructionDataset(train={len(self.train)}, "
                f"val={len(self.val)}, skipped={self.skipped})")

    def get_batch(self, split: str, batch_size: int, block_size: int,
                  device: torch.device | str = "cpu"):
        rows = self.train if split == "train" else self.val
        pick = torch.randint(len(rows), (batch_size,))

        width = min(block_size, max(len(rows[i][0]) for i in pick.tolist()))
        x = torch.full((batch_size, width), self.eos_id, dtype=torch.long)
        y = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)

        for row, i in enumerate(pick.tolist()):
            xi, yi = rows[i]
            n = min(len(xi), width)
            x[row, :n] = torch.tensor(xi[:n], dtype=torch.long)
            y[row, :n] = torch.tensor(yi[:n], dtype=torch.long)

        # Padding is masked in the targets, so a short example in a long batch
        # contributes nothing after its own stop token.
        return x.to(device), y.to(device)
