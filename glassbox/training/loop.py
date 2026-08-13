"""The training loop: AdamW, periodic held-out evaluation, best-checkpoint saving."""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from glassbox.training.data import CharDataset


@dataclass
class TrainConfig:
    max_iters: int = 3000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    seed: int = 1337
    out_dir: str = "checkpoints"
    history: list = field(default_factory=list)


@torch.no_grad()
def estimate_loss(model, dataset: CharDataset, cfg: TrainConfig, device) -> dict[str, float]:
    """Average loss over several batches from each split."""
    # A single batch is far too noisy to compare against the previous
    # evaluation, so each split is averaged over eval_iters draws. eval() also
    # disables dropout, which otherwise inflates the reported training loss and
    # makes it look worse than validation.
    out = {}
    was_training = model.training
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, model.config.block_size, device)
            _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(was_training)
    return out


def train(model, dataset: CharDataset, cfg: TrainConfig, device) -> TrainConfig:
    torch.manual_seed(cfg.seed)
    model.to(device)

    # Weight decay is applied only to matrices, never to biases or LayerNorm
    # gains. Decaying a normalization gain fights the normalization itself, and
    # decaying a bias just drags the whole layer's output toward zero.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.99),
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    start = time.time()

    model.train()
    for it in range(1, cfg.max_iters + 1):
        x, y = dataset.get_batch("train", cfg.batch_size, model.config.block_size, device)

        _, loss, _ = model(x, targets=y)

        # set_to_none frees the gradient tensors rather than filling them with
        # zeros — slightly faster, and a parameter that never received a
        # gradient shows up as None instead of a plausible-looking zero.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Rare batches produce very large gradients that would undo many steps
        # of progress in one update. Clipping rescales the whole gradient vector
        # when its norm exceeds the threshold, preserving direction.
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, dataset, cfg, device)
            elapsed = time.time() - start
            cfg.history.append((it, losses["train"], losses["val"]))
            # flush because stdout is block-buffered whenever this is piped to
            # a log file rather than a terminal, and a long run that reports
            # nothing until it exits cannot be monitored or debugged.
            print(
                f"iter {it:>5}  train {losses['train']:.4f}  "
                f"val {losses['val']:.4f}  {elapsed:6.1f}s",
                flush=True,
            )

            # Checkpoint on validation improvement, not on the latest iteration.
            # Once the model starts overfitting this keeps the best weights
            # rather than the most recent ones.
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "chars": dataset.tokenizer.chars,
                        "iter": it,
                        "val_loss": best_val,
                    },
                    out_dir / "best.pt",
                )

    print(f"best val loss {best_val:.4f}", flush=True)
    return cfg
