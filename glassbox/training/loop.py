"""The training loop: AdamW, mixed precision, accumulation, scheduling, resumable checkpoints."""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from glassbox.tokenizer.spec import tokenizer_spec
from glassbox.training.data import CharDataset
from glassbox.training.precision import select_precision
from glassbox.training.schedule import constant, cosine_with_warmup


@dataclass
class TrainConfig:
    max_iters: int = 3000
    batch_size: int = 32
    # Micro-batches accumulated before each optimizer step. The effective batch
    # is batch_size * grad_accum_steps, which is how a large batch is reached on
    # a card that cannot hold one.
    grad_accum_steps: int = 1

    learning_rate: float = 3e-4
    schedule: str = "constant"   # constant | cosine
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1

    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # Adam's moment decay rates, named here rather than buried in the optimizer
    # call. The default reproduces Phase 1, following the same rule as the
    # architecture switches: nothing changes behaviour unless it is asked for.
    # 0.95 is the GPT-3 and Llama value and is the right choice at larger scale,
    # but on this model it measured 0.024 worse — so it is opt-in, not assumed.
    betas: tuple[float, float] = (0.9, 0.99)

    eval_interval: int = 250
    eval_iters: int = 50
    seed: int = 1337
    out_dir: str = "checkpoints"

    amp: bool = True             # use mixed precision where the device supports it
    resume: bool = False         # continue from out_dir/last.pt if present

    history: list = field(default_factory=list)


def _lr_for(it: int, cfg: TrainConfig) -> float:
    if cfg.schedule == "cosine":
        return cosine_with_warmup(
            it, cfg.learning_rate, cfg.warmup_iters, cfg.max_iters, cfg.min_lr_ratio
        )
    return constant(it, cfg.learning_rate)


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


def build_optimizer(model, cfg: TrainConfig) -> torch.optim.Optimizer:
    # Weight decay is applied only to matrices, never to biases or norm gains.
    # Decaying a normalization gain fights the normalization itself, and
    # decaying a bias just drags the whole layer's output toward zero.
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=cfg.betas,
    )


def train(model, dataset: CharDataset, cfg: TrainConfig, device) -> TrainConfig:
    torch.manual_seed(cfg.seed)
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    precision = select_precision(torch.device(device), cfg.amp)
    scaler = torch.amp.GradScaler(
        torch.device(device).type, enabled=precision.use_scaler
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    start_iter = 1

    # Resume before the loop, not inside it. A Colab session can drop at any
    # moment, and everything needed to continue exactly where it stopped —
    # optimizer moments, scaler scale, iteration, best score — has to have been
    # written to disk, not just the weights.
    last_path = out_dir / "last.pt"
    if cfg.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and precision.use_scaler:
            scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt["iter"] + 1
        best_val = ckpt.get("best_val", math.inf)
        cfg.history = ckpt.get("history", [])
        print(f"resumed from iter {ckpt['iter']}, best val {best_val:.4f}", flush=True)

    tok_spec = tokenizer_spec(dataset.tokenizer)
    effective_batch = cfg.batch_size * cfg.grad_accum_steps
    print(
        f"precision   {precision.describe()}   "
        f"effective batch {effective_batch} "
        f"({cfg.batch_size} x {cfg.grad_accum_steps})   schedule {cfg.schedule}",
        flush=True,
    )

    def save(path: Path, it: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if precision.use_scaler else None,
                "model_config": model.config,
                "tokenizer": tok_spec,
                # Kept alongside the spec so Phase 1 checkpoints and scripts
                # that expect a character vocabulary keep working unchanged.
                "chars": tok_spec.get("chars"),
                "iter": it,
                "best_val": best_val,
                "val_loss": best_val,
                "history": cfg.history,
            },
            path,
        )

    start = time.time()
    model.train()

    for it in range(start_iter, cfg.max_iters + 1):
        lr = _lr_for(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for _ in range(cfg.grad_accum_steps):
            x, y = dataset.get_batch(
                "train", cfg.batch_size, model.config.block_size, device
            )
            if precision.enabled:
                with torch.autocast(precision.device_type, dtype=precision.dtype):
                    _, loss, _ = model(x, targets=y)
            else:
                _, loss, _ = model(x, targets=y)

            # Each micro-batch contributes its share. Without the division the
            # accumulated gradient would be a sum rather than a mean, so the
            # effective learning rate would scale with grad_accum_steps and
            # changing accumulation would silently change the run.
            scaler.scale(loss / cfg.grad_accum_steps).backward()

        if cfg.grad_clip > 0:
            # Gradients are still multiplied by the scaler's factor at this
            # point, so clipping them directly would compare a scaled magnitude
            # against an unscaled threshold. unscale_ divides it back out first.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, dataset, cfg, device)
            elapsed = time.time() - start
            cfg.history.append((it, losses["train"], losses["val"]))
            # flush because stdout is block-buffered whenever this is piped to
            # a log file rather than a terminal, and a long run that reports
            # nothing until it exits cannot be monitored or debugged.
            print(
                f"iter {it:>6}  train {losses['train']:.4f}  "
                f"val {losses['val']:.4f}  lr {lr:.2e}  {elapsed:7.1f}s",
                flush=True,
            )

            # Checkpoint on validation improvement, not on the latest iteration.
            # Once the model starts overfitting this keeps the best weights
            # rather than the most recent ones.
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(out_dir / "best.pt", it)

            # last.pt is written every time regardless, because its job is
            # resumption rather than quality.
            save(last_path, it)

    print(f"best val loss {best_val:.4f}", flush=True)
    return cfg
