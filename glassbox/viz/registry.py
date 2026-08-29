"""Finding, describing and loading the trained models the visualizer offers."""

import json
import threading
from pathlib import Path

import torch

from glassbox.model import GPT
from glassbox.tokenizer.spec import load_tokenizer_from_checkpoint


class ModelEntry:
    """What is known about one checkpoint without holding its weights."""

    def __init__(self, name: str, path: Path, label: str, val_loss: float,
                 params: int, config, history: list):
        self.name = name
        self.path = path
        self.label = label
        self.val_loss = val_loss
        self.params = params
        self.config = config
        self.history = history

    def as_dict(self) -> dict:
        c = self.config
        return {
            "name": self.name,
            "label": self.label,
            "val_loss": round(self.val_loss, 4) if self.val_loss is not None else None,
            "params": self.params,
            "history": self.history,
            "config": {
                "vocab_size": c.vocab_size, "block_size": c.block_size,
                "d_model": c.d_model, "n_layers": c.n_layers, "n_heads": c.n_heads,
                "n_kv_heads": c.n_kv_heads, "norm": c.norm,
                "activation": c.activation, "pos_encoding": c.pos_encoding,
                "bias": c.bias,
            },
        }


class ModelRegistry:
    """Reads every checkpoint's description once; loads weights only on demand."""

    def __init__(self, entries: list[ModelEntry]):
        self.entries = {e.name: e for e in entries}
        self._loaded: dict[str, tuple] = {}
        # The models are shared across requests and forward passes write to
        # model._hidden, so two overlapping inspections would corrupt each
        # other's hidden states. One lock is plenty for a local single-user tool.
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> list[str]:
        return list(self.entries)

    def describe(self) -> list[dict]:
        return [e.as_dict() for e in self.entries.values()]

    def get(self, name: str):
        """Return (model, tokenizer), loading and caching on first use."""
        if name not in self.entries:
            raise KeyError(f"unknown model {name!r}; have {sorted(self.entries)}")

        with self._lock:
            if name not in self._loaded:
                ckpt = torch.load(self.entries[name].path, map_location="cpu",
                                  weights_only=False)
                model = GPT(ckpt["model_config"])
                model.load_state_dict(ckpt["model"])
                model.eval()
                self._loaded[name] = (model, load_tokenizer_from_checkpoint(ckpt))
            return self._loaded[name]

    @property
    def lock(self) -> threading.Lock:
        return self._lock


def _read_entry(name: str, path: Path, label: str | None) -> ModelEntry | None:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if "model_config" not in ckpt:
        return None
    config = ckpt["model_config"]
    # The weights are read and then dropped. Startup pays one pass over the
    # files so that listing models is instant, and nothing large stays resident
    # until a model is actually asked for.
    params = GPT(config).num_parameters()
    return ModelEntry(name, path, label or name, ckpt.get("val_loss"),
                      params, config, ckpt.get("history", []))


def discover(*roots: str | Path) -> ModelRegistry:
    """Find checkpoints under the given directories."""
    entries: list[ModelEntry] = []
    seen: set[str] = set()

    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue

        # An ablation directory ships a results.json naming each variant; use it
        # for labels and to fix the order, so the UI lists them as an experiment
        # rather than alphabetically.
        labels, order = {}, []
        results = root / "results.json"
        if results.exists():
            try:
                data = json.loads(results.read_text())
                for v in data.get("variants", []):
                    labels[v["name"]] = v.get("label", v["name"])
                    order.append(v["name"])
            except Exception:
                pass

        found = {p.stem: p for p in sorted(root.glob("*.pt"))}
        for name in order + sorted(set(found) - set(order)):
            if name not in found:
                continue
            key = name if name not in seen else f"{root.name}/{name}"
            # A directory holding a single best.pt is named after the directory,
            # since "best" says nothing in a list of seven models.
            display = root.name if name == "best" else key
            entry = _read_entry(display, found[name], labels.get(name))
            if entry is not None:
                entries.append(entry)
                seen.add(display)

    return ModelRegistry(entries)
