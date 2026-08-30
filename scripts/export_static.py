"""Bake the visualizer into one self-contained HTML file.

    python scripts/export_static.py --out site/index.html

A page on GitHub Pages has no Python behind it, so every answer the server
would compute is precomputed here and inlined. The architecture view stays
fully interactive because describing a configuration is pure arithmetic and
every combination of the four switches can be enumerated. Inspection and
generation need a real forward pass, so a curated set of prompts is baked and
the page offers exactly those.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

from glassbox.model import GPTConfig
from glassbox.sampling.generate import generate
from glassbox.viz.architecture import build_diagram
from glassbox.viz.inspect import inspect as inspect_model
from glassbox.viz.registry import discover
from glassbox.viz.server import CONFIG_FIELDS, STATIC

import torch

# Short prompts on purpose: the attention payload is layers x heads x T x T, so
# it grows with the square of the prompt and a long one would dominate the file.
INSPECT_PROMPTS = ["ROMEO:\nWhat is th", "To be or not"]
GENERATE_PROMPTS = ["ROMEO:", "KING RICHARD III:"]
TINYSTORIES_INSPECT = ["Once upon a time"]
TINYSTORIES_GENERATE = ["Once upon a time, there was a little"]


def arch_key(cfg: dict) -> str:
    return "|".join(f"{k}={cfg[k]}"
                    for k in ("norm", "activation", "pos_encoding", "n_kv_heads"))


def bake_architecture(base: dict) -> dict:
    """Every combination of the four switches, so the toggles stay live."""
    out = {}
    divisors = [k for k in range(1, base["n_heads"] + 1) if base["n_heads"] % k == 0]
    for norm, act, pos, kv in itertools.product(
            ("layernorm", "rmsnorm"), ("gelu", "swiglu"),
            ("learned", "rope"), divisors):
        cfg = {**base, "norm": norm, "activation": act,
               "pos_encoding": pos, "n_kv_heads": kv}
        out[arch_key(cfg)] = build_diagram(GPTConfig(**cfg)).as_dict()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*",
                   default=["checkpoints/ablation", "checkpoints/tinystories"])
    # docs/ on the default branch is one of the two folders GitHub Pages
    # will serve from, and the only one that is not the repository root.
    p.add_argument("--out", type=str, default="docs/index.html")
    p.add_argument("--default", type=str, default="all")
    args = p.parse_args()

    registry = discover(*args.checkpoints)
    if not len(registry):
        raise SystemExit("no checkpoints found — nothing to bake")

    models = registry.describe()
    default = args.default if args.default in registry.names() else registry.names()[0]
    base_cfg = next(m for m in models if m["name"] == default)["config"]
    base = {k: v for k, v in base_cfg.items() if k in CONFIG_FIELDS}

    print(f"baking from {len(models)} models, default {default}")
    t0 = time.time()

    architecture = bake_architecture(base)
    print(f"  architecture  {len(architecture)} configurations")

    inspect_table, generate_table = {}, {}
    for entry in models:
        name = entry["name"]
        model, tokenizer = registry.get(name)
        story = name == "tinystories"
        for prompt in (TINYSTORIES_INSPECT if story else INSPECT_PROMPTS):
            inspect_table[f"{name}|{prompt}"] = inspect_model(model, tokenizer, prompt)
        for prompt in (TINYSTORIES_GENERATE if story else GENERATE_PROMPTS):
            torch.manual_seed(1337)
            ids = tokenizer.encode(prompt)
            out = generate(model, torch.tensor([ids], dtype=torch.long),
                           max_new_tokens=260, temperature=0.8, top_k=40)
            text = tokenizer.decode(out[0].tolist())
            generate_table[f"{name}|{prompt}"] = {
                "model": name, "label": entry["label"], "val_loss": entry["val_loss"],
                "prompt": prompt, "text": text, "completion": text[len(prompt):],
            }
        print(f"  {name:<12} inspected and generated")

    payload = {
        "meta": {"models": models, "default": default},
        "architecture": architecture,
        "inspect": inspect_table,
        "generate": generate_table,
    }

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    # Inlined rather than linked: the page has to work from a file:// URL with
    # nothing beside it.
    html = html.replace('<link rel="stylesheet" href="/static/app.css">',
                        f"<style>\n{css}\n</style>")
    html = html.replace(
        '<script src="/static/app.js"></script>',
        "<script>window.GLASSBOX_DATA = "
        + json.dumps(payload, separators=(",", ":"))
        + f";</script>\n<script>\n{js}\n</script>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    print(f"\n{out}  {out.stat().st_size / 1e6:.1f} MB  in {time.time() - t0:.0f}s")
    print(f"  {len(architecture)} configurations, {len(inspect_table)} inspections, "
          f"{len(generate_table)} generations")


if __name__ == "__main__":
    main()
