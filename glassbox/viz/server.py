"""A small HTTP server for the visualizer.

Standard library only. This serves one page and four endpoints to one person on
localhost; a framework would add three dependencies to a project that currently
needs torch and numpy, and "clone it and run it" is worth more than the
conveniences we would not use.

Routing is a pure function of (method, path, body), so the tests exercise every
endpoint without binding a socket.
"""

import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from glassbox.model import GPTConfig
from glassbox.sampling.generate import generate
from glassbox.viz.architecture import build_diagram
from glassbox.viz.inspect import inspect
from glassbox.viz.registry import ModelRegistry, discover

STATIC = Path(__file__).parent / "static"

# Only these may be set from a request. Anything else is rejected rather than
# ignored, so a typo in the frontend surfaces instead of silently describing a
# different architecture than the one asked for.
CONFIG_FIELDS = {
    "vocab_size", "block_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
    "norm", "activation", "pos_encoding", "bias", "dropout",
}


class Router:
    """Turns a request into (status, content_type, body). No sockets involved."""

    def __init__(self, registry: ModelRegistry, default: str | None = None):
        self.registry = registry
        names = registry.names()
        self.default = default or (names[0] if names else None)

    # ------------------------------------------------------------- helpers

    def _json(self, payload, status: int = 200):
        return status, "application/json", json.dumps(payload).encode()

    def _error(self, message: str, status: int = 400):
        return self._json({"error": message}, status)

    def _model(self, body: dict):
        name = body.get("model") or self.default
        if name is None:
            raise KeyError("no models are loaded")
        return name, *self.registry.get(name)

    # ------------------------------------------------------------- routing

    def handle(self, method: str, path: str, body: bytes = b""):
        try:
            if method == "GET":
                return self._get(path)
            if method == "POST":
                try:
                    payload = json.loads(body or b"{}")
                except json.JSONDecodeError as e:
                    return self._error(f"malformed JSON: {e}")
                if not isinstance(payload, dict):
                    return self._error("body must be a JSON object")
                return self._post(path, payload)
            return self._error(f"method {method} not allowed", 405)
        except KeyError as e:
            return self._error(str(e).strip("'\""), 404)
        except ValueError as e:
            return self._error(str(e))
        except Exception as e:  # pragma: no cover - genuinely unexpected
            return self._error(f"{type(e).__name__}: {e}", 500)

    def _get(self, path: str):
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path == "/meta":
            return self._json({
                "models": self.registry.describe(),
                "default": self.default,
            })
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        return self._error(f"no route for GET {path}", 404)

    def _static(self, name: str):
        target = (STATIC / name).resolve()
        # Refuse anything that escapes the static directory. This server binds
        # to localhost, but a path-traversal hole is not worth leaving open on
        # the argument that nobody else can reach it.
        if not str(target).startswith(str(STATIC.resolve())):
            return self._error("forbidden", 403)
        if not target.is_file():
            return self._error(f"{name} not found — has the frontend been built?", 404)
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return 200, kind, target.read_bytes()

    def _post(self, path: str, body: dict):
        if path == "/architecture":
            return self._architecture(body)
        if path == "/inspect":
            return self._inspect(body)
        if path == "/generate":
            return self._generate(body)
        return self._error(f"no route for POST {path}", 404)

    # ----------------------------------------------------------- endpoints

    def _architecture(self, body: dict):
        """Describe any config. Deliberately needs no model and no checkpoint."""
        # Accepts either {"config": {...}} or the fields at the top level.
        # "model" is a selector, not a config field, so it never counts as one.
        overrides = body.get("config")
        if overrides is None:
            overrides = {k: v for k, v in body.items() if k != "model"}
        unknown = set(overrides) - CONFIG_FIELDS
        if unknown:
            return self._error(f"unknown config fields: {sorted(unknown)}")

        base = {}
        if body.get("model"):
            entry = self.registry.entries.get(body["model"])
            if entry is None:
                return self._error(f"unknown model {body['model']!r}", 404)
            c = entry.config
            base = {f: getattr(c, f) for f in CONFIG_FIELDS if hasattr(c, f)}

        config = GPTConfig(**{**base, **{k: v for k, v in overrides.items()
                                         if k in CONFIG_FIELDS}})
        return self._json(build_diagram(config).as_dict())

    def _inspect(self, body: dict):
        prompt = body.get("prompt", "")
        if not prompt:
            return self._error("prompt is required")
        name, model, tokenizer = self._model(body)
        with self.registry.lock:
            payload = inspect(model, tokenizer, prompt,
                              top_k=int(body.get("top_k", 8)))
        payload["model"] = name
        return self._json(payload)

    def _generate(self, body: dict):
        prompt = body.get("prompt", "")
        if not prompt:
            return self._error("prompt is required")
        name, model, tokenizer = self._model(body)

        seed = body.get("seed")
        if seed is not None:
            # Two architectures compared on the same seed is the only way the
            # difference in their output means anything.
            torch.manual_seed(int(seed))

        ids = tokenizer.encode(prompt)
        if len(ids) >= model.config.block_size:
            return self._error(
                f"prompt is {len(ids)} tokens; block_size is {model.config.block_size}")

        with self.registry.lock:
            out = generate(
                model, torch.tensor([ids], dtype=torch.long),
                max_new_tokens=int(body.get("max_tokens", 200)),
                temperature=float(body.get("temperature", 0.8)),
                top_k=body.get("top_k"),
                top_p=body.get("top_p"),
            )
        text = tokenizer.decode(out[0].tolist())
        return self._json({
            "model": name,
            "label": self.registry.entries[name].label,
            "val_loss": self.registry.entries[name].val_loss,
            "prompt": prompt,
            "text": text,
            "completion": text[len(prompt):],
        })


def make_handler(router: Router):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _respond(self, result):
            status, kind, payload = result
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._respond(router.handle("GET", self.path))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self._respond(router.handle("POST", self.path, self.rfile.read(length)))

        def log_message(self, fmt, *args):
            print(f"  {self.command} {self.path}", flush=True)

    return Handler


def serve(roots, host: str = "127.0.0.1", port: int = 8000, default=None):
    registry = discover(*roots)
    router = Router(registry, default)

    # Bound before anything is announced. Printing the banner first makes a
    # failed bind look like a successful start — the listing scrolls past and
    # the traceback is above it, out of sight.
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(router))
    except OSError as e:
        raise SystemExit(
            f"cannot bind {host}:{port}: {e.strerror}. "
            f"Something else is already listening there — try --port {port + 10}."
        )

    print(f"glassbox visualizer on http://{host}:{port}")
    for entry in registry.describe():
        mark = "*" if entry["name"] == router.default else " "
        print(f" {mark} {entry['name']:<12} {entry['params']:>11,}  {entry['label']}")
    httpd.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*",
                   default=["checkpoints/ablation", "checkpoints/tinystories"])
    p.add_argument("--default", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    serve(args.checkpoints, args.host, args.port, args.default)


if __name__ == "__main__":
    main()
