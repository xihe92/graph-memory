"""The embedding model, run locally: all-MiniLM-L6-v2 through onnxruntime.

It turns a text into 384 numbers of length 1, so the dot product of two vectors is their
cosine similarity. The model files (about 90 MB) are downloaded once from Hugging Face,
at a pinned revision and checked against their SHA-256, into
~/.cache/graph-memory/models/ (or $XDG_CACHE_HOME/graph-memory/models/). After that it
runs offline on the CPU in about 5 ms per text.

Why this model: on graph-memory's own notes, newer small models ranked the right note a
little higher on paraphrased prompts, but none found more of them, the current model did
best on real prompts, and their scores crowd together where recall's thresholds are
tuned to this model's spread. Text is not padded: padding every input to 256 tokens (as
chromadb's wrapper did) gives the same vectors about eight times slower.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ModelFiles:
    repo: str
    revision: str
    onnx: tuple[str, str]  # path in the repo, sha256
    tokenizer: tuple[str, str]
    max_tokens: int


MODELS = {
    DEFAULT_MODEL: ModelFiles(
        repo="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        onnx=(
            "onnx/model.onnx",
            "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        ),
        tokenizer=(
            "tokenizer.json",
            "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        ),
        max_tokens=256,
    ),
}


class ModelDownloadError(RuntimeError):
    pass


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "graph-memory" / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: ModelFiles, file: tuple[str, str], root: Path) -> Path:
    """The file, downloaded on first use and checked against its SHA-256."""
    path, sha = file
    dst = root / spec.repo.replace("/", "--") / spec.revision[:12] / path
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{path}"
    try:
        urllib.request.urlretrieve(url, tmp)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(f"could not download {url}: {exc}") from exc
    got = _sha256(tmp)
    if got != sha:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(f"{url} has SHA-256 {got}, expected {sha}")
    tmp.replace(dst)
    return dst


class LocalEmbedder:
    """A sentence-transformers model exported to ONNX: tokenize, run, average the token
    vectors over the attention mask, scale to length 1. Thread-safe."""

    def __init__(self, name: str = DEFAULT_MODEL, root: Path | None = None) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from tokenizers import Tokenizer

        spec = MODELS[name]
        root = root or cache_dir()
        self.name = name
        self._tok = Tokenizer.from_file(str(fetch(spec, spec.tokenizer, root)))
        self._tok.enable_truncation(spec.max_tokens)
        self._tok.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        # no busy-waiting between calls: the daemon is idle almost all the time
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(fetch(spec, spec.onnx, root)), opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._lock = threading.Lock()  # the tokenizer is not safe to share across threads

    def vector(self, text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        with self._lock:
            enc = self._tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self._session.run(None, feed)[0]
        v = (out * mask[..., None]).sum(axis=1)[0] / max(int(mask.sum()), 1)
        n = float(np.linalg.norm(v))
        out32: np.ndarray[Any, np.dtype[np.float32]] = (v / n if n else v).astype(np.float32)
        return out32

    def __call__(self, text: str) -> list[float]:
        return [float(x) for x in self.vector(text)]


def default_embedder() -> LocalEmbedder:
    return LocalEmbedder(DEFAULT_MODEL)
