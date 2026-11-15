"""The embedding model's files: downloaded once, checked against their SHA-256, reused."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pytest

from graph_memory.embedding import MODELS, ModelDownloadError, ModelFiles, fetch


def _spec(content: bytes) -> ModelFiles:
    return ModelFiles(
        repo="org/model",
        revision="0123456789abcdef",
        onnx=("onnx/model.onnx", hashlib.sha256(content).hexdigest()),
        tokenizer=("tokenizer.json", "0" * 64),
        max_tokens=8,
    )


def test_a_file_is_downloaded_once_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def download(url: str, dst: Path) -> None:
        calls.append(url)
        Path(dst).write_bytes(b"weights" if "model.onnx" in url else b"tampered")

    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    spec = _spec(b"weights")
    path = fetch(spec, spec.onnx, tmp_path)
    assert path.read_bytes() == b"weights"
    assert calls == ["https://huggingface.co/org/model/resolve/0123456789abcdef/onnx/model.onnx"]
    assert fetch(spec, spec.onnx, tmp_path) == path and len(calls) == 1  # cached
    # a file that does not match its checksum is refused and not kept
    with pytest.raises(ModelDownloadError, match="SHA-256"):
        fetch(spec, spec.tokenizer, tmp_path)
    assert not list(tmp_path.rglob("tokenizer.json*"))


def test_a_failed_download_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(url: str, dst: Path) -> None:
        Path(dst).write_bytes(b"half")
        raise OSError("network is unreachable")

    monkeypatch.setattr(urllib.request, "urlretrieve", offline)
    spec = _spec(b"weights")
    with pytest.raises(ModelDownloadError, match="unreachable"):
        fetch(spec, spec.onnx, tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_the_default_model_is_pinned() -> None:
    spec = MODELS["all-MiniLM-L6-v2"]
    assert len(spec.revision) == 40 and len(spec.onnx[1]) == 64 and len(spec.tokenizer[1]) == 64
