"""The daemon as a process: what it is built from (embedder, model), the lock that keeps
it to one per memory folder, and `gm daemon`.
"""

from __future__ import annotations

import asyncio
import fcntl
import importlib
import logging
import os
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from ..config import Config
from ..embedding import MODELS, LocalEmbedder
from ..graduation import Ask
from ..knowledge import KnowledgeStore
from ..models import CommandModel, claude_code_model
from ..vectors import Embedder
from .core import Daemon
from .state import log


def _import(spec: str) -> Any:
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def build_embedder(cfg: Config) -> Embedder | None:
    if cfg.embedder == "none":
        return None
    name = "all-MiniLM-L6-v2" if cfg.embedder in ("chroma-default", "default") else cfg.embedder
    if name in MODELS:
        return LocalEmbedder(name)
    factory: Callable[[], Embedder] = _import(cfg.embedder)
    return factory()


def build_model(cfg: Config) -> Ask | None:
    """The model graduation (and note judging) calls, or None without one."""
    if not cfg.graduation:
        return None
    model = cfg.graduation_model or cfg.model
    if cfg.model_backend == "claude-code":
        return claude_code_model(model, cfg.graduation_timeout, str(cfg.home)).complete
    if cfg.model_backend == "command" and cfg.model_command:
        return CommandModel(
            cfg.model_command, timeout=cfg.graduation_timeout, cwd=str(cfg.home)
        ).complete
    return None


def _answers(sock: Path) -> bool:
    """Whether a daemon is accepting connections on `sock`."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect(str(sock))
        except OSError:
            return False
    return True


def acquire_lock(cfg: Config, wait: float) -> IO[str] | None:
    """Take the daemon lock, held for the process lifetime. A daemon that is still
    shutting down holds it for a moment after it stops answering: wait up to `wait`
    seconds for it to go, so a hook that came in during that moment gets a daemon
    instead of a timeout. A daemon that answers is running: give up at once."""
    lock = open(cfg.lock_path, "a")  # noqa: SIM115 - "a": don't wipe the holder's pid
    deadline = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if _answers(cfg.socket_path) or time.monotonic() >= deadline:
                lock.close()
                return None
            time.sleep(0.05)
    lock.truncate(0)
    lock.write(str(os.getpid()))
    lock.flush()
    return lock


def run_daemon(cfg: Config) -> int:
    """Entry point for `gm daemon`. Exits quietly if another daemon is running."""
    logging.basicConfig(
        filename=str(cfg.log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    lock = acquire_lock(cfg, wait=cfg.start_timeout)
    if lock is None:
        log.info("another daemon is running; exiting")
        return 0

    embedder = build_embedder(cfg)
    if embedder is not None:
        embedder("warm up")  # load the model before accepting requests
    daemon = Daemon(
        cfg,
        knowledge=KnowledgeStore(cfg.home, embedder),
        embedder=embedder,
        graduation_model=build_model(cfg),
    )
    asyncio.run(daemon.serve())
    return 0
