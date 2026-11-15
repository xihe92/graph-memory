"""Settings: ~/.graph-memory/config.toml (optional). GM_HOME overrides the folder,
GM_SOCKET the socket path and GM_UI_PORT the web page's port.

Example config.toml:

    embedder = "all-MiniLM-L6-v2"    # or "none", or "package.module:factory"
    model_backend = "claude-code"    # the model graduation uses: "claude-code",
                                     # "command" (model_command) or "none"
    model = "haiku"
    model_command = []               # argv for model_backend = "command"
    idle_timeout = 1800              # seconds without requests before the daemon exits
    graduation_idle = 1200           # graduate a session after this many quiet seconds
    knowledge_recall = "on"          # "on", "shadow" (computed and logged only) or "off"
    judge_rate = 0.2                 # share of turns with notes whose notes a model judges
    redact = true                    # keys, tokens and passwords out of memory (redact.py)
    redact_patterns = []             # your own regular expressions to redact as well

Settings from before the knowledge-only redesign are accepted: summarizer,
summarizer_model and summarizer_command mean model_backend, model and model_command;
the rest of them (placement, topics, per-reply summaries) are ignored.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    home: Path
    socket_path: Path
    embedder: str = "all-MiniLM-L6-v2"
    # the model graduation and note judging call
    model_backend: str = "claude-code"
    model: str = "haiku"
    model_command: tuple[str, ...] = ()
    idle_timeout: float = 1800.0
    # graduation: buffered session exchanges -> the knowledge graph
    graduation: bool = True
    graduation_model: str | None = None  # default: model
    graduation_timeout: float = 300.0
    graduation_idle: float = 1200.0  # graduate a session after this long without exchanges
    # the session buffer: a graduation that keeps failing gives up after this many tries
    # or once its oldest exchange is this old; a session untouched this long is deleted
    graduation_attempts: int = 3
    buffer_max_days: float = 3.0
    buffer_abandon_days: float = 7.0
    # recall from the knowledge graph: "on" (shown to the agent), "shadow" (computed and
    # logged, not shown) or "off"
    knowledge_recall: str = "on"
    # after a reply, a model judges the notes shown for this share of turns that had notes
    judge_rate: float = 0.2
    events_days: float = 30.0  # the event log keeps this many days; 0 keeps everything
    # secrets (keys, tokens, passwords) are replaced before anything stores or sends text;
    # redact_patterns adds regular expressions of your own, see redact.py
    redact: bool = True
    redact_patterns: tuple[str, ...] = ()
    # sessions a program drives (`claude -p`, the Agent SDK): recall into them and
    # remember them like the user's own. Off: they are usually scripts and checks.
    remember_headless: bool = False
    # the web page (`gm ui`) is served by the daemon on this local port, up while the
    # daemon is up; 0 turns it off
    ui_port: int = 7477
    save_interval: float = 15.0
    start_timeout: float = 20.0
    recall_timeout: float = 10.0

    @property
    def lock_path(self) -> Path:
        return self.home / "gm.lock"

    @property
    def log_path(self) -> Path:
        return self.home / "gm.log"


RENAMED = {
    "summarizer": "model_backend",
    "summarizer_model": "model",
    "summarizer_command": "model_command",
}
RETIRED = frozenset(
    {
        "vector_backend",
        "summarizer_timeout",
        "summary_workers",
        "recall_path_limit",
        "placement",
        "merge",
    }
)


def _override(base: Config, values: dict[str, Any]) -> Config:
    known = {f.name for f in fields(base)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return replace(base, **values)


def load_config(home: str | Path | None = None) -> Config:
    root = Path(home or os.environ.get("GM_HOME") or "~/.graph-memory").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    sock = Path(os.environ.get("GM_SOCKET") or root / "gm.sock").expanduser()
    if len(str(sock)) > 100:
        raise ValueError(f"socket path too long for a Unix socket ({len(str(sock))} chars): {sock}")

    data: dict[str, Any] = {}
    cfg_file = root / "config.toml"
    if cfg_file.exists():
        data = tomllib.loads(cfg_file.read_text(encoding="utf-8"))
    for old, new in RENAMED.items():
        if old in data:
            data.setdefault(new, data[old])
            data.pop(old)
    if data.get("model_backend") == "heuristic":  # the old no-model summariser
        data["model_backend"] = "none"
    ignored = sorted(k for k in data if k in RETIRED)
    for k in ignored:
        data.pop(k)
    if ignored:
        log.info("ignoring retired settings in %s: %s", cfg_file, ignored)
    for key in ("model_command", "redact_patterns"):
        if key in data:
            data[key] = tuple(data[key])
    cfg = _override(Config(home=root, socket_path=sock), data)
    from .redact import make_redactor

    make_redactor(cfg.redact, cfg.redact_patterns)  # a bad pattern fails here, loudly
    if os.environ.get("GM_UI_PORT"):
        cfg = replace(cfg, ui_port=int(os.environ["GM_UI_PORT"]))
    return cfg
