from __future__ import annotations

import importlib

from .base import (
    AgentAdapter,
    EndEvent,
    PromptEvent,
    StopEvent,
    SubagentResult,
    Turn,
    mentioned_files,
)
from .claude_code import ClaudeCodeAdapter

__all__ = [
    "AgentAdapter",
    "ClaudeCodeAdapter",
    "EndEvent",
    "PromptEvent",
    "StopEvent",
    "SubagentResult",
    "Turn",
    "load_adapter",
    "mentioned_files",
]

BUILTIN: dict[str, type[AgentAdapter]] = {ClaudeCodeAdapter.name: ClaudeCodeAdapter}


def load_adapter(spec: str) -> AgentAdapter:
    """A built-in name ("claude-code") or "package.module:ClassName"."""
    if spec in BUILTIN:
        return BUILTIN[spec]()
    if ":" not in spec:
        raise ValueError(f"unknown agent {spec!r}; built-in: {sorted(BUILTIN)}")
    mod, _, attr = spec.partition(":")
    cls = getattr(importlib.import_module(mod), attr)
    adapter = cls()
    if not isinstance(adapter, AgentAdapter):
        raise TypeError(f"{spec} is not an AgentAdapter")
    return adapter
