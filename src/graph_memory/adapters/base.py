"""Agent adapters: the only agent-specific code.

An adapter translates one agent product's hook events and transcript into neutral data:
    parse_prompt_event()  hook payload when the user submits a prompt -> PromptEvent
    parse_stop_event()    hook payload when the agent finishes        -> StopEvent
    read_turn()           the finished turn from the agent's transcript -> Turn
    emit_context()        hand the recall context back to the agent (stdout by default)

To support another product, subclass AgentAdapter and pass it as
`gm recall --agent your.module:YourAdapter` (same for remember).
"""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ..anchors import Access, CodeAnchor, SourceRef


class PromptEvent(BaseModel):
    session_id: str
    prompt: str
    cwd: str | None = None
    transcript_path: str | None = None
    # Not typed by the user (e.g. a background-task notification the agent product feeds
    # through the same hook). Not recorded.
    system: bool = False


class StopEvent(BaseModel):
    session_id: str
    cwd: str | None = None
    transcript_path: str | None = None


class SubagentResult(BaseModel):
    name: str
    task: str = ""
    result: str = ""


class Turn(BaseModel):
    """One finished exchange, agent-neutral. Sent from the hook client to the daemon."""

    prompt: str
    reply: str
    files: list[CodeAnchor] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    # pages the agent read on the web (URLs) and searches it ran ("search: <query>"): the
    # evidence that an answer came from documentation rather than the model's memory
    web: list[str] = Field(default_factory=list)
    subagents: list[SubagentResult] = Field(default_factory=list)
    source: SourceRef | None = None
    # Transcript id of the last AI entry *before* this prompt on the active conversation
    # chain. Lets the daemon notice rewinds (the prompt follows an older reply).
    previous_reply_ref: str | None = None
    # Working directory of the session: locates the project (git repository).
    cwd: str | None = None


class AgentAdapter(ABC):
    name: str = "agent"

    @abstractmethod
    def parse_prompt_event(self, payload: dict[str, Any]) -> PromptEvent: ...

    @abstractmethod
    def parse_stop_event(self, payload: dict[str, Any]) -> StopEvent: ...

    @abstractmethod
    def read_turn(self, event: StopEvent) -> Turn | None:
        """The turn that just finished, or None if it can't be read."""

    def is_headless(self, payload: dict[str, Any]) -> bool:
        """True when a program, not the user, is driving the session (the product's
        non-interactive mode). Such sessions are not recalled or remembered unless the
        config's `remember_headless` is set."""
        return False

    def compact_summary(self, transcript_path: str | None) -> str:
        """The agent's own summary from its latest context compaction, if the product
        writes one to its transcript; "" otherwise."""
        return ""

    def emit_context(self, text: str) -> None:
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()

    def session_key(self, session_id: str) -> str:
        """Namespace session ids per product so two agents can't collide."""
        return f"{self.name}:{session_id}"


_PATH_IN_TEXT = re.compile(
    r"(?<![\w/.-])((?:~?/)?(?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|ts|tsx|js|jsx|go|rs|java|kt|dart|"
    r"swift|rb|c|h|cc|cpp|hpp|cs|sql|sh|toml|yaml|yml|json|md|html|css))\b"
)


def path_mentions(text: str) -> list[str]:
    """Source-file paths named in free text, in order, without duplicates."""
    return list(dict.fromkeys(m.group(1) for m in _PATH_IN_TEXT.finditer(text)))


def mentioned_files(text: str) -> list[CodeAnchor]:
    """File paths named in free text (a prompt) as MENTION anchors."""
    return [CodeAnchor(path=p, access=Access.MENTION) for p in path_mentions(text)]
