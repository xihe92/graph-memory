"""Claude Code adapter.

Hook payloads (JSON on stdin) carry session_id, transcript_path, cwd and, for
UserPromptSubmit, the prompt. Text printed to stdout by a UserPromptSubmit hook is added
to the model's context.

The transcript is JSONL. Conversation entries have `uuid` and `parentUuid`; following
`parentUuid` back from the newest entry gives the active conversation, which skips
anything abandoned by a rewind. Assistant content is split across entries (one block
each: text, thinking, tool_use); tool results come back as user entries with
`tool_result` blocks. This format is internal to Claude Code, so parsing is defensive.

Entries that look like user turns but are not:
- A slash command is stored as `<command-name>/x</command-name><command-args>..`, then
  (for skills and custom commands) a meta entry with the expanded instructions. It is
  read as the prompt `/x args`. Built-in commands such as /model are followed by their
  `<local-command-stdout>` output instead, and are skipped.
- A background-task notification (`origin.kind == "task-notification"`) wakes the agent
  with no user involved. It is skipped: the reply that follows belongs to the turn that
  started the task, so that turn is read again, now ending at the newer reply.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..anchors import Access, CodeAnchor, SourceRef
from .base import AgentAdapter, PromptEvent, StopEvent, SubagentResult, Turn, path_mentions

WRITE_TOOLS = {
    "Edit": "file_path",
    "Write": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
}
READ_TOOLS = {"Read": "file_path"}
SHELL_TOOLS = {"Bash": "command"}
SUBAGENT_TOOLS = {"Task", "Agent"}
WEB_TOOLS = {"WebFetch": ("url", ""), "WebSearch": ("query", "search: ")}

# A shell command that changes files: redirection to a file, in-place edits, scripts that
# write, moves/copies/deletes, formatters. Crude, but most edits made through a shell
# (heredoc scripts, sed -i) would otherwise leave no trace of which files they touched.
_SHELL_WRITES = re.compile(
    r"(?<![0-9&])>>?\s*(?!/dev/null|&)\S|\bsed\s+-i|\btee\b|write_text|write_bytes|"
    r"\.write\(|open\([^)]*['\"][wa]['\"]|\b(?:mv|cp|rm|touch|patch)\s|"
    r"\b(?:ruff format|black|prettier --write|gofmt -w)\b"
)
_CD = re.compile(r"^\s*cd\s+(['\"]?)([^'\"&;|\s]+)\1\s*(?:&&|;)")
_COMMAND_NAME = re.compile(r"<command-name>\s*(.*?)\s*</command-name>", re.S)
_COMMAND_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.S)
_NOTIFICATION = "<task-notification>"
# Claude Code's entrypoint for print mode (`claude -p`) and the Agent SDK: a program
# drove the session, not the user. Hooks see it in their environment.
HEADLESS_ENTRYPOINTS = ("sdk-",)


class ClaudeCodeAdapter(AgentAdapter):
    name = "claude-code"

    # Claude Code delivers these through UserPromptSubmit too, but they are not user turns:
    # the transcript stores them as attachments.
    SYSTEM_PROMPT_PREFIXES = (_NOTIFICATION,)

    def is_headless(self, payload: dict[str, Any]) -> bool:
        entry = os.environ.get("CLAUDE_CODE_ENTRYPOINT", "")
        return entry.startswith(HEADLESS_ENTRYPOINTS)

    def parse_prompt_event(self, payload: dict[str, Any]) -> PromptEvent:
        prompt = str(payload.get("prompt", ""))
        return PromptEvent(
            session_id=str(payload["session_id"]),
            prompt=prompt,
            cwd=payload.get("cwd"),
            transcript_path=payload.get("transcript_path"),
            system=prompt.lstrip().startswith(self.SYSTEM_PROMPT_PREFIXES),
        )

    def parse_stop_event(self, payload: dict[str, Any]) -> StopEvent:
        return StopEvent(
            session_id=str(payload["session_id"]),
            cwd=payload.get("cwd"),
            transcript_path=payload.get("transcript_path"),
        )

    def compact_summary(self, transcript_path: str | None) -> str:
        if not transcript_path:
            return ""
        path = Path(transcript_path).expanduser()
        return _latest_compact_summary(path) if path.exists() else ""

    def read_turn(self, event: StopEvent, *, wait: float = 1.0) -> Turn | None:
        if not event.transcript_path:
            return None
        path = Path(event.transcript_path).expanduser()
        deadline = time.monotonic() + wait
        while True:  # the final entry can land a moment after the Stop hook fires
            turn = _read_turn(path, event.cwd)
            if turn is not None or time.monotonic() >= deadline:
                return turn
            time.sleep(0.1)


def _latest_compact_summary(path: Path) -> str:
    text = ""
    for entry in _load(path):
        if entry.get("isCompactSummary"):
            text = _text(_blocks(entry)) or text
    return text


def _load(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(e, dict) and e.get("uuid"):
                    entries.append(e)
    except OSError:
        return []
    return entries


def _blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _is_main(entry: dict[str, Any]) -> bool:
    return not entry.get("isSidechain") and not entry.get("isMeta")


def _user_text(entry: dict[str, Any]) -> str | None:
    """The text of a main-agent user entry that is not a tool result; else None."""
    if entry.get("type") != "user" or not _is_main(entry):
        return None
    blocks = _blocks(entry)
    if not blocks or any(b.get("type") == "tool_result" for b in blocks):
        return None
    return _text(blocks).lstrip()


def _is_notification(entry: dict[str, Any]) -> bool:
    origin = entry.get("origin")
    if isinstance(origin, dict) and origin.get("kind") == "task-notification":
        return entry.get("type") == "user"
    text = _user_text(entry)
    return text is not None and text.startswith(_NOTIFICATION)


def _command(text: str) -> str | None:
    """`/name args` for a stored slash command, else None."""
    m = _COMMAND_NAME.search(text)
    if not m or not m.group(1):
        return None
    name = m.group(1) if m.group(1).startswith("/") else "/" + m.group(1)
    a = _COMMAND_ARGS.search(text)
    args = " ".join(a.group(1).split()) if a else ""
    return f"{name} {args}" if args else name


def _prompt_text(entry: dict[str, Any], after: dict[str, Any] | None) -> str | None:
    """The prompt this entry gave, or None if it is not a prompt. `after` is the entry
    that follows it on the conversation chain."""
    text = _user_text(entry)
    if not text or _is_notification(entry):
        return None
    if text.startswith("<command-"):
        # a built-in command answers with its own output, not with the model
        local = after is not None and (_user_text(after) or "").startswith("<local-command")
        return None if local else _command(text)
    if text.startswith("<local-command"):
        return None
    return text


def _is_reply(entry: dict[str, Any]) -> bool:
    return entry.get("type") == "assistant" and _is_main(entry)


def _text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text").strip()


def _tool_result_text(block: dict[str, Any]) -> str:
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
    return ""


def _rel(path: str, cwd: str | None) -> str:
    if cwd and os.path.isabs(path):
        try:
            rel = os.path.relpath(path, cwd)
        except ValueError:
            return path
        if not rel.startswith(".."):
            return rel
    return path


def _shell_files(command: str, cwd: str | None) -> list[CodeAnchor]:
    """Source files named in a shell command, relative to the session's directory."""
    base = cwd
    m = _CD.match(command)
    if m:
        target = os.path.expanduser(m.group(2))
        base = target if os.path.isabs(target) else os.path.join(cwd or "", target)
    access = Access.WRITE if _SHELL_WRITES.search(command) else Access.READ
    out: list[CodeAnchor] = []
    for raw in path_mentions(command)[:50]:
        full = raw if os.path.isabs(raw) or not base else os.path.join(base, raw)
        out.append(CodeAnchor(path=_rel(os.path.normpath(full), cwd), access=access))
    return out


def _read_turn(path: Path, cwd: str | None) -> Turn | None:
    entries = _load(path)
    if not entries:
        return None
    by_id = {e["uuid"]: e for e in entries}

    # Newest main-agent reply = end of the turn that just finished.
    tail = next((e for e in reversed(entries) if _is_reply(e)), None)
    if tail is None:
        return None
    return _turn_at(by_id, tail, path, cwd)


def read_turns(path: Path, cwd: str | None = None) -> list[tuple[str, Turn]]:
    """Every turn on the session's active conversation, oldest first, each with the time
    of its last reply (ISO). Follows the chain back from the newest entry, across
    compactions (a compaction's boundary links to what came before through
    `logicalParentUuid`), so rewound branches are left out. For rebuilding what a
    session taught after its buffered exchanges were graduated and deleted."""
    entries = _load(path)
    if not entries:
        return []
    by_id = {e["uuid"]: e for e in entries}
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur: dict[str, Any] | None = entries[-1]
    while cur is not None and cur["uuid"] not in seen:
        seen.add(cur["uuid"])
        chain.append(cur)
        parent = cur.get("parentUuid") or cur.get("logicalParentUuid")
        cur = by_id.get(parent) if parent else None
    chain.reverse()
    # a turn ends with the last reply before the next prompt
    out: list[tuple[str, Turn]] = []
    tail: dict[str, Any] | None = None
    for n, e in enumerate(chain):
        nxt = chain[n + 1] if n + 1 < len(chain) else None
        if _prompt_text(e, nxt) is not None and tail is not None:
            turn = _turn_at(by_id, tail, path, cwd or tail.get("cwd"))
            if turn is not None:
                out.append((str(tail.get("timestamp") or ""), turn))
            tail = None
        if _is_reply(e):
            tail = e
    if tail is not None:
        turn = _turn_at(by_id, tail, path, cwd or tail.get("cwd"))
        if turn is not None:
            out.append((str(tail.get("timestamp") or ""), turn))
    return out


def find_transcript(session_id: str, root: Path | None = None) -> Path | None:
    """Where Claude Code keeps a session's transcript (`~/.claude/projects/<dir>/<id>.jsonl`,
    or under CLAUDE_CONFIG_DIR), by its id or an unambiguous prefix; None when there is
    none (they are cleaned up after 30 days by default) or the prefix is ambiguous."""
    base = root or Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    hits = {p.stem: p for p in (base / "projects").glob(f"*/{session_id}*.jsonl")}
    if session_id in hits:
        return hits[session_id]
    return next(iter(hits.values())) if len(hits) == 1 else None


def _turn_at(
    by_id: dict[str, dict[str, Any]], tail: dict[str, Any], path: Path, cwd: str | None
) -> Turn | None:
    """The turn whose last reply is `tail`: back along the chain to its prompt."""
    # Walk back along the active chain to the prompt that started this turn.
    chain: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = tail
    prompt_entry: dict[str, Any] | None = None
    prompt = ""
    while cur is not None:
        text = _prompt_text(cur, chain[-1] if chain else None)
        if text is not None:
            prompt_entry, prompt = cur, text
            break
        chain.append(cur)
        parent = cur.get("parentUuid")
        cur = by_id.get(parent) if parent else None
    if prompt_entry is None:
        return None
    chain.reverse()

    # Previous turn's last reply on the active chain (for rewind detection).
    prev_reply: str | None = None
    parent = prompt_entry.get("parentUuid")
    cur = by_id.get(parent) if parent else None
    while cur is not None:
        if _is_reply(cur):
            prev_reply = cur["uuid"]
            break
        parent = cur.get("parentUuid")
        cur = by_id.get(parent) if parent else None

    files: list[CodeAnchor] = []
    commands: list[str] = []
    web: list[str] = []
    subagent_calls: dict[str, SubagentResult] = {}
    last_tool_result_idx = -1
    for i, e in enumerate(chain):
        for b in _blocks(e):
            btype = b.get("type")
            if btype == "tool_use" and _is_reply(e):
                name = str(b.get("name", ""))
                inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                assert isinstance(inp, dict)
                if name in WRITE_TOOLS and inp.get(WRITE_TOOLS[name]):
                    files.append(
                        CodeAnchor(path=_rel(str(inp[WRITE_TOOLS[name]]), cwd), access=Access.WRITE)
                    )
                elif name in READ_TOOLS and inp.get(READ_TOOLS[name]):
                    files.append(
                        CodeAnchor(path=_rel(str(inp[READ_TOOLS[name]]), cwd), access=Access.READ)
                    )
                elif name in SHELL_TOOLS and inp.get(SHELL_TOOLS[name]):
                    command = str(inp[SHELL_TOOLS[name]])
                    commands.append(" ".join(command.split())[:300])
                    files.extend(_shell_files(command, cwd))
                elif name in WEB_TOOLS and inp.get(WEB_TOOLS[name][0]):
                    key, prefix = WEB_TOOLS[name]
                    web.append(prefix + " ".join(str(inp[key]).split())[:300])
                elif name in SUBAGENT_TOOLS:
                    subagent_calls[str(b.get("id"))] = SubagentResult(
                        name=str(inp.get("subagent_type") or inp.get("description") or "subagent"),
                        task=str(inp.get("prompt") or inp.get("description") or "")[:2000],
                    )
            elif btype == "tool_result":
                last_tool_result_idx = i
                tid = str(b.get("tool_use_id"))
                if tid in subagent_calls:
                    subagent_calls[tid].result = _tool_result_text(b)[:20000]

    # The reply is the text written after the last tool result; earlier text was
    # narration between tool calls.
    final = [e for e in chain[last_tool_result_idx + 1 :] if _is_reply(e)]
    reply = "\n\n".join(t for t in (_text(_blocks(e)) for e in final) if t)
    if not reply:
        reply = "\n\n".join(t for t in (_text(_blocks(e)) for e in chain if _is_reply(e)) if t)

    return Turn(
        prompt=prompt.strip(),
        reply=reply,
        files=files,
        commands=commands,
        web=list(dict.fromkeys(web))[:20],
        subagents=[s for s in subagent_calls.values() if s.result],
        source=SourceRef(path=str(path), first=prompt_entry["uuid"], last=tail["uuid"]),
        previous_reply_ref=prev_reply,
        cwd=cwd,
    )
