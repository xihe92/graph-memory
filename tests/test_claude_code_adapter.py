from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graph_memory.adapters import ClaudeCodeAdapter, StopEvent, load_adapter
from graph_memory.anchors import Access


def _e(uuid: str, parent: str | None, type_: str, content: Any, **kw: Any) -> dict[str, Any]:
    role = "assistant" if type_ == "assistant" else "user"
    return {
        "uuid": uuid,
        "parentUuid": parent,
        "type": type_,
        "isSidechain": False,
        "message": {"role": role, "content": content},
        **kw,
    }


def _write(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [json.dumps(e) for e in entries]
    lines.insert(2, '{"type": "ai-title", "title": "noise without uuid"}')
    lines.insert(3, "not json at all")
    path.write_text("\n".join(lines) + "\n")


def _transcript(tmp_path: Path) -> Path:
    p = tmp_path / "t.jsonl"
    _write(
        p,
        [
            _e("u1", None, "user", "first question"),
            _e("a1", "u1", "assistant", [{"type": "text", "text": "first answer"}]),
            # abandoned by a rewind:
            _e("u2", "a1", "user", "abandoned question"),
            _e("a2", "u2", "assistant", [{"type": "text", "text": "abandoned answer"}]),
            # the rewound conversation continues from a1:
            _e("u3", "a1", "user", "fix the race in /repo/svc/limiter.py"),
            _e("m1", "u3", "user", "<system-reminder>meta</system-reminder>", isMeta=True),
            _e("a3", "m1", "assistant", [{"type": "text", "text": "Let me look."}]),
            _e(
                "a4",
                "a3",
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "/repo/svc/limiter.py"},
                    }
                ],
            ),
            _e(
                "r1",
                "a4",
                "user",
                [{"type": "tool_result", "tool_use_id": "t1", "content": "code"}],
            ),
            _e(
                "a5",
                "r1",
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "Edit",
                        "input": {"file_path": "/repo/svc/limiter.py"},
                    }
                ],
            ),
            _e("r2", "a5", "user", [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]),
            _e(
                "a6",
                "r2",
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "t3",
                        "name": "Bash",
                        "input": {"command": "pytest  -q\n tests"},
                    }
                ],
            ),
            _e(
                "r3",
                "a6",
                "user",
                [{"type": "tool_result", "tool_use_id": "t3", "content": "1 passed"}],
            ),
            _e(
                "a7",
                "r3",
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "t4",
                        "name": "Agent",
                        "input": {"subagent_type": "Explore", "prompt": "find callers"},
                    }
                ],
            ),
            _e(
                "s1",
                "a7",
                "assistant",
                [{"type": "text", "text": "sidechain chatter"}],
                isSidechain=True,
            ),
            _e(
                "r4",
                "a7",
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t4",
                        "content": [{"type": "text", "text": "called from api.py"}],
                    }
                ],
            ),
            _e("a8", "r4", "assistant", [{"type": "thinking", "thinking": "hmm"}]),
            _e("a9", "a8", "assistant", [{"type": "text", "text": "Fixed it with a Lua script."}]),
        ],
    )
    return p


def test_read_turn(tmp_path: Path) -> None:
    p = _transcript(tmp_path)
    turn = ClaudeCodeAdapter().read_turn(
        StopEvent(session_id="x", transcript_path=str(p), cwd="/repo")
    )
    assert turn is not None
    assert turn.prompt == "fix the race in /repo/svc/limiter.py"
    assert turn.reply == "Fixed it with a Lua script."
    assert [(f.path, f.access) for f in turn.files] == [
        ("svc/limiter.py", Access.READ),
        ("svc/limiter.py", Access.WRITE),
    ]
    assert turn.commands == ["pytest -q tests"]
    assert [(s.name, s.task, s.result) for s in turn.subagents] == [
        ("Explore", "find callers", "called from api.py")
    ]
    assert turn.source is not None and (turn.source.first, turn.source.last) == ("u3", "a9")
    assert turn.previous_reply_ref == "a1"  # rewind visible: not a2


def test_unreadable_transcript_returns_none(tmp_path: Path) -> None:
    ad = ClaudeCodeAdapter()
    assert (
        ad.read_turn(StopEvent(session_id="x", transcript_path=str(tmp_path / "nope")), wait=0)
        is None
    )
    assert ad.read_turn(StopEvent(session_id="x"), wait=0) is None


def test_events_and_session_key() -> None:
    ad = load_adapter("claude-code")
    ev = ad.parse_prompt_event({"session_id": "abc", "prompt": "hi", "cwd": "/r"})
    assert ev.prompt == "hi" and ad.session_key(ev.session_id) == "claude-code:abc"
    assert isinstance(load_adapter("graph_memory.adapters:ClaudeCodeAdapter"), ClaudeCodeAdapter)


def test_end_events_tell_a_compaction_from_a_session_end() -> None:
    ad = ClaudeCodeAdapter()
    end = ad.parse_end_event({"session_id": "abc", "transcript_path": "/t/abc.jsonl"})
    assert end is not None and end.session_id == "abc" and not end.compacted
    assert end.transcript_path == "/t/abc.jsonl"
    after = ad.parse_end_event({"session_id": "abc", "source": "compact"})
    assert after is not None and after.compacted  # SessionStart after a compaction
    before = ad.parse_end_event({"session_id": "abc", "trigger": "auto"})
    assert before is not None and before.compacted  # PreCompact
    started = ad.parse_end_event({"session_id": "abc", "source": "startup"})
    assert started is not None and not started.compacted
    assert ad.parse_end_event({}) is None


def test_compact_summary_is_the_latest_one(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write(
        p,
        [
            _e("1", None, "user", "first summary", isCompactSummary=True),
            _e("2", "1", "assistant", [{"type": "text", "text": "ok"}]),
            _e("3", "2", "user", "second summary", isCompactSummary=True),
        ],
    )
    ad = ClaudeCodeAdapter()
    assert ad.compact_summary(str(p)) == "second summary"
    assert ad.compact_summary(str(tmp_path / "missing.jsonl")) == ""
    assert ad.compact_summary(None) == ""


# --------------------------------------------------------------------------- not typed turns


def _turn(tmp_path: Path, entries: list[dict[str, Any]]) -> Any:
    p = tmp_path / "x.jsonl"
    _write(p, entries)
    return ClaudeCodeAdapter().read_turn(StopEvent(session_id="x", transcript_path=str(p)), wait=0)


def _command(uuid: str, parent: str | None, name: str, args: str = "") -> dict[str, Any]:
    text = (
        f"<command-message>{name.lstrip('/')}</command-message>\n"
        f"<command-name>{name}</command-name>\n<command-args>{args}</command-args>"
    )
    return _e(uuid, parent, "user", text)


def _text_reply(uuid: str, parent: str, text: str) -> dict[str, Any]:
    return _e(uuid, parent, "assistant", [{"type": "text", "text": text}])


def test_a_skill_command_is_the_prompt(tmp_path: Path) -> None:
    # as `claude -p "/trip-planner:plan lisbon-weekend"` writes it
    turn = _turn(
        tmp_path,
        [
            _command("c1", None, "/trip-planner:plan", "lisbon-weekend"),
            _e("m1", "c1", "user", "Base directory for this skill: ...", isMeta=True),
            _e(
                "a1",
                "m1",
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "/s.java"},
                    }
                ],
            ),
            _e("r1", "a1", "user", [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]),
            _text_reply("a2", "r1", "# 217. Contains Duplicate"),
        ],
    )
    assert turn.prompt == "/trip-planner:plan lisbon-weekend"
    assert turn.reply == "# 217. Contains Duplicate"
    assert turn.source.first == "c1"


def test_a_command_without_arguments_or_slash(tmp_path: Path) -> None:
    text = "<command-name>init</command-name>\n<command-args></command-args>"
    turn = _turn(tmp_path, [_e("c1", None, "user", text), _text_reply("a1", "c1", "Wrote it.")])
    assert turn.prompt == "/init"


def test_a_built_in_command_is_not_a_prompt(tmp_path: Path) -> None:
    out = "<local-command-stdout>Set model to haiku</local-command-stdout>"
    entries = [
        _e("u1", None, "user", "start the tests in the background"),
        _text_reply("a1", "u1", "Started."),
        _command("c1", "a1", "/model", "haiku"),
        _e("o1", "c1", "user", out),
    ]
    # no reply after /model: the newest turn is still the first one
    assert _turn(tmp_path, entries).prompt == "start the tests in the background"
    # a reply after it (here, to a notification) is not an answer to /model either
    entries += [
        _e("n1", "o1", "user", "<task-notification>done</task-notification>"),
        _text_reply("a2", "n1", "All tests pass."),
    ]
    assert _turn(tmp_path, entries).prompt == "start the tests in the background"


def test_a_task_notification_continues_the_turn_that_started_the_task(tmp_path: Path) -> None:
    turn = _turn(
        tmp_path,
        [
            _e("u1", None, "user", "run the suite in the background"),
            _e(
                "a1",
                "u1",
                "assistant",
                [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}}],
            ),
            _e("r1", "a1", "user", [{"type": "tool_result", "tool_use_id": "t1", "content": "bg"}]),
            _text_reply("a2", "r1", "Started; I'll report back."),
            _e(
                "n1",
                "a2",
                "user",
                "<task-notification>\n<task-id>b1</task-id>",
                origin={"kind": "task-notification", "producer": "session-task"},
            ),
            _text_reply("a3", "n1", "All 136 tests pass."),
        ],
    )
    assert turn.prompt == "run the suite in the background"
    assert turn.reply == "Started; I'll report back.\n\nAll 136 tests pass."
    assert (turn.source.first, turn.source.last) == ("u1", "a3")
    assert turn.commands == ["pytest"]


def test_a_notification_is_known_by_its_origin_even_if_the_text_changes(tmp_path: Path) -> None:
    turn = _turn(
        tmp_path,
        [
            _e("u1", None, "user", "watch the deploy"),
            _text_reply("a1", "u1", "Watching."),
            _e("n1", "a1", "user", "Task b1 finished", origin={"kind": "task-notification"}),
            _text_reply("a2", "n1", "Deployed."),
        ],
    )
    assert turn.prompt == "watch the deploy"


def test_headless_sessions_are_known_by_the_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = ClaudeCodeAdapter()
    assert not ad.is_headless({})
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    assert not ad.is_headless({})
    for entry in ("sdk-cli", "sdk-py", "sdk-ts"):
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", entry)
        assert ad.is_headless({})


def test_a_command_entry_without_a_name_is_not_a_prompt(tmp_path: Path) -> None:
    turn = _turn(
        tmp_path,
        [
            _e("u1", None, "user", "real question"),
            _e("c1", "u1", "user", "<command-message>half written</command-message>"),
            _text_reply("a1", "c1", "answer"),
        ],
    )
    assert turn.prompt == "real question"


def test_pages_read_and_searches_run_are_kept(tmp_path: Path) -> None:
    def tool(
        uuid: str, parent: str, tid: str, name: str, inp: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            _e(
                uuid,
                parent,
                "assistant",
                [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
            ),
            _e(
                f"r{uuid}",
                uuid,
                "user",
                [{"type": "tool_result", "tool_use_id": tid, "content": "x"}],
            ),
        ]

    turn = _turn(
        tmp_path,
        [
            _e("u1", None, "user", "how do sqlite foreign keys work?"),
            *tool("a1", "u1", "t1", "WebSearch", {"query": "sqlite  foreign key pragma"}),
            *tool(
                "a2",
                "ra1",
                "t2",
                "WebFetch",
                {"url": "https://www.sqlite.org/foreignkeys.html", "prompt": "x"},
            ),
            *tool(
                "a3", "ra2", "t3", "WebFetch", {"url": "https://www.sqlite.org/foreignkeys.html"}
            ),
            _text_reply("a4", "ra3", "Each reference field holds one parent."),
        ],
    )
    assert turn.web == [
        "search: sqlite foreign key pragma",
        "https://www.sqlite.org/foreignkeys.html",
    ]
    assert turn.reply == "Each reference field holds one parent."
