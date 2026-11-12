"""`gm migrate`: a memory home in the old format becomes knowledge-only, losing nothing
that was not graduated, with the old files archived before they are deleted."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from graph_memory import cli
from graph_memory.buffer import SessionBuffer
from graph_memory.client import call
from graph_memory.config import load_config
from graph_memory.knowledge import KnowledgeStore
from graph_memory.migrate import apply, plan

from .test_daemon import _start


def _node(home: Path, meta: dict[str, Any], content: str) -> None:
    (home / "nodes").mkdir(exist_ok=True)
    text = (
        f"---\n{json.dumps(meta, indent=2)}\n---\n\n# {meta['role']}\n\n---\n"
        f"\n<!-- gme:content -->\n{content}"
    )
    (home / "nodes" / f"{meta['node_id']}.md").write_text(text)


def old_home(home: Path) -> None:
    """Two sessions in a project topic: `a` graduated, `b` not; plus a subagent result."""
    topic = {"node_id": "t1", "role": "topic", "session_id": "topics",
             "topic_key": "repo:/w/limiter", "created_at": "2026-11-01T09:00:00Z"}  # fmt: skip
    _node(home, topic, "doc")
    rows = [
        ("a", "pa1", "ra1", "2026-11-01T10:00:00Z", "add a lock", "done"),
        ("b", "pb1", "rb1", "2026-11-02T10:00:00Z", "add retries", "backoff added"),
        ("b", "pb2", "rb2", "2026-11-02T10:05:00Z", "and tests?", "green"),
    ]
    for s, pid, rid, ts, prompt, reply in rows:
        _node(home, {"node_id": pid, "role": "user", "session_id": f"claude-code:{s}",
                     "parent_id": "t1", "created_at": ts}, prompt)  # fmt: skip
        _node(
            home,
            {
                "node_id": rid,
                "role": "agent",
                "session_id": f"claude-code:{s}",
                "parent_id": pid,
                "created_at": ts,
                "code_context": [{"path": "svc/limiter.py", "access": "write"}],
                "commands": ["pytest"],
                "source": {"path": "/t.jsonl", "first": "u1", "last": "a1"},
            },
            reply,
        )
    _node(home, {"node_id": "sub", "role": "agent", "agent_name": "Explore",
                 "session_id": "claude-code:b", "parent_id": "rb1",
                 "created_at": "2026-11-02T10:01:00Z"}, "subagent notes")  # fmt: skip
    (home / "graph.json").write_text(json.dumps({"schema_version": 2, "nodes": []}))
    (home / "chroma").mkdir()
    (home / "chroma" / "chroma.sqlite3").write_bytes(b"\0" * 1000)
    knowledge = {
        "schema": 1,
        "entities": [],
        "items": [
            {
                "item_id": "i1",
                "type": "decision",
                "statement": "svc/limiter.py guards acquire() with a SETNX lock",
                "evidence": ["ra1"],
                "segments": ["g1"],
            }
        ],
        "edges": [],
        "segments": [
            {
                "segment_id": "g1",
                "session": "claude-code:a",
                "exchanges": ["ra1"],
                "start": "2026-11-01T10:00:00Z",
                "end": "2026-11-01T10:00:00Z",
                "trigger": "session_end",
                "compact_summary": "x" * 500,
            }
        ],
        "cursors": {"claude-code:a": "2026-11-01T10:00:00+00:00"},
    }
    (home / "knowledge.json").write_text(json.dumps(knowledge))


def test_plan_and_apply(short_dir: Path) -> None:
    old_home(short_dir)
    p = plan(short_dir)
    assert p.needed and p.nodes == 8 and p.exchanges == 3
    assert {s: [x.prompt for x in xs] for s, xs in p.pending.items()} == {
        "claude-code:b": ["add retries", "and tests?"]
    }
    assert p.items == 1 and p.items_without_sources == 1
    text = p.text()
    assert "2 exchanges never graduated, from 1 sessions" in text and "archive-" in text

    res = apply(p)
    assert res["migrated"] and res["buffered"] == 2
    assert res["removed"] == ["nodes", "graph.json", "chroma"]
    for f in ("nodes", "graph.json", "chroma"):
        assert not (short_dir / f).exists()

    buf = SessionBuffer(short_dir)
    xs = buf.exchanges("claude-code:b")
    assert [(x.prompt, x.reply, x.project) for x in xs] == [
        ("add retries", "backoff added", "repo:/w/limiter"),
        ("and tests?", "green", "repo:/w/limiter"),
    ]
    assert xs[0].files[0].path == "svc/limiter.py" and xs[0].source and xs[0].source.last == "a1"

    raw = json.loads((short_dir / "knowledge.json").read_text())
    assert "cursors" not in raw and "evidence" not in raw["items"][0]
    assert raw["segments"][0]["exchanges"] == 1 and "compact_summary" not in raw["segments"][0]
    (item,) = KnowledgeStore(short_dir).items.values()
    assert [(s.session, s.until.isoformat()) for s in item.sources] == [
        ("claude-code:a", "2026-11-01T10:00:00+00:00")
    ]

    with tarfile.open(res["archive"]) as tar:
        names = tar.getnames()
        old = json.load(tar.extractfile("knowledge.before-migration.json"))  # type: ignore[arg-type]
    assert {"nodes", "graph.json", "chroma", "knowledge.before-migration.json"} <= set(names)
    assert "nodes/rb2.md" in names and old["cursors"]
    # run again: nothing left to do, and the buffer isn't filled twice
    assert not plan(short_dir).needed and apply(plan(short_dir)) == {"migrated": False}
    assert "Nothing to migrate" in plan(short_dir).text()


def test_gm_migrate_refuses_while_the_daemon_runs(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old_home(short_dir)
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["migrate", "--dry-run"]) == 0
    assert "never graduated" in capsys.readouterr().out
    assert (short_dir / "graph.json").exists()  # a dry run changes nothing

    cfg = load_config(short_dir)
    daemon, t = _start(cfg)
    try:
        assert call(cfg, "status", autostart=False)["needs_migration"] is True
        assert cli.main(["migrate"]) == 1
        assert "stop it first" in capsys.readouterr().err
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)
    assert cli.main(["migrate"]) == 0
    assert "Done: 2 exchanges moved to the buffer" in capsys.readouterr().out
    assert cli.main(["migrate"]) == 0
    assert "Nothing to migrate" in capsys.readouterr().out
