"""Graduating a session by hand, and rebuilding what a session taught from its transcript
under the current rules."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from graph_memory import cli
from graph_memory.adapters.claude_code import find_transcript, read_turns
from graph_memory.buffer import Summary
from graph_memory.client import call
from graph_memory.config import Config
from graph_memory.knowledge import Item, ItemType, Source

from .test_claude_code_adapter import _e
from .test_daemon import _wait
from .test_daemon_buffer import _events, _graduate, _remember, cfg, daemon_with  # noqa: F401
from .test_graduation import LOCK, Script
from .test_monitoring import _req

SID = "5e55-1011-aaaa"


def _reply(uuid: str, parent: str, text: str, ts: str) -> dict[str, Any]:
    return _e(
        uuid, parent, "assistant", [{"type": "text", "text": text}], timestamp=ts, cwd="/w/limiter"
    )


def _transcript(root: Path) -> Path:
    """Three turns on the active chain, a rewound branch, a compaction in between."""
    d = root / "projects" / "-w-limiter"
    d.mkdir(parents=True)
    entries = [
        _e("u1", None, "user", "add a lock to the limiter", timestamp="2026-11-01T10:00:00Z"),
        _reply("a1", "u1", "Added a SETNX lock around acquire().", "2026-11-01T10:01:00Z"),
        _e("u2", "a1", "user", "rewound question", timestamp="2026-11-01T10:02:00Z"),
        _reply("a2", "u2", "abandoned answer", "2026-11-01T10:03:00Z"),
        _e("u3", "a1", "user", "and retries?", timestamp="2026-11-01T10:05:00Z"),
        _reply("a3", "u3", "acquire() retries 5 times with backoff.", "2026-11-01T10:06:00Z"),
        # a compaction: the boundary has no parent, only a logical one
        {
            "uuid": "c1",
            "parentUuid": None,
            "logicalParentUuid": "a3",
            "type": "system",
            "subtype": "compact_boundary",
            "timestamp": "2026-11-01T11:00:00Z",
        },  # fmt: skip
        _e("u4", "c1", "user", "still waiting question", timestamp="2026-11-01T11:05:00Z"),
        _reply("a4", "u4", "still waiting answer", "2026-11-01T11:06:00Z"),
    ]
    p = d / f"{SID}.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return p


def test_read_turns_follows_the_active_chain_across_a_compaction(tmp_path: Path) -> None:
    p = _transcript(tmp_path)
    turns = read_turns(p)
    assert [(ts[:16], t.prompt) for ts, t in turns] == [
        ("2026-11-01T10:01", "add a lock to the limiter"),
        ("2026-11-01T10:06", "and retries?"),
        ("2026-11-01T11:06", "still waiting question"),
    ]
    assert turns[0][1].cwd == "/w/limiter" and turns[1][1].source.first == "u3"  # type: ignore[union-attr]
    assert find_transcript(SID, tmp_path) == p
    assert find_transcript("5e55", tmp_path) == p  # a prefix
    assert find_transcript("nope", tmp_path) is None
    (p.parent / "5e55-other.jsonl").write_text("")
    assert find_transcript("5e55", tmp_path) is None  # ambiguous
    assert read_turns(tmp_path / "missing.jsonl") == []


def _old_session(d: Any, session: str) -> None:
    """What the old rules left: two summaries that kept nothing, an item it did learn,
    and one exchange still waiting (the turn after the compaction)."""
    from graph_memory.anchors import SourceRef
    from graph_memory.buffer import Exchange

    for n in (1, 2):
        ex = d.buffer.append(session, Exchange(prompt=f"old {n}", reply="r"))
        d.buffer.summarize(
            session,
            [ex.id],
            Summary(trigger="idle", segment=f"seg{n}", text="Nothing worth keeping was learned."),
        )
    d.buffer.append(
        session,
        Exchange(
            prompt="still waiting question",
            reply="still waiting answer",
            source=SourceRef(first="u4"),
        ),
    )
    lock = LOCK["items"][0]
    d.knowledge.add_item(
        Item(
            type=ItemType.DECISION,
            statement=lock["statement"],
            sources=[Source(session=session, until=d.buffer.records(session)[0].ts)],
        )
    )


def test_a_rebuild_replaces_the_old_summaries_and_counts_nothing_twice(
    cfg: Config,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    _transcript(tmp_path)
    session = f"claude-code:{SID}"
    ask = Script([{**LOCK, "summary": "Added a SETNX lock and retries to the limiter."}])
    with daemon_with(cfg, ask) as d:
        _old_session(d, session)
        lock = next(iter(d.knowledge.items.values()))
        res = call(cfg, "regraduate", {"session": "5e55"}, autostart=False)
        assert res["session"] == session and res["queued"]
        _wait(lambda: any(e.get("trigger") == "rebuild" for e in _events(cfg, "graduate")))
        _wait(d.settled)
        (ev,) = [e for e in _events(cfg, "graduate") if e.get("trigger") == "rebuild"]
        assert ev["ok"] and ev["exchanges"] == 2 and ev["replaced"] == 2
        # the turns that had graduated, not the rewound one, not the one still waiting
        (extract,) = [p for p in ask.prompts if "distil" in p]
        assert "add a lock" in extract and "and retries?" in extract
        assert "rewound" not in extract and "still waiting" not in extract
        # what the session taught before is matched, not strengthened; the rest is new
        assert lock.strength == 1.0 and len(d.knowledge.items) == 2
        recs = d.buffer.records(session)
        assert [r.kind for r in recs] == ["summary", "exchange"]
        summary = recs[0]
        assert isinstance(summary, Summary)
        assert summary.trigger == "rebuild" and summary.exchanges == 2
        assert summary.text == "Added a SETNX lock and retries to the limiter."
        assert summary.folded == ["seg1", "seg2"]
        # the Sessions page: one summary, no separate nodes for what it replaced
        (view,) = call(cfg, "sessions", autostart=False)["sessions"]
        assert [r["kind"] for r in view["records"]] == ["summary", "exchange"]
        assert recs[1].prompt == "still waiting question"  # type: ignore[union-attr]


def test_a_rebuild_of_an_ended_session_keeps_no_buffer(
    cfg: Config,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    _transcript(tmp_path)
    with daemon_with(cfg, Script([LOCK])) as d:
        # the last turn is still in progress (its prompt arrived, its reply didn't)
        call(cfg, "recall", {"session": f"claude-code:{SID}", "prompt": "still waiting question"})
        call(cfg, "regraduate", {"session": SID}, autostart=False)
        _wait(lambda: bool(_events(cfg, "graduate")))
        _wait(d.settled)
        assert len(d.knowledge.items) == 2 and d.buffer.sessions() == []
        (extract,) = [p for p in d.ask.prompts if "distil" in p]  # type: ignore[union-attr]
        assert "and retries?" in extract and "still waiting" not in extract
        with pytest.raises(RuntimeError, match="no transcript"):
            call(cfg, "regraduate", {"session": "zzzz"}, autostart=False)


def test_graduate_a_session_now_from_gm_and_the_page(
    cfg: Config,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from graph_memory.webui import UIServer

    monkeypatch.setenv("GM_HOME", str(cfg.home))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with daemon_with(
        cfg, Script([{**LOCK, "summary": "a lock"}, {"entities": [], "items": []}])
    ) as d:
        _remember(cfg, "claude-code:abc123", "add a lock", "done")
        _remember(cfg, "claude-code:abd456", "other", "x")
        assert cli.main(["knowledge", "graduate", "ab"]) == 1
        assert "ambiguous session" in capsys.readouterr().err
        assert cli.main(["knowledge", "graduate"]) == 2
        assert cli.main(["knowledge", "graduate", "abc", "--wait"]) == 0
        out = capsys.readouterr().out
        assert (
            "graduating claude-code:abc123 (1 exchanges)" in out
            and "done: 1 exchanges -> 2 items" in out
        )
        assert [r.kind for r in d.buffer.records("claude-code:abc123")] == ["summary"]
        assert cli.main(["knowledge", "graduate", "abc"]) == 0
        assert "nothing waiting" in capsys.readouterr().out

        ui = UIServer(cfg, "127.0.0.1", 0)
        import threading

        threading.Thread(target=ui.serve_forever, daemon=True).start()
        try:
            status, res = _req(
                ui, "/api/graduate", {"session": "claude-code:abd456"}, token=ui.token
            )
            assert status == 200 and res["queued"] and res["waiting"] == 1
            _wait(lambda: d.buffer.exchanges("claude-code:abd456") == [])
            status, _ = _req(ui, "/api/graduate", {"session": "nope"}, token=ui.token)
            assert status == 400
        finally:
            ui.shutdown()
            ui.server_close()


def test_a_rebuild_whose_merge_fails_changes_nothing(
    cfg: Config,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    _transcript(tmp_path)
    session = f"claude-code:{SID}"
    boom = RuntimeError("model down")
    with daemon_with(cfg, Script([LOCK], merges=[boom, boom])) as d:
        _old_session(d, session)
        before = [r.model_dump() for r in d.buffer.records(session)]
        call(cfg, "regraduate", {"session": SID}, autostart=False)
        _wait(lambda: bool(_events(cfg, "graduate")))
        _wait(d.settled)
        (ev,) = _events(cfg, "graduate")
        assert not ev["ok"] and "merge call failed twice" in ev["error"]
        assert len(d.knowledge.items) == 1  # nothing added
        assert [r.model_dump() for r in d.buffer.records(session)] == before


def _item(text: str, kind: str = "decision") -> dict[str, Any]:
    return {"type": kind, "statement": text, "entities": ["svc/limiter.py"], "scope": "project"}


def _waiting_session(d: Any, session: str) -> None:
    from graph_memory.anchors import SourceRef
    from graph_memory.buffer import Exchange

    d.buffer.append(
        session,
        Exchange(prompt="still waiting question", reply="x", source=SourceRef(first="u4")),
    )


SETNX = "svc/limiter.py guards acquire() with a Redis SETNX lock"
LUA = "svc/limiter.py guards acquire() with a Lua script instead of the SETNX lock"


def test_a_long_session_is_replayed_in_order(
    cfg: Config,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parts: the second retires what the first learned, as live graduation would."""
    from graph_memory import graduation

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(graduation, "CHUNK_CHARS", 100)  # one turn per part
    _transcript(tmp_path)
    session = f"claude-code:{SID}"
    ask = Script(
        [
            {"entities": [], "items": [_item(SETNX)], "summary": "added a SETNX lock"},
            {"entities": [], "items": [_item(LUA)], "summary": "replaced it with Lua"},
        ],
        merges=[{"items": [{"id": "N1", "invalidates": [{"id": "K1", "reason": "Lua now"}]}]}],
    )
    with daemon_with(cfg, ask) as d:
        _waiting_session(d, session)
        call(cfg, "regraduate", {"session": SID}, autostart=False)
        _wait(lambda: bool(_events(cfg, "graduate")))
        _wait(d.settled)
        (ev,) = _events(cfg, "graduate")
        assert ev["ok"] and ev["parts"] == 2 and ev["new"] == 2 and ev["retired"] == 0
        by = {i.statement: i for i in d.knowledge.items.values()}
        assert not by[SETNX].valid and by[SETNX].invalidated_by == by[LUA].item_id
        extracts = [p for p in ask.prompts if "distil" in p]
        assert "<earlier>\nadded a SETNX lock\n</earlier>" in extracts[1]
        (summary,) = d.buffer.summaries(session)
        assert summary.text == "replaced it with Lua" and summary.graduations == 2


def test_a_long_session_is_previewed_first_unless_forced(
    cfg: Config,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Replaying a long session can bring back many stale items: it is previewed first."""
    from graph_memory import daemon, graduation

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("GM_HOME", str(cfg.home))
    monkeypatch.setattr(graduation, "CHUNK_CHARS", 100)  # one turn per part
    monkeypatch.setattr(daemon, "REBUILD_DIRECT_PARTS", 1)
    _transcript(tmp_path)
    session = f"claude-code:{SID}"
    run = [
        {"entities": [], "items": [_item(SETNX)], "summary": "added a SETNX lock"},
        {"entities": [], "items": [_item(LUA)], "summary": "replaced it with Lua"},
    ]
    merge = {"items": [{"id": "N1", "invalidates": [{"id": "K1", "reason": "Lua now"}]}]}
    with daemon_with(cfg, Script(run + run, merges=[merge, merge])) as d:
        _waiting_session(d, session)
        assert cli.main(["knowledge", "regraduate", "5e55", "--wait"]) == 0
        out = capsys.readouterr().out
        assert "previewed first" in out and "long session (2 parts)" in out
        assert "gm knowledge regraduate 5e55-101 --apply" in out
        (ev,) = _events(cfg, "graduate")
        assert ev["ok"] and ev["preview"] and ev["long"]
        assert d.knowledge.items == {}  # nothing changed yet
        assert cli.main(["knowledge", "regraduate", "5e55", "--force", "--wait"]) == 0
        assert "previewed first" not in capsys.readouterr().out
        assert not _events(cfg, "graduate")[-1]["preview"]
        assert sorted(i.statement for i in d.knowledge.items.values()) == [LUA, SETNX]


def test_history_is_matched_not_revived(
    cfg: Config,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    _transcript(tmp_path)
    ask = Script([{"entities": [], "items": [_item(SETNX), _item(LUA)]}])
    with daemon_with(cfg, ask) as d:
        old = d.knowledge.add_item(Item(type=ItemType.DECISION, statement=SETNX))
        d.knowledge.retract(old.item_id, "replaced by Lua")
        call(cfg, "regraduate", {"session": SID}, autostart=False)
        _wait(lambda: bool(_events(cfg, "graduate")))
        _wait(d.settled)
        assert not old.valid  # matched by its statement, still history
        assert sorted(i.statement for i in d.knowledge.items.values()) == [LUA, SETNX]
        (merge,) = [p for p in ask.prompts if "distil" not in p]
        assert "(NO LONGER TRUE since " in merge and "replaced by Lua)" in merge


def test_a_preview_changes_nothing_until_it_is_applied(
    cfg: Config,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("GM_HOME", str(cfg.home))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    _transcript(tmp_path)
    session = f"claude-code:{SID}"
    ask = Script(
        [
            {"entities": [], "items": [_item(LUA)], "summary": "Lua now"},
            {"entities": [], "items": [_item(LUA)], "summary": "Lua now"},
        ],
        merges=[
            {"items": [{"id": "N1", "invalidates": [{"id": "K1", "reason": "Lua now"}]}]},
            {"items": [{"id": "N1", "invalidates": [{"id": "K1", "reason": "Lua now"}]}]},
        ],
    )
    with daemon_with(cfg, ask) as d:
        _old_session(d, session)
        (old_lock,) = d.knowledge.items.values()  # what the session taught before
        before = ([r.model_dump() for r in d.buffer.records(session)], len(d.knowledge.items))
        assert cli.main(["knowledge", "regraduate", "5e55", "--preview", "--wait"]) == 0
        out = capsys.readouterr().out
        assert "would add 1 items" in out and "retire 1" in out and LUA in out
        assert "gm knowledge regraduate 5e55-101 --apply" in out
        # nothing changed yet
        assert (
            [r.model_dump() for r in d.buffer.records(session)],
            len(d.knowledge.items),
        ) == before
        assert old_lock.valid

        assert cli.main(["knowledge", "regraduate", "5e55", "--apply"]) == 0
        assert "1 new items, 1 no longer true, 2 summaries replaced" in capsys.readouterr().out
        k = d.knowledge
        assert not k.items[old_lock.item_id].valid and any(
            i.statement == LUA for i in k.items.values()
        )
        assert [r.kind for r in d.buffer.records(session)] == ["summary", "exchange"]
        assert not (cfg.home / "rebuild-preview").exists() or not any(
            (cfg.home / "rebuild-preview").iterdir()
        )
        assert cli.main(["knowledge", "regraduate", "5e55", "--apply"]) == 1
        assert "no rebuild preview" in capsys.readouterr().err

        # a preview that memory moved on from is refused
        call(cfg, "regraduate", {"session": SID, "preview": True}, autostart=False)
        _wait(
            lambda: (
                (cfg.home / "rebuild-preview").exists()
                and any((cfg.home / "rebuild-preview").glob("*/preview.json"))
            )
        )
        _wait(d.settled)
        d.knowledge.add_item(Item(type=ItemType.FACT, statement="learned meanwhile"))
        assert cli.main(["knowledge", "regraduate", "5e55", "--apply"]) == 1
        assert "memory changed since the preview" in capsys.readouterr().err


def test_resummarize_folds_a_sessions_summaries_and_leaves_knowledge_alone(
    cfg: Config,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from graph_memory.buffer import Exchange

    monkeypatch.setenv("GM_HOME", str(cfg.home))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    asked: list[str] = []

    def write(prompt: str) -> str:
        asked.append(prompt)
        return "  The whole session,\n in one summary.  "

    session = "claude-code:fold1234"
    with daemon_with(cfg, write) as d:
        for n, text in enumerate(["Nothing worth keeping was learned.", "[fact] early", "later"]):
            ex = d.buffer.append(session, Exchange(prompt=f"q{n}", reply="r"))
            d.buffer.summarize(
                session, [ex.id], Summary(trigger="idle", segment=f"g{n}", exchanges=2, text=text)
            )
        d.buffer.append(session, Exchange(prompt="waiting", reply="r"))
        now = d.buffer.records(session)[0].ts
        d.knowledge.add_item(
            Item(
                type=ItemType.DECISION,
                statement="the current design",
                sources=[Source(session=session, until=now)],
            )
        )
        gone = d.knowledge.add_item(
            Item(
                type=ItemType.DECISION,
                statement="the first design",
                sources=[Source(session=session, until=now)],
            )
        )
        d.knowledge.retract(gone.item_id, "replaced")
        before = len(d.knowledge.items), d.knowledge.version

        assert cli.main(["knowledge", "resummarize", "fold"]) == 0
        assert "3 summaries folded into one" in capsys.readouterr().out
        recs = d.buffer.records(session)
        assert [r.kind for r in recs] == ["summary", "exchange"]
        s = recs[0]
        assert isinstance(s, Summary)
        assert s.text == "The whole session, in one summary." and s.trigger == "folded"
        assert (s.exchanges, s.graduations, s.segment, s.folded) == (6, 3, "g2", ["g0", "g1"])
        (prompt,) = asked
        assert "the current design" in prompt and "the first design" not in prompt
        assert "Nothing worth keeping" not in prompt and "later" in prompt
        assert (len(d.knowledge.items), d.knowledge.version) == before  # knowledge untouched

        assert cli.main(["knowledge", "resummarize", "fold"]) == 0
        assert "nothing to fold" in capsys.readouterr().out
