"""Summaries: a brief of what memory knows about one project or topic, written by the
model for a person. Every point cites its notes; the brief is cached, never knowledge;
the page is told how many notes changed since."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from graph_memory import cli
from graph_memory.client import RemoteError, call
from graph_memory.config import Config
from graph_memory.graduation import GraduationError
from graph_memory.knowledge import Item, ItemType, KnowledgeStore
from graph_memory.summarize import build_prompt, changed_since, parse, select, stamp, summarize
from graph_memory.webui import UIServer

from .conftest import topic_embed
from .test_daemon import _wait
from .test_daemon_buffer import _events, cfg, daemon_with  # noqa: F401
from .test_graduation import Script
from .test_monitoring import _req

REPO = "repo:/w/limiter"


def _notes(k: KnowledgeStore) -> list[Item]:
    add = k.add_item
    return [
        add(
            Item(
                type=ItemType.DECISION,
                statement="limiter guards acquire with a Redis lock",
                why="workers raced on the counter",
                scope=REPO,
            )
        ),
        add(Item(type=ItemType.PITFALL, statement="the lock must expire", scope=REPO)),
        add(Item(type=ItemType.OPEN, statement="backoff is not tuned yet", scope=REPO)),
    ]


def _answer(**sections: list[dict[str, Any]]) -> dict[str, Any]:
    return {"overview": "The limiter is  mostly done.", **sections}


BRIEF = _answer(
    decisions=[{"text": "A Redis lock guards acquire, because workers raced.", "notes": ["K1"]}],
    pitfalls=[{"text": "The lock must expire.", "notes": ["k2", "K2", "K9"]}],
    open=[
        {"text": "Backoff is not tuned.", "notes": ["K3"]},
        {"text": "It also needs sharding.", "notes": []},  # cites nothing: dropped
        {"text": "And a rewrite.", "notes": ["K7"]},  # cites no note it was given: dropped
    ],
)


# --------------------------------------------------------------------------- the brief


def test_the_prompt_lists_the_notes_and_the_answer_must_cite_them(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    lock, expire, backoff = _notes(k)
    k.dispute(expire.item_id, "the docs say it never expires")
    items, left_out = select([backoff, expire, lock])
    assert items == [lock, expire, backoff] and left_out == 0  # by type, not as given
    prompt = build_prompt("the project limiter", items)
    assert "all\nthe notes memory has about the project limiter" in prompt
    assert "K1 [decision, " in prompt and "(why: workers raced on the counter)" in prompt
    assert "disputed: the docs say it never expires] the lock must expire" in prompt

    got = parse(BRIEF, items)
    assert got["overview"] == "The limiter is mostly done."
    assert [(s["key"], s["title"]) for s in got["sections"]] == [
        ("decisions", "Decisions and why"),
        ("pitfalls", "Pitfalls"),
        ("open", "Open items"),
    ]
    assert got["sections"][1]["points"] == [
        {"text": "The lock must expire.", "notes": [expire.item_id]}
    ]
    assert [p["text"] for p in got["sections"][2]["points"]] == ["Backoff is not tuned."]
    with pytest.raises(GraduationError, match="no point that cites a note"):
        parse(_answer(open=[{"text": "made up", "notes": []}]), items)

    # a bad answer is asked again once
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "I can't" if len(asked) == 1 else json.dumps(BRIEF)

    assert summarize("the project limiter", items, ask)["sections"][0]["key"] == "decisions"
    assert len(asked) == 2


def test_a_full_prompt_leaves_out_the_least_alive_and_changes_are_counted(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    lock, expire, backoff = _notes(k)
    lock.strength = 4.0
    kept, left_out = select([lock, expire, backoff], budget=120)
    assert kept == [lock] and left_out == 2

    seen = {i.item_id: stamp(i) for i in (lock, expire, backoff)}
    assert changed_since(seen, [lock, expire, backoff]) == 0
    lock.uses += 3  # being used is no change to what it says
    assert changed_since(seen, [lock, expire, backoff]) == 0
    k.dispute(expire.item_id, "wrong")
    new = k.add_item(Item(type=ItemType.FACT, statement="it runs on three nodes", scope=REPO))
    assert changed_since(seen, [lock, expire, new]) == 3  # one disputed, one gone, one new


# --------------------------------------------------------------------------- the daemon


def _summary(cfg: Config, scope: str) -> dict[str, Any]:  # noqa: F811
    return call(cfg, "knowledge_summary", {"scope": scope}, autostart=False)  # type: ignore[no-any-return]


def _summarize(cfg: Config, scope: str, **kw: Any) -> dict[str, Any]:  # noqa: F811
    return call(cfg, "knowledge_summarize", {"scope": scope, **kw}, autostart=False)  # type: ignore[no-any-return]


def test_the_daemon_writes_caches_and_dates_a_summary(cfg: Config) -> None:  # noqa: F811
    again = _answer(open=[{"text": "Backoff is still open.", "notes": ["K2"]}])
    ask = Script([], merges=[BRIEF, RuntimeError("model is down"), again])
    with daemon_with(cfg, ask) as d:
        k = d.knowledge
        lock, expire, backoff = _notes(k)
        elsewhere = k.add_item(Item(type=ItemType.FACT, statement="the user likes tea"))
        assert _summary(cfg, "limiter") == {
            "scope": REPO,
            "name": "limiter",
            "items": 3,
            "running": False,
            "error": None,
            "summary": None,
            "changed": 0,
        }
        res = _summarize(cfg, "limiter")
        assert res == {"scope": REPO, "queued": True, "running": True, "items": 3}
        _wait(d.settled)
        got = _summary(cfg, REPO)
        s = got["summary"]
        assert not got["running"] and got["changed"] == 0 and got["error"] is None
        assert (s["name"], s["items"], s["left_out"]) == ("limiter", 3, 0)
        assert s["sections"][0]["points"][0]["notes"] == [lock.item_id]
        assert set(s["notes"]) == {lock.item_id, expire.item_id, backoff.item_id}
        assert "about the project limiter" in ask.prompts[-1]
        assert elsewhere.statement not in ask.prompts[-1]  # only this scope's notes
        # a view, not knowledge: nothing was added to memory
        assert len(k.heads) == 4 and set(k.summaries()) == {REPO}
        (ev,) = _events(cfg, "summarize")
        assert ev["ok"] and (ev["scope"], ev["items"], ev["points"]) == (REPO, 3, 3)

        # current: asking again does not call the model
        assert _summarize(cfg, "limiter")["queued"] is False
        assert len(ask.prompts) == 1
        # a note is retracted: the page is told, and the old brief stays until refreshed
        k.retract(expire.item_id, "no longer true")
        assert _summary(cfg, "limiter")["changed"] == 1
        assert _summarize(cfg, "limiter")["queued"] is True  # stale: written again
        _wait(d.settled)
        failed = _summary(cfg, "limiter")
        assert "model is down" in failed["error"] and failed["summary"]["items"] == 3
        assert _events(cfg, "summarize")[-1]["ok"] is False
        _summarize(cfg, "limiter")
        _wait(d.settled)
        fresh = _summary(cfg, "limiter")
        assert fresh["error"] is None and fresh["changed"] == 0
        assert fresh["summary"]["sections"][0]["points"][0]["notes"] == [backoff.item_id]
        assert "the lock must expire" not in ask.prompts[-1]  # only what is still true

        with pytest.raises(RemoteError, match="unknown scope"):
            _summarize(cfg, "nowhere")
        k.retract(elsewhere.item_id, "gone")
        with pytest.raises(RemoteError, match="no notes"):
            _summarize(cfg, "global")
    with daemon_with(cfg, None) as bare:
        _notes(bare.knowledge)
        with pytest.raises(RemoteError, match="no model"):
            _summarize(cfg, "limiter")


def test_gm_knowledge_summarize_prints_the_brief(
    cfg: Config,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GM_HOME", str(cfg.home))
    ask = Script([], merges=[BRIEF, BRIEF])
    with daemon_with(cfg, ask) as d:
        k = d.knowledge
        lock, expire, _ = _notes(k)
        k.add_item(Item(type=ItemType.FACT, statement=topic_fact, scope="topic:Redis Cluster"))
        assert cli.main(["knowledge", "summarize", "limiter"]) == 0
        out = capsys.readouterr().out
        assert "Summary of project limiter: 3 notes, written " in out
        assert "The limiter is mostly done." in out and "DECISIONS AND WHY" in out
        assert f"- The lock must expire. [{expire.item_id[:8]}]" in out
        assert f"because workers raced. [{lock.item_id[:8]}]" in out
        assert cli.main(["knowledge", "summarize", "limiter"]) == 0  # cached
        assert len(ask.prompts) == 1 and "DECISIONS AND WHY" in capsys.readouterr().out
        # a topic name with a space, as separate words; --force writes it again
        assert cli.main(["knowledge", "summarize", "topic:redis", "cluster", "--force"]) == 0
        assert "Summary of topic Redis Cluster: 1 note," in capsys.readouterr().out
        assert cli.main(["knowledge", "summarize"]) == 2
        assert cli.main(["knowledge", "summarize", "nowhere"]) == 1


topic_fact = "a Redis Cluster has 16384 hash slots"


def test_the_web_api_starts_and_reads_a_summary(cfg: Config) -> None:  # noqa: F811
    with daemon_with(cfg, Script([], merges=[BRIEF])) as d:
        _notes(d.knowledge)
        ui = UIServer(cfg, "127.0.0.1", 0)
        threading.Thread(target=ui.serve_forever, daemon=True).start()
        try:
            assert _req(ui, "/api/ksummarize", {"scope": REPO}, token="nope")[0] == 403
            status, res = _req(ui, "/api/ksummarize", {"scope": REPO}, token=ui.token)
            assert status == 200 and res["queued"] is True
            _wait(d.settled)
            status, got = _req(ui, "/api/ksummary?scope=limiter&poll=1", token=ui.token)
            assert status == 200 and got["summary"]["overview"] == "The limiter is mostly done."
            status, err = _req(ui, "/api/ksummary?scope=nowhere", token=ui.token)
            assert status == 400 and b"unknown scope" in err
            status, page = _req(ui, "/knowledge")
            assert status == 200 and b'data-act="brief-open"' in page
        finally:
            ui.shutdown()
            ui.server_close()
