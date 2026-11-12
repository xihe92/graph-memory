from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from graph_memory.client import call
from graph_memory.config import Config, load_config
from graph_memory.daemon import Daemon
from graph_memory.events import EventLog, events_path, parse_since, read_events
from graph_memory.knowledge import KnowledgeStore
from graph_memory.stats import compute_stats, format_stats
from graph_memory.webui import UIServer

from .conftest import topic_embed, turn

# --------------------------------------------------------------------------- event log


def test_event_log_appends_rotates_and_filters(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path, max_bytes=200)
    for i in range(10):
        log.emit("hook", hook="recall", ms=float(i), ok=True)
    assert path.with_name("events.jsonl.1").exists()  # rotated at least once
    path.open("a").write("{broken\n")
    got = list(read_events(path, kinds=["hook"]))
    assert [e["ms"] for e in got] == sorted(e["ms"] for e in got)  # oldest first
    assert got[-1]["ms"] == 9.0
    assert list(read_events(path, since=datetime.now(UTC) + timedelta(minutes=1))) == []


def test_parse_since() -> None:
    assert parse_since("all") is None
    since = parse_since("24h")
    assert since is not None and timedelta(hours=23) < datetime.now(UTC) - since
    with pytest.raises(ValueError):
        parse_since("3 days")


# --------------------------------------------------------------------------- stats


def test_compute_and_format_stats() -> None:
    evs = [
        {"kind": "hook", "hook": "recall", "ms": 40, "ok": True, "ts": "a"},
        {"kind": "hook", "hook": "recall", "ms": 60, "ok": False, "ts": "b"},
        {"kind": "request", "method": "remember", "ms": 5, "ok": True},
        {"kind": "remember", "session": "s", "rewound": 0, "continued": False},
        {"kind": "remember", "session": "s", "rewound": 2, "continued": False},
        {"kind": "remember", "session": "s", "rewound": 0, "continued": True},
        {"kind": "forget", "session": "s"},
        {
            "kind": "graduate",
            "ok": True,
            "trigger": "compact",
            "exchanges": 5,
            "items": 3,
            "new": 2,
            "calls": 2,
            "ms": 4000,
        },  # fmt: skip
        {
            "kind": "graduate",
            "ok": True,
            "trigger": "session_end",
            "exchanges": 2,
            "items": 1,
            "new": 1,
            "calls": 1,
            "ms": 2000,
        },  # fmt: skip
        {"kind": "graduate", "ok": False, "error": "ModelError: timeout", "given_up": False},
        {"kind": "graduate", "ok": False, "error": "ModelError: timeout", "given_up": True},
        {"kind": "buffer_dropped", "exchanges": 4},
        {"kind": "session_cleared", "session": "s"},
        {"kind": "daemon", "op": "start"},
    ]
    st = compute_stats(evs)
    h, b, g = st["health"], st["buffer"], st["graduation"]
    assert h["hooks"]["recall"]["count"] == 2 and h["hooks"]["recall"]["errors"] == 1
    assert h["daemon_starts"] == 1
    assert (b["remembered"], b["rewinds"], b["continued"], b["forgotten"], b["dropped"]) == (
        3,
        1,
        1,
        1,
        4,
    )
    assert (g["ok"], g["failed"], g["given_up"]) == (2, 2, 1)
    assert (g["exchanges"], g["items"], g["new_items"], g["model_calls"]) == (7, 4, 3, 3)
    assert g["by_trigger"] == {"compact": 1, "session_end": 1}
    assert g["errors"] == {"ModelError": 2} and g["sessions_cleared"] == 1
    status = {"knowledge": {"items": 9, "invalidated": 1, "entities": 4},
              "buffer": {"exchanges": 2, "sessions": 1}}  # fmt: skip
    text = format_stats(st, status)
    assert "9 knowledge items (1 no longer true)" in text and "2 exchanges waiting" in text
    assert "2 ok, 2 failed (1 given up)" in text
    assert "3 new items, 1 seen again, from 7 exchanges (3 model calls)" in text
    assert "failures       ModelError 2" in text
    assert format_stats(compute_stats([])) == "No events recorded in this window yet."


# --------------------------------------------------------------------------- daemon + web UI


@pytest.fixture
def stack(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, UIServer]]:
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.delenv("GM_SOCKET", raising=False)
    cfg = load_config()
    daemon = Daemon(
        cfg,
        knowledge=KnowledgeStore(cfg.home, topic_embed),
        embedder=topic_embed,
        events=EventLog(events_path(cfg.home)),
    )
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_until_complete, args=(daemon.serve(),), daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while not cfg.socket_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    ui = UIServer(cfg, "127.0.0.1", 0)
    ut = threading.Thread(target=ui.serve_forever, daemon=True)
    ut.start()
    try:
        yield cfg, ui
    finally:
        ui.shutdown()
        ui.server_close()
        call(cfg, "shutdown", autostart=False)
        t.join(5)


def _req(
    ui: UIServer, path: str, body: Any = None, *, token: str | None = None, host: str | None = None
) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(ui.url.rstrip("/") + path, data=data)
    if token is not None:
        req.add_header("X-GM-Token", token)
    if host is not None:
        req.add_header("Host", host)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            return r.status, json.loads(raw) if r.headers[
                "Content-Type"
            ] == "application/json" else raw
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_web_ui_end_to_end(stack: tuple[Config, UIServer]) -> None:
    cfg, ui = stack
    status, page = _req(ui, "/")
    assert status == 200 and ui.token.encode() in page and b"__GM_TOKEN__" not in page
    status, js = _req(ui, "/vendor/3d-force-graph.min.js")
    assert status == 200 and b"ForceGraph3D" in js[:400]
    assert _req(ui, "/nope")[0] == 404

    # no token, wrong token, wrong Host: all refused
    assert _req(ui, "/api/sessions")[0] == 403
    assert _req(ui, "/api/sessions", token="nope")[0] == 403
    assert _req(ui, "/api/sessions", token=ui.token, host="evil.example:80")[0] == 421
    assert _req(ui, "/api/forget", {"id": "x"}, token="nope")[0] == 403

    a = call(cfg, "remember", {"session": "s", "turn": turn("redis race", "lua")}, autostart=False)
    call(cfg, "remember", {"session": "s", "turn": turn("weather", "rain")}, autostart=False)
    status, view = _req(ui, "/api/sessions", token=ui.token)
    assert status == 200
    (sess,) = view["sessions"]
    assert sess["session"] == "s" and [r["prompt"] for r in sess["records"]] == [
        "redis race",
        "weather",
    ]
    status, same = _req(ui, f"/api/sessions?since={view['version']}", token=ui.token)
    assert same == {"version": view["version"], "unchanged": True}

    status, res = _req(ui, "/api/forget", {"session": "s", "id": a["exchange"]}, token=ui.token)
    assert status == 200 and res == {"exchange": a["exchange"]}
    status, err = _req(ui, "/api/forget", {"session": "s", "id": "zzzz"}, token=ui.token)
    assert status == 400
    status, view = _req(ui, "/api/sessions", token=ui.token)
    assert [r["prompt"] for r in view["sessions"][0]["records"]] == ["weather"]

    status, st = _req(ui, "/api/stats?since=all", token=ui.token)
    assert st["buffer"]["remembered"] == 2 and st["buffer"]["forgotten"] == 1
    assert st["status"]["buffer"]["exchanges"] == 1
    assert _req(ui, "/api/stats?since=yesterday", token=ui.token)[0] == 400
    assert _req(ui, "/api/nope", token=ui.token)[0] == 404
    assert _req(ui, "/api/nope", {"a": 1}, token=ui.token)[0] == 404


def test_task_notifications_are_not_prompts() -> None:
    from graph_memory.adapters.claude_code import ClaudeCodeAdapter

    ad = ClaudeCodeAdapter()
    ev = ad.parse_prompt_event({"session_id": "s", "prompt": "<task-notification>\n<task-id>x"})
    assert ev.system is True
    assert ad.parse_prompt_event({"session_id": "s", "prompt": "fix the bug"}).system is False
