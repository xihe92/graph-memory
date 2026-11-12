"""Marking knowledge no longer true by hand, and undoing it: the store, the daemon, `gm
knowledge retract|restore`, the web API, plus the other admin commands of `gm`."""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from graph_memory import cli
from graph_memory.client import DaemonUnavailable, call
from graph_memory.config import Config, load_config
from graph_memory.daemon import Daemon, scope_names
from graph_memory.knowledge import Item, ItemType, KnowledgeStore, Relation
from graph_memory.knowledge_recall import KnowledgeIndex, recall, tokens
from graph_memory.webui import UIServer

from .conftest import topic_embed
from .test_daemon import _start
from .test_graduation import Script
from .test_monitoring import _req

REPO = "repo:/work/limiter"


def _item(k: KnowledgeStore, text: str, **kw: object) -> Item:
    return k.add_item(Item(type=ItemType.FACT, statement=text, **kw))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- store


def test_retract_keeps_history_and_stops_recall(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a = _item(k, "redis lua script guards acquire in the limiter", scope=REPO)
    v = k.version
    k.retract(a.item_id, "replaced outside any session")
    assert not a.valid and a.invalid_at and a.expired_at and a.invalidated_by is None
    assert a.invalid_reason == "replaced outside any session" and k.version > v
    assert k.find_statement(a.statement) is None
    got = recall(k, KnowledgeIndex(k), "redis lua script acquire", project=REPO)
    assert a.item_id not in {i.item_id for i in got.items}
    assert k.stats()["invalidated"] == 1

    first = a.invalid_at
    k.retract(a.item_id, "again")  # already no longer true: nothing changes
    assert a.invalid_at == first and a.invalid_reason == "replaced outside any session"

    k.save()
    again = KnowledgeStore(tmp_path, topic_embed).items[a.item_id]
    assert not again.valid and again.invalid_reason == "replaced outside any session"


def test_retract_without_reason_says_it_was_by_hand(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a = _item(k, "redis lua script")
    k.retract(a.item_id)
    assert a.invalid_reason == "marked no longer true by hand"


def test_restore_undoes_a_model_invalidation_and_its_edge(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    old = _item(k, "the limiter uses a redis lua script")
    new = _item(k, "the limiter uses a redis SETNX lock")
    k.add_edge(new.item_id, old.item_id, Relation.REFINES, 0.6)
    k.invalidate(old.item_id, new, "lock replaced the script")
    assert old.invalidated_by == new.item_id and old.invalid_reason == "lock replaced the script"
    assert (new.item_id, old.item_id, "supersedes") in k.edges

    k.restore(old.item_id)
    assert old.valid and old.invalidated_by is None and old.expired_at is None
    assert old.invalid_reason == ""
    assert (new.item_id, old.item_id, "supersedes") not in k.edges
    assert (new.item_id, old.item_id, "refines") in k.edges  # other edges stay
    v = k.version
    k.restore(old.item_id)  # already true: nothing changes
    assert k.version == v
    got = recall(k, KnowledgeIndex(k), "redis lua script limiter")
    assert old.item_id in {i.item_id for i in got.items}


def test_scope_names_are_short_and_unique() -> None:
    got = scope_names(
        ["global", "repo:/a/x/api", "repo:/b/y/api", "repo:/a/x/api", "repo:/w/limiter", "global"]
    )
    assert got == [
        {"key": "global", "name": "global", "items": 2},
        {"key": "repo:/a/x/api", "name": "x/api", "items": 2},
        {"key": "repo:/b/y/api", "name": "y/api", "items": 1},
        {"key": "repo:/w/limiter", "name": "limiter", "items": 1},
    ]
    assert scope_names([]) == []
    # the same directory name all the way up: the full paths tell them apart
    same = scope_names(["repo:/api", "repo:/x/api"])
    assert [s["name"] for s in same] == ["api", "x/api"]


def test_hyphenated_words_match_their_parts() -> None:
    assert tokens("Per-tenant limits") == ["per-tenant", "tenant", "limit"]
    assert tokens("svc/rate-limit.py") == ["svc/rate-limit.py"]  # paths stay whole


# --------------------------------------------------------------------------- daemon, CLI, web


@pytest.fixture
def admin(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, Daemon]]:
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.delenv("GM_SOCKET", raising=False)
    k = KnowledgeStore(short_dir, topic_embed)
    _item(k, "the limiter uses a redis lua script", scope=REPO, item_id="aaaa1111")
    _item(k, "the user prefers short answers", item_id="aaaa2222")
    _item(k, "pytest needs redis from docker compose", scope=REPO, item_id="bbbb3333")
    k.save()
    cfg = replace(load_config(), save_interval=0.05)
    daemon, t = _start(cfg, Script([]))
    try:
        yield cfg, daemon
    finally:
        with contextlib.suppress(DaemonUnavailable, OSError):  # `gm stop` may have run
            call(cfg, "shutdown", autostart=False)
        t.join(5)


def _gm(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_daemon_retract_and_restore(admin: tuple[Config, Daemon]) -> None:
    cfg, daemon = admin
    res = call(cfg, "knowledge_retract", {"id": "bbbb", "reason": "tests mock redis now"})
    assert res == {
        "id": "bbbb3333",
        "valid": False,
        "statement": "pytest needs redis from docker compose",
    }
    view = call(cfg, "knowledge")
    gone = next(i for i in view["items"] if i["id"] == "bbbb3333")
    assert not gone["valid"] and gone["invalid_reason"] == "tests mock redis now"
    assert view["scopes"] == [
        {"key": "global", "name": "global", "items": 1},
        {"key": REPO, "name": "limiter", "items": 2},
    ]
    assert call(cfg, "knowledge_restore", {"id": "bbbb3333"})["valid"] is True
    events = (cfg.home / "events.jsonl").read_text()
    assert '"knowledge_retract"' in events and '"knowledge_restore"' in events
    assert daemon.knowledge_dirty or not daemon.knowledge.items["bbbb3333"].invalid_at


@pytest.mark.parametrize("ref", ["aaaa", "", "zzzz"])
def test_retract_refuses_an_unclear_id(admin: tuple[Config, Daemon], ref: str) -> None:
    cfg, daemon = admin
    with pytest.raises(RuntimeError, match="ambiguous|no knowledge item"):
        call(cfg, "knowledge_retract", {"id": ref})
    assert all(i.valid for i in daemon.knowledge.items.values())


def test_detail_with_an_empty_id_is_an_error(admin: tuple[Config, Daemon]) -> None:
    cfg, _ = admin
    with pytest.raises(RuntimeError):
        call(cfg, "knowledge_item", {"id": ""})


def test_gm_knowledge_commands(
    admin: tuple[Config, Daemon], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg, daemon = admin
    code, out, _ = _gm(capsys, "knowledge", "retract", "aaaa1", "--reason", "switched to SETNX")
    assert code == 0 and "aaaa1111 is now no longer true" in out
    it = daemon.knowledge.items["aaaa1111"]
    assert not it.valid and it.invalid_reason == "switched to SETNX"

    code, out, _ = _gm(capsys, "knowledge")
    assert code == 0 and "lua script" not in out and "short answers" in out
    assert "2 items (1 no longer true)" in out
    code, out, _ = _gm(capsys, "knowledge", "list", "--all")
    assert "lua script  [no longer true]" in out

    code, out, _ = _gm(capsys, "knowledge", "show", "aaaa1")
    assert code == 0 and json.loads(out)["item"]["invalid_reason"] == "switched to SETNX"

    code, out, _ = _gm(capsys, "knowledge", "restore", "aaaa1111")
    assert code == 0 and "is now true again" in out and it.valid

    for action in ("show", "retract", "restore"):
        code, _, err = _gm(capsys, "knowledge", action)
        assert code == 2 and "needs an id" in err
    code, _, err = _gm(capsys, "knowledge", "retract", "aaaa")
    assert code == 1 and "ambiguous" in err
    code, _, err = _gm(capsys, "knowledge", "restore", "nope")
    assert code == 1 and "no knowledge item" in err


def test_gm_admin_commands(
    admin: tuple[Config, Daemon], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg, _ = admin
    code, out, _ = _gm(capsys, "status")
    assert code == 0 and json.loads(out)["knowledge"]["items"] == 3
    with pytest.raises(SystemExit):
        cli.main(["links"])  # link review went with the conversation graph
    capsys.readouterr()
    code, out, _ = _gm(capsys, "stats", "--since", "all", "--json")
    assert code == 0 and json.loads(out)["status"]["knowledge"]["items"] == 3
    code, out, _ = _gm(capsys, "stats", "--since", "all")
    assert code == 0 and out.strip()
    code, _, err = _gm(capsys, "stats", "--since", "yesterday")
    assert code == 2 and err
    code, out, _ = _gm(capsys, "stop")
    assert code == 0 and "stopped" in out
    assert not cfg.socket_path.exists()  # it returns once the daemon is gone


def test_gm_stop_reports_a_daemon_that_does_not_go(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.delenv("GM_SOCKET", raising=False)
    (short_dir / "config.toml").write_text("start_timeout = 0.2\n")
    (short_dir / "gm.sock").write_text("")  # a socket that never goes away
    monkeypatch.setattr(cli, "call", lambda *a, **kw: None)
    code, _, err = _gm(capsys, "stop")
    assert code == 1 and "still shutting down" in err


def test_gm_without_a_daemon(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.delenv("GM_SOCKET", raising=False)
    for argv in (["status"], ["stop"]):
        code, _, err = _gm(capsys, *argv)
        assert code == 1 and "not running" in err
    code, out, _ = _gm(capsys, "stats", "--json")
    assert code == 0 and json.loads(out)["status"] is None


def test_web_retract_and_restore(admin: tuple[Config, Daemon]) -> None:
    cfg, daemon = admin
    ui = UIServer(cfg, "127.0.0.1", 0)
    threading.Thread(target=ui.serve_forever, daemon=True).start()
    try:
        status, page = _req(ui, "/knowledge")
        assert status == 200 and ui.token.encode() in page
        # a script's address changes with its content, so a browser never pairs a new
        # page with a script it kept from an older version
        import hashlib
        import re

        from graph_memory.webui import _asset

        src = re.search(rb'src="(/kview\.js\?v=([0-9a-f]{10}))"', page)
        assert src and src.group(2).decode() == hashlib.sha256(_asset("kview.js")).hexdigest()[:10]
        status, js = _req(ui, src.group(1).decode())
        assert status == 200 and b"KView" in js and b"selectable" in js
        status, js = _req(ui, "/idle.js")
        assert status == 200 and b"quietWhenStill" in js

        assert _req(ui, "/api/kretract", {"id": "aaaa1111"}, token="nope")[0] == 403
        status, res = _req(
            ui, "/api/kretract", {"id": "aaaa1111", "reason": "done"}, token=ui.token
        )
        assert status == 200 and res["valid"] is False
        assert daemon.knowledge.items["aaaa1111"].invalid_reason == "done"
        status, view = _req(ui, "/api/knowledge", token=ui.token)
        assert [s["name"] for s in view["scopes"]] == ["global", "limiter"]
        status, res = _req(ui, "/api/krestore", {"id": "aaaa1111"}, token=ui.token)
        assert status == 200 and res["valid"] is True
        status, err = _req(ui, "/api/kretract", {"id": "aaaa"}, token=ui.token)
        assert status == 400 and b"ambiguous" in err
    finally:
        ui.shutdown()
        ui.server_close()


def test_gm_delete_review_confirm_and_undelete(
    admin: tuple[Config, Daemon],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import sys

    from graph_memory.knowledge import Volatility

    cfg, daemon = admin
    k = daemon.knowledge
    k.add_edge("aaaa1111", "aaaa2222", Relation.DEPENDS_ON, 0.9)
    k.add_edge("aaaa2222", "bbbb3333", Relation.DEPENDS_ON, 0.9)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # not a terminal: never prompts
    code, out, err = _gm(capsys, "knowledge", "delete", "aaaa2")
    assert code == 1 and "short answers" in out and "--yes" in err
    assert "aaaa2222" in k.items
    code, out, _ = _gm(capsys, "knowledge", "delete", "aaaa2", "--yes", "--reason", "noise")
    assert code == 0 and "deleted 1 items" in out and "1 bridge edges" in out
    assert "aaaa2222" not in k.items
    assert ("aaaa1111", "bbbb3333", "depends_on") in k.edges
    batch = out.split("gm knowledge undelete ")[1].split()[0]

    # review: a disputed item, an overdue one, and the deletion to undo
    k.dispute("aaaa1111", "the reply showed SETNX")
    k.items["bbbb3333"].volatility = Volatility.VOLATILE
    k.items["bbbb3333"].checked_at = k.items["bbbb3333"].created_at.replace(year=2020)
    code, out, _ = _gm(capsys, "knowledge", "review")
    assert code == 0 and "disputed: the reply showed SETNX" in out
    assert "not re-checked since 2020-" in out and "[not recalled]" in out
    assert batch in out and "short answers" in out
    code, out, _ = _gm(capsys, "knowledge", "confirm", "bbbb")
    assert code == 0 and "confirmed" in out and k.items["bbbb3333"].recallable

    code, out, _ = _gm(capsys, "knowledge", "undelete", batch)
    assert code == 0 and "put back 1 items" in out and "aaaa2222" in k.items
    assert ("aaaa1111", "bbbb3333", "depends_on") not in k.edges
    code, _, err = _gm(capsys, "knowledge", "undelete", batch)
    assert code == 1 and "no deletion" in err
    code, _, err = _gm(capsys, "knowledge", "undelete")
    assert code == 2 and "needs a batch id" in err
    events = (cfg.home / "events.jsonl").read_text()
    assert '"knowledge_delete"' in events and '"knowledge_undelete"' in events
    assert '"knowledge_confirm"' in events


def test_web_delete_confirm_review_and_undelete(admin: tuple[Config, Daemon]) -> None:
    cfg, daemon = admin
    ui = UIServer(cfg, "127.0.0.1", 0)
    threading.Thread(target=ui.serve_forever, daemon=True).start()
    try:
        assert _req(ui, "/api/kdelete", {"ids": ["aaaa1111"]}, token="nope")[0] == 403
        status, res = _req(
            ui, "/api/kdelete", {"ids": ["aaaa1111", "bbbb3333"], "reason": "noise"}, token=ui.token
        )
        assert status == 200 and len(res["deleted"]) == 2
        assert set(daemon.knowledge.items) == {"aaaa2222"}
        status, review = _req(ui, "/api/kreview", token=ui.token)
        assert status == 200 and review["deleted"][0]["batch"] == res["batch"]
        status, back = _req(ui, "/api/kundelete", {"batch": res["batch"]}, token=ui.token)
        assert status == 200 and back["restored"] == 2 and len(daemon.knowledge.items) == 3
        daemon.knowledge.dispute("aaaa2222", "maybe not")
        status, ok = _req(ui, "/api/kconfirm", {"id": "aaaa2222"}, token=ui.token)
        assert status == 200 and daemon.knowledge.items["aaaa2222"].disputes == 0
        status, view = _req(ui, "/api/knowledge", token=ui.token)
        mine = next(i for i in view["items"] if i["id"] == "aaaa2222")
        assert mine["origin"] == "" and mine["volatility"] == "stable" and not mine["overdue"]
        status, err = _req(ui, "/api/kdelete", {"ids": []}, token=ui.token)
        assert status == 400
    finally:
        ui.shutdown()
        ui.server_close()


def test_a_daemon_that_hangs_up_is_unavailable(short_dir: Path) -> None:
    """A daemon that accepts and closes without answering (as while shutting down) is
    reported as unavailable, not as a JSON error."""
    import socket

    cfg = load_config(short_dir)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(cfg.socket_path))
    srv.listen(1)

    def hang_up() -> None:
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.close()

    t = threading.Thread(target=hang_up, daemon=True)
    t.start()
    try:
        with pytest.raises(DaemonUnavailable, match="closed the connection"):
            call(cfg, "status", autostart=False)
    finally:
        t.join(5)
        srv.close()
