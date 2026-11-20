from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from graph_memory import cli, clock
from graph_memory.client import DaemonUnavailable, RemoteError, call
from graph_memory.config import Config, load_config
from graph_memory.daemon import Daemon
from graph_memory.events import EventLog, events_path
from graph_memory.knowledge import Entity, Item, ItemType, KnowledgeStore

from .conftest import topic_embed


def _start(cfg: Config, ask: Callable[[str], str] | None = None) -> tuple[Daemon, threading.Thread]:
    daemon = Daemon(
        cfg,
        knowledge=KnowledgeStore(cfg.home, topic_embed),
        embedder=topic_embed,
        graduation_model=ask,
        events=EventLog(events_path(cfg.home)),
    )
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_until_complete, args=(daemon.serve(),), daemon=True)
    t.start()
    _wait(daemon.ready.is_set)
    return daemon, t


@pytest.fixture
def cfg(short_dir: Path) -> Config:
    return replace(load_config(short_dir), save_interval=0.05)


@pytest.fixture
def running(cfg: Config) -> Iterator[Daemon]:
    daemon, t = _start(cfg)
    yield daemon
    try:
        call(cfg, "shutdown", autostart=False)
    except DaemonUnavailable:
        pass
    t.join(5)


def _wait(pred, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def seed_arsenal(home: Path) -> None:
    """A knowledge store with one item a football prompt recalls."""
    k = KnowledgeStore(home, topic_embed)
    team = k.add_entity(Entity(name="Arsenal", kind="other"))
    for text in ("arsenal won the match, saka scored", "saka is arsenal's top goal scorer"):
        k.add_item(Item(type=ItemType.FACT, statement=text, entities=[team.entity_id]))
    k.save()


def test_recall_and_remember(cfg: Config, running: Daemon) -> None:
    r = call(cfg, "recall", {"session": "s", "prompt": "fix the redis race"}, autostart=False)
    assert r == {"context": {}}  # nothing known yet
    m = call(
        cfg,
        "remember",
        {"session": "s", "turn": {"prompt": "fix the redis race", "reply": "lua script"}},
        autostart=False,
    )
    assert m["rewound"] == 0 and not m["continued"]
    (ex,) = running.buffer.exchanges("s")
    assert ex.id == m["exchange"] and ex.reply == "lua script"
    st = call(cfg, "status", autostart=False)
    assert st["buffer"]["exchanges"] == 1 and st["knowledge"]["items"] == 0
    assert not (cfg.home / "graph.json").exists() and not (cfg.home / "nodes").exists()


def test_errors_are_reported(cfg: Config, running: Daemon) -> None:
    with pytest.raises(RemoteError, match="unknown method"):
        call(cfg, "nope", autostart=False)


def test_idle_shutdown(cfg: Config) -> None:
    daemon, t = _start(replace(cfg, idle_timeout=0.2))
    t.join(5)
    assert not t.is_alive() and not cfg.socket_path.exists()


def test_hook_commands_print_knowledge_and_never_fail(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed_arsenal(cfg.home)
    daemon, t = _start(cfg)
    monkeypatch.setenv("GM_HOME", str(cfg.home))
    try:
        payload = {"session_id": "new", "prompt": "how many goals has saka scored for arsenal"}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        assert cli.main(["recall"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("<graph-memory>\nKnowledge from earlier sessions")
        assert "saka" in out and "Memory of" not in out  # no topic documents any more

        assert daemon.knowledge.days.count() == 1  # a prompt you typed: a day you worked

        # a background-task notification the next day is not a prompt you typed
        monkeypatch.setattr(clock, "_offset", timedelta(days=1))
        note = {"session_id": "new", "prompt": "<task-notification>done</task-notification>"}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(note)))
        assert cli.main(["recall"]) == 0
        assert daemon.knowledge.days.count() == 1
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        assert cli.main(["recall"]) == 0
        capsys.readouterr()
        assert daemon.knowledge.days.count() == 2

        monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))
        assert cli.main(["remember"]) == 0  # logged, not raised

        monkeypatch.setenv("GM_DISABLE", "1")
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        assert cli.main(["recall"]) == 0
        assert capsys.readouterr().out == ""
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)


def test_headless_sessions_are_skipped_unless_configured(
    cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seed_arsenal(cfg.home)
    daemon, t = _start(cfg)
    monkeypatch.setenv("GM_HOME", str(cfg.home))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"uuid": "u1", "type": "user", "message": {"content": "arsenal score?"}})
        + "\n"
        + json.dumps(
            {
                "uuid": "a1",
                "parentUuid": "u1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "2-1 to arsenal"}]},
            }
        )
        + "\n"
    )
    prompt = {"session_id": "bot", "prompt": "how many goals has saka scored for arsenal"}
    stop = {"session_id": "bot", "transcript_path": str(transcript)}

    def hooks() -> tuple[str, int]:
        before = daemon.buffer.stats()["exchanges"]
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(prompt)))
        assert cli.main(["recall"]) == 0
        out = capsys.readouterr().out
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(stop)))
        assert cli.main(["remember"]) == 0
        return out, daemon.buffer.stats()["exchanges"] - before

    try:
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
        assert hooks() == ("", 0)  # nothing shown, nothing stored
        assert daemon.knowledge.days.count() == 0  # and not a day you worked

        (cfg.home / "config.toml").write_text("remember_headless = true\n")
        out, added = hooks()
        assert "<graph-memory>" in out and added == 1
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)


def test_end_to_end_autostart_subprocess(short_dir: Path) -> None:
    """Real `gm` processes: the first hook call starts the daemon on its own."""
    (short_dir / "config.toml").write_text(
        'embedder = "none"\nmodel_backend = "none"\nidle_timeout = 30\n'
    )
    env = {**os.environ, "GM_HOME": str(short_dir)}
    env.pop("GM_DISABLE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    transcript = short_dir / "t.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {
                    "uuid": "u1",
                    "parentUuid": None,
                    "type": "user",
                    "message": {"content": "hello there"},
                },
                {
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Hi. How can I help?"}]},
                },
            ]
        )
        + "\n"
    )

    def gm(
        *args: str, payload: dict[str, object] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "graph_memory", *args],
            input=json.dumps(payload) if payload is not None else "",
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    try:
        assert gm("recall", payload={"session_id": "s", "prompt": "hello there"}).returncode == 0
        assert (
            gm(
                "remember", payload={"session_id": "s", "transcript_path": str(transcript)}
            ).returncode
            == 0
        )
        st = json.loads(gm("status").stdout)
        assert st["buffer"]["exchanges"] == 1 and st["knowledge"]["graduation"] is False
        # no model: the session's exchanges go when it ends, nothing learned
        gm("graduate", payload={"session_id": "s"})
        deadline = time.monotonic() + 10
        while json.loads(gm("status").stdout)["buffer"]["sessions"]:
            assert time.monotonic() < deadline
            time.sleep(0.1)
    finally:
        gm("stop")


# ---------------------------------------------------------------- restarting while one stops


def _hold_lock(cfg: Config):  # type: ignore[no-untyped-def]
    import fcntl

    held = open(cfg.lock_path, "a")  # noqa: SIM115
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return held


def test_a_new_daemon_waits_for_one_that_is_shutting_down(cfg: Config) -> None:
    from graph_memory.daemon import acquire_lock

    held = _hold_lock(cfg)  # holds the lock, answers nothing: a daemon on its way out
    threading.Timer(0.3, held.close).start()
    t0 = time.monotonic()
    lock = acquire_lock(cfg, wait=5)
    assert lock is not None and time.monotonic() - t0 >= 0.25
    assert cfg.lock_path.read_text() == str(os.getpid())
    lock.close()


def test_a_new_daemon_gives_up_on_one_that_never_goes(cfg: Config) -> None:
    from graph_memory.daemon import acquire_lock

    held = _hold_lock(cfg)
    try:
        assert acquire_lock(cfg, wait=0.2) is None
    finally:
        held.close()


def test_a_new_daemon_leaves_a_running_one_alone(cfg: Config, running: Daemon) -> None:
    from graph_memory.daemon import acquire_lock

    held = _hold_lock(cfg)
    try:
        t0 = time.monotonic()
        assert acquire_lock(cfg, wait=5) is None
        assert time.monotonic() - t0 < 1  # it answers, so no waiting
    finally:
        held.close()


def test_shutdown_stops_taking_requests_before_closing_the_page(cfg: Config) -> None:
    from graph_memory.daemon import _answers

    daemon, t = _start(cfg)
    assert _answers(cfg.socket_path)
    stop_ui, seen = daemon._stop_ui, []

    async def recording_stop_ui() -> None:
        seen.append(_answers(cfg.socket_path))
        await stop_ui()

    daemon._stop_ui = recording_stop_ui  # type: ignore[method-assign]
    call(cfg, "shutdown", autostart=False)
    # the moment shutdown is acknowledged, a caller finds no daemon (and would start one)
    with pytest.raises(DaemonUnavailable):
        call(cfg, "status", autostart=False)
    t.join(5)
    assert seen == [False] and not cfg.socket_path.exists()


def test_a_request_already_in_is_answered_before_the_daemon_goes(cfg: Config) -> None:
    import socket

    daemon, t = _start(cfg)
    early = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    early.connect(str(cfg.socket_path))  # connected, request not sent yet
    _wait(lambda: daemon._handling)
    call(cfg, "shutdown", autostart=False)
    early.sendall(b'{"method": "ping"}\n')  # arrives while the daemon is stopping
    reply = json.loads(early.makefile().readline())
    early.close()
    assert reply == {"ok": True, "result": {"pid": os.getpid()}}
    t.join(5)
    assert not t.is_alive()
