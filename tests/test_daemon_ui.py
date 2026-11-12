"""The web page lives and dies with the daemon; a page left open doesn't keep it awake."""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import stat
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from graph_memory import cli
from graph_memory.client import DaemonUnavailable, call
from graph_memory.config import Config, load_config
from graph_memory.daemon import Daemon
from graph_memory.webui import UIServer, ui_token

from .test_daemon import _start, _wait
from .test_graduation import Script
from .test_monitoring import _req


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _refused(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


@pytest.fixture
def served(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, Daemon]]:
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.delenv("GM_SOCKET", raising=False)
    monkeypatch.setenv("GM_UI_PORT", str(_free_port()))
    cfg = load_config()
    daemon, t = _start(cfg, Script([]))
    _wait(lambda: daemon.ui is not None)
    try:
        yield cfg, daemon
    finally:
        with contextlib.suppress(DaemonUnavailable, OSError):
            call(cfg, "shutdown", autostart=False)
        t.join(5)


def test_the_daemon_serves_the_page_while_it_runs(served: tuple[Config, Daemon]) -> None:
    cfg, daemon = served
    ui = daemon.ui
    assert ui is not None
    assert call(cfg, "status", autostart=False)["ui"] == f"http://127.0.0.1:{cfg.ui_port}/"
    status, page = _req(ui, "/help")
    assert status == 200 and ui.token.encode() in page
    status, view = _req(ui, "/api/sessions", token=ui.token)
    assert status == 200 and view["sessions"] == []

    call(cfg, "shutdown", autostart=False)
    _wait(lambda: _refused(cfg.ui_port))  # the page went down with the daemon


def test_a_taken_port_leaves_the_daemon_running_without_a_page(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        cfg = replace(load_config(short_dir), ui_port=busy.getsockname()[1])
        daemon, t = _start(cfg, Script([]))
        try:
            assert call(cfg, "status", autostart=False)["ui"] is None
            events = (cfg.home / "events.jsonl").read_text()
            assert '"kind": "ui", "op": "unavailable"' in events
        finally:
            call(cfg, "shutdown", autostart=False)
            t.join(5)


def test_the_token_is_kept_so_open_pages_survive_restarts(tmp_path: Path) -> None:
    first = ui_token(tmp_path)
    assert ui_token(tmp_path) == first and len(first) >= 32
    assert stat.S_IMODE(os.stat(tmp_path / "ui.token").st_mode) == 0o600
    (tmp_path / "ui.token").write_text("short")  # damaged: replaced
    assert ui_token(tmp_path) != "short"


def test_polls_do_not_keep_the_daemon_awake(served: tuple[Config, Daemon]) -> None:
    cfg, daemon = served
    ui = daemon.ui
    assert ui is not None

    def idle_after(path: str) -> bool:
        daemon._last_activity = 0.0
        assert _req(ui, path, token=ui.token)[0] == 200
        return daemon._last_activity == 0.0

    assert idle_after("/api/sessions?poll=1")
    assert idle_after("/api/help?since=all&poll=1")
    assert idle_after("/api/stats?poll=1")
    assert not idle_after("/api/sessions")  # opening or using the page counts
    daemon._last_activity = 0.0
    call(cfg, "ping", autostart=False, passive=True)
    assert daemon._last_activity == 0.0
    call(cfg, "ping", autostart=False)
    assert daemon._last_activity > 0.0


def test_gm_ui_opens_the_daemons_page(
    served: tuple[Config, Daemon],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg, _ = served
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    assert cli.main(["ui"]) == 0
    url = f"http://127.0.0.1:{cfg.ui_port}/"
    assert opened == [url] and url in capsys.readouterr().out
    assert cli.main(["ui", "--no-open"]) == 0 and len(opened) == 1


def test_gm_ui_runs_its_own_server_when_asked_or_when_the_daemon_has_none(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ran: list[dict[str, Any]] = []
    monkeypatch.setattr("graph_memory.webui.serve_ui", lambda cfg, **kw: ran.append(kw) or 0)
    cfg = load_config(short_dir)  # GM_UI_PORT=0 from conftest: no page in the daemon
    daemon, t = _start(cfg, Script([]))
    try:
        assert cli._ui(cfg, argparse.Namespace(port=None, no_open=True)) == 0
        assert "serves no page" in capsys.readouterr().err
        assert cli._ui(cfg, argparse.Namespace(port=9999, no_open=True)) == 0
        assert ran == [
            {"port": 7477, "open_browser": False},
            {"port": 9999, "open_browser": False},
        ]
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)
    stopped = replace(cfg, start_timeout=0.2, socket_path=cfg.home / "none.sock")
    monkeypatch.setattr("graph_memory.client.spawn_daemon", lambda cfg: None)
    assert cli._ui(stopped, argparse.Namespace(port=None, no_open=True)) == 1


def test_the_ui_port_setting(short_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GM_UI_PORT")
    assert load_config(short_dir).ui_port == 7477
    (short_dir / "config.toml").write_text("ui_port = 0\n")
    assert load_config(short_dir).ui_port == 0
    monkeypatch.setenv("GM_UI_PORT", "8123")
    assert load_config(short_dir).ui_port == 8123


def test_a_standalone_server_still_works(short_dir: Path) -> None:
    ui = UIServer(load_config(short_dir), "127.0.0.1", 0)
    threading.Thread(target=ui.serve_forever, daemon=True).start()
    try:
        assert _req(ui, "/")[0] == 200
    finally:
        ui.shutdown()
        ui.server_close()
