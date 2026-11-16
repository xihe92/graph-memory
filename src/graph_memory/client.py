"""Talks to the daemon over its Unix socket; starts it on demand."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from typing import Any

from .config import Config


class DaemonUnavailable(RuntimeError):
    pass


class RemoteError(RuntimeError):
    pass


def _call_once(
    cfg: Config, method: str, params: dict[str, Any], timeout: float, passive: bool = False
) -> Any:
    req: dict[str, Any] = {"method": method, "params": params}
    if passive:
        req["passive"] = True
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(cfg.socket_path))
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = bytearray()
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    if not buf.strip():  # e.g. the daemon is shutting down
        raise DaemonUnavailable("the daemon closed the connection without answering")
    resp = json.loads(buf)
    if not resp.get("ok"):
        raise RemoteError(resp.get("error", "unknown error"))
    return resp.get("result")


def spawn_daemon(cfg: Config) -> None:
    env = {**os.environ, "GM_HOME": str(cfg.home), "GM_SOCKET": str(cfg.socket_path)}
    env.pop("GM_DISABLE", None)
    with open(cfg.log_path, "a") as log:
        subprocess.Popen(
            [sys.executable, "-m", "graph_memory", "daemon"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,  # survives the hook process exiting
            env=env,
        )


def call(
    cfg: Config,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
    autostart: bool = True,
    passive: bool = False,
) -> Any:
    """`passive`: the request doesn't count as activity, so it never keeps an idle daemon
    from stopping (a web page left open, polling)."""
    params = params or {}
    try:
        return _call_once(cfg, method, params, timeout, passive)
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        if not autostart:
            raise DaemonUnavailable(str(exc)) from exc
    spawn_daemon(cfg)
    deadline = time.monotonic() + cfg.start_timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            return _call_once(cfg, method, params, timeout, passive)
        except (FileNotFoundError, ConnectionRefusedError):
            continue
    raise DaemonUnavailable(
        f"daemon did not start within {cfg.start_timeout:.0f}s; see {cfg.log_path}"
    )


def knowledge_items(cfg: Config, *, passive: bool = False) -> dict[str, dict[str, Any]]:
    """Item id -> {type, statement, valid}, from the running daemon; {} without one."""
    try:
        view = call(cfg, "knowledge", {"lean": True}, autostart=False, passive=passive)
    except (DaemonUnavailable, RemoteError):
        return {}
    return {
        str(i["id"]): {"type": i["type"], "statement": i["statement"], "valid": i["valid"]}
        for i in view.get("items", [])
    }
