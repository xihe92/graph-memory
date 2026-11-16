"""`gm ui`: a local web page for looking at the graph and correcting it.

The daemon serves the page on `ui_port` for as long as it runs (`gm ui --port N` runs a
server of its own instead). The page talks to this small HTTP server, which forwards to
the daemon over its socket, so the daemon stays the only process that changes the graph.
Stats and review state are read straight from the event log.

Bound to 127.0.0.1 only. Every API call must carry the token embedded in the page, and
the Host header must be the local address, so other sites open in the browser can't read
or change the memory (no CORS headers are ever sent). The token is kept in the memory
home (readable only by the user), so a page left open keeps working after the daemon
restarts.

A page's background refreshes send `poll=1`: they are passive, so a page left open never
keeps an idle daemon running, and never starts a stopped one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .client import DaemonUnavailable, RemoteError, call, knowledge_items
from .config import Config
from .events import events_path, parse_since, read_events
from .stats import compute_stats, memory_help

log = logging.getLogger(__name__)

DEFAULT_PORT = 7477
_STATIC = {
    "/kview.js": ("kview.js", "text/javascript"),
    "/idle.js": ("idle.js", "text/javascript"),
    "/vendor/3d-force-graph.min.js": ("vendor/3d-force-graph.min.js", "text/javascript"),
    "/vendor/force-graph.min.js": ("vendor/force-graph.min.js", "text/javascript"),
    "/layout.js": ("layout.js", "text/javascript"),
    "/vendor/d3-force-3d.bundle.js": ("vendor/d3-force-3d.bundle.js", "text/javascript"),
}


def _since(q: dict[str, str]) -> dict[str, Any]:
    """The Sessions page sends the buffer version it has: unchanged, the daemon says so."""
    return {"since": int(q["since"])} if q.get("since", "").isdigit() else {}


# GET /api/<route> -> (daemon method, its params from the query, passive when polled)
_GET: dict[str, tuple[str, Callable[[dict[str, str]], dict[str, Any] | None], bool]] = {
    "sessions": ("sessions", _since, True),
    "knowledge": ("knowledge", lambda q: {"lean": True}, False),
    "ksearch": ("knowledge_search", lambda q: {"q": q.get("q", "")}, False),
    "kitem": ("knowledge_item", lambda q: {"id": q.get("id", "")}, False),
    "kreview": ("knowledge_review", lambda q: None, False),
    "ksummary": ("knowledge_summary", lambda q: {"scope": q.get("scope", "")}, True),
}

# POST /api/<route> -> (daemon method, the body fields it takes)
_POST: dict[str, tuple[str, tuple[str, ...]]] = {
    "forget": ("forget", ("session", "id")),
    "kretract": ("knowledge_retract", ("id", "reason")),
    "krestore": ("knowledge_restore", ("id",)),
    "graduate": ("graduate_now", ("session",)),
    "kconfirm": ("knowledge_confirm", ("id",)),
    "kdelete": ("knowledge_delete", ("ids", "reason")),
    "kundelete": ("knowledge_undelete", ("batch",)),
    "ksummarize": ("knowledge_summarize", ("scope", "force")),
}


def _asset(name: str) -> bytes:
    return resources.files("graph_memory.ui").joinpath(name).read_bytes()


def ui_token(home: Path) -> str:
    """The page's API token, created once and kept in `home` (mode 0600)."""
    path = home / "ui.token"
    try:
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 32:
            return token
    except FileNotFoundError:
        pass
    token = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    return token


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, cfg: Config, host: str, port: int) -> None:
        super().__init__((host, port), _Handler)
        self.cfg = cfg
        self.token = ui_token(cfg.home)
        bound_port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}"}

    def index(self, page: str = "index.html") -> bytes:
        """The page with its token, and each script's address ending in a hash of the
        script: browsers keep scripts for a day, and a page from a newer version must
        not run with a script kept from an older one."""
        html = _asset(page).replace(b"__GM_TOKEN__", self.token.encode())
        for path, (name, _) in _STATIC.items():
            version = hashlib.sha256(_asset(name)).hexdigest()[:10]
            html = html.replace(f'src="{path}"'.encode(), f'src="{path}?v={version}"'.encode())
        return html

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


class _Handler(BaseHTTPRequestHandler):
    server: UIServer

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("ui: " + format, *args)

    # ------------------------------------------------------------------ plumbing

    def _send(self, status: int, body: bytes, ctype: str, *, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self.server.allowed_hosts

    def _authorized(self) -> bool:
        return self._host_ok() and secrets.compare_digest(
            self.headers.get("X-GM-Token", ""), self.server.token
        )

    def _daemon(
        self, method: str, params: dict[str, Any] | None = None, *, passive: bool = False
    ) -> None:
        try:
            res = call(
                self.server.cfg,
                method,
                params,
                timeout=30,
                passive=passive,
                autostart=not passive,
            )
            self._json(200, res)
        except RemoteError as exc:
            self._json(400, {"error": str(exc)})
        except DaemonUnavailable as exc:
            self._json(503, {"error": f"daemon unavailable: {exc}"})

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if not self._host_ok():
            self._send(HTTPStatus.MISDIRECTED_REQUEST, b"bad host", "text/plain")
            return
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, self.server.index(), "text/html; charset=utf-8")
            return
        if url.path in ("/knowledge", "/help"):
            page = "knowledge.html" if url.path == "/knowledge" else "help.html"
            self._send(200, self.server.index(page), "text/html; charset=utf-8")
            return
        if url.path in _STATIC:
            name, ctype = _STATIC[url.path]
            self._send(200, _asset(name), ctype, cache=True)
            return
        if not url.path.startswith("/api/"):
            self._send(404, b"not found", "text/plain")
            return
        if not self._authorized():
            self._json(403, {"error": "forbidden"})
            return
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        route = url.path[len("/api/") :]
        poll = q.get("poll") == "1"
        if route in _GET:
            method, params, passive = _GET[route]
            self._daemon(method, params(q), passive=passive and poll)
        elif route == "stats":
            self._stats(q.get("since", "7d"), poll)
        elif route == "help":
            self._help(q.get("since", "7d"), poll)
        else:
            self._json(404, {"error": "unknown endpoint"})

    def _stats(self, since_text: str, poll: bool = False) -> None:
        """Counts from the event log, and the daemon's status if it runs."""
        try:
            since = parse_since(since_text)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        st = compute_stats(read_events(events_path(self.server.cfg.home), since=since))
        try:
            st["status"] = call(self.server.cfg, "status", autostart=False, passive=poll)
        except DaemonUnavailable:
            st["status"] = None
        self._json(200, st)

    def _help(self, since_text: str, poll: bool = False) -> None:
        """How often memory gave Claude notes, and which notes the replies used."""
        try:
            since = parse_since(since_text)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        cfg = self.server.cfg
        data = memory_help(read_events(events_path(cfg.home), since=since))
        data["items"] = knowledge_items(cfg, passive=poll)
        data["mode"] = cfg.knowledge_recall
        self._json(200, data)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if not self._authorized():
            self._json(403, {"error": "forbidden"})
            return
        length = min(int(self.headers.get("Content-Length") or 0), 1 << 20)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        route = urlparse(self.path).path[len("/api/") :]
        if route not in _POST:
            self._json(404, {"error": "unknown endpoint"})
            return
        method, keys = _POST[route]
        self._daemon(method, {k: body.get(k) for k in keys})


def serve_ui(
    cfg: Config, *, port: int = DEFAULT_PORT, open_browser: bool = True, host: str = "127.0.0.1"
) -> int:
    try:
        server = UIServer(cfg, host, port)
    except OSError as exc:
        print(f"could not listen on {host}:{port}: {exc} (try --port)")
        return 1
    print(f"graph-memory UI on {server.url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(server.url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
