"""The daemon: one long-running local process per memory folder, put together from its
parts (state.py, then one module per concern), serving the socket and running the
background tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import threading
import time
from datetime import timedelta
from typing import Any

from .. import clock
from ..events import prune_events
from ..storage import json_note, sqlite_problem
from .admin import Admin
from .graduating import Graduating
from .rebuild import Rebuilds
from .routes import ROUTES
from .state import FOLDED, REBUILD, RESCOPE, SUMMARIZE, Job, log
from .summaries import Summaries
from .turns import Turns
from .views import Views

MAX_REQUEST_BYTES = 32 * 1024 * 1024
TIMED_METHODS = frozenset({"recall", "remember"})  # the ones a hook waits on


class Daemon(Turns, Graduating, Rebuilds, Admin, Summaries, Views):
    """The daemon: the parts put together, serving requests on the socket and running
    the background tasks (saving, graduating, maintenance, the idle stop)."""

    # ------------------------------------------------------------------ lifecycle

    async def serve(self) -> None:
        sock = self.cfg.socket_path
        with contextlib.suppress(FileNotFoundError):
            sock.unlink()  # stale; we hold the lock, so no live daemon owns it
        server = await asyncio.start_unix_server(
            self._handle, path=str(sock), limit=MAX_REQUEST_BYTES
        )
        os.chmod(sock, 0o600)
        self._server = server
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, self.stop)

        tasks = [
            asyncio.create_task(self._saver(), name="saver"),
            asyncio.create_task(self._idle_watch(), name="idle"),
            asyncio.create_task(self._graduator(), name="graduator"),
            asyncio.create_task(self._maintainer(), name="maintenance"),
        ]
        self.sweep(trigger="sweep")  # sessions left over from before this daemon started
        log.info("daemon listening on %s (pid %d)", sock, os.getpid())
        self.events.emit("daemon", op="start", pid=os.getpid(), storage=self.knowledge.backend.name)
        note = json_note(self.knowledge.backend.name)
        if note:
            log.warning(note)
        self._start_ui()
        self.ready.set()
        try:
            await self._stop.wait()
        finally:
            # No new requests (stop() already closed the listener), and the ones already
            # in finish before anything they could reach is closed: no caller sees a
            # half-stopped daemon or a dropped connection.
            server.close()
            await server.wait_closed()
            if self._handling:
                await asyncio.wait(set(self._handling), timeout=5)
            await self._stop_ui()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # everything once more, as insurance against a change that was never marked
            self.knowledge.save(full=True)
            self.knowledge.close()
            with contextlib.suppress(FileNotFoundError):
                sock.unlink()
            self.events.emit("daemon", op="stop", pid=os.getpid())
            log.info("daemon stopped")

    def stop(self) -> None:
        """Stop at once taking connections: a caller that comes now finds no daemon and
        starts a new one, which waits for this one to go (`acquire_lock`)."""
        if self._server is not None:
            self._server.close()
        self._stop.set()

    def _start_ui(self) -> None:
        """Serve the web page on `ui_port` while the daemon runs. A port already in use
        (another program, or a `gm ui --port` server) leaves the daemon without a page."""
        if not self.cfg.ui_port:
            return
        from ..webui import UIServer

        try:
            self.ui = UIServer(self.cfg, "127.0.0.1", self.cfg.ui_port)
        except OSError as exc:
            log.warning("web page not served: port %d: %s", self.cfg.ui_port, exc)
            self.events.emit("ui", op="unavailable", port=self.cfg.ui_port, error=str(exc))
            return
        threading.Thread(target=self.ui.serve_forever, name="web-ui", daemon=True).start()
        log.info("web page on %s", self.ui.url)
        self.events.emit("ui", op="start", url=self.ui.url)

    async def _stop_ui(self) -> None:
        ui, self.ui = self.ui, None
        if ui is not None:
            await asyncio.to_thread(ui.shutdown)
            ui.server_close()

    # ------------------------------------------------------------------ requests

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # busy while answering, so the idle watch never stops the daemon mid-request; a
        # passive request (a web page polling) doesn't count as activity
        self._busy += 1
        active = True
        task = asyncio.current_task()
        if task is not None:
            self._handling.add(task)
        try:
            line = await reader.readline()
            started = time.perf_counter()
            method = "?"
            error: str | None = None
            try:
                req = json.loads(line)
                active = not req.get("passive")
                if active:
                    self._last_activity = time.monotonic()
                method = str(req["method"])
                result = self._dispatch(method, dict(req.get("params") or {}))
                resp: dict[str, Any] = {"ok": True, "result": result}
            except Exception as exc:  # report every failure to the caller
                log.exception("request failed")
                error = f"{type(exc).__name__}: {exc}"
                resp = {"ok": False, "error": error}
            if method in TIMED_METHODS or error is not None:
                self.events.emit(
                    "request",
                    method=method,
                    ms=round((time.perf_counter() - started) * 1000, 1),
                    ok=error is None,
                    error=error,
                )
            writer.write(json.dumps(resp).encode() + b"\n")
            await writer.drain()
        finally:
            if task is not None:
                self._handling.discard(task)
            self._busy -= 1
            if active:
                self._last_activity = time.monotonic()
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _dispatch(self, method: str, p: dict[str, Any]) -> Any:
        route = ROUTES.get(method)
        if route is None:
            raise ValueError(f"unknown method {method!r}")
        return route(self, p)

    def status(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "ui": self.ui.url if self.ui is not None else None,
            "buffer": self.buffer.stats(),
            # the conversation graph from before the redesign is still on disk, or the
            # memory is still in the JSON files from before SQLite and can move
            "needs_migration": (self.cfg.home / "graph.json").exists()
            or (self.knowledge.backend.name == "json" and sqlite_problem() is None),
            "storage": self.knowledge.backend.name,
            "storage_note": json_note(self.knowledge.backend.name),
            "knowledge": {
                **self.knowledge.stats(),
                "graduation": self.ask is not None,
                "queued": self.grad_queue.qsize(),
                "graduating": self._graduating,
            },
        }

    # ------------------------------------------------------------------ background

    async def _saver(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.save_interval)
            if self.knowledge_dirty:
                self.knowledge_dirty = False
                try:
                    self.knowledge.save()
                except Exception:
                    self.knowledge_dirty = True
                    log.exception("knowledge save failed")

    async def _idle_watch(self) -> None:
        interval = min(15.0, max(0.05, self.cfg.idle_timeout / 4))
        while True:
            await asyncio.sleep(interval)
            idle = time.monotonic() - self._last_activity
            if (
                idle >= self.cfg.idle_timeout
                and self._busy == 0
                and self.grad_queue.empty()
                and self._graduating is None
                and not self._bg
            ):
                log.info("idle for %.0fs; shutting down", idle)
                self.stop()
                return

    async def _maintainer(self) -> None:
        interval = max(1.0, min(60.0, self.cfg.graduation_idle / 4))
        while True:
            await asyncio.sleep(interval)
            self.maintain()

    def maintain(self) -> dict[str, int]:
        """One maintenance pass: graduate quiet sessions, clear abandoned ones, move
        knowledge that faded to dormant. Runs every minute or so; the simulator calls it
        directly after moving the clock."""
        queued = self.sweep()
        faded = self.knowledge.sweep()
        if faded:
            self.knowledge_dirty = True
            self.events.emit("knowledge_dormant", items=faded)
        return {"queued": queued, "dormant": len(faded), "events_pruned": self.prune_events()}

    def prune_events(self) -> int:
        """Keep `events_days` of the event log; runs at most once a day."""
        now = clock.now()
        path = self.events.path
        if (
            path is None
            or self.cfg.events_days <= 0  # 0: keep everything
            or (self._pruned is not None and now - self._pruned < timedelta(days=1))
        ):
            return 0
        self._pruned = now
        try:
            return prune_events(path, now - timedelta(days=self.cfg.events_days))
        except OSError:
            log.exception("pruning the event log failed")
            return 0

    def settled(self) -> bool:
        """No background work left: graduation, use checks."""
        return self.grad_queue.empty() and self._graduating is None and not self._bg

    async def _graduator(self) -> None:
        while True:
            job = await self.grad_queue.get()
            self._graduating = job.session
            try:
                await self._run(job)
            except Exception:
                log.exception("graduation bookkeeping failed for %s", job.session)
            finally:
                self._graduating = None
                self._grad_queued[job.session] -= 1
                if self._grad_queued[job.session] <= 0:
                    del self._grad_queued[job.session]
                self.grad_queue.task_done()

    async def _run(self, job: Job) -> None:
        """One queued job: a session's graduation, or a job queued by hand."""
        if job.trigger == REBUILD:
            await self._rebuild(job)
        elif job.trigger == FOLDED:
            await self._fold_summaries(job)
        elif job.trigger == RESCOPE:
            await self._rescope(job)
        elif job.trigger == SUMMARIZE:
            await self._brief(job)
        else:
            await self._graduate(job)
