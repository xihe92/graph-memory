"""The daemon's protocol: each method a request can name, and what it calls.

    {"method": "<name>", "params": {...}}\\n  ->  {"ok": true, "result": ...}\\n

A method not in ROUTES is an error, as is a missing param.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Daemon

Route = Callable[["Daemon", dict[str, Any]], Any]


def _graduate(d: Daemon, p: dict[str, Any]) -> Any:
    """The hooks' graduations (a compaction, the session's end) and `gm graduate`."""
    if p.get("all"):
        return {"queued": d.sweep(min_idle=0.0, trigger="manual")}
    if p.get("trigger") == "compact":
        d.compacted(str(p["session"]))
    return {
        "queued": d.enqueue_graduation(
            str(p["session"]),
            str(p.get("trigger") or "manual"),
            d.redact(str(p.get("compact_summary") or "")),
        )
    }


def _sessions(d: Daemon, p: dict[str, Any]) -> Any:
    """The Sessions page; a page that has the current version is told so."""
    if p.get("since") == d.buffer.version and p.get("since") is not None:
        return {"version": d.buffer.version, "unchanged": True}
    return d.sessions_view()


def _rebuild_preview(d: Daemon, p: dict[str, Any]) -> Any:
    return {k: v for k, v in d.rebuild_preview(str(p["session"])).items() if k != "base"}


def _shutdown(d: Daemon, p: dict[str, Any]) -> None:
    d.stop()  # this answer still goes out: serve() waits for it


def _ids(p: dict[str, Any]) -> list[str]:
    return [str(i) for i in p.get("ids") or []]


ROUTES: dict[str, Route] = {
    # the daemon
    "ping": lambda d, p: {"pid": os.getpid()},
    "status": lambda d, p: d.status(),
    "maintain": lambda d, p: d.maintain(),
    "settled": lambda d, p: {"settled": d.settled()},
    "shutdown": _shutdown,
    # the hooks
    "recall": lambda d, p: d.recall(
        cwd=str(p["cwd"]) if p.get("cwd") else None,
        prompt=str(p["prompt"]),
        session=str(p["session"]),
    ),
    "remember": lambda d, p: d.remember(str(p["session"]), dict(p["turn"])),
    "graduate": _graduate,
    # sessions, by hand
    "forget": lambda d, p: d.forget(str(p["session"]), str(p["id"])),
    "graduate_now": lambda d, p: d.graduate_now(str(p["session"])),
    "regraduate": lambda d, p: d.regraduate(
        str(p["session"]),
        bool(p.get("preview")),
        bool(p.get("force")),
        str(p.get("agent") or "claude-code"),
    ),
    "rebuild_preview": _rebuild_preview,
    "apply_rebuild": lambda d, p: d.apply_rebuild(str(p["session"])),
    "resummarize": lambda d, p: d.resummarize(str(p["session"])),
    # reading memory
    "sessions": _sessions,
    "knowledge": lambda d, p: d.knowledge_view(lean=bool(p.get("lean"))),
    "knowledge_search": lambda d, p: d.knowledge_search(str(p.get("q") or "")),
    "knowledge_item": lambda d, p: d.knowledge_detail(str(p["id"])),
    "knowledge_review": lambda d, p: d.knowledge_review(),
    # correcting memory
    "knowledge_retract": lambda d, p: d.knowledge_retract(str(p["id"]), str(p.get("reason") or "")),
    "knowledge_restore": lambda d, p: d.knowledge_restore(str(p["id"])),
    "knowledge_confirm": lambda d, p: d.knowledge_confirm(str(p["id"])),
    "knowledge_delete": lambda d, p: d.knowledge_delete(_ids(p), str(p.get("reason") or "")),
    "knowledge_undelete": lambda d, p: d.knowledge_undelete(str(p["batch"])),
    "knowledge_scope": lambda d, p: d.knowledge_scope(_ids(p), str(p["to"])),
    "rescope": lambda d, p: d.rescope(),
    "rescope_preview": lambda d, p: d.rescope_preview(),
    "apply_rescope": lambda d, p: d.apply_rescope(),
    # briefs
    "knowledge_summarize": lambda d, p: d.knowledge_summarize(
        str(p.get("scope") or ""), bool(p.get("force"))
    ),
    "knowledge_summary": lambda d, p: d.knowledge_summary(str(p.get("scope") or "")),
}
