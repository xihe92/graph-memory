"""Briefs: the model writes a summary of one scope's notes for a person to read (see
summarize.py); the daemon queues it, caches it, and says when it is out of date.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .. import clock
from ..graduation import GraduationError
from ..knowledge import Item, topic_of
from .scopes import scope_names
from .state import SUMMARIZE, DaemonState, Job, log

SUMMARIZE_JOB = "memory:summarize:"  # + the scope: the queue's name for that job


class Summaries(DaemonState):
    """`gm knowledge summarize` and the Knowledge page's brief."""

    def _scope_items(self, scope: str) -> list[Item]:
        return list(self.knowledge.scan(valid=True, scope=scope))

    def _scope_label(self, scope: str) -> tuple[str, str]:
        """(short name, how the prompt and the page call it)."""
        if scope == "global":
            return "global", "the user themself and how they like to work (global notes)"
        name = topic_of(scope)
        if name:
            return name, f"the topic {name}"
        names = {
            x["key"]: x["name"] for x in scope_names(h.scope for h in self.knowledge.heads.values())
        }
        title = names.get(scope, scope.rsplit("/", 1)[-1])
        return title, f"the project {title}"

    def _summaries(self) -> dict[str, Any]:
        return self.knowledge.summaries()

    def knowledge_summarize(self, to: str, force: bool = False) -> dict[str, Any]:
        """Queue a brief of one scope's valid items (see summarize.py). Nothing is queued
        when the scope's brief is current, unless `force`, or when one is being written."""
        from ..summarize import changed_since

        scope = self._scope_ref(to)
        items = self._scope_items(scope)
        if not items:
            raise KeyError(f"memory has no notes in {to!r} to summarize")
        if self.ask is None:
            raise GraduationError("no model to ask")
        job = SUMMARIZE_JOB + scope
        if self._grad_queued.get(job):
            return {"scope": scope, "queued": False, "running": True, "items": len(items)}
        have = self._summaries().get(scope)
        if have and not force and changed_since(have.get("notes") or {}, items) == 0:
            return {"scope": scope, "queued": False, "running": False, "items": len(items)}
        self._summary_errors.pop(scope, None)
        self._grad_queued[job] += 1
        self.grad_queue.put_nowait(Job(job, SUMMARIZE, scope=scope))
        return {"scope": scope, "queued": True, "running": True, "items": len(items)}

    async def _brief(self, job: Job) -> None:
        from ..summarize import select, stamp, summarize

        scope = job.scope
        started = time.perf_counter()
        assert self.ask is not None
        items, left_out = select(self._scope_items(scope))
        name, label = self._scope_label(scope)
        try:
            if not items:
                raise KeyError("no notes left in this scope")
            brief = await asyncio.to_thread(summarize, label, items, self.ask)
        except Exception as exc:
            log.warning("summarizing %s failed: %s", scope, exc)
            self._summary_errors[scope] = f"{type(exc).__name__}: {exc}"[:300]
            self.events.emit(SUMMARIZE, ok=False, scope=scope, error=self._summary_errors[scope])
            return
        brief.update(
            scope=scope,
            name=name,
            at=clock.now().isoformat(),
            items=len(items),
            left_out=left_out,
            notes={i.item_id: stamp(i) for i in items},
        )
        valid = {h.scope for h in self.knowledge.heads.values() if h.valid}
        kept = {k: v for k, v in self._summaries().items() if k in valid}  # drop empty scopes
        kept[scope] = brief
        self.knowledge.set_summaries(kept)
        self.knowledge_dirty = False
        self.events.emit(
            SUMMARIZE,
            ok=True,
            scope=scope,
            items=len(items),
            left_out=left_out,
            points=sum(len(s["points"]) for s in brief["sections"]),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def knowledge_summary(self, to: str) -> dict[str, Any]:
        """The cached brief of a scope, how many notes changed since it was written, and
        whether one is being written now."""
        from ..summarize import changed_since

        scope = self._scope_ref(to)
        items = self._scope_items(scope)
        have = self._summaries().get(scope)
        return {
            "scope": scope,
            "name": self._scope_label(scope)[0],
            "items": len(items),
            "running": bool(self._grad_queued.get(SUMMARIZE_JOB + scope)),
            "error": self._summary_errors.get(scope),
            "summary": have,
            "changed": changed_since(have.get("notes") or {}, items) if have else 0,
        }
