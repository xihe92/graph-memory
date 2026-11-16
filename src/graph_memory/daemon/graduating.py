"""Graduation: a session's waiting exchanges, once it goes quiet, compacts or ends, are
turned into knowledge by the model and leave the buffer as a summary. One job runs at a
time (see Daemon._graduator), so the knowledge store has a single writer.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .. import clock
from ..buffer import Exchange, Summary
from ..graduation import (
    apply,
    extract_step,
    known_entities,
    known_topics,
    merge_step,
    plan_merge,
    segment_input,
)
from ..knowledge import Segment
from .state import ENDING, NOTHING_LEARNED, NOTHING_PREFIX, DaemonState, Job, log


class Graduating(DaemonState):
    """Queue sessions for graduation, and graduate them."""

    def sweep(self, *, min_idle: float | None = None, trigger: str = "idle") -> int:
        """Queue the sessions whose newest exchange waited `min_idle` seconds (default:
        graduation_idle), and the sessions untouched for buffer_abandon_days."""
        now = clock.now()
        idle = self.cfg.graduation_idle if min_idle is None else min_idle
        queued = 0
        for session in self.buffer.sessions():
            recs = self.buffer.records(session)
            last = max((r.ts for r in recs), default=None)
            if last is None:
                self.buffer.delete(session)
                continue
            quiet = (now - last).total_seconds()
            if quiet >= self.cfg.buffer_abandon_days * 86400:
                queued += self.enqueue_graduation(session, "abandoned")
            elif any(isinstance(r, Exchange) for r in recs) and quiet >= idle:
                queued += self.enqueue_graduation(session, trigger)
        return queued

    def enqueue_graduation(self, session: str, trigger: str, compact_summary: str = "") -> bool:
        if self._grad_queued[session] and not compact_summary and trigger not in ENDING:
            return False  # one waiting already covers the same exchanges
        pending = self.buffer.exchanges(session)
        self._grad_queued[session] += 1
        self.grad_queue.put_nowait(
            Job(session, trigger, compact_summary, pending[-1].id if pending else None)
        )
        return True

    def graduate_now(self, ref: str) -> dict[str, Any]:
        """Graduate what a session has waiting now; the session goes on."""
        session = self._session_ref(ref)
        waiting = len(self.buffer.exchanges(session))
        queued = self.enqueue_graduation(session, "manual") if waiting else False
        return {"session": session, "waiting": waiting, "queued": queued}

    async def _graduate(self, job: Job) -> None:
        """Graduate what the job covers of a session's waiting exchanges into knowledge,
        and leave a summary in their place."""
        session = job.session
        pending = self.buffer.exchanges(session)
        if job.upto is not None:
            idx = next((i for i, x in enumerate(pending) if x.id == job.upto), None)
            pending = pending[: idx + 1] if idx is not None else pending
        else:
            pending = []
        ending = job.trigger in ENDING
        if ending:
            self._live.pop(session, None)
            self._asking.pop(session, None)
        if not pending:
            if ending and not self.buffer.exchanges(session):
                self.buffer.delete(session)
                self.events.emit("session_cleared", session=session, trigger=job.trigger)
            return
        if self.ask is None:
            # no model: nothing can be learned, so nothing is kept past the session
            if ending:
                self.buffer.delete(session)
                self.events.emit("session_cleared", session=session, trigger=job.trigger)
            return
        # a compaction's own summary covers the whole session; otherwise the graduation
        # reads the summary the session's last graduation (or compaction) left
        earlier = ""
        if not job.compact_summary:
            last = self.buffer.summaries(session)
            if last and not last[-1].text.startswith(NOTHING_PREFIX):
                earlier = last[-1].text
        started = time.perf_counter()
        seg = segment_input(session, pending, job.trigger, job.compact_summary, earlier)
        try:
            known = known_entities(self.knowledge, seg)
            topics = known_topics(self.knowledge)
            ex = await asyncio.to_thread(extract_step, seg, self.ask, self.embedder, known, topics)
            plan = plan_merge(self.knowledge, seg, ex)
            merged = await asyncio.to_thread(merge_step, plan, self.ask)
            segment = apply(self.knowledge, plan, merged, started)
        except Exception as exc:
            self._graduation_failed(job, pending, exc, started)
            return
        # the knowledge is on disk before the exchanges leave the buffer: a crash in
        # between graduates them again, and merging removes the duplicates
        self.knowledge.save()
        self.knowledge_dirty = False
        self._failures.pop(session, None)
        covered = [x.id for x in pending]
        if ending and len(self.buffer.exchanges(session)) == len(covered):
            self.buffer.delete(session)
        else:
            self._summarize(job, pending, segment, ex.summary)
        self.events.emit(
            "graduate",
            session=session,
            trigger=job.trigger,
            segment=segment.segment_id,
            exchanges=segment.exchanges,
            items=len(segment.items),
            new=len(segment.new),
            calls=segment.calls,
            ms=segment.ms,
            ok=True,
            cleared=ending and not self.buffer.records(session),
        )
        log.info(
            "graduated %s: %d exchanges -> %d items", session, segment.exchanges, len(segment.items)
        )

    def _summarize(self, job: Job, pending: list[Exchange], segment: Segment, written: str) -> None:
        """Replace the graduated exchanges with a summary record: Claude's own after a
        compaction, else the summary of the session so far that the graduation wrote. A
        graduation folds into the summary just before it, so a session keeps one summary
        that grows, not one per pause; a compaction starts a new one."""
        session = job.session
        covered = [x.id for x in pending]
        recs = self.buffer.records(session)
        at = next((n for n, r in enumerate(recs) if r.id == covered[0]), 0)
        prev = recs[at - 1] if at > 0 else None
        if isinstance(prev, Summary) and prev.trigger != "compact" and job.trigger != "compact":
            folded = prev.model_copy(
                update={
                    "ts": clock.now(),
                    "trigger": job.trigger,
                    "end": pending[-1].ts,
                    "exchanges": prev.exchanges + len(pending),
                    "segment": segment.segment_id,
                    "items": list(dict.fromkeys([*prev.items, *segment.items])),
                    "text": written or prev.text,
                    "graduations": prev.graduations + 1,
                    "folded": [*prev.folded, *([prev.segment] if prev.segment else [])],
                }
            )
            self.buffer.summarize(session, covered, folded, into=prev.id)
            return
        text = job.compact_summary if job.trigger == "compact" else written
        if not text:  # a model that wrote no summary: what it learned, if anything
            learned = self.knowledge.get_items(segment.items)
            text = "\n".join(f"[{i.type.value}] {i.statement}" for i in learned) or NOTHING_LEARNED
        self.buffer.summarize(
            session,
            covered,
            Summary(
                trigger=job.trigger,
                start=pending[0].ts,
                end=pending[-1].ts,
                exchanges=len(pending),
                segment=segment.segment_id,
                items=segment.items,
                text=text,
            ),
        )

    def _graduation_failed(
        self, job: Job, pending: list[Exchange], exc: Exception, started: float
    ) -> None:
        """Keep the exchanges for the next try, unless this session has failed too often
        or its oldest exchange is too old: then give up on them, so a broken model can't
        make the buffer grow without limit."""
        session = job.session
        self._failures[session] += 1
        age_days = (clock.now() - pending[0].ts).total_seconds() / 86400
        give_up = (
            self._failures[session] >= self.cfg.graduation_attempts
            or age_days >= self.cfg.buffer_max_days
        )
        log.warning("graduation failed for %s: %s", session, exc)
        self.events.emit(
            "graduate",
            session=session,
            trigger=job.trigger,
            ok=False,
            ms=round((time.perf_counter() - started) * 1000, 1),
            error=f"{type(exc).__name__}: {exc}"[:300],
            attempt=self._failures[session],
            given_up=give_up,
        )
        if give_up:
            self.buffer.drop(session, [x.id for x in pending])
            self._failures.pop(session, None)
            if job.trigger in ENDING and not self.buffer.exchanges(session):
                self.buffer.delete(session)
            self.events.emit(
                "buffer_dropped", session=session, exchanges=len(pending), reason=str(exc)[:200]
            )
