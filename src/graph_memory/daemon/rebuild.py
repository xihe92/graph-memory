"""Graduating a session again by hand: from its transcript under the current rules
(a rebuild, previewed on a copy of memory first when it is long), or folding its
summaries into one without touching knowledge.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from .. import clock
from ..adapters.base import mentioned_files
from ..buffer import Exchange, Summary
from ..graduation import (
    SUMMARY_CHARS,
    GraduationError,
    apply,
    build_resummarize_prompt,
    extract_step,
    known_entities,
    known_topics,
    merge_step,
    parts,
    plan_merge,
    segment_input,
)
from ..knowledge import KnowledgeStore, Segment
from ..knowledge_recall import KnowledgeIndex
from ..projects import project_of
from .state import FOLDED, NOTHING_LEARNED, NOTHING_PREFIX, REBUILD, DaemonState, Job, log

REBUILD_DIRECT_PARTS = 2  # a rebuild in more parts is previewed first unless forced


class Rebuilds(DaemonState):
    """`gm regraduate` and `gm resummarize`."""

    def regraduate(self, ref: str, preview: bool = False, force: bool = False) -> dict[str, Any]:
        """Queue a rebuild of what a session taught, read again from its transcript. With
        `preview` it runs on a copy of memory and changes nothing until `apply_rebuild`.
        A long session (more than REBUILD_DIRECT_PARTS parts) is always previewed first
        unless `force`: its replay is where stale items slip in."""
        from ..adapters.claude_code import find_transcript

        sid = ref.strip().split(":", 1)[-1]
        path = find_transcript(sid)
        if path is None:
            raise KeyError(f"no transcript (or more than one) for session {ref!r}")
        session = f"claude-code:{path.stem}"
        self._grad_queued[session] += 1
        job = Job(session, REBUILD, transcript=str(path), preview=preview, force=force)
        self.grad_queue.put_nowait(job)
        return {
            "session": session,
            "transcript": str(path),
            "queued": True,
            "preview": preview,
            "long_parts": REBUILD_DIRECT_PARTS,
        }

    def _preview_dir(self, session: str) -> Path:
        return self.cfg.home / "rebuild-preview" / quote(session, safe="")

    def _rebuild_input(self, job: Job, turns: list[Any]) -> list[Exchange]:
        """The transcript's turns that already graduated and left the buffer: not the ones
        still waiting (they graduate as usual), not a turn still in progress."""
        session = job.session
        waiting = {x.source.first for x in self.buffer.exchanges(session) if x.source}
        asking = self._asking.get(session)
        xs = []
        for ts, t in turns:
            if t.source is not None and t.source.first in waiting:
                continue
            # after a restart the waiting prompt is known only as far as the log kept it
            said = " ".join(t.prompt.split())
            if asking is not None and said.startswith(" ".join(asking[0].split())):
                continue
            files = [*mentioned_files(t.prompt), *t.files]
            project = project_of(files, t.cwd, use_cwd=False) or (
                project_of([], t.cwd) if t.cwd else None
            )
            xs.append(
                Exchange(
                    prompt=t.prompt,
                    reply=t.reply,
                    ts=_parse_time(ts) or clock.now(),
                    files=t.files,
                    commands=t.commands,
                    web=t.web,
                    project=project,
                    source=t.source,
                )
            )
        return xs

    async def _rebuild(self, job: Job) -> None:
        """Graduate a session again under the current rules, from the turns in its
        transcript, replayed in order: one graduation per part, each reading the summary
        the one before wrote, so a later decision retires an earlier one as it would have
        live. What the session taught before is matched (history too), not counted twice.

        For real, it changes memory part by part and then replaces the session's summaries
        with one. As a preview, it runs on a copy in `rebuild-preview/<session>/` and
        writes what it would change; `apply_rebuild` swaps the copy in."""
        from ..adapters.claude_code import read_turns

        session, started = job.session, time.perf_counter()
        if self.ask is None or job.transcript is None:
            return
        turns = [
            (ts, self._redacted(t))
            for ts, t in await asyncio.to_thread(read_turns, Path(job.transcript))
        ]
        xs = self._rebuild_input(job, turns)
        todo = parts(xs)
        # a long session whose design moved on replays into many stale items: look first
        long = len(todo) > REBUILD_DIRECT_PARTS and not job.preview and not job.force
        preview = job.preview or long
        base = {
            "items": sorted(self.knowledge.heads),
            "valid": sorted(self.knowledge.ids(valid=True)),
            "segments": sorted(self.knowledge.segment_ids()),
        }
        if preview:
            pdir = self._preview_dir(session)
            shutil.rmtree(pdir, ignore_errors=True)
            pdir.mkdir(parents=True)
            self.knowledge.copy_to(pdir)  # saved first; then a copy to change freely
            self.knowledge_dirty = False
            store = KnowledgeStore(pdir, self.embedder)
        else:
            store = self.knowledge
        done: list[Segment] = []
        summary = ""
        try:
            for part in todo:
                seg = segment_input(session, part, REBUILD, earlier=summary)
                known = known_entities(store, seg)
                ex = await asyncio.to_thread(
                    extract_step, seg, self.ask, self.embedder, known, known_topics(store)
                )
                plan = plan_merge(store, seg, ex)
                merged = await asyncio.to_thread(merge_step, plan, self.ask)
                if plan.prompt is not None and merged is None:
                    # without the merge, what the session taught before would be added
                    # again as new items: stop here
                    raise GraduationError("the merge call failed twice")
                done.append(apply(store, plan, merged, time.perf_counter()))
                summary = ex.summary or summary
                store.save()
                if store is self.knowledge:
                    self.knowledge_dirty = False
            # read while the preview's copy is still open
            result = _rebuild_diff(base, store, done, summary, session, len(xs))
        except Exception as exc:
            log.warning("rebuilding %s failed: %s", session, exc)
            self.events.emit(
                "graduate",
                session=session,
                trigger=REBUILD,
                ok=False,
                preview=preview,
                parts_done=len(done),
                error=f"{type(exc).__name__}: {exc}"[:300],
                ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return
        finally:
            if store is not self.knowledge:
                store.close()
        if preview:
            (self._preview_dir(session) / "preview.json").write_text(json.dumps(result, indent=1))
        elif done:
            result["replaced"] = self._rebuild_summary(session, done, summary, xs)
        self.events.emit(
            "graduate",
            session=session,
            trigger=REBUILD,
            ok=True,
            preview=preview,
            long=long,
            exchanges=len(xs),
            parts=len(done),
            items=len({i for g in done for i in g.items}),
            new=len(result["new"]),
            retired=len(result["retired"]),
            calls=sum(g.calls for g in done),
            ms=round((time.perf_counter() - started) * 1000, 1),
            replaced=result.get("replaced", 0),
        )
        log.info("rebuilt %s%s: %d exchanges in %d parts", session,
                 " (preview)" if preview else "", len(xs), len(done))  # fmt: skip

    def _rebuild_summary(
        self, session: str, done: list[Segment], summary: str, xs: list[Exchange]
    ) -> int:
        """Replace the session's summary records with one for the whole rebuild; an ended
        session keeps no buffer."""
        if session not in self.buffer.sessions() or not done:
            return 0
        old = self.buffer.summaries(session)
        items = list(dict.fromkeys(i for g in done for i in g.items))
        learned = self.knowledge.get_items(items)
        text = summary or "\n".join(f"[{i.type.value}] {i.statement}" for i in learned)
        return self.buffer.replace_summaries(
            session,
            Summary(
                trigger=REBUILD,
                start=xs[0].ts,
                end=xs[-1].ts,
                exchanges=len(xs),
                segment=done[-1].segment_id,
                items=items,
                text=text or NOTHING_LEARNED,
                graduations=len(done),
                # every earlier graduation of the session: the rebuild covers them all
                folded=list(
                    dict.fromkeys(
                        [g for s in old for g in (*s.folded, s.segment) if g]
                        + [
                            g.segment_id
                            for g in self.knowledge.segments(session=session)
                            if g.segment_id != done[-1].segment_id
                        ]
                    )
                ),
            ),
        )

    def rebuild_preview(self, ref: str) -> dict[str, Any]:
        """What a previewed rebuild would change."""
        session = f"claude-code:{ref.strip().split(':', 1)[-1]}"
        hits = [
            p
            for p in (self.cfg.home / "rebuild-preview").glob("*/preview.json")
            if unquote(p.parent.name).split(":", 1)[-1].startswith(session.split(":", 1)[1])
        ]
        if len(hits) != 1:
            raise KeyError(f"{'no' if not hits else 'more than one'} rebuild preview for {ref!r}")
        return dict(json.loads(hits[0].read_text()))

    def apply_rebuild(self, ref: str) -> dict[str, Any]:
        """Swap a previewed rebuild in, if memory has not learned or lost anything since
        the preview started (counters such as uses since then are not carried over)."""
        result = self.rebuild_preview(ref)
        session = result["session"]
        k = self.knowledge
        if sorted(k.heads) != result["base"]["items"] or (
            sorted(k.segment_ids()) != result["base"]["segments"]
        ):
            raise ValueError("memory changed since the preview: run the preview again")
        pdir = self._preview_dir(session)
        preview = KnowledgeStore(pdir, self.embedder)
        done = preview.segments(ids=result["segments"])
        # the preview's memory replaces this one: close it, copy the preview over it, open
        self.knowledge.close()
        preview.copy_to(self.cfg.home)
        preview.close()
        self.knowledge = KnowledgeStore(self.cfg.home, self.embedder)
        self.kindex = KnowledgeIndex(self.knowledge)
        self.knowledge_dirty = False
        xs = [
            Exchange(prompt="", reply="", ts=datetime.fromisoformat(t))
            for t in (result["start"], result["end"])
        ]
        result["replaced"] = self._rebuild_summary(session, done, result["summary"], xs)
        shutil.rmtree(pdir, ignore_errors=True)
        self.events.emit(
            "rebuild_applied", session=session, new=len(result["new"]), replaced=result["replaced"]
        )
        return {k2: v for k2, v in result.items() if k2 != "base"}

    def resummarize(self, ref: str) -> dict[str, Any]:
        """Queue folding a session's summaries into one (see _fold_summaries)."""
        session = self._session_ref(ref)
        n = len(self.buffer.summaries(session))
        if n < 2:
            return {"session": session, "summaries": n, "queued": False}
        self._grad_queued[session] += 1
        self.grad_queue.put_nowait(Job(session, FOLDED))
        return {"session": session, "summaries": n, "queued": True}

    async def _fold_summaries(self, job: Job) -> None:
        """Replace a session's summary records with one, written by the model from what
        the session taught and the summaries along the way; knowledge is not touched. For
        a session whose summaries were written under older rules, when graduating it again
        would do more harm than good (a long session whose design changed as it went)."""
        session, started = job.session, time.perf_counter()
        old = self.buffer.summaries(session)
        if len(old) < 2:
            return
        k = self.knowledge
        taught = sorted(
            k.get_items(i for i in k.taught_by(session) if k.heads[i].valid),
            key=lambda i: max(s.until for s in i.sources if s.session == session),
        )
        texts = [s.text for s in old if s.text and not s.text.startswith(NOTHING_PREFIX)]
        try:
            if self.ask is None:
                raise GraduationError("no model to write the summary")
            prompt = build_resummarize_prompt(taught, texts)
            text = " ".join((await asyncio.to_thread(self.ask, prompt)).split())[:SUMMARY_CHARS]
            if not text:
                raise GraduationError("the model wrote no summary")
        except Exception as exc:
            log.warning("folding the summaries of %s failed: %s", session, exc)
            self.events.emit("summaries_folded", session=session, ok=False, error=f"{exc}"[:300])
            return
        segs = [g for s in old for g in (*s.folded, s.segment) if g]
        segs += [g.segment_id for g in k.segments(session=session)]
        segs = list(dict.fromkeys(segs))
        self.buffer.replace_summaries(
            session,
            Summary(
                trigger=FOLDED,
                start=min((s.start for s in old if s.start), default=None),
                end=max((s.end for s in old if s.end), default=None),
                exchanges=sum(s.exchanges for s in old),
                segment=segs[-1] if segs else None,
                items=list(dict.fromkeys(i for s in old for i in s.items)),
                text=text,
                graduations=sum(s.graduations for s in old),
                folded=segs[:-1],
            ),
        )
        self.events.emit(
            "summaries_folded",
            session=session,
            ok=True,
            replaced=len(old),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )


def _rebuild_diff(
    base: dict[str, list[str]],
    store: KnowledgeStore,
    done: list[Segment],
    summary: str,
    session: str,
    exchanges: int,
) -> dict[str, Any]:
    """What a rebuild changed, against the item ids memory had before it."""
    before = set(base["items"])

    def briefs(ids: Iterable[str]) -> list[dict[str, Any]]:
        return [
            {"id": it.item_id, "type": it.type.value, "statement": it.statement, "valid": it.valid}
            for it in store.get_items(ids)
        ]

    touched = list(dict.fromkeys(i for g in done for i in g.items))
    heads = store.heads
    return {
        "session": session,
        "exchanges": exchanges,
        "parts": len(done),
        "segments": [g.segment_id for g in done],
        "start": done[0].start.isoformat() if done else clock.now().isoformat(),
        "end": done[-1].end.isoformat() if done else clock.now().isoformat(),
        "summary": summary,
        "base": base,
        "new": briefs(i for i in heads if i not in before),
        "matched": briefs(i for i in touched if i in before),
        # true before the rebuild, no longer true after it
        "retired": briefs(i for i in base["valid"] if i in heads and not heads[i].valid),
    }


def _parse_time(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None
    except ValueError:
        return None
