"""A turn as the hooks see it: the notes recalled for a prompt (UserPromptSubmit), then
the finished exchange stored for graduation and its reply checked for which notes it
used, and sometimes judged (Stop).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from .. import clock
from ..adapters.base import Turn, mentioned_files
from ..buffer import Exchange, Summary
from ..judge import build_judge_prompt, verdicts
from ..knowledge import TOPIC_PREFIX, at_risk, in_view, named_topics
from ..knowledge_recall import note, used_by
from ..knowledge_recall import recall as knowledge_recall
from ..projects import project_of
from ..words import tokens
from .state import DaemonState, log

TOPIC_RECORDS = 40  # the session's latest exchanges and summaries read for its topics
PROMPT_IN_EVENT = 200  # characters of the prompt kept on knowledge_recall events


class Turns(DaemonState):
    """Recall for a prompt; remember the exchange; check and judge the notes it was shown."""

    def recall(self, session: str, prompt: str, cwd: str | None) -> dict[str, Any]:
        """The UserPromptSubmit hook: the notes for this prompt, as context for Claude
        (none in shadow mode, where recall is only logged)."""
        prompt = self.redact(prompt)
        here = project_of([], cwd) if cwd else None
        self._seen(session, here)
        # a prompt you typed (the hook skips programs and notifications): today is a day
        # you worked, which notes age by
        if self.knowledge.mark_active(here):
            self.knowledge_dirty = True
        self._asking[session] = (prompt, clock.now())
        context: dict[str, Any] = {}
        if self.cfg.knowledge_recall in ("shadow", "on"):
            lines = self.recall_knowledge(session, prompt, cwd)
            if lines and self.cfg.knowledge_recall == "on":
                context["knowledge"] = lines
        return {"context": context}

    def recall_knowledge(self, session: str, prompt: str, cwd: str | None) -> list[str]:
        """Knowledge items for this prompt, as lines; logged either way. Leaves out items
        already shown in this session and items the session can still see (learned only
        from its own exchanges since its last compaction)."""
        try:
            k = self.knowledge
            shown = self._shown.setdefault(session, set())
            since = self._last_compaction(session)
            taught = [i for i in k.taught_by(session) if k.heads[i].valid]
            own = {i.item_id for i in k.get_items(taught) if in_view(i, session, since)}
            first = session not in self._greeted
            self._greeted.add(session)
            here = project_of([], cwd) if cwd else None
            topics = self._session_topics(session, prompt)
            got = knowledge_recall(
                k,
                self.kindex,
                prompt,
                project=here,
                exclude=shown | own,
                standing=first,
                topics=topics,
            )
            if self.cfg.knowledge_recall == "on":
                shown.update(i.item_id for i in got.items)
                k.mark_recalled(i.item_id for i in got.items)
            # replaced or cleared on every prompt: a turn interrupted before its reply
            # must not have its notes checked against the next prompt's reply
            if got.items:
                self._recalled[session] = [i.item_id for i in got.items]
            else:
                self._recalled.pop(session, None)
            self.events.emit(
                "knowledge_recall",
                session=session,
                mode=self.cfg.knowledge_recall,
                prompt=" ".join(prompt.split())[:PROMPT_IN_EVENT],
                items=[i.item_id for i in got.items],
                scores=got.scores,
                why=got.why,
                entities=got.entities,
                seeds=len(got.seeds),
                reason=got.reason,
                topics=sorted(t[len(TOPIC_PREFIX) :] for t in topics),
                ms=got.ms,
            )
            # each note says when it was last confirmed and where it came from; an item from
            # another project (an open item the prompt asked about) says which
            return [note(i, here) for i in got.items]
        except Exception:
            log.exception("knowledge recall failed")
            return []

    def _session_topics(self, session: str, prompt: str) -> set[str]:
        """The topics memory knows that this session is about: named in the prompt, in
        the exchanges it has waiting, or in its summaries."""
        self.kindex.fresh()
        scopes = self.kindex.topics()  # topics with something recall can show
        if not scopes:
            return set()
        texts = [prompt]
        for r in self.buffer.records(session)[-TOPIC_RECORDS:]:
            texts.append(r.text if isinstance(r, Summary) else f"{r.prompt}\n{r.reply[:3000]}")
        return named_topics(scopes, "\n".join(texts))

    def remember(self, session: str, turn_data: dict[str, Any]) -> dict[str, Any]:
        """Store the finished exchange in the session's buffer, then check which recalled
        notes the reply used."""
        turn = self._redacted(Turn.model_validate(turn_data))
        files = [*mentioned_files(turn.prompt), *turn.files]
        project = project_of(files, turn.cwd, use_cwd=False) or (
            project_of([], turn.cwd) if turn.cwd else None
        )
        self._seen(session, project)
        self._asking.pop(session, None)
        result: dict[str, Any] = {"rewound": 0, "continued": False}
        chain = self.buffer.exchanges(session)
        first = turn.source.first if turn.source else None
        same = next(
            (
                x
                for x in chain
                if first and x.source and x.source.first == first and x.prompt == turn.prompt
            ),
            None,
        )
        if same is not None:
            # the turn went on after a background task finished: the transcript reads
            # back as the same prompt with a longer reply
            ex = same.model_copy(
                update={
                    "reply": turn.reply,
                    "source": turn.source,
                    "files": turn.files or same.files,
                    "commands": turn.commands or same.commands,
                    "web": turn.web or same.web,
                }
            )
            if ex != same:
                self.buffer.update(session, ex)
            result["continued"] = True
        else:
            if turn.previous_reply_ref:
                # the prompt follows an older reply than the newest: the user rewound,
                # and what came after that reply was discarded
                prev = next(
                    (x for x in chain if x.source and x.source.last == turn.previous_reply_ref),
                    None,
                )
                if prev is not None:
                    dropped = self.buffer.drop_after(session, prev.id)
                    if dropped:
                        result["rewound"] = len(dropped)
                        self.events.emit("rewind", session=session, dropped=[d.id for d in dropped])
            ex = self.buffer.append(
                session,
                Exchange(
                    prompt=turn.prompt,
                    reply=turn.reply,
                    files=turn.files,
                    commands=turn.commands,
                    web=turn.web,
                    project=project,
                    source=turn.source,
                ),
            )
        result["exchange"] = ex.id
        result["project"] = project
        self.events.emit(
            "remember",
            session=session,
            exchange=ex.id,
            project=project,
            continued=result["continued"],
            rewound=result["rewound"],
        )
        recalled = self._recalled.pop(session, None)
        if recalled:
            self._background(self._check_use(session, recalled, turn.reply, ex.id))
            # a note that could have gone stale, or that nobody checked against a source,
            # is always judged; the rest, a sample
            stale_risk = any(at_risk(i) for i in self.knowledge.get_items(recalled))
            if (
                self.judge is not None
                and self.cfg.knowledge_recall == "on"
                and (stale_risk or self._rng() < self.cfg.judge_rate)
            ):
                self._background(self._judge(session, recalled, turn.prompt, turn.reply))
        return result

    def _background(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _check_use(
        self, session: str, item_ids: list[str], reply: str, node: str | None = None
    ) -> None:
        """After the reply: which recalled items did it use? Those get stronger, and items
        used together get linked. Counted in shadow mode too: there it measures whether
        recall picked what the reply turned out to need."""
        try:
            embed = self.embedder
            vec = await asyncio.to_thread(embed, reply[:4000]) if embed else None
            k = self.knowledge
            words = set(tokens(reply))
            items = k.get_items(item_ids)
            df = k.word_df(t for i in items for t in tokens(i.statement))  # one query for all
            used = [
                i.item_id
                for i in items
                if used_by(self.kindex, i, words, vec, k.vectors.get(i.item_id), df)
            ]
            k.mark_used(used)
            self.knowledge_dirty = True
            self.events.emit(
                "knowledge_use",
                session=session,
                mode=self.cfg.knowledge_recall,
                recalled=item_ids,
                used=used,
                node=node,
            )
        except Exception:
            log.exception("knowledge use check failed")

    async def _judge(self, session: str, item_ids: list[str], prompt: str, reply: str) -> None:
        """A model labels each note shown for this turn useful, neutral, misleading or
        contradicted. A contradicted note is disputed: shown with a warning from then on,
        retracted when it happens again."""
        assert self.judge is not None
        items = self.knowledge.get_items(item_ids)
        if not items:
            return
        started = time.perf_counter()
        try:
            answer = await asyncio.to_thread(self.judge, build_judge_prompt(prompt, reply, items))
            found = verdicts(answer, items)
            labels = {i: lab for i, (lab, _) in found.items()}
            self.events.emit(
                "knowledge_judge",
                session=session,
                items=[i.item_id for i in items],
                labels=labels,
                ms=round((time.perf_counter() - started) * 1000, 1),
                ok=True,
            )
            for item_id, (lab, why) in found.items():
                if lab != "contradicted" or not self.knowledge.has(item_id):
                    continue
                reason = why or f"contradicted by the reply to: {' '.join(prompt.split())[:80]}"
                it = self.knowledge.dispute(item_id, reason)
                self.knowledge_dirty = True
                self.events.emit(
                    "knowledge_dispute",
                    session=session,
                    item=item_id,
                    reason=reason,
                    disputes=it.disputes,
                    retracted=not it.valid,
                )
        except Exception as exc:
            log.warning("judging notes failed: %s", exc)
            self.events.emit(
                "knowledge_judge",
                session=session,
                items=[i.item_id for i in items],
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )

    def forget(self, session: str, exchange_id: str) -> dict[str, Any]:
        """Remove an exchange recorded by mistake, before it can graduate."""
        ex = self.buffer.remove(session, exchange_id)
        if ex is None:
            raise KeyError(f"no exchange {exchange_id!r} waiting in {session!r}")
        self.events.emit(
            "forget", session=session, exchange=exchange_id, text=" ".join(ex.prompt.split())[:120]
        )
        return {"exchange": exchange_id}

    def compacted(self, session: str) -> None:
        """The session's context was compacted: what it was shown and what it learned
        before is no longer in its context, so both may be recalled again."""
        self._compacted[session] = clock.now()
        self._shown.pop(session, None)
        self._greeted.discard(session)
        self.events.emit("compacted", session=session)

    def _last_compaction(self, session: str) -> datetime | None:
        """In memory since this daemon started; else from the session's buffer, whose
        compaction summaries survive a restart."""
        if session in self._compacted:
            return self._compacted[session]
        times = [s.ts for s in self.buffer.summaries(session) if s.trigger == "compact"]
        return max(times) if times else None
