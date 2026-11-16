"""What the web page and `gm knowledge` read: the Sessions page's chains, every item,
search, one item's detail, and what needs a person's review. Read only.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any

from .. import clock
from ..buffer import Exchange, Summary
from ..knowledge import Entity, Item, ItemEdge
from .scopes import scope_names
from .state import LIVE, DaemonState

# A session with nothing to show (nothing buffered or graduated, no prompt in progress)
# is shown only this long, not LIVE: one that just started, not a script's run hours ago
JUST_STARTED = timedelta(minutes=30)


class Views(DaemonState):
    """Read-only views of memory, for the page and the CLI."""

    def sessions_view(self) -> dict[str, Any]:
        """For the Sessions page: each live session's chain, oldest first: what already
        graduated from it (each graduation, with the items it taught), then what waits in
        the buffer, then the prompt still waiting for its reply. And the sessions that
        graduated in the last three days."""
        k = self.knowledge
        now = clock.now()
        buffered = dict(self.buffer.iter_all())
        graduated_from = {g.session for g in k.segments.values()}
        keys = set(buffered) | {
            s
            for s, (seen, _) in self._live.items()
            if now - seen < JUST_STARTED
            or (now - seen < LIVE and (s in graduated_from or s in self._asking))
        }
        live = []
        for session in keys:
            recs = buffered.get(session, [])
            seen, known_project = self._live.get(session, (None, None))
            projects = Counter(r.project for r in recs if isinstance(r, Exchange) and r.project)
            graduated = sorted(
                (g for g in k.segments.values() if g.session == session),
                key=lambda g: g.start,
            )
            # a buffered summary already stands for its graduation
            summarized = {
                seg for r in recs if isinstance(r, Summary) for seg in (r.segment, *r.folded) if seg
            }
            chain: list[dict[str, Any]] = [
                {
                    "kind": "graduated",
                    "id": "g:" + g.segment_id,
                    "trigger": g.trigger,
                    "start": g.start.isoformat(),
                    "end": g.end.isoformat(),
                    "ts": g.created_at.isoformat(),
                    "exchanges": g.exchanges,
                    "items": [self._item_brief(i, g.segment_id) for i in g.items if i in k.items],
                }
                for g in graduated
                if g.segment_id not in summarized
            ]
            for r in recs:
                rec = r.model_dump(mode="json")
                if isinstance(r, Summary):  # what it learned, readable on the page
                    rec["items"] = [
                        self._item_brief(i, r.segment or "") for i in r.items if i in k.items
                    ]
                chain.append(rec)
            asking = self._asking.get(session)
            if asking is not None:
                chain.append(
                    {
                        "kind": "asking",
                        "id": "q:" + session,
                        "prompt": asking[0],
                        "ts": asking[1].isoformat(),
                    }
                )
            times = [x for x in (seen, *(r.ts for r in recs)) if x is not None]
            live.append(
                {
                    "session": session,
                    "project": projects.most_common(1)[0][0] if projects else known_project,
                    "last": max(times).isoformat() if times else None,
                    "waiting": sum(1 for r in recs if isinstance(r, Exchange)),
                    "records": chain,
                }
            )
        cutoff = clock.now() - timedelta(days=3)
        recent = sorted(
            (s for s in k.segments.values() if s.created_at >= cutoff),
            key=lambda s: s.created_at,
            reverse=True,
        )
        return {
            "version": self.buffer.version,
            "sessions": sorted(live, key=lambda s: s["last"] or "", reverse=True),
            "graduated": [
                {
                    "segment": s.segment_id,
                    "session": s.session,
                    "trigger": s.trigger,
                    "exchanges": s.exchanges,
                    "at": s.created_at.isoformat(),
                    "items": [
                        {
                            "id": i,
                            "type": k.items[i].type.value,
                            "statement": k.items[i].statement,
                            "new": k.items[i].segments[0] == s.segment_id,
                        }
                        for i in s.items
                        if i in k.items
                    ],
                }
                for s in recent[:30]
            ],
        }

    def _item_brief(self, item_id: str, segment_id: str) -> dict[str, Any]:
        """An item as the Sessions page lists it under a graduation or summary."""
        it = self.knowledge.items[item_id]
        return {
            "id": item_id,
            "type": it.type.value,
            "statement": it.statement,
            "new": bool(it.segments) and it.segments[0] == segment_id,
            "valid": it.valid,
        }

    def knowledge_view(self, lean: bool = False) -> dict[str, Any]:
        """Every item, entity and edge. `lean` (the Knowledge page) leaves out what only an
        item's detail shows (reason, sources, provenance, review dates): the page loads
        those on click, by id, and searches through the daemon."""
        k = self.knowledge
        now = clock.now()

        def item(i: Item) -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": i.item_id,
                "type": i.type.value,
                "statement": i.statement,
                "scope": i.scope,
                "entities": i.entities,
                "valid": i.valid,
                "strength": i.strength,
                "area": i.area.value,
                "uses": i.uses,
                "created_at": i.created_at.isoformat(),
                "invalid_at": i.invalid_at.isoformat() if i.invalid_at else None,
                "invalidated_by": i.invalidated_by,
                "invalid_reason": i.invalid_reason,
                "disputes": i.disputes,
                "overdue": i.overdue(now),
            }
            if not lean:
                out.update(
                    why=i.why,
                    vitality=round(i.vitality(now), 2),
                    sources=[s.model_dump(mode="json") for s in i.sources],
                    origin=i.origin.value,
                    ref=i.ref,
                    applies_to=i.applies_to,
                    volatility=i.volatility.value,
                    checked=i.checked.isoformat(),
                    dispute_reason=i.dispute_reason,
                    recallable=i.recallable,
                )
            return out

        def entity(e: Entity) -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": e.entity_id,
                "name": e.name,
                "kind": e.kind.value,
                "mentions": e.mentions,
            }
            if not lean:
                out["summary"] = e.summary
            return out

        def edge(e: ItemEdge) -> dict[str, Any]:
            out: dict[str, Any] = {
                "source": e.source,
                "target": e.target,
                "relation": e.relation.value,
                "weight": e.weight,
                "reason": e.reason,
            }
            if not lean:
                out["origin"] = e.origin
            return out

        return {
            "version": k.version,
            "stats": k.stats(),
            "scopes": scope_names(i.scope for i in k.items.values()),
            "entities": [entity(e) for e in k.entities.values()],
            "items": [item(i) for i in k.items.values()],
            "edges": [edge(e) for e in k.edges.values()],
        }

    def knowledge_search(self, q: str) -> dict[str, Any]:
        """Items whose statement or reason contains the words, case aside, or that are about
        an entity whose name does; and those entities. What the Knowledge page's search
        box highlights."""
        q = q.strip().lower()
        k = self.knowledge
        if not q:
            return {"q": q, "items": [], "entities": []}
        ents = {e.entity_id for e in k.entities.values() if q in e.name.lower()}
        items = [
            i.item_id
            for i in k.items.values()
            if q in f"{i.statement} {i.why}".lower() or ents.intersection(i.entities)
        ]
        return {"q": q, "items": items, "entities": sorted(ents)}

    def knowledge_detail(self, ref: str) -> dict[str, Any]:
        k = self.knowledge
        iid = next((i for i in k.items if ref and i.startswith(ref)), None)
        if iid is None:
            eid = next((e for e in k.entities if ref and e.startswith(ref)), None)
            if eid is None:
                raise KeyError(ref)
            e = k.entities[eid]
            return {
                "version": k.version,
                "entity": e.model_dump(mode="json"),
                "items": [i.model_dump(mode="json") for i in k.items.values() if eid in i.entities],
            }
        item = k.items[iid]
        due = item.review_due()
        return {
            "version": k.version,
            "item": item.model_dump(mode="json"),
            "vitality": round(item.vitality(), 3),
            "checked": item.checked.isoformat(),
            "review_due": due.isoformat() if due else None,
            "overdue": item.overdue(),
            "recallable": item.recallable,
            "entities": [
                k.entities[e].model_dump(mode="json") for e in item.entities if e in k.entities
            ],
            "neighbours": [
                {
                    "id": other,
                    "relation": e.relation.value,
                    "weight": e.weight,
                    "origin": e.origin,
                    "reason": e.reason,
                    "outgoing": e.source == iid,
                    "statement": k.items[other].statement,
                    "valid": k.items[other].valid,
                }
                for other, e in k.neighbours(iid)
                if other in k.items
            ],
            # how it was learned: each graduation that stated it (the exchanges are gone)
            "learned": [
                {
                    "session": s.session,
                    "trigger": s.trigger,
                    "exchanges": s.exchanges,
                    "from": s.start.isoformat(),
                    "to": s.end.isoformat(),
                    "at": s.created_at.isoformat(),
                }
                for s in (k.segments[sid] for sid in item.segments if sid in k.segments)
            ],
        }

    def knowledge_review(self) -> dict[str, Any]:
        """What a person should look at: disputed items, items past their review date,
        and the deletions that can still be undone."""
        now = clock.now()
        items = []
        for i in self.knowledge.valid_items():
            if i.disputes:
                why = f"disputed: {i.dispute_reason}" if i.dispute_reason else "disputed"
            elif i.overdue(now):
                why = f"not re-checked since {i.checked.date().isoformat()}"
            else:
                continue
            items.append(
                {
                    "id": i.item_id,
                    "type": i.type.value,
                    "statement": i.statement,
                    "why": why,
                    "recalled": i.recallable,
                }
            )
        return {
            "items": items,
            "deleted": [
                {
                    "batch": b["batch"],
                    "at": b["at"],
                    "reason": b["reason"],
                    "statements": [r["item"]["statement"] for r in b["items"]],
                }
                for b in reversed(self.knowledge.deleted())
            ],
        }
