"""What the web page and `gm knowledge` read: the Sessions page's chains, every item,
search, one item's detail, and what needs a person's review. Read only.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
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
        graduated_from = k.graduated_sessions()
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
            graduated = sorted(k.segments(session=session), key=lambda g: g.start)
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
                    "items": self._item_briefs(g.items, g.segment_id),
                }
                for g in graduated
                if g.segment_id not in summarized
            ]
            for r in recs:
                rec = r.model_dump(mode="json")
                if isinstance(r, Summary):  # what it learned, readable on the page
                    rec["items"] = self._item_briefs(r.items, r.segment or "")
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
        recent = sorted(k.segments_since(cutoff), key=lambda s: s.created_at, reverse=True)[:30]
        taught = {i.item_id: i for i in k.get_items(i for s in recent for i in s.items)}
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
                            "type": taught[i].type.value,
                            "statement": taught[i].statement,
                            "new": taught[i].segments[0] == s.segment_id,
                        }
                        for i in s.items
                        if i in taught
                    ],
                }
                for s in recent
            ],
        }

    def _item_briefs(self, item_ids: list[str], segment_id: str) -> list[dict[str, Any]]:
        """Items as the Sessions page lists them under a graduation or summary."""
        got = {i.item_id: i for i in self.knowledge.get_items(item_ids)}
        return [
            {
                "id": i,
                "type": got[i].type.value,
                "statement": got[i].statement,
                "new": bool(got[i].segments) and got[i].segments[0] == segment_id,
                "valid": got[i].valid,
            }
            for i in item_ids
            if i in got
        ]

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

        if lean:  # read in SQL, field by field: the page loads every note
            return {
                "version": k.version,
                "stats": k.stats(),
                "scopes": scope_names(h.scope for h in k.heads.values()),
                "entities": [
                    {"id": i, "name": n.name, "kind": n.kind.value, "mentions": n.mentions}
                    for i, n in k.names.items()
                ],
                "items": self._page_items(now),
                "edges": [
                    {"source": a, "target": b, "relation": r, "weight": w, "reason": why}
                    for a, b, r, w, why in k.fields("edge", ("weight", "reason"))
                ],
            }
        return {
            "version": k.version,
            "stats": k.stats(),
            "scopes": scope_names(h.scope for h in k.heads.values()),
            "entities": [entity(e) for e in k.scan_entities()],
            "items": [item(i) for i in k.scan()],
            "edges": [edge(e) for e in k.scan_edges()],
        }

    def _page_items(self, now: datetime) -> list[dict[str, Any]]:
        """Every item as the Knowledge page lists it (knowledge_view's lean fields)."""
        heads = self.knowledge.heads
        out = []
        names = [
            *("statement", "entities", "strength", "area", "uses", "created_at"),
            *("invalid_at", "invalidated_by", "invalid_reason", "disputes"),
        ]
        for row in self.knowledge.fields("item", names):
            i, stmt, ents, strength, area, uses, created, invalid, by, why, disputes = row
            h = heads[i]
            out.append(
                {
                    "id": i,
                    "type": h.type.value,
                    "statement": stmt,
                    "scope": h.scope,
                    "entities": json.loads(ents),
                    "valid": h.valid,
                    "strength": float(strength),
                    "area": area,
                    "uses": uses,
                    "created_at": datetime.fromisoformat(created).isoformat(),
                    "invalid_at": datetime.fromisoformat(invalid).isoformat() if invalid else None,
                    "invalidated_by": by,
                    "invalid_reason": why,
                    "disputes": disputes,
                    "overdue": h.overdue(now),
                }
            )
        return out

    def knowledge_search(self, q: str) -> dict[str, Any]:
        """Items whose statement or reason contains the words, case aside, or that are about
        an entity whose name does; and those entities. What the Knowledge page's search
        box highlights."""
        q = q.strip().lower()
        k = self.knowledge
        if not q:
            return {"q": q, "items": [], "entities": []}
        ents = {i for i, e in k.names.items() if q in e.name.lower()}
        items = [
            i
            for i, stmt, why, about in k.texts()
            if q in f"{stmt} {why}".lower() or ents.intersection(about)
        ]
        return {"q": q, "items": items, "entities": sorted(ents)}

    def knowledge_detail(self, ref: str) -> dict[str, Any]:
        k = self.knowledge
        iid = next((i for i in k.heads if ref and i.startswith(ref)), None)
        if iid is None:
            eid = next((e for e in k.names if ref and e.startswith(ref)), None)
            e = k.entity(eid) if eid is not None else None
            if eid is None or e is None:
                raise KeyError(ref)
            return {
                "version": k.version,
                "entity": e.model_dump(mode="json"),
                "items": [
                    i.model_dump(mode="json")
                    for i in k.get_items(k.about([eid], include_invalid=True))
                ],
            }
        item = k.item(iid)
        if item is None:
            raise KeyError(ref)
        near = k.neighbours(iid)
        others = {i.item_id: i for i in k.get_items(o for o, _ in near)}
        due = item.review_due()
        return {
            "version": k.version,
            "item": item.model_dump(mode="json"),
            "vitality": round(item.vitality(), 3),
            "checked": item.checked.isoformat(),
            "review_due": due.isoformat() if due else None,
            "overdue": item.overdue(),
            "recallable": item.recallable,
            "entities": [e.model_dump(mode="json") for e in k.get_entities(item.entities)],
            "neighbours": [
                {
                    "id": other,
                    "relation": e.relation.value,
                    "weight": e.weight,
                    "origin": e.origin,
                    "reason": e.reason,
                    "outgoing": e.source == iid,
                    "statement": others[other].statement,
                    "valid": others[other].valid,
                }
                for other, e in near
                if other in others
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
                for s in k.segments(ids=item.segments)
            ],
        }

    def knowledge_review(self) -> dict[str, Any]:
        """What a person should look at: disputed items, items past their review date,
        and the deletions that can still be undone."""
        now = clock.now()
        k = self.knowledge
        heads = k.heads
        look = set(k.disputed()) | {i for i, h in heads.items() if h.valid and h.overdue(now)}
        items = []
        for i in k.get_items(k.in_order(look)):
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
