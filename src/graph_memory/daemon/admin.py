"""Correcting knowledge by hand: retract, restore, confirm, delete and undelete items, set
their scope, or have the model propose every item's scope (rescope) for a person to
apply.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .. import clock
from ..files import atomic_write
from ..graduation import GraduationError
from ..knowledge import topic_of
from .scopes import scope_names
from .state import RESCOPE, DaemonState, Job, log

RESCOPE_JOB = "memory:rescope"  # the queue's name for that job (it is no session)
RESCOPE_FILE = "rescope.json"


class Admin(DaemonState):
    """`gm knowledge` and the Knowledge page's buttons."""

    def _item_ref(self, ref: str) -> str:
        """The item id an id or unambiguous prefix names."""
        ref = ref.strip()
        hits = [i for i in self.knowledge.heads if ref and i.startswith(ref)]
        if len(hits) != 1:
            raise KeyError(f"{'no' if not hits else 'ambiguous'} knowledge item {ref!r}")
        return hits[0]

    def knowledge_retract(self, ref: str, reason: str) -> dict[str, Any]:
        it = self.knowledge.retract(self._item_ref(ref), reason)
        self.knowledge_dirty = True
        self.events.emit("knowledge_retract", item=it.item_id, reason=it.invalid_reason)
        return {"id": it.item_id, "valid": it.valid, "statement": it.statement}

    def knowledge_restore(self, ref: str) -> dict[str, Any]:
        it = self.knowledge.restore(self._item_ref(ref))
        self.knowledge_dirty = True
        self.events.emit("knowledge_restore", item=it.item_id)
        return {"id": it.item_id, "valid": it.valid, "statement": it.statement}

    def knowledge_confirm(self, ref: str) -> dict[str, Any]:
        """A person checked the item and it still holds."""
        it = self.knowledge.confirm(self._item_ref(ref))
        self.knowledge_dirty = True
        self.events.emit("knowledge_confirm", item=it.item_id)
        return {"id": it.item_id, "valid": it.valid, "statement": it.statement}

    def knowledge_delete(self, refs: list[str], reason: str) -> dict[str, Any]:
        """Delete items for good (see KnowledgeStore.delete); undone by its batch id."""
        ids = list(dict.fromkeys(self._item_ref(r) for r in refs))
        if not ids:
            raise KeyError("no items to delete")
        batch = self.knowledge.delete(ids, reason)
        self.knowledge.save()  # the log of the deletion and the store agree on disk
        self.knowledge_dirty = False
        self.events.emit(
            "knowledge_delete",
            batch=batch["batch"],
            items=ids,
            bridges=len(batch["bridges"]),
            entities=len(batch["entities"]),
            reason=reason,
        )
        return {
            "batch": batch["batch"],
            "deleted": [r["item"]["statement"] for r in batch["items"]],
            "bridges": len(batch["bridges"]),
            "entities": [r["entity"]["name"] for r in batch["entities"]],
        }

    def knowledge_undelete(self, batch_id: str) -> dict[str, Any]:
        batch = self.knowledge.undelete(batch_id.strip())
        self.knowledge.save()
        self.knowledge_dirty = False
        self.events.emit("knowledge_undelete", batch=batch["batch"])
        return {"batch": batch["batch"], "restored": len(batch["items"])}

    def knowledge_scope(self, refs: list[str], to: str) -> dict[str, Any]:
        """Set items' scope by hand: "global", "topic:<name>", or a project by its name
        or key."""
        k = self.knowledge
        ids = list(dict.fromkeys(self._item_ref(r) for r in refs))
        if not ids:
            raise KeyError("no items to scope")
        scope = self._scope_ref(to)
        for i in ids:
            k.edit(i).scope = scope
        k.touch(*ids)
        k.save()
        self.knowledge_dirty = False
        self.events.emit("knowledge_scope", items=ids, scope=scope)
        return {"ids": ids, "scope": scope}

    def rescope(self) -> dict[str, Any]:
        """Queue the model's pass over every valid item's scope (see rescope.py); it
        writes a preview and changes nothing until `apply_rescope`."""
        if self.ask is None:
            raise GraduationError("no model to ask")
        self._grad_queued[RESCOPE_JOB] += 1
        self.grad_queue.put_nowait(Job(RESCOPE_JOB, RESCOPE))
        return {"queued": True, "items": len(self.knowledge.ids(valid=True))}

    async def _rescope(self, job: Job) -> None:
        from ..rescope import inputs, propose

        started = time.perf_counter()
        assert self.ask is not None
        try:
            changes = await asyncio.to_thread(propose, *inputs(self.knowledge), self.ask)
        except Exception as exc:
            log.warning("rescoping failed: %s", exc)
            self.events.emit(RESCOPE, ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
            return
        plan = {"at": clock.now().isoformat(), "changes": changes}
        atomic_write(self.cfg.home / RESCOPE_FILE, json.dumps(plan, indent=1))
        self.events.emit(
            RESCOPE,
            ok=True,
            preview=True,
            changes=len(changes),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def rescope_preview(self) -> dict[str, Any]:
        path = self.cfg.home / RESCOPE_FILE
        if not path.exists():
            raise KeyError("no rescope preview: run `gm knowledge rescope` first")
        plan: dict[str, Any] = json.loads(path.read_text())
        plan["scopes"] = scope_names(
            [c["from"] for c in plan["changes"]] + [c["to"] for c in plan["changes"]]
        )
        return plan

    def apply_rescope(self) -> dict[str, Any]:
        """Apply the preview to the items still as it saw them (one changed by hand or by
        a graduation since keeps its scope)."""
        plan = self.rescope_preview()
        k = self.knowledge
        done, kept = [], []
        for c in plan["changes"]:
            h = k.heads.get(c["id"])
            if h is None or h.scope != c["from"]:
                kept.append(c["id"])
                continue
            name = topic_of(c["to"])
            k.edit(c["id"]).scope = (k.topic_key(name) or c["to"]) if name else c["to"]
            k.touch(c["id"])
            done.append(c["id"])
        k.save()
        self.knowledge_dirty = False
        (self.cfg.home / RESCOPE_FILE).unlink(missing_ok=True)
        self.events.emit(RESCOPE, ok=True, preview=False, changed=len(done), skipped=len(kept))
        return {"changed": len(done), "skipped": len(kept)}
