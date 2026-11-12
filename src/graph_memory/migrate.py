"""`gm migrate`: from the conversation graph to knowledge-only memory, once.

Before: every exchange kept for good in a conversation graph (`nodes/`, `graph.json`,
`chroma/`), plus the knowledge graph. After: the knowledge graph only, and a buffer for
exchanges not yet graduated.

    1. exchanges that never graduated move into the session buffer; the daemon graduates
       them when it next starts, as sessions left over from before it started
    2. knowledge.json is rewritten in the new format: each item gets its sources (session,
       newest exchange that taught it) from its graduation segments; the links to
       exchanges and the per-session cursors go
    3. the old files, and knowledge.json as it was, are packed into
       archive-<date>.tar.gz, which is checked before anything is deleted
    4. the old files are deleted

Run it with the daemon stopped. `--dry-run` shows the plan and changes nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import clock
from .anchors import CodeAnchor, SourceRef
from .buffer import Exchange, SessionBuffer
from .knowledge import KNOWLEDGE_FILE, KnowledgeStore

OLD_FILES = ("nodes", "graph.json", "chroma")
CONTENT_MARKER = "<!-- gme:content -->"


@dataclass
class Plan:
    home: Path
    nodes: int = 0
    exchanges: int = 0
    old_bytes: int = 0
    pending: dict[str, list[Exchange]] = field(default_factory=dict)  # session -> to buffer
    items: int = 0
    items_without_sources: int = 0
    archive: Path | None = None

    @property
    def needed(self) -> bool:
        return any((self.home / f).exists() for f in OLD_FILES)

    def text(self) -> str:
        if not self.needed:
            return "Nothing to migrate: no conversation graph in " + str(self.home)
        waiting = sum(len(v) for v in self.pending.values())
        lines = [
            f"Conversation graph in {self.home}: {self.nodes} nodes ({self.exchanges} "
            f"exchanges), {self.old_bytes / 1e6:.1f} MB",
            f"  {waiting} exchanges never graduated, from {len(self.pending)} sessions: they "
            "move into the session buffer and graduate when the daemon next starts",
        ]
        for s, xs in sorted(self.pending.items(), key=lambda t: -len(t[1])):
            lines.append(f"    {len(xs):>4}  {s}")
        lines += [
            f"  knowledge: {self.items} items; {self.items_without_sources} get their sources "
            "from their graduation segments; links to exchanges and cursors are dropped",
            f"  archive: {self.archive} (the old files and knowledge.json as it was)",
            "  then deleted: " + ", ".join(f for f in OLD_FILES if (self.home / f).exists()),
        ]
        return "\n".join(lines)


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _read_node(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
        end = text.index("\n---\n", 4)
        meta: dict[str, Any] = json.loads(text[4:end])
        body = text[end + 5 :]
        meta["content"] = body[body.index(CONTENT_MARKER) + len(CONTENT_MARKER) + 1 :]
        return meta
    except (OSError, ValueError):
        return None


def _when(v: Any) -> datetime:
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def plan(home: Path) -> Plan:
    p = Plan(home, archive=home / f"archive-{clock.now().date().isoformat()}.tar.gz")
    if not p.needed:
        return p
    p.old_bytes = sum(_size(home / f) for f in OLD_FILES if (home / f).exists())
    nodes: dict[str, dict[str, Any]] = {}
    if (home / "nodes").is_dir():
        for f in (home / "nodes").glob("*.md"):
            meta = _read_node(f)
            if meta and meta.get("node_id"):
                nodes[str(meta["node_id"])] = meta
    p.nodes = len(nodes)

    raw: dict[str, Any] = {}
    if (home / KNOWLEDGE_FILE).exists():
        raw = json.loads((home / KNOWLEDGE_FILE).read_text(encoding="utf-8"))
    # how far each session graduated: the old cursors, and the newest segment end of each
    # session (a new-format daemon that saved knowledge.json first dropped the cursors)
    cursors = {k: _when(v) for k, v in (raw.get("cursors") or {}).items()}
    for seg in raw.get("segments") or []:
        session, end = seg.get("session"), seg.get("end")
        if session and end:
            cursors[session] = max(cursors.get(session, _when(end)), _when(end))
    p.items = len(raw.get("items") or [])
    p.items_without_sources = sum(1 for i in raw.get("items") or [] if not i.get("sources"))

    def topic_key(meta: dict[str, Any]) -> str | None:
        seen = set()
        cur: dict[str, Any] | None = meta
        while cur is not None and cur.get("node_id") not in seen:
            seen.add(cur.get("node_id"))
            if cur.get("role") == "topic":
                key = cur.get("topic_key")
                return str(key) if key and str(key).startswith("repo:") else None
            parent = cur.get("parent_id")
            cur = nodes.get(str(parent)) if parent else None
        return None

    for nid, meta in nodes.items():
        if meta.get("role") != "agent" or meta.get("agent_name"):
            continue  # exchanges are the main agent's replies
        p.exchanges += 1
        session = str(meta.get("session_id") or "")
        when = _when(meta["created_at"])
        if session in cursors and when <= cursors[session]:
            continue  # graduated already
        prompt = nodes.get(str(meta.get("parent_id") or ""), {})
        src = meta.get("source")
        p.pending.setdefault(session, []).append(
            Exchange(
                id=nid[:12],
                ts=when,
                prompt=str(prompt.get("content") or "").strip(),
                reply=str(meta.get("content") or "").strip(),
                files=[CodeAnchor.model_validate(a) for a in meta.get("code_context") or []],
                commands=[str(c) for c in meta.get("commands") or []],
                project=topic_key(meta),
                source=SourceRef.model_validate(src) if src else None,
            )
        )
    for xs in p.pending.values():
        xs.sort(key=lambda x: x.ts)
    return p


def apply(p: Plan) -> dict[str, Any]:
    """Carry out the plan. Order matters: nothing is deleted until the archive is checked."""
    home = p.home
    if not p.needed:
        return {"migrated": False}
    assert p.archive is not None
    # 1. exchanges that never graduated wait in the buffer
    buf = SessionBuffer(home)
    have = {s: {x.id for x in buf.exchanges(s)} for s in buf.sessions()}
    moved = 0
    for session, xs in p.pending.items():
        for x in xs:
            if x.id not in have.get(session, set()):
                buf.append(session, x)
                moved += 1
    # 2. archive the old files with knowledge.json as it was, and check the archive
    members = [f for f in OLD_FILES if (home / f).exists()]
    tmp = p.archive.with_name(p.archive.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for f in members:
            tar.add(home / f, arcname=f)
        if (home / KNOWLEDGE_FILE).exists():
            tar.add(home / KNOWLEDGE_FILE, arcname="knowledge.before-migration.json")
    with tarfile.open(tmp, "r:gz") as tar:
        names = set(tar.getnames())
    expected = {f for f in members} | (
        {"knowledge.before-migration.json"} if (home / KNOWLEDGE_FILE).exists() else set()
    )
    if not expected <= names:
        tmp.unlink()
        raise RuntimeError(f"archive check failed: missing {sorted(expected - names)}")
    os.replace(tmp, p.archive)
    # 3. knowledge in the new format (loading fills the sources from the segments)
    if (home / KNOWLEDGE_FILE).exists():
        KnowledgeStore(home).save()
    # 4. the old files go
    for f in members:
        target = home / f
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return {
        "migrated": True,
        "buffered": moved,
        "archive": str(p.archive),
        "archive_bytes": p.archive.stat().st_size,
        "removed": members,
    }
