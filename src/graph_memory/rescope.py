"""Scoping knowledge again: the model reads items and says where each should be recalled.

An item is recalled anywhere ("global"), in one project ("repo:/path"), or when a session
is about one subject ("topic:SQLite"). Graduation picks the scope as it learns an item;
this pass fixes items learned before topics existed, or scoped wrongly. It proposes; a
person applies (`gm knowledge rescope`, then `--apply`).

    items ──batches──▶ model ──▶ {item: scope}  (preview)  ──apply──▶ store
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from .graduation import GraduationError, _json_obj
from .knowledge import TOPIC_PREFIX, Item, KnowledgeStore, topic_of, topic_scope
from .projects import project_title

log = logging.getLogger(__name__)

BATCH = 40  # items per model call
TRIES = 2

RESCOPE_PROMPT = """You decide where each item of a long-term memory is recalled. The memory
belongs to one user who works with an AI assistant; each item was learned in a session.

Scopes:
- "project": it only matters for one project's own code, design, setup or work,
  including how that project uses a technology. Name the project by its P id.
- "topic": it is about one specific technology, product, library, service or subject
  (PostgreSQL, Redis, Kubernetes, SQLite) and matters whenever that subject comes up, in
  any project, but not otherwise. Decisions, open questions and preferences in work on
  that subject count too. Give the subject's usual name in "topic" (one name, no
  version: "PostgreSQL", not "PostgreSQL 16"), reusing a known topic when it is the same subject.
- "global": only for the user themself, their machine, or how they like to work in
  general ("explain choices in plain terms"): it matters whatever they work on. Never
  for an item about a technology or about work on one: that is "topic" (or "project").

Keep an item's current scope unless another one is clearly better. A project of its
own (P id) beats a topic when the item is about that project's own work, even when it
names a technology.

Projects:
{projects}
Known topics: {topics}

Items (id, type, current scope, statement):
{items}

Reply with ONLY a JSON object, no prose, no code fence, one entry per item:
{{"items": [{{"id": "K1", "scope": "<project|topic|global>", "project": "<P id or empty>",
             "topic": "<name or empty>"}}]}}
"""

Ask = Callable[[str], str]


def _scope_label(scope: str, projects: dict[str, str]) -> str:
    if scope == "global":
        return "global"
    name = topic_of(scope)
    if name:
        return f"topic {name}"
    return f"project {projects.get(scope, project_title(scope))}"


def build_prompt(items: Sequence[Item], projects: Sequence[str], topics: Sequence[str]) -> str:
    pids = {p: f"P{n + 1}" for n, p in enumerate(projects)}
    lines = []
    for n, i in enumerate(items):
        why = f" (why: {i.why})" if i.why else ""
        lines.append(
            f"K{n + 1} [{i.type.value}] [{_scope_label(i.scope, pids)}] {i.statement}{why}"
        )
    return RESCOPE_PROMPT.format(
        projects="\n".join(f"{pids[p]} {project_title(p)} ({p})" for p in projects) or "(none)",
        topics=", ".join(topics) if topics else "(none yet)",
        items="\n".join(lines),
    )


def parse(raw: dict[str, Any], items: Sequence[Item], projects: Sequence[str]) -> dict[str, str]:
    """{item id: scope} for the items the answer gives a usable scope for."""
    out: dict[str, str] = {}
    for d in raw.get("items") or []:
        if not isinstance(d, dict):
            continue
        ref = str(d.get("id") or "").strip().upper().removeprefix("K")
        if not ref.isdigit() or not 1 <= int(ref) <= len(items):
            continue
        item = items[int(ref) - 1]
        said = str(d.get("scope") or "").strip().lower()
        if said == "global":
            out[item.item_id] = "global"
        elif said == "topic":
            key = topic_scope(str(d.get("topic") or ""))
            if key:
                out[item.item_id] = key
        elif said == "project":
            p = str(d.get("project") or "").strip().upper().removeprefix("P")
            if p.isdigit() and 1 <= int(p) <= len(projects):
                out[item.item_id] = projects[int(p) - 1]
    return out


def propose(store: KnowledgeStore, ask: Ask, batch: int = BATCH) -> list[dict[str, Any]]:
    """The changes the model proposes for the valid items, in batches; topics found in
    one batch are offered to the next so the same subject keeps one name. Runs in a
    worker thread; reads the store, changes nothing."""
    items = sorted(store.valid_items(), key=lambda i: i.created_at)
    projects = sorted({i.scope for i in store.items.values() if i.scope.startswith("repo:")})
    topics = {k: v[len(TOPIC_PREFIX) :] for k, v in store.topics().items()}  # lower -> name
    changes = []
    for start in range(0, len(items), batch):
        part = items[start : start + batch]
        prompt = build_prompt(part, projects, sorted(topics.values(), key=str.lower))
        for attempt in range(TRIES):
            try:
                got = parse(_json_obj(ask(prompt)), part, projects)
                break
            except GraduationError:
                if attempt == TRIES - 1:
                    raise
                log.warning("rescope answer did not parse; asking again")
        for item in part:
            to = got.get(item.item_id)
            name = topic_of(to) if to else None
            if name:  # one spelling per topic
                name = topics.setdefault(name.lower(), name)
                to = TOPIC_PREFIX + name
            if to and to != item.scope:
                changes.append(
                    {"id": item.item_id, "statement": item.statement, "from": item.scope, "to": to}
                )
    return changes
