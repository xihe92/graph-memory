"""A summary of what memory knows about one project or topic, for a person to read.

The model reads the valid items of one scope and writes a short brief in fixed sections.
Every point cites the items it came from, so a reader can check it, and a point citing
nothing is dropped. The brief is a view: it is cached in a file, never stored as
knowledge and never recalled into a session. Each cached brief remembers the items it
was written from, so the page can say how many changed since.

    items of one scope ──▶ model ──▶ {overview, sections: points citing items}  (cached)
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

from .active_days import ActiveDays
from .graduation import GraduationError, _json_obj
from .knowledge import Item, ItemType

log = logging.getLogger(__name__)

MAX_CHARS = 120_000  # of item text in one prompt; the least alive items past it are left out
POINT_CHARS = 400
OVERVIEW_CHARS = 700
SECTION_POINTS = 8
TRIES = 2

# section key -> the heading a person sees, in reading order
SECTIONS = {
    "standing": "Where it stands",
    "decisions": "Decisions and why",
    "how": "How it works",
    "preferences": "Preferences",
    "pitfalls": "Pitfalls",
    "open": "Open items",
}

_TYPE_ORDER = [
    ItemType.DECISION,
    ItemType.FACT,
    ItemType.LEARNED,
    ItemType.PROCEDURE,
    ItemType.PREFERENCE,
    ItemType.PITFALL,
    ItemType.OPEN,
]

SUMMARY_PROMPT = """You summarize one part of a long-term memory for the person it belongs
to. The memory holds notes learned in their sessions with an AI assistant. Below are all
the notes memory has about {label}.

Write a brief that lets them see quickly what memory knows about it:
- "overview": 2-4 sentences on what this is and where it stands now.
- "standing": the current state in 3-5 points: what is done, what is in progress.
- "decisions": the decisions that shape it, each with its reason.
- "how": how it works: the facts and procedures someone needs to work on it.
- "preferences": how the user wants things done here.
- "pitfalls": what went wrong before and what to avoid.
- "open": what is unfinished or undecided.

Rules:
- Say only what the notes say. Add nothing from your own knowledge.
- Every point ends up with "notes": the K ids of the notes it rests on. A point with no
  note behind it must not be written.
- This is a brief, not a list of the notes. Merge notes about the same thing into one
  point, and leave out history, numbers and detail that would not matter to someone
  picking this work up today. At most {points} points in a section, the most important
  first. Leave a section empty when no note fits it.
- When a later note shows an earlier one is done or settled, report only the current
  state.
- A note marked "disputed" may be wrong: say so in the point, or leave it out.
- Plain sentences, one per point, no markdown.

Notes (id, type, last confirmed, statement):
{items}

Reply with ONLY a JSON object, no prose, no code fence:
{{"overview": "<2-4 sentences>",
  "standing": [{{"text": "<one sentence>", "notes": ["K1", "K7"]}}],
  "decisions": [], "how": [], "preferences": [], "pitfalls": [], "open": []}}
"""

Ask = Callable[[str], str]


def stamp(item: Item) -> str:
    """What a brief saw of an item: it changes when the item's words or its standing do."""
    text = f"{item.statement}|{item.why}|{'disputed' if item.disputes else ''}"
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def select(
    items: Sequence[Item], days: ActiveDays | None = None, budget: int = MAX_CHARS
) -> tuple[list[Item], int]:
    """The items a brief reads, by type then by how alive they are (aged by `days`, the
    store's active days), and how many were left out because the prompt was full (the
    least alive go first)."""
    days = days or ActiveDays()
    kept: list[Item] = []
    used = 0
    for it in sorted(items, key=lambda i: -i.vitality(days)):
        size = len(it.statement) + len(it.why) + 40
        if kept and used + size > budget:
            continue
        kept.append(it)
        used += size
    order = {t: n for n, t in enumerate(_TYPE_ORDER)}
    kept.sort(key=lambda i: (order.get(i.type, len(order)), -i.vitality(days), i.created_at))
    return kept, len(items) - len(kept)


def build_prompt(label: str, items: Sequence[Item]) -> str:
    lines = []
    for n, i in enumerate(items):
        tags = [i.type.value, i.checked.date().isoformat()]
        if i.disputes:
            tags.append("disputed" + (f": {i.dispute_reason}" if i.dispute_reason else ""))
        why = f" (why: {i.why})" if i.why else ""
        lines.append(f"K{n + 1} [{', '.join(tags)}] {i.statement}{why}")
    return SUMMARY_PROMPT.format(label=label, points=SECTION_POINTS, items="\n".join(lines))


def parse(raw: dict[str, Any], items: Sequence[Item]) -> dict[str, Any]:
    """The brief as it is kept: an overview and the sections that have points, each point
    with the ids of the items it cites. A point that cites no item it was given is
    dropped; a brief with no point left is an error."""
    sections = []
    for key, title in SECTIONS.items():
        points = []
        for d in raw.get(key) or []:
            if not isinstance(d, dict):
                continue
            text = " ".join(str(d.get("text") or "").split())[:POINT_CHARS]
            cited: list[str] = []
            refs = d.get("notes")
            for ref in refs if isinstance(refs, list) else []:
                num = str(ref).strip().upper().removeprefix("K")
                if num.isdigit() and 1 <= int(num) <= len(items):
                    iid = items[int(num) - 1].item_id
                    if iid not in cited:
                        cited.append(iid)
            if text and cited:
                points.append({"text": text, "notes": cited})
        if points:
            sections.append({"key": key, "title": title, "points": points[:SECTION_POINTS]})
    if not sections:
        raise GraduationError("the summary has no point that cites a note")
    overview = " ".join(str(raw.get("overview") or "").split())[:OVERVIEW_CHARS]
    return {"overview": overview, "sections": sections}


def summarize(label: str, items: Sequence[Item], ask: Ask) -> dict[str, Any]:
    """Ask the model for the brief of `items` (already selected). Runs in a worker
    thread; `items` are read, never changed."""
    prompt = build_prompt(label, items)
    for attempt in range(TRIES):
        try:
            return parse(_json_obj(ask(prompt)), items)
        except GraduationError:
            if attempt == TRIES - 1:
                raise
            log.warning("summary answer was not usable; asking again")
    raise AssertionError("unreachable")


def changed_since(notes: dict[str, str], items: Sequence[Item]) -> int:
    """How many items differ from what a brief was written from: new in the scope, gone
    from it, or changed."""
    now = {i.item_id: stamp(i) for i in items}
    gone = sum(1 for k in notes if k not in now)
    return gone + sum(1 for k, v in now.items() if notes.get(k) != v)
