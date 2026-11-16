"""Graduation: turn a run of one session's buffered exchanges into knowledge in the
persistent area.

    segment ─▶ 1. extract (model call per chunk) ─▶ entities + items + a summary of the
                  session so far (it replaces the previous one, which the call reads)
            ─▶ 2. resolve entities without a model where possible:
                  exact name ▶ 3-gram Jaccard ≥ 0.9 (guarded) ▶ embedding candidates
            ─▶ 3. merge items (one model call for the whole segment): for each new item,
                  duplicate of / makes false / related to (typed, weighted), against
                  existing items about the same entities or similar in meaning
            ─▶ 4. apply: reinforce duplicates, invalidate what became false, add items,
                  entities and edges; co-occurrence edges need no call

The steps that call the model or the embedder take no store access (`extract_step`,
`merge_step`); the steps that read and write the store take no model (`plan_merge`,
`apply`). The daemon runs the first kind in threads and the second on its loop.
Rules adapted from Graphiti (Apache-2.0): its entity extraction exclusions, its
deterministic dedup, and its contradiction test ("both cannot be true at once").
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import buffer
from .knowledge import (
    CO_OCCUR_WEIGHT,
    ORIGIN_LABEL,
    TOPIC_PREFIX,
    WEIGHTS,
    Entity,
    EntityKind,
    Item,
    ItemType,
    KnowledgeStore,
    Origin,
    Relation,
    Segment,
    Source,
    Volatility,
    fuzzy_key,
    normalize_name,
    topic_of,
    topic_scope,
)
from .projects import project_title
from .vectors import Embedder

log = logging.getLogger(__name__)

Ask = Callable[[str], str]  # a model call: prompt in, answer out

PROMPT_CHARS = 1000
REPLY_HEAD = 1800
REPLY_TAIL = 700
CHUNK_CHARS = 40000
ENTITY_HINTS = 40
ENTITY_CANDIDATES = 5
ENTITY_MIN_SIM = 0.6  # Graphiti's candidate floor
ITEM_CANDIDATES = 8
ITEM_MIN_SIM = 0.5
MAX_EXISTING = 60
MAX_STATEMENT_WORDS = 60
SUMMARY_CHARS = 1200  # the session summary a graduation writes
EXTRACT_TRIES = 2  # model calls per chunk when an answer doesn't parse
EARLIER_CHARS = 3000  # of the previous summary, given to the next graduation
# an origin that read or saw the thing beats one that only said it
TRUST = {
    Origin.UNKNOWN: 0,
    Origin.ASSISTANT: 1,
    Origin.USER: 2,
    Origin.DOCS: 3,
    Origin.CODE: 3,
    Origin.OBSERVED: 3,
}


class GraduationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- input


@dataclass(frozen=True)
class Exchange:
    id: str
    when: datetime
    prompt: str
    reply: str
    files: tuple[str, ...] = ()
    web: tuple[str, ...] = ()  # pages read, searches run


@dataclass(frozen=True)
class SegmentInput:
    session: str
    exchanges: tuple[Exchange, ...]
    trigger: str
    project: str | None = None  # topic key "repo:/path" when the work was in a project
    project_name: str | None = None
    compact_summary: str = ""
    earlier: str = ""  # the session's summary from its previous graduation


def _clip_reply(text: str) -> str:
    text = text.strip()
    if len(text) <= REPLY_HEAD + REPLY_TAIL:
        return text
    return f"{text[:REPLY_HEAD]}\n[…]\n{text[-REPLY_TAIL:]}"


def segment_input(
    session: str,
    exchanges: Sequence[buffer.Exchange],
    trigger: str,
    compact_summary: str = "",
    earlier: str = "",
) -> SegmentInput:
    """What graduation reads: the session's buffered exchanges, clipped, the project most
    of them were in, and what the session's earlier graduations summarised."""
    xs = tuple(
        Exchange(
            id=x.id,
            when=x.ts,
            prompt=x.prompt[:PROMPT_CHARS],
            reply=_clip_reply(x.reply),
            files=tuple(a.path for a in x.files)[:15],
            web=tuple(x.web)[:10],
        )
        for x in exchanges
    )
    projects = Counter(x.project for x in exchanges if x.project)
    key = projects.most_common(1)[0][0] if projects else None
    return SegmentInput(
        session,
        xs,
        trigger,
        key,
        project_title(key) if key else None,
        compact_summary,
        earlier[-EARLIER_CHARS:],
    )


def parts(xs: Sequence[buffer.Exchange], budget: int | None = None) -> list[list[buffer.Exchange]]:
    """A long run of exchanges split, in order, into graduations of about one chunk each:
    replayed one after another, a later one can retire what an earlier one learned."""
    budget = budget or CHUNK_CHARS
    out: list[list[buffer.Exchange]] = [[]]
    size = 0
    for x in xs:
        n = min(len(x.prompt), PROMPT_CHARS) + min(len(x.reply), REPLY_HEAD + REPLY_TAIL) + 200
        if out[-1] and size + n > budget:
            out.append([])
            size = 0
        out[-1].append(x)
        size += n
    return [p for p in out if p]


def chunks(seg: SegmentInput, budget: int = CHUNK_CHARS) -> list[list[tuple[int, Exchange]]]:
    out: list[list[tuple[int, Exchange]]] = [[]]
    size = 0
    for i, x in enumerate(seg.exchanges):
        n = len(x.prompt) + len(x.reply) + 200
        if out[-1] and size + n > budget:
            out.append([])
            size = 0
        out[-1].append((i, x))
        size += n
    return [c for c in out if c]


# --------------------------------------------------------------------------- prompts

EXTRACT_PROMPT = """You distil a work session between a user and an AI assistant into
long-term memory. The session may be coding, debugging, research or learning: what the
user found out by asking counts as much as code that was written.

Below are exchanges (X1, X2, …){project_line}. Write down knowledge worth having in FUTURE
sessions with this user. An item is worth keeping when it is:
- durable: still true next week (not "the tests are running now")
- useful later: knowing it would change, shorten or keep consistent what the assistant
  does or answers next time
- specific: names the concrete thing (file, function, tool, setting, value, version)

Good items: decisions with their reason; facts about the environment, setup or codebase;
the user's preferences and corrections ("don't do X", "always Y"); pitfalls that cost
time and their fix; procedures that worked; open items: work left unfinished, questions
still open, options being weighed and the current leaning ("considering A over B for X,
leaning A because …"); and what the user learned or established by asking how a tool,
library, API or concept works (type "learned"), written as it applies to the user's own
work: "SQLite has no array column type, so a recipe's tags need a recipe_tag join
table", not a textbook paragraph.
When the session finishes, drops or settles something that was left open or still to do,
write a fact saying it is done or dropped ("X is now done"), even if the code or git
history shows it too: that is how memory learns the open item is closed.
Skip: narration of what was done step by step, greetings, test counts, explanations with
no bearing on the user's work or question, and items about whether something was checked
or verified: "origin" below records that; an answer that wasn't checked is still kept,
with origin "assistant".

When the session changed its mind, record only the final state, and say why in "why".
Write each statement so it stands alone without the conversation: name the thing
instead of "it", "this", "the bug" (max {max_words} words, present tense).

For each item, also say where it came from and how long it will hold:
- "origin": "user" (the user stated it; an answer to the user's question is not from
  the user), "docs" (the assistant read it in documentation during these exchanges),
  "code" (read in source code during these exchanges), "observed" (seen in tool
  output, a test run, the running system), "assistant" (the assistant explained it from
  its own knowledge without checking a source). Choose docs or code only when the
  exchange shows the source was read (a "read on the web" page, a file read, or a reply
  quoting what it read); when unsure, assistant.
- "ref": the URL or file it was read in, else "".
- "applies_to": the version or context it holds for ("SQLite 3", "Python 3.12"), else "".
- "volatility": "stable" (a concept, decision, preference or open item: these change
  when the user changes them, and later sessions say so), "versioned" (how a release of
  a tool or API behaves), "volatile" (latest versions, prices, the current status of
  something outside the user's control: likely to change within months).

Entities are the named things items are about: files (path as written), code symbols,
components of the system, tools/libraries/services, projects, people by role ("the
user"), the machine/environment. Every item names 1-4 entities.
NEVER extract as entities: pronouns, abstract concepts ("performance", "quality",
"design"), generic nouns ("code", "the project", "function", "bug", "test", "user
request"), actions or sentence fragments. An entity must be specific enough to tell
apart from other things of its kind. Reuse a known entity's exact name when it is the
same thing. Known entities: {known}

Last, "summary": 2-4 sentences on the whole session so far, not only these exchanges:
the problem, what was decided or learned, what is still open. It replaces the earlier
summary below, so carry its points forward.

Reply with ONLY a JSON object, no prose, no code fence:
{{"entities": [{{"name": "<name>", "kind": "<{kinds}>",
                "summary": "<what it is, max 15 words>"}}],
  "items": [{{"type": "<{types}>", "statement": "<the knowledge>",
             "why": "<reason or evidence, may be empty>",
             "entities": ["<entity name>", …], "scope": "<project|topic|global>",
             "topic": "", "evidence": ["X3", …], "origin": "<{origins}>", "ref": "",
             "applies_to": "", "volatility": "<{volatilities}>"}}],
  "summary": "<2-4 sentences>"}}
"scope" decides when the item is recalled again:
{scope_rules}
- "topic": it is about one specific technology, product, library, service or subject
  (PostgreSQL, Redis, Kubernetes, SQLite) and matters whenever that subject comes up, in
  any project, but not otherwise. Decisions, open questions and preferences in work on
  that subject count too. Put the subject's usual name in "topic" (one name, no
  version: "PostgreSQL", not "PostgreSQL 16" or "PostgreSQL indexes"), reusing a known topic when
  it is the same subject. Known topics: {topics}
- "global": only for the user themself, their machine, or how they like to work in
  general ("explain choices in plain terms"): it matters whatever they work on. Never
  for an item about a technology or about work on one: that is "topic" (or "project").
Most sessions yield 0-8 items; an empty list is a fine answer.
{summary_block}{earlier_block}
<exchanges>
{exchanges}
</exchanges>
"""

SCOPE_PROJECT = """- "project": it only matters for the project {project}: its own code, design,
  setup or work, including how it uses a technology ("this project keeps its cache in
  SQLite")."""
SCOPE_NO_PROJECT = """- "project": not available here, these exchanges are in no project: choose
  "topic" or "global"."""

RESUMMARIZE_PROMPT = """You write the summary a long-term memory keeps of one work session
between a user and an AI assistant: 2-4 sentences on the whole session, ending where it
stands now: the problem, what was decided or learned, what is still open. Below is what
the session taught, oldest first (current knowledge only), then the summaries written
along the way. Where they disagree, the knowledge is current. Reply with only the summary.

<learned>
{learned}
</learned>

<summaries>
{summaries}
</summaries>
"""


def build_resummarize_prompt(items: Sequence[Item], summaries: Sequence[str]) -> str:
    learned = "\n".join(f"- [{i.type.value}] {i.statement}" for i in items[-80:])
    return RESUMMARIZE_PROMPT.format(
        learned=learned or "(nothing)",
        summaries="\n\n".join(s[:1500] for s in summaries)[-6000:] or "(none)",
    )


SUMMARY_BLOCK = """
The assistant's own summary of the conversation up to here (from compacting its context):
<summary>
{summary}
</summary>
"""

MERGE_PROMPT = """You maintain a long-term knowledge graph for a coding assistant. New
items (N1, N2, …) were just learned. Existing items (K1, K2, …) are already in memory,
with the date they were learned.

For each new item decide:
- "duplicate_of": the K or N id of an item stating the same knowledge (minor wording
  differences are fine; more detail is fine), else null. New items come from different
  parts of one session and often repeat each other: mark a later N item as duplicate_of
  the earlier N item that says the same thing.
- "invalidates": existing K items this new item makes FALSE, with a short reason. Only
  when both cannot be true at the same time, e.g. a tool was replaced, a setting was
  changed, a decision was reversed, an open/still-to-do item was done or dropped, or a
  tool's behaviour changed in a new version. A K item marked "from the assistant" is
  made false by a new item read in docs or code, or observed, that says otherwise. For
  each K item marked DISPUTED, check whether a new item states what is true instead:
  if so, that new item invalidates it. An open item that says where an open question or
  choice stands now makes the earlier open item about the same question false: memory
  keeps the latest state. NEVER for added detail, partial progress, a new related fact,
  or something only proposed. Most items invalidate nothing. A K item marked NO LONGER
  TRUE is history: a new item saying the same thing is a duplicate_of it.
- "related": up to 4 K or N items that someone using this item should also know,
  with "relation" one of depends_on, causes, refines, alternative, same_area and
  "strength" strong, medium or weak. Skip loose associations.
{entity_block}
Reply with ONLY a JSON object, no prose, no code fence:
{{"items": [{{"id": "N1", "duplicate_of": null,
             "invalidates": [{{"id": "K2", "reason": "<why it is no longer true>"}}],
             "related": [{{"id": "K1", "relation": "refines",
                          "strength": "medium"}}]}}]{entity_field}}}

<new_items>
{new}
</new_items>

<existing_items>
{existing}
</existing_items>
"""

EARLIER_BLOCK = """
What this session covered before these exchanges (the summary its last graduation wrote;
what was learned from it is already in memory):
<earlier>
{earlier}
</earlier>
"""

ENTITY_BLOCK = """
Also, for each entity question (Q1, Q2, …): a newly named entity and existing entities
that might be the same thing. Answer "same_as" with the C id if it is the same real
thing (same file, same component under another name), else null.

<entity_questions>
{questions}
</entity_questions>
"""

ENTITY_FIELD = ',\n  "entities": [{"q": "Q1", "same_as": "C2"}]'


def _json_obj(text: str) -> dict[str, Any]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise GraduationError(f"no JSON object in model output: {text[:200]!r}")
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError as exc:
        raise GraduationError(f"bad JSON from model: {exc}") from exc
    if not isinstance(obj, dict):
        raise GraduationError("model output is not an object")
    return obj


def build_extract_prompt(
    seg: SegmentInput,
    chunk: Sequence[tuple[int, Exchange]],
    known: Sequence[str],
    earlier: str | None = None,
    topics: Sequence[str] = (),
) -> str:
    """`earlier`: the summary so far; the session's previous one for the first chunk, the
    one the previous chunk wrote for the others."""
    earlier = seg.earlier if earlier is None else earlier
    parts = []
    for i, x in chunk:
        files = f"\n[files: {', '.join(x.files)}]" if x.files else ""
        if x.web:
            files += f"\n[read on the web: {', '.join(x.web)}]"
        parts.append(
            f"X{i + 1} ({x.when.date().isoformat()})\nUSER: {x.prompt}\nASSISTANT: {x.reply}{files}"
        )
    return EXTRACT_PROMPT.format(
        project_line=f" in the project {seg.project_name}" if seg.project_name else "",
        max_words=40,
        known=", ".join(known) if known else "(none yet)",
        scope_rules=SCOPE_PROJECT.format(project=seg.project_name)
        if seg.project_name
        else SCOPE_NO_PROJECT,
        topics=", ".join(topics) if topics else "(none yet)",
        kinds="|".join(k.value for k in EntityKind),
        types="|".join(t.value for t in ItemType),
        origins="|".join(o.value for o in Origin if o.value),
        volatilities="|".join(v.value for v in Volatility),
        summary_block=SUMMARY_BLOCK.format(summary=seg.compact_summary[:6000])
        if seg.compact_summary
        else "",
        earlier_block=EARLIER_BLOCK.format(earlier=earlier) if earlier else "",
        exchanges="\n\n".join(parts),
    )


# --------------------------------------------------------------------------- step 1


@dataclass
class Extracted:
    entities: dict[str, dict[str, Any]]  # normalized name -> {name, kind, summary, vector}
    items: list[dict[str, Any]]  # {type, statement, why, entities, scope, evidence,
    #                               origin, ref, applies_to, volatility, vector}
    calls: int = 0
    summary: str = ""  # of the session so far, written by the last chunk's call


def _parse_extract(raw: dict[str, Any], seg: SegmentInput, ex: Extracted) -> None:
    for e in raw.get("entities") or []:
        if not isinstance(e, dict) or not str(e.get("name", "")).strip():
            continue
        name = str(e["name"]).strip()[:120]
        try:
            kind = EntityKind(str(e.get("kind", "other")).lower())
        except ValueError:
            kind = EntityKind.OTHER
        ex.entities.setdefault(
            fuzzy_key(name),
            {"name": name, "kind": kind, "summary": str(e.get("summary") or "")[:200]},
        )
    for it in raw.get("items") or []:
        if not isinstance(it, dict):
            continue
        statement = " ".join(str(it.get("statement", "")).split())
        if len(statement.split()) < 3:
            continue
        try:
            itype = ItemType(str(it.get("type", "")).lower())
        except ValueError:
            continue
        words = statement.split()
        if len(words) > MAX_STATEMENT_WORDS:
            statement = " ".join(words[:MAX_STATEMENT_WORDS])
        names = [str(n).strip() for n in it.get("entities") or [] if str(n).strip()][:4]
        for n in names:  # an item may name an entity it didn't list
            ex.entities.setdefault(
                fuzzy_key(n), {"name": n, "kind": EntityKind.OTHER, "summary": ""}
            )
        evidence = []
        for ref in it.get("evidence") or []:
            m = re.fullmatch(r"X?(\d+)", str(ref).strip(), re.I)
            if m and 1 <= int(m.group(1)) <= len(seg.exchanges):
                evidence.append(seg.exchanges[int(m.group(1)) - 1].id)
        said = str(it.get("scope", "")).lower().strip()
        topic = topic_scope(str(it.get("topic") or ""))
        if said == "project" and seg.project:
            scope = seg.project
        elif said in ("topic", "project") and topic:  # no project: the topic, if it named one
            scope = topic
        else:
            scope = "global"
        try:
            origin = Origin(str(it.get("origin") or "assistant").lower())
        except ValueError:
            origin = Origin.ASSISTANT
        try:
            volatility = Volatility(str(it.get("volatility") or "stable").lower())
        except ValueError:
            volatility = Volatility.STABLE
        ex.items.append(
            {
                "type": itype,
                "statement": statement,
                "why": " ".join(str(it.get("why") or "").split())[:400],
                "entities": [fuzzy_key(n) for n in names],
                "scope": scope,
                "evidence": list(dict.fromkeys(evidence)),
                "origin": origin,
                "ref": str(it.get("ref") or "").strip()[:300],
                "applies_to": " ".join(str(it.get("applies_to") or "").split())[:80],
                "volatility": volatility,
            }
        )
    summary = " ".join(str(raw.get("summary") or "").split())
    if summary:
        ex.summary = summary[:SUMMARY_CHARS]


def _merge_file_names(ex: Extracted) -> None:
    """`engine.py` and `src/pkg/engine.py` in one segment are one file, when exactly one
    longer path ends with the short name."""
    files = {k: e["name"] for k, e in ex.entities.items() if e["kind"] is EntityKind.FILE}
    remap: dict[str, str] = {}
    for k, name in files.items():
        longer = [k2 for k2, n2 in files.items() if k2 != k and n2.endswith("/" + name)]
        if len(longer) == 1:
            remap[k] = longer[0]
    for k in remap:
        ex.entities.pop(k, None)
    for it in ex.items:
        it["entities"] = list(dict.fromkeys(remap.get(n, n) for n in it["entities"]))


def extract_step(
    seg: SegmentInput,
    ask: Ask,
    embedder: Embedder | None,
    known: Sequence[str],
    topics: Sequence[str] = (),
) -> Extracted:
    """Model calls and embeddings only; touches no store. Runs in a worker thread.
    `topics`: the topic names memory already has, for the extractor to reuse."""
    ex = Extracted({}, [])
    for chunk in chunks(seg):
        prompt = build_extract_prompt(seg, chunk, known, ex.summary or seg.earlier, topics)
        for attempt in range(EXTRACT_TRIES):
            ex.calls += 1
            try:
                raw = _json_obj(ask(prompt))
                break
            except GraduationError:
                # an answer that doesn't parse (prose, a broken quote) usually parses the
                # second time; one bad chunk shouldn't throw away a long graduation
                if attempt == EXTRACT_TRIES - 1:
                    raise
                log.warning("extraction answer did not parse; asking again")
        _parse_extract(raw, seg, ex)
    _merge_file_names(ex)
    if embedder is not None:
        for e in ex.entities.values():
            text = f"{e['name']} ({e['kind'].value}): {e['summary']}" if e["summary"] else e["name"]
            e["vector"] = embedder(text)
        for it in ex.items:
            it["vector"] = embedder(f"{it['statement']} {it['why']}".strip())
    return ex


def known_entities(store: KnowledgeStore, seg: SegmentInput) -> list[str]:
    """Entity names to offer the extractor so it reuses them: those named in the segment's
    text, then the most mentioned. Cheap: a substring scan, no embedding."""
    text = " ".join(f"{x.prompt} {x.reply} {' '.join(x.files)}" for x in seg.exchanges).lower()
    named = [e for e in store.entities.values() if normalize_name(e.name) in text]
    rest = sorted((e for e in store.entities.values() if e not in named), key=lambda e: -e.mentions)
    return [e.name for e in [*named, *rest][:ENTITY_HINTS]]


def known_topics(store: KnowledgeStore) -> list[str]:
    """Topic names memory already has, for the extractor to reuse their spelling."""
    return sorted((s[len(TOPIC_PREFIX) :] for s in store.topics().values()), key=str.lower)


# --------------------------------------------------------------------------- step 2


@dataclass
class MergePlan:
    seg: SegmentInput
    ex: Extracted
    entity_ids: dict[str, str] = field(default_factory=dict)  # norm name -> existing id
    questions: list[tuple[str, list[str]]] = field(default_factory=list)  # (norm, [ids])
    new_items: list[dict[str, Any]] = field(default_factory=list)
    reinforce: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # (id, item)
    existing: list[str] = field(default_factory=list)  # K ids, in order
    prompt: str | None = None


def plan_merge(store: KnowledgeStore, seg: SegmentInput, ex: Extracted) -> MergePlan:
    """Store reads, no model. Resolve what can be resolved deterministically and build the
    one merge prompt for the rest (None when nothing needs the model)."""
    plan = MergePlan(seg, ex)
    # a rebuild reads history: what is no longer true is matched too, not revived
    history = seg.trigger == "rebuild"

    def candidate(i: str) -> bool:
        return i in store.items and (store.items[i].valid or history)

    for norm, e in ex.entities.items():
        hit = store.entity_by_name(e["name"])
        if hit is None and e["kind"] is not EntityKind.FILE:
            hit = store.fuzzy_entity(e["name"])
        if hit is not None:
            plan.entity_ids[norm] = hit.entity_id
            continue
        cands = [
            eid
            for eid, _ in store.similar(
                e.get("vector"), "entity", ENTITY_CANDIDATES, ENTITY_MIN_SIM
            )
        ]
        if cands:
            plan.questions.append((norm, cands))

    for it in ex.items:
        dup = store.find_statement(it["statement"], include_invalid=history)
        if dup is not None:
            plan.reinforce.append((dup.item_id, it))
        else:
            plan.new_items.append(it)

    # candidates: items about the same (already known) entities, and items close in meaning
    chosen: dict[str, float] = {}
    for it in plan.new_items:
        ids = {plan.entity_ids[n] for n in it["entities"] if n in plan.entity_ids}
        local: dict[str, float] = {}
        for i, s in store.similar(it.get("vector"), "item", ITEM_CANDIDATES * 2, ITEM_MIN_SIM):
            if candidate(i):
                local[i] = s
        for other in store.items_about(ids, include_invalid=history):
            local.setdefault(other.item_id, 0.4)
        for i, s in sorted(local.items(), key=lambda t: -t[1])[: ITEM_CANDIDATES + 4]:
            chosen[i] = max(chosen.get(i, 0.0), s)
    plan.existing = [i for i, _ in sorted(chosen.items(), key=lambda t: -t[1])[:MAX_EXISTING]]

    if not plan.new_items or (not plan.existing and len(plan.new_items) < 2 and not plan.questions):
        return plan
    plan.prompt = build_merge_prompt(store, plan)
    return plan


def build_merge_prompt(store: KnowledgeStore, plan: MergePlan) -> str:
    new = "\n".join(
        f"N{i + 1} [{it['type'].value}{_from(it['origin'])}] {it['statement']}"
        + (f" (why: {it['why']})" if it["why"] else "")
        for i, it in enumerate(plan.new_items)
    )

    def known(k: Item) -> str:
        line = f"[{k.type.value}, {k.created_at.date()}{_from(k.origin)}] {k.statement}"
        if not k.valid:
            since = k.invalid_at.date() if k.invalid_at else "?"
            line += f" (NO LONGER TRUE since {since}: {k.invalid_reason or 'replaced'})"
        elif k.disputes:
            line += f" (DISPUTED: {k.dispute_reason or 'a reply contradicted it'})"
        return line

    existing = (
        "\n".join(f"K{i + 1} {known(store.items[k])}" for i, k in enumerate(plan.existing))
        or "(none)"
    )
    entity_block = entity_field = ""
    if plan.questions:
        qs = []
        for qi, (norm, cands) in enumerate(plan.questions):
            e = plan.ex.entities[norm]
            opts = "; ".join(
                f"C{ci + 1}: {store.entity_text(store.entities[c])}" for ci, c in enumerate(cands)
            )
            qs.append(f"Q{qi + 1}: {e['name']} ({e['kind'].value}): {e['summary']} — {opts}")
        entity_block = ENTITY_BLOCK.format(questions="\n".join(qs))
        entity_field = ENTITY_FIELD
    return MERGE_PROMPT.format(
        new=new,
        existing=existing,
        entity_block=entity_block,
        entity_field=entity_field,
    )


def _from(origin: Origin) -> str:
    label = ORIGIN_LABEL.get(origin)
    return f", {label}" if label else ""


def merge_step(plan: MergePlan, ask: Ask) -> dict[str, Any] | None:
    """The merge model call; touches no store. None when there was nothing to ask, or
    when it failed (graduation then keeps the items, without dedup or typed edges)."""
    if plan.prompt is None:
        return None
    for _ in range(EXTRACT_TRIES):
        try:
            plan.ex.calls += 1
            return _json_obj(ask(plan.prompt))
        except Exception as exc:  # degrade, don't lose the extracted knowledge
            log.warning("merge call failed: %s", exc)
    return None


# --------------------------------------------------------------------------- step 3


def _ref(ref: object, prefix: str, n: int) -> int | None:
    m = re.fullmatch(rf"{prefix}(\d+)", str(ref).strip(), re.I)
    if m and 1 <= int(m.group(1)) <= n:
        return int(m.group(1)) - 1
    return None


def apply(
    store: KnowledgeStore, plan: MergePlan, merged: dict[str, Any] | None, started: float
) -> Segment:
    """Store writes, no model. Returns the segment record."""
    w = _Writes(store, plan)
    w.entities(merged)
    w.topic_spellings()
    w.reinforce()
    decisions = _decisions(merged, len(plan.new_items))
    created, dup_of_new = w.new_items(decisions)
    w.links(decisions, created, dup_of_new)
    w.co_occurrence()
    return w.finish(started)


def _decisions(merged: dict[str, Any] | None, new_items: int) -> dict[int, dict[str, Any]]:
    """The model's decisions per new item, by its index."""
    decisions: dict[int, dict[str, Any]] = {}
    for d in (merged or {}).get("items") or []:
        if isinstance(d, dict):
            n = _ref(d.get("id"), "N", new_items)
            if n is not None:
                decisions[n] = d
    return decisions


class _Writes:
    """One graduation's writes to the store, in the order `apply` makes them."""

    def __init__(self, store: KnowledgeStore, plan: MergePlan) -> None:
        self.store, self.plan, self.seg = store, plan, plan.seg
        seg = plan.seg
        self.segment = Segment(
            session=seg.session,
            exchanges=len(seg.exchanges),
            start=seg.exchanges[0].when,
            end=seg.exchanges[-1].when,
            trigger=seg.trigger,
        )
        self.when = {x.id: x.when for x in seg.exchanges}

    def entities(self, merged: dict[str, Any] | None) -> None:
        """Model answers to the open questions (is it an entity memory has?), then create
        what is still new."""
        store, plan = self.store, self.plan
        for ans in (merged or {}).get("entities") or []:
            if not isinstance(ans, dict):
                continue
            q = _ref(ans.get("q"), "Q", len(plan.questions))
            if q is None:
                continue
            norm, cands = plan.questions[q]
            c = _ref(ans.get("same_as"), "C", len(cands))
            if c is not None and cands[c] in store.entities:
                plan.entity_ids[norm] = cands[c]
        for norm, e in plan.ex.entities.items():
            if norm in plan.entity_ids:
                store.merge_entity(plan.entity_ids[norm], e["name"], e["summary"])
            else:
                ent = store.add_entity(
                    Entity(name=e["name"], kind=e["kind"], summary=e["summary"]), e.get("vector")
                )
                plan.entity_ids[norm] = ent.entity_id

    def topic_spellings(self) -> None:
        """One spelling per topic: memory's, else the first one here."""
        spelled = self.store.topics()
        for it in [*self.plan.new_items, *(x for _, x in self.plan.reinforce)]:
            name = topic_of(it["scope"])
            if name:
                it["scope"] = spelled.setdefault(name.lower(), it["scope"])

    def reinforce(self) -> None:
        """The items the plan matched to ones memory has."""
        for item_id, it in self.plan.reinforce:
            self.store.reinforce(
                item_id, self.segment.segment_id, self._source(it), boost=not self._again(item_id)
            )
            self._provenance(item_id, it)
            self.segment.items.append(item_id)

    def new_items(
        self, decisions: dict[int, dict[str, Any]]
    ) -> tuple[dict[int, str], dict[int, int]]:
        """Each new item: a duplicate of one memory has, or of one before it here, or
        added. Returns new item -> its id, and new item -> the earlier one it repeats."""
        store, plan, segment = self.store, self.plan, self.segment
        created: dict[int, str] = {}
        dup_of_new: dict[int, int] = {}
        for n, it in enumerate(plan.new_items):
            d = decisions.get(n, {})
            k = _ref(d.get("duplicate_of"), "K", len(plan.existing))
            if k is not None and plan.existing[k] in store.items:
                store.reinforce(
                    plan.existing[k],
                    segment.segment_id,
                    self._source(it),
                    boost=not self._again(plan.existing[k]),
                )
                self._provenance(plan.existing[k], it)
                created[n] = plan.existing[k]
                segment.items.append(plan.existing[k])
                continue
            other = _ref(d.get("duplicate_of"), "N", len(plan.new_items))
            if other is not None and other < n and other in created:
                dup_of_new[n] = other
                store.reinforce(created[other], segment.segment_id, self._source(it))
                created[n] = created[other]
                continue
            item = store.add_item(
                Item(
                    type=it["type"],
                    statement=it["statement"],
                    why=it["why"],
                    scope=it["scope"],
                    entities=self._entity_ids(it),
                    segments=[segment.segment_id],
                    sources=[self._source(it)],
                    valid_at=self._valid_at(it),
                    origin=it["origin"],
                    ref=it["ref"],
                    applies_to=it["applies_to"],
                    volatility=it["volatility"],
                ),
                it.get("vector"),
            )
            created[n] = item.item_id
            segment.items.append(item.item_id)
        return created, dup_of_new

    def links(
        self,
        decisions: dict[int, dict[str, Any]],
        created: dict[int, str],
        dup_of_new: dict[int, int],
    ) -> None:
        """What each item added here retires, and what it relates to."""
        for n, d in decisions.items():
            if n not in created or n in dup_of_new:
                continue
            me = created[n]
            if me in self.plan.existing:  # was a duplicate of an existing item
                continue
            self._invalidates(me, d)
            self._related(me, d, created)

    def _invalidates(self, me: str, d: dict[str, Any]) -> None:
        store, plan = self.store, self.plan
        for inv in d.get("invalidates") or []:
            if not isinstance(inv, dict):
                continue
            k = _ref(inv.get("id"), "K", len(plan.existing))
            if k is not None and plan.existing[k] in store.items:
                store.invalidate(
                    plan.existing[k], store.items[me], str(inv.get("reason") or "")[:300]
                )

    def _related(self, me: str, d: dict[str, Any], created: dict[int, str]) -> None:
        store, plan = self.store, self.plan
        for rel in (d.get("related") or [])[:4]:
            if not isinstance(rel, dict):
                continue
            try:
                relation = Relation(str(rel.get("relation", "same_area")))
            except ValueError:
                relation = Relation.SAME_AREA
            if relation in (Relation.CO_OCCURS, Relation.CO_USED, Relation.SUPERSEDES):
                relation = Relation.SAME_AREA
            weight = WEIGHTS.get(str(rel.get("strength", "medium")).lower(), WEIGHTS["medium"])
            k = _ref(rel.get("id"), "K", len(plan.existing))
            target = plan.existing[k] if k is not None else None
            if target is None:
                o = _ref(rel.get("id"), "N", len(plan.new_items))
                target = created.get(o) if o is not None else None
            if target is not None and target in store.items:
                store.add_edge(me, target, relation, weight)

    def co_occurrence(self) -> None:
        """Learned in the same segment and about a shared, specific entity: linked."""
        store = self.store
        ids = list(dict.fromkeys(self.segment.items))
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1 :]:
                ia, ib = store.items[a], store.items[b]
                if any(not store.hub(e) for e in set(ia.entities) & set(ib.entities)):
                    store.add_edge(a, b, Relation.CO_OCCURS, CO_OCCUR_WEIGHT, "cooccur")
        self.segment.items = ids

    def finish(self, started: float) -> Segment:
        """The segment record, with what it taught, stored."""
        store, segment = self.store, self.segment
        ids = segment.items
        segment.new = [i for i in ids if store.items[i].segments[:1] == [segment.segment_id]]
        segment.calls = self.plan.ex.calls
        segment.ms = round((time.perf_counter() - started) * 1000, 1)
        store.add_segment(segment)
        return segment

    # ------------------------------------------------------------------ per item

    def _entity_ids(self, it: dict[str, Any]) -> list[str]:
        plan = self.plan
        return list(
            dict.fromkeys(plan.entity_ids[n] for n in it["entities"] if n in plan.entity_ids)
        )

    def _valid_at(self, it: dict[str, Any]) -> datetime:
        times = [self.when[e] for e in it["evidence"] if e in self.when]
        return max(times) if times else self.segment.end

    def _source(self, it: dict[str, Any]) -> Source:
        return Source(session=self.seg.session, until=self._valid_at(it))

    def _again(self, item_id: str) -> bool:
        """A rebuild reads a session's turns a second time: what that session already
        taught is the same evidence, not a new statement of it."""
        return self.seg.trigger == "rebuild" and any(
            s.session == self.seg.session for s in self.store.items[item_id].sources
        )

    def _provenance(self, item_id: str, it: dict[str, Any]) -> None:
        """A restatement from a better source (docs over the assistant's word) upgrades
        where the item is said to come from."""
        old = self.store.items[item_id]
        if TRUST[it["origin"]] > TRUST[old.origin]:
            old.origin = it["origin"]
            old.ref = it["ref"] or old.ref
        old.applies_to = old.applies_to or it["applies_to"]
        if old.scope == "global" and topic_of(it["scope"]):
            old.scope = it["scope"]  # restated as about one subject: recall it only there
        self.store.touch(item_id)  # what it says about itself may have changed: save it


def graduate(
    store: KnowledgeStore,
    seg: SegmentInput,
    ask: Ask,
    embedder: Embedder | None,
) -> Segment:
    """All steps in one thread: for the CLI backfill and tests. The daemon interleaves
    them with its event loop instead."""
    started = time.perf_counter()
    ex = extract_step(seg, ask, embedder, known_entities(store, seg), known_topics(store))
    plan = plan_merge(store, seg, ex)
    return apply(store, plan, merge_step(plan, ask), started)
