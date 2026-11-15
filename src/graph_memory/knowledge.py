"""The persistent area: a global knowledge graph built from graduated session chains.

    entity  ◀─about─  item  ─related(weight, type)─▶  item
                        └─sources─▶ (session, newest exchange that taught it)

An *entity* is a named thing the work is about: a file, a code symbol, a component, a
tool, a project, the user, the environment. Entities are anchors; they carry no claims.

An *item* is one piece of knowledge worth remembering (a decision and why, a fact about
the environment, a preference or correction from the user, a pitfall, a procedure, an
open item, something the user learned). An item that a later one makes false gets
`invalid_at` (when it stopped being true) and `expired_at` (when the memory learned so),
following Graphiti's bi-temporal edges, and stays for history.

Knowledge goes stale. Each item records where it came from (`origin`: the user, docs,
source code, something observed, or the assistant's own explanation), what it applies to
(`applies_to`, e.g. "SQLite 3") and how likely it is to change (`volatility`). A versioned
or volatile item is due for review some time after it was last confirmed (`checked_at`);
a volatile one past that is no longer recalled. A note the live judge finds contradicted
is *disputed*: shown with a warning, and retracted when contradicted again.

A person can also delete items outright (noise, never worth keeping). Neighbours that
were linked only through them get a weaker *bridge* edge, so recall still finds one from
the other; each deletion is logged and can be undone.

Stored in `<home>/memory.db` (SQLite, with the vectors searched in place), or in the JSON
files from before it until `gm migrate` (see storage.py). The store keeps everything in
memory and records what changed, so a save writes only the rows that did.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from . import clock
from .storage import DB_FILE as DB_FILE  # the file names are imported from here elsewhere
from .storage import DELETED_FILE as DELETED_FILE
from .storage import KNOWLEDGE_FILE as KNOWLEDGE_FILE
from .storage import SUMMARY_FILE as SUMMARY_FILE
from .storage import VECTORS_FILE as VECTORS_FILE
from .storage import Backend, open_backend
from .vectors import Embedder


class ItemType(StrEnum):
    DECISION = "decision"  # chosen, with the reason
    FACT = "fact"  # how something is: environment, codebase, tool behaviour
    PREFERENCE = "preference"  # how the user wants things done, incl. corrections
    PITFALL = "pitfall"  # something that failed or misleads, and the fix
    PROCEDURE = "procedure"  # how to do something that worked
    OPEN = "open"  # an unfinished item, an open question, options still being weighed
    LEARNED = "learned"  # how a tool, library or concept works, as it applies to the user's work


class Origin(StrEnum):
    """Where an item's knowledge came from, which says how far to trust it."""

    UNKNOWN = ""  # learned before origins were recorded
    USER = "user"  # the user said it
    DOCS = "docs"  # read in documentation during the session
    CODE = "code"  # read in source code during the session
    OBSERVED = "observed"  # seen in tool output, a test run, the running system
    ASSISTANT = "assistant"  # the assistant's own explanation, not checked against a source


# how notes say where they came from
ORIGIN_LABEL = {
    Origin.USER: "from the user",
    Origin.DOCS: "from docs",
    Origin.CODE: "from source code",
    Origin.OBSERVED: "observed",
    Origin.ASSISTANT: "from the assistant",
}


class Volatility(StrEnum):
    STABLE = "stable"  # a concept, a decision, a preference: changes only when someone changes it
    VERSIONED = "versioned"  # behaviour of a release of a tool or API: changes with new versions
    VOLATILE = "volatile"  # "the latest version", prices, current status: changes within months


class EntityKind(StrEnum):
    FILE = "file"
    SYMBOL = "symbol"  # function, class, module, config key
    COMPONENT = "component"  # a named part of a system: "summariser", "web UI"
    TOOL = "tool"  # external program, library, service
    PROJECT = "project"
    PERSON = "person"  # the user, a teammate (by role)
    ENVIRONMENT = "environment"  # machine, OS, account setup
    OTHER = "other"


class Relation(StrEnum):
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"  # a pitfall and what it leads to, a decision and its consequence
    REFINES = "refines"  # adds detail to the other
    ALTERNATIVE = "alternative"  # a different option for the same thing
    SAME_AREA = "same_area"  # about the same part of the work
    CO_OCCURS = "co_occurs"  # learned together and share an entity (no model call)
    CO_USED = "co_used"  # recalled together and both used by the same reply (learned)
    SUPERSEDES = "supersedes"  # this item made the other no longer true


WEIGHTS = {"strong": 0.9, "medium": 0.6, "weak": 0.3}


class Area(StrEnum):
    """Where an item lives, after the JVM's generations: every item starts as a survivor;
    one seen again or used enough is promoted to old; one that goes long unused fades to
    dormant (kept, but no longer recalled) until it is seen or used again."""

    SURVIVOR = "survivor"
    OLD = "old"
    DORMANT = "dormant"


# a dormant item keeps fading at the fast rate; with the slow one its vitality would
# jump back above the floor the moment it went dormant
HALF_LIFE_DAYS = {Area.SURVIVOR: 14.0, Area.OLD: 90.0, Area.DORMANT: 14.0}
PROMOTE_AT = 2.0  # strength at which a survivor becomes old
DORMANT_BELOW = 0.25  # vitality under which an item stops being recalled
USE_BOOST = 0.5  # strength gained when a recalled item is used by the reply
SEEN_BOOST = 1.0  # strength gained when a later session states it again
CO_USE_STEP = 0.15
CO_OCCUR_WEIGHT = 0.25
# days after it was last confirmed that an item is due for review; stable items never are
REVIEW_DAYS = {Volatility.VERSIONED: 180.0, Volatility.VOLATILE: 30.0}
DISPUTE_LIMIT = 2  # contradictions that retract an item
JOURNAL_MAX = 20_000  # changes kept for the recall index; past it, it rebuilds
BRIDGE_MAX = 6  # bridge edges added per group of deleted items
BRIDGE_MIN = 0.05  # weaker bridges are not worth an edge
# a chain of one of these through a deleted item still holds without it
TRANSITIVE = frozenset({"depends_on", "causes", "refines"})
TOPIC_PREFIX = "topic:"
TOPIC_CHARS = 40


def topic_scope(name: str) -> str | None:
    """The scope key for a topic name ("SQLite" -> "topic:SQLite"), None for no name."""
    name = " ".join(str(name).replace(TOPIC_PREFIX, "", 1).split())[:TOPIC_CHARS].strip(" .:")
    return TOPIC_PREFIX + name if name else None


def topic_of(scope: str) -> str | None:
    """The topic a scope names ("topic:SQLite" -> "SQLite"), None for a project or global."""
    return scope[len(TOPIC_PREFIX) :] if scope.startswith(TOPIC_PREFIX) else None


def topic_forms(name: str) -> set[str]:
    """The ways a text may write a topic: "Claude Code", "claude-code", "claude_code"."""
    base = " ".join(name.lower().split())
    return {base, base.replace(" ", "-"), base.replace(" ", "_"), re.sub(r"[-_ ]", "", base)} | {
        re.sub(r"[-_]", " ", base)
    }


def named_topics(scopes: Iterable[str], text: str) -> set[str]:
    """The topic scopes among `scopes` whose topic `text` names, as a word."""
    low = text.lower()
    out = set()
    for scope in set(scopes):
        name = topic_of(scope)
        if name and any(
            re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", low)
            for f in topic_forms(name)
            if len(f) >= 2
        ):
            out.add(scope)
    return out


def _now() -> datetime:
    return clock.now()


def _id() -> str:
    return uuid.uuid4().hex[:16]


class Entity(BaseModel):
    entity_id: str = Field(default_factory=_id)
    name: str
    kind: EntityKind = EntityKind.OTHER
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    mentions: int = 1


class Source(BaseModel):
    """A session an item was learned from, and the time of the newest exchange in it that
    taught it. Recall skips an item while every source is the current session and newer
    than its last compaction: the session can still see what taught it."""

    session: str
    until: datetime


class Item(BaseModel):
    item_id: str = Field(default_factory=_id)
    type: ItemType
    statement: str
    why: str = ""
    # where it is recalled: "global" (anywhere), a project key such as "repo:/path" (in
    # that project), or a topic such as "topic:SQLite" (when the session is about it)
    scope: str = "global"
    entities: list[str] = Field(default_factory=list)  # entity ids
    segments: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    valid_at: datetime = Field(default_factory=_now)  # when it became true
    invalid_at: datetime | None = None  # when it stopped being true
    created_at: datetime = Field(default_factory=_now)  # when the memory learned it
    expired_at: datetime | None = None  # when the memory learned it is no longer true
    invalidated_by: str | None = None  # the item that replaced it; None when retracted by hand
    invalid_reason: str = ""
    strength: float = 1.0  # raised when seen again or used; see vitality()
    last_seen: datetime = Field(default_factory=_now)
    area: Area = Area.SURVIVOR
    uses: int = 0  # times recalled and then used by the reply
    recalls: int = 0  # times recalled
    last_used: datetime | None = None
    origin: Origin = Origin.UNKNOWN
    ref: str = ""  # the doc URL or file it came from, when there is one
    applies_to: str = ""  # the version or context it holds for, e.g. "SQLite 3"
    volatility: Volatility = Volatility.STABLE
    checked_at: datetime | None = None  # last stated again or confirmed; None: created_at
    disputes: int = 0  # times found contradicted since last confirmed
    disputed_at: datetime | None = None
    dispute_reason: str = ""

    @property
    def valid(self) -> bool:
        return self.invalid_at is None

    @property
    def checked(self) -> datetime:
        return self.checked_at or self.created_at

    def review_due(self) -> datetime | None:
        days = REVIEW_DAYS.get(self.volatility)
        return self.checked + timedelta(days=days) if days is not None else None

    def overdue(self, now: datetime | None = None) -> bool:
        due = self.review_due()
        return due is not None and (now or _now()) >= due

    def vitality(self, now: datetime | None = None) -> float:
        """Strength decayed by time since it was last seen or used, with the half-life of
        its area: 14 days for a survivor, 90 for an old item."""
        ref = max(self.last_seen, self.last_used or self.last_seen)
        age = ((now or _now()) - ref).total_seconds() / 86400
        return float(self.strength * 0.5 ** (max(age, 0.0) / HALF_LIFE_DAYS[self.area]))

    @property
    def recallable(self) -> bool:
        """Valid, not faded, and not a volatile item past its review: "the latest version
        is 8.2" a month on misleads more than it helps."""
        return (
            self.valid
            and self.area is not Area.DORMANT
            and not (self.volatility is Volatility.VOLATILE and self.overdue())
        )


class ItemEdge(BaseModel):
    source: str
    target: str
    relation: Relation
    weight: float
    origin: str = "llm"  # "llm", "cooccur", "co_use", "invalidation", "bridge"
    reason: str = ""
    created_at: datetime = Field(default_factory=_now)


class Segment(BaseModel):
    """A run of one session's exchanges that was graduated together. The exchanges
    themselves are gone once graduated; this records when, how many, and what came of
    them."""

    segment_id: str = Field(default_factory=_id)
    session: str
    exchanges: int = 0
    start: datetime
    end: datetime
    trigger: str  # session_end, compact, idle, abandoned, manual
    items: list[str] = Field(default_factory=list)  # items created or reinforced
    new: list[str] = Field(default_factory=list)  # of those, the ones it created
    created_at: datetime = Field(default_factory=_now)
    ms: float | None = None
    calls: int = 0

    @field_validator("exchanges", mode="before")
    @classmethod
    def _count(cls, v: Any) -> Any:
        return len(v) if isinstance(v, list) else v  # older files listed the exchange ids


def add_source(item: Item, session: str, until: datetime) -> None:
    """Record that `session` taught the item, up to `until`; one entry per session."""
    for src in item.sources:
        if src.session == session:
            src.until = max(src.until, until)
            return
    item.sources.append(Source(session=session, until=until))


def at_risk(item: Item) -> bool:
    """A note worth judging every time it is shown: it could have gone stale (versioned or
    volatile), or it is something learned from the assistant's own explanation, never
    checked against a source."""
    return item.volatility is not Volatility.STABLE or (
        item.type is ItemType.LEARNED and item.origin is Origin.ASSISTANT
    )


def in_view(item: Item, session: str, compacted: datetime | None) -> bool:
    """True when the session can still see everything that taught the item: every
    source is this session, after its last compaction. Compared by the exchange time,
    not by when the item was learned: a compaction triggers graduation, so items it
    yields are learned just after it, about exchanges it removed from view."""
    return bool(item.sources) and all(
        s.session == session and (compacted is None or s.until > compacted) for s in item.sources
    )


# --------------------------------------------------------------------------- names

_SPACE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    n = _SPACE.sub(" ", name.strip().lower().strip("`'\""))
    return n[4:] if n.startswith("the ") and len(n) > 4 else n


_NON_ALNUM = re.compile(r"[^a-z0-9' ]")


def _fuzzy_form(norm: str) -> str:
    """Graphiti's fuzzy normal form: punctuation becomes space, spaces collapse."""
    return _SPACE.sub(" ", _NON_ALNUM.sub(" ", norm)).strip()


def _shingles(norm: str, n: int = 3) -> set[str]:
    s = _fuzzy_form(norm).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _entropy(s: str) -> float:
    s = s.replace(" ", "")
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def fuzzy_safe(norm: str) -> bool:
    """Graphiti's guard: short or repetitive names match fuzzily by accident, so they go
    to exact matching or the model instead (under 6 characters with fewer than 2 words,
    or character entropy below 1.5)."""
    form = _fuzzy_form(norm)
    if len(form.replace(" ", "")) < 6 and len(form.split()) < 2:
        return False
    return _entropy(form) >= 1.5


FUZZY_JACCARD = 0.9


def fuzzy_key(name: str) -> str:
    """Names equal after dropping punctuation ("graph_memory daemon" and "graph-memory
    daemon") are the same entity, whatever their length."""
    return _fuzzy_form(normalize_name(name))


# --------------------------------------------------------------------------- store


class KnowledgeStore:
    def __init__(
        self, home: Path, embedder: Embedder | None = None, backend: Backend | None = None
    ) -> None:
        self.home = home
        self.embedder = embedder
        self.backend = backend or open_backend(home)
        self.vectors = self.backend.vectors
        self.entities: dict[str, Entity] = {}
        self.items: dict[str, Item] = {}
        self.edges: dict[tuple[str, str, str], ItemEdge] = {}
        self.segments: dict[str, Segment] = {}
        self._by_name: dict[str, str] = {}  # normalized name or alias -> entity id
        self._shingle_cache: dict[str, set[str]] = {}
        self.version = 0  # bumped on every change
        # What changed, for the recall index to update itself from (see touch()); a new
        # epoch means "everything", after a bulk change such as a deletion
        self.journal: list[tuple[str, ...]] = []
        self.epoch = 0
        self._due: set[str] = set()  # volatile items past their review, see sweep()
        # what changed since the last save, by kind; an edge by (source, target, relation)
        self._dirty: dict[str, set[Any]] = {k: set() for k in ("item", "entity", "edge", "segment")}
        self._full_save = False  # a bulk change (delete, undelete): the next save writes all
        self._vec_meta: set[str] = set()  # items whose scope or recallability to re-file
        self._load()
        self._due = self._due_now()

    # ------------------------------------------------------------------ change log

    def touch(self, *item_ids: str) -> None:
        """Record that these items changed in a way recall may see: validity, area,
        review, scope, text or entities. Every change goes through a method of the store
        that calls this; code outside the store that sets a field itself calls it too."""
        self.version += 1
        self.journal.extend(("item", i) for i in item_ids)
        self._dirty["item"].update(item_ids)
        self._vec_meta.update(item_ids)  # filed again before the next search or save
        self._due.difference_update(item_ids)  # the next sweep looks at them afresh
        self._trim_journal()

    def _touch_pair(self, a: str, b: str) -> None:
        self.version += 1
        self.journal.append(("pair", a, b))
        # every edge the two may share, either way: the save writes the ones that exist
        # and removes the rest
        for rel in Relation:
            self._dirty["edge"].update({(a, b, rel.value), (b, a, rel.value)})
        self._trim_journal()

    def _touch_entity(self, entity_id: str) -> None:
        self.version += 1
        self.journal.append(("entity", entity_id))
        self._dirty["entity"].add(entity_id)
        self._trim_journal()

    def changed_everything(self) -> None:
        """A bulk change: readers of the journal start over, and the next save writes
        everything."""
        self.version += 1
        self.epoch += 1
        self.journal = []
        self._full_save = True

    def _trim_journal(self) -> None:
        if len(self.journal) > JOURNAL_MAX:
            self.changed_everything()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        data = self.backend.load()
        for e in data.get("entities", []):
            ent = Entity.model_validate(e)
            self.entities[ent.entity_id] = ent
            self._index_names(ent)
        for i in data.get("items", []):
            it = Item.model_validate(i)
            # items reinforced before areas existed become old; a dormant item stays
            # dormant (only being stated again or used revives it)
            if it.area is Area.SURVIVOR and it.strength >= PROMOTE_AT:
                it.area = Area.OLD
                self._dirty["item"].add(it.item_id)
            self.items[it.item_id] = it
        for e in data.get("edges", []):
            edge = ItemEdge.model_validate(e)
            self.edges[(edge.source, edge.target, edge.relation.value)] = edge
        for s in data.get("segments", []):
            seg = Segment.model_validate(s)
            self.segments[seg.segment_id] = seg
        for it in self.items.values():
            if not it.sources:  # learned before items kept their sources
                for sid in it.segments:
                    if sid in self.segments:
                        add_source(it, self.segments[sid].session, self.segments[sid].end)
                        self._dirty["item"].add(it.item_id)

    def collections(self) -> dict[str, dict[Any, Any]]:
        """Everything the backend stores, by kind."""
        return {
            "entity": self.entities,
            "item": self.items,
            "edge": self.edges,
            "segment": self.segments,
        }

    def changed(self, kind: str) -> set[Any]:
        """The ids of one kind written or removed since the last save."""
        return self._dirty[kind]

    def vector_meta(self, kind: str, key: str) -> dict[str, str]:
        """What is kept beside a vector: its kind, and for an item its scope and whether
        it can be recalled now (a database searches by them)."""
        it = self.items.get(key) if kind == "item" else None
        if it is None:
            return {"kind": kind}
        return {"kind": kind, "scope": it.scope, "live": "1" if it.recallable else "0"}

    def _file_vectors(self) -> None:
        """Bring each changed item's vector row up to date with its scope and whether it
        can be recalled (done late: a method marks an item before changing it)."""
        if self._vec_meta:
            items = self.items
            self.vectors.set_meta(
                {i: (items[i].scope, items[i].recallable) for i in self._vec_meta if i in items}
            )
            self._vec_meta.clear()

    def save(self, full: bool = False) -> None:
        """Write what changed since the last save (everything with `full`), with the
        vectors, in one go."""
        full = full or self._full_save
        self._file_vectors()
        self.backend.save(self, full)
        for ids in self._dirty.values():
            ids.clear()
        self._full_save = False

    def add_segment(self, segment: Segment) -> Segment:
        self.version += 1
        self.segments[segment.segment_id] = segment
        self._dirty["segment"].add(segment.segment_id)
        return segment

    def summaries(self) -> dict[str, Any]:
        """The cached summaries of projects and topics (see summarize.py)."""
        return self.backend.summaries()

    def set_summaries(self, data: dict[str, Any]) -> None:
        """Replace the cached summaries; saved at once, with whatever else changed, so
        what is on disk is always one consistent state."""
        self.backend.set_summaries(data)
        self.save()

    def copy_to(self, dest: Path) -> None:
        """Save, then put a copy of this memory in `dest` (for a preview to change)."""
        self.save()
        self.backend.copy_to(dest)

    def close(self) -> None:
        self.backend.close()

    # ------------------------------------------------------------------ vectors

    def embed(self, text: str) -> list[float] | None:
        return list(self.embedder(text)) if self.embedder else None

    @staticmethod
    def entity_text(e: Entity) -> str:
        return f"{e.name} ({e.kind.value}): {e.summary}" if e.summary else e.name

    @staticmethod
    def item_text(i: Item) -> str:
        return f"{i.statement} {i.why}".strip()

    def _index(self, key: str, text: str, kind: str, vector: list[float] | None = None) -> None:
        v = vector if vector is not None else self.embed(text)
        if v is not None:
            self.vectors.upsert(key, v, self.vector_meta(kind, key))
            if kind == "item" and key not in self.items:
                self._vec_meta.add(key)  # filed once the item is in

    def similar(
        self,
        vector: list[float] | None,
        kind: str,
        k: int,
        min_score: float,
        among: set[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """The nearest of a kind, among the given ids when a set is given. `scopes` (with
        `among` the visible items of those scopes) lets a database filter in the scan."""
        if vector is None:
            return []
        self._file_vectors()
        return [
            (nid, s)
            for nid, s in self.vectors.query(
                vector, k, where={"kind": kind}, among=among, scopes=scopes
            )
            if s >= min_score
        ]

    # ------------------------------------------------------------------ entities

    def _index_names(self, e: Entity) -> None:
        for n in (e.name, *e.aliases):
            self._by_name.setdefault(normalize_name(n), e.entity_id)

    def add_entity(self, e: Entity, vector: list[float] | None = None) -> Entity:
        self.version += 1
        self.entities[e.entity_id] = e
        self._dirty["entity"].add(e.entity_id)
        self._index_names(e)
        self._index(e.entity_id, self.entity_text(e), "entity", vector)
        return e

    def merge_entity(self, target_id: str, name: str, summary: str = "") -> Entity:
        self.version += 1
        e = self.entities[target_id]
        e.mentions += 1
        self._dirty["entity"].add(target_id)
        if normalize_name(name) != normalize_name(e.name) and name not in e.aliases:
            e.aliases.append(name)
            self._index_names(e)
            self._touch_entity(target_id)
        if summary and not e.summary:
            e.summary = summary
            self._index(e.entity_id, self.entity_text(e), "entity")
        return e

    def entity_by_name(self, name: str) -> Entity | None:
        eid = self._by_name.get(normalize_name(name))
        if eid is None:
            key = fuzzy_key(name)
            eid = next((v for k, v in self._by_name.items() if _fuzzy_form(k) == key), None)
        if eid is None and "/" not in name.strip("/") and "." in name:
            # a bare file name matches the one known path that ends with it
            tail = "/" + name.strip()
            paths = [
                e
                for e in self.entities.values()
                if e.kind is EntityKind.FILE and e.name.endswith(tail)
            ]
            if len(paths) == 1:
                return paths[0]
        return self.entities.get(eid) if eid else None

    def hub(self, entity_id: str, limit: int = 8) -> bool:
        """Too general to say two items belong together: the project, a person, or an
        entity that many items are about."""
        e = self.entities.get(entity_id)
        if e is None or e.kind in (EntityKind.PROJECT, EntityKind.PERSON):
            return True
        return sum(1 for i in self.valid_items() if entity_id in i.entities) > limit

    def fuzzy_entity(self, name: str) -> Entity | None:
        """Graphiti's deterministic pass after exact match: Jaccard over character
        3-grams, at least 0.9. Exact Jaccard rather than MinHash: a personal memory
        has thousands of entities, not millions."""
        norm = normalize_name(name)
        if not fuzzy_safe(norm):
            return None
        mine = _shingles(norm)
        best: tuple[float, str] | None = None
        for key, eid in self._by_name.items():
            if not fuzzy_safe(key):
                continue
            sh = self._shingle_cache.get(key)
            if sh is None:
                sh = self._shingle_cache[key] = _shingles(key)
            j = _jaccard(mine, sh)
            if j >= FUZZY_JACCARD and (best is None or j > best[0]):
                best = (j, eid)
        return self.entities[best[1]] if best else None

    # ------------------------------------------------------------------ items

    def add_item(self, item: Item, vector: list[float] | None = None) -> Item:
        self.items[item.item_id] = item
        self._index(item.item_id, self.item_text(item), "item", vector)
        self.touch(item.item_id)
        return item

    def valid_items(self) -> Iterator[Item]:
        return (i for i in self.items.values() if i.valid)

    def topics(self) -> dict[str, str]:
        """The topic scopes memory has items in, by lower-case name: "sqlite" -> "topic:SQLite"."""
        out: dict[str, str] = {}
        for i in self.items.values():
            name = topic_of(i.scope)
            if name:
                out.setdefault(name.lower(), i.scope)
        return out

    def topic_key(self, name: str) -> str | None:
        """The scope key for a topic, reusing the spelling memory already has."""
        key = topic_scope(name)
        if key is None:
            return None
        return self.topics().get(key[len(TOPIC_PREFIX) :].lower(), key)

    def items_about(self, entity_ids: Iterable[str], include_invalid: bool = False) -> list[Item]:
        want = set(entity_ids)
        pool = self.items.values() if include_invalid else self.valid_items()
        return [i for i in pool if want.intersection(i.entities)]

    def find_statement(self, statement: str, include_invalid: bool = False) -> Item | None:
        norm = normalize_name(statement)
        pool = self.items.values() if include_invalid else self.valid_items()
        return next((i for i in pool if normalize_name(i.statement) == norm), None)

    def reinforce(
        self, item_id: str, segment_id: str, source: Source | None = None, *, boost: bool = True
    ) -> Item:
        """A later graduation stated the item again: stronger (unless `boost` is off: the
        same evidence read twice), and one more source."""
        self.touch(item_id)
        it = self.items[item_id]
        if source is not None:
            add_source(it, source.session, source.until)
        if boost:
            it.strength += SEEN_BOOST
        it.last_seen = _now()
        self._confirm(it)
        self._promote(it)
        if segment_id not in it.segments:
            it.segments.append(segment_id)
        return it

    @staticmethod
    def _confirm(it: Item) -> None:
        it.checked_at = _now()
        it.disputes = 0
        it.disputed_at = None
        it.dispute_reason = ""

    def confirm(self, item_id: str) -> Item:
        """A person (or a later session) confirms the item still holds: its review starts
        over and any dispute is cleared."""
        self.touch(item_id)
        it = self.items[item_id]
        self._confirm(it)
        return it

    def dispute(self, item_id: str, reason: str = "") -> Item:
        """The live judge found the note contradicted by what the session saw. The first
        time it is shown with a warning; at DISPUTE_LIMIT it is retracted."""
        self.touch(item_id)
        it = self.items[item_id]
        if not it.valid:
            return it
        it.disputes += 1
        it.disputed_at = _now()
        it.dispute_reason = reason[:300]
        if it.disputes >= DISPUTE_LIMIT:
            self.retract(item_id, f"contradicted {it.disputes} times; last: {reason}"[:300])
        return it

    # ------------------------------------------------------------------ lifecycle

    def _promote(self, it: Item) -> None:
        if it.area is Area.DORMANT:
            it.area = Area.OLD if it.strength >= PROMOTE_AT else Area.SURVIVOR
        elif it.area is Area.SURVIVOR and it.strength >= PROMOTE_AT:
            it.area = Area.OLD

    def mark_recalled(self, item_ids: Iterable[str]) -> None:
        self.version += 1  # nothing recall reads, but a page caching the item sees it
        for i in item_ids:
            self._dirty["item"].add(i)
            if i in self.items:
                self.items[i].recalls += 1

    def mark_used(self, item_ids: Sequence[str]) -> None:
        """Recalled items the reply used: stronger, and linked to each other (Hebbian:
        items used together get a co_used edge that grows each time)."""
        if not item_ids:
            return
        self.touch(*(i for i in item_ids if i in self.items))
        now = _now()
        for i in item_ids:
            it = self.items.get(i)
            if it is None:
                continue
            it.uses += 1
            it.strength += USE_BOOST
            it.last_used = now
            self._promote(it)
        ids = [i for i in item_ids if i in self.items]
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1 :]:
                key = next(
                    (k for k in [(a, b, "co_used"), (b, a, "co_used")] if k in self.edges), None
                )
                if key is None:
                    self.add_edge(a, b, Relation.CO_USED, CO_USE_STEP, "co_use")
                else:
                    edge = self.edges[key]
                    edge.weight = round(min(1.0, edge.weight + CO_USE_STEP), 3)
                    self._touch_pair(a, b)

    def sweep(self, now: datetime | None = None) -> list[str]:
        """Move items whose vitality fell under the floor to dormant. Returns their ids.
        Volatile items that fell due for review (or were confirmed) since the last sweep
        are logged too: they stop or start being recallable with no other change."""
        faded = []
        for it in self.valid_items():
            if it.area is not Area.DORMANT and it.vitality(now) < DORMANT_BELOW:
                it.area = Area.DORMANT
                faded.append(it.item_id)
        due = self._due_now(now)
        changed = [*faded, *(due ^ self._due)]
        self._due = due
        if changed:
            self.touch(*changed)
        return faded

    def _due_now(self, now: datetime | None = None) -> set[str]:
        return {
            i.item_id
            for i in self.items.values()
            if i.volatility is Volatility.VOLATILE and i.valid and i.overdue(now)
        }

    def invalidate(self, old_id: str, new: Item, reason: str = "") -> None:
        """Graphiti's temporal invalidation: the old item stopped being true when the new
        one became true. It stays in the graph for history."""
        self.touch(old_id)
        old = self.items[old_id]
        if not old.valid:
            return
        old.invalid_at = max(new.valid_at, old.valid_at)
        old.expired_at = _now()
        old.invalidated_by = new.item_id
        old.invalid_reason = reason
        self.add_edge(new.item_id, old_id, Relation.SUPERSEDES, 1.0, "invalidation", reason)

    def retract(self, item_id: str, reason: str = "") -> Item:
        """A person marks an item no longer true by hand: for when no later session states
        what replaced it, or the model missed the contradiction. Kept for history."""
        it = self.items[item_id]
        if it.valid:
            self.touch(item_id)
            it.invalid_at = it.expired_at = _now()
            it.invalidated_by = None
            it.invalid_reason = reason or "marked no longer true by hand"
        return it

    def restore(self, item_id: str) -> Item:
        """Undo an invalidation, by hand or by the model: the item is true again and the
        supersedes edges pointing at it are dropped."""
        it = self.items[item_id]
        if not it.valid:
            self.touch(item_id)
            it.invalid_at = it.expired_at = it.invalidated_by = None
            it.invalid_reason = ""
            for key in [k for k in self.edges if k[1] == item_id and k[2] == "supersedes"]:
                del self.edges[key]
                self._dirty["edge"].add(key)
        return it

    # ------------------------------------------------------------------ edges

    def add_edge(
        self,
        source: str,
        target: str,
        relation: Relation,
        weight: float,
        origin: str = "llm",
        reason: str = "",
    ) -> None:
        if source == target:
            return
        self._touch_pair(source, target)
        if relation is not Relation.SUPERSEDES:
            # undirected relations are stored once, in a canonical direction
            rev = self.edges.get((target, source, relation.value))
            if rev is not None:
                rev.weight = max(rev.weight, weight)
                return
            if relation is Relation.CO_OCCURS and self.related(source, target):
                return  # a typed edge already says more
            if relation not in (Relation.CO_OCCURS, Relation.CO_USED):
                for key in [(source, target, "co_occurs"), (target, source, "co_occurs")]:
                    self.edges.pop(key, None)
        key = (source, target, relation.value)
        cur = self.edges.get(key)
        if cur is not None:
            cur.weight = max(cur.weight, weight)
            return
        self.edges[key] = ItemEdge(
            source=source,
            target=target,
            relation=relation,
            weight=round(weight, 3),
            origin=origin,
            reason=reason,
        )

    def related(self, a: str, b: str) -> bool:
        return any(
            {e.source, e.target} == {a, b}
            and e.relation not in (Relation.CO_OCCURS, Relation.CO_USED)
            for e in self.edges.values()
        )

    def neighbours(self, item_id: str) -> list[tuple[str, ItemEdge]]:
        out = []
        for e in self.edges.values():
            if e.source == item_id:
                out.append((e.target, e))
            elif e.target == item_id:
                out.append((e.source, e))
        return out

    # ------------------------------------------------------------------ views

    def stats(self) -> dict[str, Any]:
        valid = [i for i in self.items.values() if i.valid]
        return {
            "entities": len(self.entities),
            "items": len(valid),
            "invalidated": len(self.items) - len(valid),
            "edges": len(self.edges),
            "segments": len(self.segments),
            "by_type": dict(Counter(i.type.value for i in valid)),
            "by_area": dict(Counter(i.area.value for i in valid)),
        }

    # ------------------------------------------------------------------ deleting

    def delete(self, item_ids: Iterable[str], reason: str = "") -> dict[str, Any]:
        """Remove items for good (a person's clean-up of noise). Their edges go with them;
        neighbours linked only through them get a bridge edge; entities no item names any
        more go too. The batch is logged in full, so `undelete` can put it back."""
        gone = {i for i in item_ids if i in self.items}
        if not gone:
            raise KeyError("no such items")
        bridges = self._bridges(gone)
        batch: dict[str, Any] = {
            "batch": _id(),
            "at": _now().isoformat(),
            "reason": reason,
            "items": [],
            "entities": [],
            "edges": [],
            "bridges": [],
            "unlinked": [],
        }
        for i in sorted(gone):
            batch["items"].append(
                {"item": self.items[i].model_dump(mode="json"), "vector": self.vectors.get(i)}
            )
        for key, e in list(self.edges.items()):
            if e.source in gone or e.target in gone:
                batch["edges"].append(e.model_dump(mode="json"))
                del self.edges[key]
        for i in gone:
            del self.items[i]
            self.vectors.remove(i)
        for it in self.items.values():
            if it.invalidated_by in gone:
                batch["unlinked"].append([it.item_id, it.invalidated_by])
                it.invalidated_by = None
        named = {e for it in self.items.values() for e in it.entities}
        for eid in [e for e in self.entities if e not in named]:
            ent = self.entities.pop(eid)
            batch["entities"].append(
                {"entity": ent.model_dump(mode="json"), "vector": self.vectors.get(eid)}
            )
            self.vectors.remove(eid)
        if batch["entities"]:
            self._reindex_names()
        for b in bridges:
            self.edges[(b.source, b.target, b.relation.value)] = b
            batch["bridges"].append([b.source, b.target, b.relation.value])
        self.changed_everything()
        self.backend.log_deletion(batch)
        return batch

    def deleted(self) -> list[dict[str, Any]]:
        """Logged deletions, oldest first."""
        return self.backend.deletions()

    def undelete(self, batch_id: str) -> dict[str, Any]:
        """Put a deleted batch back as it was: items, entities, edges; its bridges go."""
        batches = self.deleted()
        batch = next((b for b in batches if b["batch"] == batch_id), None)
        if batch is None:
            raise KeyError(f"no deletion {batch_id!r}")
        for rec in batch["entities"]:
            ent = Entity.model_validate(rec["entity"])
            if ent.entity_id not in self.entities:
                self.entities[ent.entity_id] = ent
                if rec.get("vector"):
                    self.vectors.upsert(ent.entity_id, rec["vector"], {"kind": "entity"})
        self._reindex_names()
        for rec in batch["items"]:
            it = Item.model_validate(rec["item"])
            self.items[it.item_id] = it
            if rec.get("vector"):
                self.vectors.upsert(it.item_id, rec["vector"], self.vector_meta("item", it.item_id))
            else:
                self._index(it.item_id, self.item_text(it), "item")
        for src, tgt, rel in batch["bridges"]:
            e = self.edges.get((src, tgt, rel))
            if e is not None and e.origin == "bridge":
                del self.edges[(src, tgt, rel)]
        for raw in batch["edges"]:
            e = ItemEdge.model_validate(raw)
            if e.source in self.items and e.target in self.items:
                self.edges[(e.source, e.target, e.relation.value)] = e
        for item_id, by in batch["unlinked"]:
            if item_id in self.items and self.items[item_id].invalidated_by is None:
                self.items[item_id].invalidated_by = by
        self.changed_everything()
        self.backend.drop_deletion(batch_id)
        return batch

    def _reindex_names(self) -> None:
        self._by_name = {}
        self._shingle_cache = {}
        for e in self.entities.values():
            self._index_names(e)

    def _bridges(self, gone: set[str]) -> list[ItemEdge]:
        """Edges that keep neighbours of the deleted items within reach of each other.

        Each group of deleted items linked to each other acts as one node: for each pair
        of valid items linked to the group, not already close (a direct edge, a shared
        specific entity or a common neighbour), a bridge as strong as the product of
        their strongest links into the group. The relation carries over when both links
        chain the same transitive relation (A depends on X, X depends on B), else it is
        same_area. At most BRIDGE_MAX per group, strongest first."""
        adj: dict[str, list[tuple[str, ItemEdge]]] = defaultdict(list)
        for e in self.edges.values():
            if e.relation is Relation.SUPERSEDES:
                continue
            adj[e.source].append((e.target, e))
            adj[e.target].append((e.source, e))
        out: list[ItemEdge] = []
        seen: set[str] = set()
        for start in sorted(gone):
            if start in seen:
                continue
            group, stack = {start}, [start]
            while stack:
                u = stack.pop()
                for v, _ in adj[u]:
                    if v in gone and v not in group:
                        group.add(v)
                        stack.append(v)
            seen |= group
            border: dict[str, ItemEdge] = {}
            for u in group:
                for v, e in adj[u]:
                    if v in gone or v not in self.items or not self.items[v].valid:
                        continue
                    if v not in border or e.weight > border[v].weight:
                        border[v] = e
            pairs = []
            names = sorted(border)
            for n, a in enumerate(names):
                for b in names[n + 1 :]:
                    w = border[a].weight * border[b].weight
                    if w >= BRIDGE_MIN and not self._close(a, b, gone, adj):
                        pairs.append((w, a, b))
            pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
            through = "; ".join(sorted(self.items[g].statement[:60] for g in group))[:200]
            for w, a, b in pairs[:BRIDGE_MAX]:
                relation, src, dst = _compose(a, border[a], b, border[b])
                out.append(
                    ItemEdge(
                        source=src,
                        target=dst,
                        relation=relation,
                        weight=round(w, 3),
                        origin="bridge",
                        reason=f"linked through deleted: {through}",
                    )
                )
        return out

    def _close(
        self, a: str, b: str, gone: set[str], adj: dict[str, list[tuple[str, ItemEdge]]]
    ) -> bool:
        na = {v for v, _ in adj[a] if v not in gone}
        if b in na:
            return True
        if na & {v for v, _ in adj[b] if v not in gone}:
            return True
        shared = set(self.items[a].entities) & set(self.items[b].entities)
        return any(not self.hub(e) for e in shared)


def _compose(a: str, ea: ItemEdge, b: str, eb: ItemEdge) -> tuple[Relation, str, str]:
    """The relation a bridge between a and b keeps, and its direction."""
    if ea.relation == eb.relation and ea.relation.value in TRANSITIVE:
        if ea.source == a and eb.target == b:  # a → group → b
            return ea.relation, a, b
        if eb.source == b and ea.target == a:  # b → group → a
            return ea.relation, b, a
    return Relation.SAME_AREA, a, b
