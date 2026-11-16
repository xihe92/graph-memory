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

Stored in `<home>/memory.db` (SQLite, with the vectors and words searched in place; a
memory in the JSON files from before it is moved there by `gm migrate`). The store keeps
only small fields in memory, reads the rest by id, and keeps what changed until a save
writes those rows (see KnowledgeStore).
"""

from __future__ import annotations

import json
import math
import re
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import clock
from .storage import DB_FILE as DB_FILE  # the file names are imported from here elsewhere
from .storage import DELETED_FILE as DELETED_FILE
from .storage import KNOWLEDGE_FILE as KNOWLEDGE_FILE
from .storage import SUMMARY_FILE as SUMMARY_FILE
from .storage import VECTORS_FILE as VECTORS_FILE
from .storage import SqliteBackend, open_backend
from .vectors import Embedder
from .words import tokens


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


class ReadEntity(Entity):
    """An entity as read from memory.db: frozen, like ReadItem."""

    model_config = ConfigDict(frozen=True)


@dataclass(slots=True)
class EntityName:
    """What the store keeps in memory for each entity: what finds it by name."""

    name: str
    kind: EntityKind
    aliases: tuple[str, ...]
    mentions: int


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
        return _review_due(self.volatility, self.checked)

    def overdue(self, now: datetime | None = None) -> bool:
        return _overdue(self.volatility, self.checked, now)

    def vitality(self, now: datetime | None = None) -> float:
        """Strength decayed by time since it was last seen or used, with the half-life of
        its area: 14 days for a survivor, 90 for an old item."""
        return _vitality(self.strength, self.area, self.last_seen, self.last_used, now)

    @property
    def recallable(self) -> bool:
        """Valid, not faded, and not a volatile item past its review: "the latest version
        is 8.2" a month on misleads more than it helps."""
        return _recallable(self.valid, self.area, self.volatility, self.checked)


class ReadItem(Item):
    """An item as read from memory.db: frozen, so a change made to it by mistake fails
    loudly instead of being lost. `KnowledgeStore.edit()` gives one to change."""

    model_config = ConfigDict(frozen=True)


def _review_due(volatility: Volatility, checked: datetime) -> datetime | None:
    days = REVIEW_DAYS.get(volatility)
    return checked + timedelta(days=days) if days is not None else None


def _overdue(volatility: Volatility, checked: datetime, now: datetime | None = None) -> bool:
    due = _review_due(volatility, checked)
    return due is not None and (now or _now()) >= due


def _vitality(
    strength: float,
    area: Area,
    last_seen: datetime,
    last_used: datetime | None,
    now: datetime | None = None,
) -> float:
    ref = max(last_seen, last_used or last_seen)
    age = ((now or _now()) - ref).total_seconds() / 86400
    return float(strength * 0.5 ** (max(age, 0.0) / HALF_LIFE_DAYS[area]))


def _recallable(valid: bool, area: Area, volatility: Volatility, checked: datetime) -> bool:
    return (
        valid
        and area is not Area.DORMANT
        and not (volatility is Volatility.VOLATILE and _overdue(volatility, checked))
    )


@dataclass(slots=True)
class Head:
    """What the store keeps in memory for each item: the fields recall, aging and upkeep
    read for every item. The rest of an item is read from memory.db by id."""

    type: ItemType
    scope: str
    valid: bool
    area: Area
    strength: float
    last_seen: datetime
    last_used: datetime | None
    volatility: Volatility
    checked: datetime
    created_at: datetime
    entities: tuple[str, ...]
    sessions: tuple[str, ...]  # the sessions that taught it
    norm: int  # a hash of its statement, normalized, to find a restatement
    chars: int  # the length of its statement and reason, for recall's budget

    @classmethod
    def of(cls, it: Item) -> Head:
        return cls(
            it.type,
            sys.intern(it.scope),
            it.valid,
            it.area,
            it.strength,
            it.last_seen,
            it.last_used,
            it.volatility,
            it.checked,
            it.created_at,
            tuple(sys.intern(e) for e in it.entities),
            tuple(sys.intern(s.session) for s in it.sources),
            hash(normalize_name(it.statement)),
            len(it.statement) + len(it.why),
        )

    def overdue(self, now: datetime | None = None) -> bool:
        return _overdue(self.volatility, self.checked, now)

    def vitality(self, now: datetime | None = None) -> float:
        return _vitality(self.strength, self.area, self.last_seen, self.last_used, now)

    @property
    def recallable(self) -> bool:
        return _recallable(self.valid, self.area, self.volatility, self.checked)


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

_MISSING: Any = object()
Key = tuple[str, str, str]  # an edge: (source, target, relation)


def _when(v: Any) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


class KnowledgeStore:
    """The knowledge graph in memory.db. In memory it keeps only what recall and upkeep
    read for every item: a Head per item, each entity's names, and the weight of each
    edge. The rest is read from memory.db by id.

    Reading: `item(id)` and `get_items(ids)` give frozen copies, `scan()` streams many.
    Changing: `edit(id)` gives the item to change; `touch(id)` once it is changed (the
    store's own methods do both). Changed objects are kept until save() writes them;
    flush() writes them early, into the open transaction, for a query to see."""

    def __init__(
        self, home: Path, embedder: Embedder | None = None, backend: SqliteBackend | None = None
    ) -> None:
        self.home = home
        self.embedder = embedder
        self.backend = backend or open_backend(home)
        self.vectors = self.backend.vectors
        self.heads: dict[str, Head] = {}  # in the order the items were added
        self._seq: dict[str, int] = {}  # item -> its place in that order
        self._next = 0
        self.names: dict[str, EntityName] = {}
        self.links: dict[Key, float] = {}  # edge -> weight
        self._at: dict[str, dict[Key, None]] = defaultdict(dict)  # item -> its edges, in order
        self._about: dict[str, set[str]] = defaultdict(set)  # entity -> items, any validity
        self._taught: dict[str, set[str]] = defaultdict(set)  # session -> items it taught
        self._by_norm: dict[int, set[str]] = defaultdict(set)  # statement hash -> items
        self._by_name: dict[str, str] = {}  # normalized name or alias -> entity id
        self._shingle_cache: dict[str, set[str]] = {}
        self.version = 0  # bumped on every change
        # What changed, for the recall index to update itself from (see touch()); a new
        # epoch means "everything", after a bulk change such as a deletion
        self.journal: list[tuple[str, ...]] = []
        self.epoch = 0
        self._due: set[str] = set()  # volatile items past their review, see sweep()
        # objects changed since the last save, by key (None: removed); and the keys not
        # yet written into the open transaction
        self._items: dict[str, Item | None] = {}
        self._entities: dict[str, Entity | None] = {}
        self._edges: dict[Key, ItemEdge | None] = {}
        self._segments: dict[str, Segment] = {}
        self._unflushed: dict[tuple[str, Any], None] = {}  # in the order changed
        self._words: set[str] = set()  # items whose words to write again, see sync_words()
        self._worded: set[str] = set()  # ... written since the last commit
        self._vec_meta: set[str] = set()  # items whose scope or recallability to re-file
        self._docs, self._doc_len = 0, 0  # the word index: notes, and their total length
        self._topics: tuple[int, dict[str, str]] | None = None
        self._load()
        self._due = self._due_now()

    # ------------------------------------------------------------------ change log

    def touch(self, *item_ids: str) -> None:
        """Record that these items changed in a way recall may see: validity, area,
        review, scope, text or entities. Called once the change is made: their heads are
        read again from the changed objects."""
        self.version += 1
        self.journal.extend(("item", i) for i in item_ids)
        for i in item_ids:
            if i not in self._items and i in self.heads:
                self.edit(i)  # kept for the save
            self._refile(i)
            self._unflushed[("item", i)] = None
        self._words.update(item_ids)
        self._vec_meta.update(item_ids)  # filed again before the next search or save
        self._due.difference_update(item_ids)  # the next sweep looks at them afresh
        self._trim_journal()

    def _touch_pair(self, a: str, b: str) -> None:
        self.version += 1
        self.journal.append(("pair", a, b))
        self._trim_journal()

    def _touch_entity(self, entity_id: str) -> None:
        self.version += 1
        self.journal.append(("entity", entity_id))
        self._trim_journal()

    def changed_everything(self) -> None:
        """A bulk change: readers of the journal start over."""
        self.version += 1
        self.epoch += 1
        self.journal = []

    def _trim_journal(self) -> None:
        if len(self.journal) > JOURNAL_MAX:
            self.changed_everything()

    # ------------------------------------------------------------------ heads

    def _refile(self, i: str) -> None:
        """Bring an item's head, and the maps built from heads, up to date with its
        changed object (or its removal)."""
        obj = self._items.get(i, _MISSING)
        if obj is _MISSING:
            return
        old = self.heads.get(i)
        if old is not None:
            self._unfile(i, old)
        if obj is None:
            self.heads.pop(i, None)
            self._seq.pop(i, None)
            return
        head = Head.of(obj)
        if old is None:
            self._seq[i] = self._next
            self._next += 1
        self.heads[i] = head
        self._file(i, head)

    def _file(self, i: str, h: Head) -> None:
        for e in h.entities:
            self._about[e].add(i)
        for s in h.sessions:
            self._taught[s].add(i)
        self._by_norm[h.norm].add(i)

    def _unfile(self, i: str, h: Head) -> None:
        for m, keys in ((self._about, h.entities), (self._taught, h.sessions)):
            for k in keys:
                got = m.get(k)
                if got is not None:
                    got.discard(i)
                    if not got:
                        del m[k]
        got = self._by_norm.get(h.norm)
        if got is not None:
            got.discard(i)
            if not got:
                del self._by_norm[h.norm]

    def in_order(self, ids: Iterable[str]) -> list[str]:
        """The given items that exist, in the order they were added."""
        seq = self._seq
        return sorted((i for i in set(ids) if i in seq), key=seq.__getitem__)

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        b = self.backend
        for row in b.entity_heads():
            eid, name, kind, aliases, mentions = row
            self.names[sys.intern(eid)] = EntityName(
                name, EntityKind(kind), tuple(json.loads(aliases or "[]")), int(mentions or 1)
            )
        for eid, n in self.names.items():
            self._index_names_of(eid, n)
        promote, sourceless = [], []
        for row in b.item_heads():
            head = self._head_of_row(row)
            i = sys.intern(row[0])
            self.heads[i] = head
            self._seq[i] = self._next
            self._next += 1
            self._file(i, head)
            if head.area is Area.SURVIVOR and head.strength >= PROMOTE_AT:
                promote.append(i)
            if not head.sessions:
                sourceless.append(i)
        for src, tgt, rel, w in b.edge_weights():
            self._set_link((sys.intern(src), sys.intern(tgt), sys.intern(rel)), float(w))
        self._docs, self._doc_len = b.doc_stats()
        # an older memory.db, or one just moved in: its word index is built once, by the
        # first sync_words() (into tables that are empty)
        self._build_words = b.needs_words
        if self._build_words:
            self._words.update(self.heads)
        # items reinforced before areas existed become old; a dormant item stays dormant
        # (only being stated again or used revives it)
        for i in promote:
            self.edit(i).area = Area.OLD
            self._refile(i)
            self._unflushed[("item", i)] = None
        # items learned before they kept their sources get them from their graduations
        for i in sourceless:
            it = self.edit(i)
            rows = b.bodies("segment", it.segments)
            segs = [Segment.model_validate_json(rows[s]) for s in it.segments if s in rows]
            for seg in segs:
                add_source(it, seg.session, seg.end)
            if segs:
                self._refile(i)
            else:
                del self._items[i]
                self._unflushed.pop(("item", i), None)

    @staticmethod
    def _head_of_row(row: tuple[Any, ...]) -> Head:
        """A head from the columns SqliteBackend.item_heads() reads."""
        (_, typ, scope, valid, area, strength, seen, used, vol, checked, created, ents, stmt) = row[
            :13
        ]
        sources = json.loads(row[13] or "[]")
        chars = row[14]
        created_at = _when(created)
        assert created_at is not None
        return Head(
            ItemType(typ),
            sys.intern(scope),
            bool(valid),
            Area(area or Area.SURVIVOR.value),
            float(strength if strength is not None else 1.0),
            _when(seen) or created_at,
            _when(used),
            Volatility(vol or Volatility.STABLE.value),
            _when(checked) or created_at,
            created_at,
            tuple(sys.intern(e) for e in json.loads(ents or "[]")),
            tuple(sys.intern(s["session"]) for s in sources),
            hash(normalize_name(stmt or "")),
            int(chars or 0),
        )

    def flush(self) -> None:
        """Write the changed objects not written yet into the open transaction, so a
        query sees them. save() commits them."""
        if not self._unflushed:
            return
        b = self.backend
        pending: dict[str, dict[Any, Any]] = {
            "item": self._items,
            "entity": self._entities,
            "edge": self._edges,
            "segment": self._segments,
        }
        for kind, key in self._unflushed:
            obj = pending[kind].get(key, _MISSING)
            if obj is _MISSING:
                continue
            if obj is None:
                b.remove(kind, key)
            else:
                b.write(kind, obj)
        self._unflushed.clear()

    def save(self) -> None:
        """Write everything changed since the last save, with the vectors and the word
        index, in one transaction."""
        try:
            self._file_vectors()
            self.flush()
            self.sync_words()
            self.backend.commit()
        except BaseException:
            self.backend.rollback()
            # the transaction is gone: everything changed since the last save is written
            # again next time
            for kind, objs in (
                ("item", self._items),
                ("entity", self._entities),
                ("edge", self._edges),
                ("segment", self._segments),
            ):
                self._unflushed.update(dict.fromkeys((kind, k) for k in objs))
            self._words |= self._worded
            self._docs, self._doc_len = self.backend.doc_stats()
            raise
        self._worded.clear()
        # what was changed is read-only from here on: a change to it would be lost
        for obj in self._items.values():
            if obj is not None:
                obj.__class__ = ReadItem
        for ent in self._entities.values():
            if ent is not None:
                ent.__class__ = ReadEntity
        self._items.clear()
        self._entities.clear()
        self._edges.clear()
        self._segments.clear()

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
        """Close memory.db, committing what was written to it; changes not saved are
        lost, as when the process stops."""
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

    def vector_meta(self, kind: str, key: str) -> dict[str, str]:
        """What is kept beside a vector: its kind, and for an item its scope and whether
        it can be recalled now (a database searches by them)."""
        h = self.heads.get(key) if kind == "item" else None
        if h is None:
            return {"kind": kind}
        return {"kind": kind, "scope": h.scope, "live": "1" if h.recallable else "0"}

    def _file_vectors(self) -> None:
        """Bring each changed item's vector row up to date with its scope and whether it
        can be recalled (done late: a method marks an item before changing it)."""
        if self._vec_meta:
            heads = self.heads
            self.vectors.set_meta(
                {i: (heads[i].scope, heads[i].recallable) for i in self._vec_meta if i in heads}
            )
            self._vec_meta.clear()

    def _index(self, key: str, text: str, kind: str, vector: list[float] | None = None) -> None:
        v = vector if vector is not None else self.embed(text)
        if v is not None:
            self.vectors.upsert(key, v, self.vector_meta(kind, key))
            if kind == "item":
                self._vec_meta.add(key)  # filed again once the item is in

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
        `among` the visible items of those scopes) lets the database filter in the scan."""
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

    # ------------------------------------------------------------------ words

    def sync_words(self) -> None:
        """Write the word index again for the items changed since it was last written:
        each recallable item's words (its statement, reason and entity names) and length.
        Recall's word search reads it in SQL (see recall's bm25)."""
        if not self._words:
            return
        ids, self._words = sorted(self._words), set()
        heads = self.heads
        for n in range(0, len(ids), 500):
            part = ids[n : n + 500]
            items = {
                it.item_id: it
                for it in self.get_items(i for i in part if i in heads and heads[i].recallable)
            }
            for i in part:
                h = heads.get(i)
                it = items.get(i)
                tf: Counter[str] | None = None
                if h is not None and it is not None:
                    names = " ".join(self.names[e].name for e in it.entities if e in self.names)
                    tf = Counter(tokens(f"{it.statement} {it.why} {names}"))
                before, after = self.backend.set_words(i, tf, h.scope if h else "")
                self._docs += (after is not None) - (before is not None)
                self._doc_len += (after or 0) - (before or 0)
        self._worded.update(ids)
        if self._build_words:
            self.backend.words_built()
            self._build_words = False

    @property
    def doc_count(self) -> int:
        """How many items the word index holds."""
        self.sync_words()
        return self._docs

    def word_df(self, terms: Iterable[str]) -> dict[str, int]:
        """How many items in the word index hold each word."""
        self.sync_words()
        return self.backend.word_df(terms)

    def word_scores(
        self,
        query: Sequence[str],
        k: int,
        scopes: Sequence[str] | None = None,
        exclude: Iterable[str] = (),
    ) -> list[tuple[str, float, int]]:
        """BM25 over the word index: the best k items (of `scopes`, without `exclude`)
        as (id, score, how many of the query's words it holds). Word statistics are those
        of the whole index, so a score means the same in every scope."""
        self.sync_words()
        n = self._docs
        if not n:
            return []
        terms = list(dict.fromkeys(query))
        df = self.backend.word_df(terms)
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in terms if t in df}
        return self.backend.word_scores(idf, self._doc_len / n, k, scopes, exclude)

    def word_differences(self) -> list[str]:
        """How the word index differs from one built from scratch now: empty when no
        change was missed. For tests and the simulator."""
        self.sync_words()
        want: dict[tuple[str, str], tuple[int, int, str]] = {}
        docs = 0
        total = 0
        for it in self.scan():
            h = self.heads[it.item_id]
            if not h.recallable:
                continue
            names = " ".join(self.names[e].name for e in it.entities if e in self.names)
            tf = Counter(tokens(f"{it.statement} {it.why} {names}"))
            dl = sum(tf.values())
            docs += 1
            total += dl
            for t, f in tf.items():
                want[(t, it.item_id)] = (f, dl, h.scope)
        have = {(t, i): (f, dl, s) for t, i, f, dl, s in self.backend.word_rows()}
        out = []
        if have != want:
            bad = sorted({i for k in set(have) ^ set(want) for i in [k[1]]})
            bad += sorted({k[1] for k in set(have) & set(want) if have[k] != want[k]})
            out.append(f"words of {len(set(bad))} items, e.g. {bad[0] if bad else '?'}")
        if (self._docs, self._doc_len) != (docs, total) or self.backend.doc_stats() != (
            docs,
            total,
        ):
            out.append("word index size")
        return out

    # ------------------------------------------------------------------ entities

    def _index_names_of(self, eid: str, n: EntityName) -> None:
        for name in (n.name, *n.aliases):
            self._by_name.setdefault(normalize_name(name), eid)

    def _set_name(self, e: Entity) -> None:
        self.names[e.entity_id] = EntityName(e.name, e.kind, tuple(e.aliases), e.mentions)
        self._index_names_of(e.entity_id, self.names[e.entity_id])

    def entity(self, entity_id: str) -> Entity | None:
        obj = self._entities.get(entity_id, _MISSING)
        if obj is not _MISSING:
            return obj  # type: ignore[no-any-return]
        if entity_id not in self.names:
            return None
        body = self.backend.bodies("entity", [entity_id]).get(entity_id)
        return ReadEntity.model_validate_json(body) if body else None

    def get_entities(self, ids: Iterable[str]) -> list[Entity]:
        """The given entities that exist, in the order given."""
        want = list(dict.fromkeys(ids))
        missing = [e for e in want if e not in self._entities and e in self.names]
        rows = self.backend.bodies("entity", missing)
        out: list[Entity] = []
        for e in want:
            obj = self._entities.get(e, _MISSING)
            if obj is _MISSING:
                obj = ReadEntity.model_validate_json(rows[e]) if e in rows else None
            if obj is not None:
                out.append(obj)
        return out

    def edit_entity(self, entity_id: str) -> Entity:
        obj = self._entities.get(entity_id, _MISSING)
        if obj is None:
            raise KeyError(entity_id)
        if obj is _MISSING:
            body = self.backend.bodies("entity", [entity_id]).get(entity_id)
            if body is None or entity_id not in self.names:
                raise KeyError(entity_id)
            obj = Entity.model_validate_json(body)
            self._entities[entity_id] = obj
        self._unflushed[("entity", entity_id)] = None
        return obj  # type: ignore[no-any-return]

    def add_entity(self, e: Entity, vector: list[float] | None = None) -> Entity:
        self.version += 1
        if type(e) is not Entity:
            e = Entity.model_validate(e.model_dump())
        self._entities[e.entity_id] = e
        self._unflushed[("entity", e.entity_id)] = None
        self._set_name(e)
        self._index(e.entity_id, self.entity_text(e), "entity", vector)
        return e

    def merge_entity(self, target_id: str, name: str, summary: str = "") -> Entity:
        self.version += 1
        e = self.edit_entity(target_id)
        e.mentions += 1
        if normalize_name(name) != normalize_name(e.name) and name not in e.aliases:
            e.aliases.append(name)
            self._set_name(e)
            self._touch_entity(target_id)
        else:
            self._set_name(e)
        if summary and not e.summary:
            e.summary = summary
            self._index(e.entity_id, self.entity_text(e), "entity")
        return e

    def entity_id_by_name(self, name: str) -> str | None:
        eid = self._by_name.get(normalize_name(name))
        if eid is None:
            key = fuzzy_key(name)
            eid = next((v for k, v in self._by_name.items() if _fuzzy_form(k) == key), None)
        if eid is None and "/" not in name.strip("/") and "." in name:
            # a bare file name matches the one known path that ends with it
            tail = "/" + name.strip()
            paths = [
                i
                for i, n in self.names.items()
                if n.kind is EntityKind.FILE and n.name.endswith(tail)
            ]
            if len(paths) == 1:
                return paths[0]
        return eid if eid in self.names else None

    def entity_by_name(self, name: str) -> Entity | None:
        eid = self.entity_id_by_name(name)
        return self.entity(eid) if eid else None

    def hub(self, entity_id: str, limit: int = 8) -> bool:
        """Too general to say two items belong together: the project, a person, or an
        entity that many items are about."""
        n = self.names.get(entity_id)
        if n is None or n.kind in (EntityKind.PROJECT, EntityKind.PERSON):
            return True
        heads = self.heads
        return sum(1 for i in self._about.get(entity_id, ()) if heads[i].valid) > limit

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
        return self.entity(best[1]) if best else None

    # ------------------------------------------------------------------ items

    def has(self, item_id: str) -> bool:
        return item_id in self.heads

    def item(self, item_id: str) -> Item | None:
        """The item, or None: the changed object when it is changed, else a frozen copy
        read from memory.db."""
        obj = self._items.get(item_id, _MISSING)
        if obj is not _MISSING:
            return obj  # type: ignore[no-any-return]
        if item_id not in self.heads:
            return None
        body = self.backend.bodies("item", [item_id]).get(item_id)
        return ReadItem.model_validate_json(body) if body else None

    def get_items(self, ids: Iterable[str]) -> list[Item]:
        """The given items that exist, in the order given, read in one query."""
        want = list(dict.fromkeys(ids))
        missing = [i for i in want if i not in self._items and i in self.heads]
        rows = self.backend.bodies("item", missing)
        out: list[Item] = []
        for i in want:
            obj = self._items.get(i, _MISSING)
            if obj is _MISSING:
                obj = ReadItem.model_validate_json(rows[i]) if i in rows else None
            if obj is not None:
                out.append(obj)
        return out

    def scan(
        self, *, valid: bool | None = None, scope: str | None = None, type: ItemType | None = None
    ) -> Iterator[Item]:
        """Every item (or those matching), read from memory.db in the order added. For
        pages, commands and upkeep: a recall never scans."""
        self.flush()
        where: list[str] = []
        args: list[Any] = []
        if valid is not None:
            where.append("valid = ?")
            args.append(int(valid))
        if scope is not None:
            where.append("scope = ?")
            args.append(scope)
        if type is not None:
            where.append("type = ?")
            args.append(type.value)
        for body in self.backend.scan("item", " AND ".join(where), args):
            it = ReadItem.model_validate_json(body)
            obj = self._items.get(it.item_id, _MISSING)
            yield it if obj is _MISSING else obj

    def edit(self, item_id: str) -> Item:
        """The item to change. Call touch(item_id) once it is changed (a change recall
        does not see, such as a count, needs no touch: it is saved all the same)."""
        obj = self._items.get(item_id, _MISSING)
        if obj is None:
            raise KeyError(item_id)
        if obj is _MISSING:
            body = (
                self.backend.bodies("item", [item_id]).get(item_id)
                if item_id in self.heads
                else None
            )
            if body is None:
                raise KeyError(item_id)
            obj = Item.model_validate_json(body)
            self._items[item_id] = obj
        self._unflushed[("item", item_id)] = None
        return obj  # type: ignore[no-any-return]

    def add_item(self, item: Item, vector: list[float] | None = None) -> Item:
        if type(item) is not Item:
            item = Item.model_validate(item.model_dump())
        self._items[item.item_id] = item
        self._refile(item.item_id)
        self._index(item.item_id, self.item_text(item), "item", vector)
        self.touch(item.item_id)
        return item

    def ids(self, valid: bool | None = None) -> list[str]:
        """Item ids in the order added, or only the valid (invalid) ones."""
        if valid is None:
            return list(self.heads)
        return [i for i, h in self.heads.items() if h.valid is valid]

    def topics(self) -> dict[str, str]:
        """The topic scopes memory has items in, by lower-case name: "sqlite" -> "topic:SQLite"."""
        if self._topics is not None and self._topics[0] == self.version:
            return dict(self._topics[1])
        out: dict[str, str] = {}
        for h in self.heads.values():
            name = topic_of(h.scope)
            if name:
                out.setdefault(name.lower(), h.scope)
        self._topics = (self.version, out)
        return dict(out)

    def topic_key(self, name: str) -> str | None:
        """The scope key for a topic, reusing the spelling memory already has."""
        key = topic_scope(name)
        if key is None:
            return None
        return self.topics().get(key[len(TOPIC_PREFIX) :].lower(), key)

    def about(self, entity_ids: Iterable[str], include_invalid: bool = False) -> list[str]:
        """The items about any of these entities, in the order added."""
        ids: set[str] = set()
        for e in entity_ids:
            ids |= self._about.get(e, set())
        if not include_invalid:
            heads = self.heads
            ids = {i for i in ids if heads[i].valid}
        return self.in_order(ids)

    def taught_by(self, session: str) -> list[str]:
        """The items a session taught, in the order added."""
        return self.in_order(self._taught.get(session, ()))

    def find_statement(self, statement: str, include_invalid: bool = False) -> Item | None:
        """The first item (in the order added) that says the same, written the same."""
        norm = normalize_name(statement)
        heads = self.heads
        cands = [
            i
            for i in self.in_order(self._by_norm.get(hash(norm), ()))
            if include_invalid or heads[i].valid
        ]
        return next(
            (it for it in self.get_items(cands) if normalize_name(it.statement) == norm), None
        )

    def reinforce(
        self, item_id: str, segment_id: str, source: Source | None = None, *, boost: bool = True
    ) -> Item:
        """A later graduation stated the item again: stronger (unless `boost` is off: the
        same evidence read twice), and one more source."""
        it = self.edit(item_id)
        if source is not None:
            add_source(it, source.session, source.until)
        if boost:
            it.strength += SEEN_BOOST
        it.last_seen = _now()
        self._confirm(it)
        self._promote(it)
        if segment_id not in it.segments:
            it.segments.append(segment_id)
        self.touch(item_id)
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
        it = self.edit(item_id)
        self._confirm(it)
        self.touch(item_id)
        return it

    def dispute(self, item_id: str, reason: str = "") -> Item:
        """The live judge found the note contradicted by what the session saw. The first
        time it is shown with a warning; at DISPUTE_LIMIT it is retracted."""
        it = self.edit(item_id)
        if not it.valid:
            self.touch(item_id)
            return it
        it.disputes += 1
        it.disputed_at = _now()
        it.dispute_reason = reason[:300]
        self.touch(item_id)
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
            if i in self.heads:
                self.edit(i).recalls += 1

    def mark_used(self, item_ids: Sequence[str]) -> None:
        """Recalled items the reply used: stronger, and linked to each other (Hebbian:
        items used together get a co_used edge that grows each time)."""
        if not item_ids:
            return
        ids = [i for i in item_ids if i in self.heads]
        now = _now()
        for i in ids:
            it = self.edit(i)
            it.uses += 1
            it.strength += USE_BOOST
            it.last_used = now
            self._promote(it)
        self.touch(*ids)
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1 :]:
                key = next(
                    (k for k in [(a, b, "co_used"), (b, a, "co_used")] if k in self.links), None
                )
                if key is None:
                    self.add_edge(a, b, Relation.CO_USED, CO_USE_STEP, "co_use")
                else:
                    self._weigh(key, round(min(1.0, self.links[key] + CO_USE_STEP), 3))
                    self._touch_pair(a, b)

    def sweep(self, now: datetime | None = None) -> list[str]:
        """Move items whose vitality fell under the floor to dormant. Returns their ids.
        Volatile items that fell due for review (or were confirmed) since the last sweep
        are logged too: they stop or start being recallable with no other change."""
        faded = [
            i
            for i, h in self.heads.items()
            if h.valid and h.area is not Area.DORMANT and h.vitality(now) < DORMANT_BELOW
        ]
        for i in faded:
            self.edit(i).area = Area.DORMANT
        due = self._due_now(now)
        changed = [*faded, *(due ^ self._due)]
        self._due = due
        if changed:
            self.touch(*changed)
        return faded

    def _due_now(self, now: datetime | None = None) -> set[str]:
        return {
            i
            for i, h in self.heads.items()
            if h.volatility is Volatility.VOLATILE and h.valid and h.overdue(now)
        }

    def invalidate(self, old_id: str, new: Item, reason: str = "") -> None:
        """Graphiti's temporal invalidation: the old item stopped being true when the new
        one became true. It stays in the graph for history."""
        old = self.edit(old_id)
        if not old.valid:
            self.touch(old_id)
            return
        old.invalid_at = max(new.valid_at, old.valid_at)
        old.expired_at = _now()
        old.invalidated_by = new.item_id
        old.invalid_reason = reason
        self.touch(old_id)
        self.add_edge(new.item_id, old_id, Relation.SUPERSEDES, 1.0, "invalidation", reason)

    def retract(self, item_id: str, reason: str = "") -> Item:
        """A person marks an item no longer true by hand: for when no later session states
        what replaced it, or the model missed the contradiction. Kept for history."""
        it = self.item(item_id)
        if it is None:
            raise KeyError(item_id)
        if it.valid:
            it = self.edit(item_id)
            it.invalid_at = it.expired_at = _now()
            it.invalidated_by = None
            it.invalid_reason = reason or "marked no longer true by hand"
            self.touch(item_id)
        return it

    def restore(self, item_id: str) -> Item:
        """Undo an invalidation, by hand or by the model: the item is true again and the
        supersedes edges pointing at it are dropped."""
        it = self.item(item_id)
        if it is None:
            raise KeyError(item_id)
        if not it.valid:
            it = self.edit(item_id)
            it.invalid_at = it.expired_at = it.invalidated_by = None
            it.invalid_reason = ""
            self.touch(item_id)
            for key in [
                k for k in self._at.get(item_id, {}) if k[1] == item_id and k[2] == "supersedes"
            ]:
                self._remove_edge(key)
        return it

    # ------------------------------------------------------------------ edges

    def _set_link(self, key: Key, weight: float) -> None:
        self.links[key] = weight
        self._at[key[0]][key] = None
        self._at[key[1]][key] = None

    def _remove_edge(self, key: Key) -> None:
        if key not in self.links:
            return
        del self.links[key]
        for end in (key[0], key[1]):
            at = self._at.get(end)
            if at is not None:
                at.pop(key, None)
                if not at:
                    del self._at[end]
        self._edges[key] = None
        self._unflushed[("edge", key)] = None

    def edge(self, key: Key) -> ItemEdge | None:
        obj = self._edges.get(key, _MISSING)
        if obj is not _MISSING:
            return obj  # type: ignore[no-any-return]
        if key not in self.links:
            return None
        body = self.backend.bodies("edge", [key]).get(key)
        return ItemEdge.model_validate_json(body) if body else None

    def _weigh(self, key: Key, weight: float) -> None:
        """A new weight for an edge memory has."""
        if self.links.get(key) == weight:
            return
        e = self._edges.get(key)
        if e is None:
            body = self.backend.bodies("edge", [key]).get(key)
            if body is None:
                return
            e = ItemEdge.model_validate_json(body)
            self._edges[key] = e
        e.weight = weight
        self.links[key] = weight
        self._unflushed[("edge", key)] = None

    def _put_edge(self, e: ItemEdge) -> None:
        key = (e.source, e.target, e.relation.value)
        self._edges[key] = e
        self._set_link(key, e.weight)
        self._unflushed[("edge", key)] = None

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
            rev = (target, source, relation.value)
            if rev in self.links:
                self._weigh(rev, max(self.links[rev], weight))
                return
            if relation is Relation.CO_OCCURS and self.related(source, target):
                return  # a typed edge already says more
            if relation not in (Relation.CO_OCCURS, Relation.CO_USED):
                for key in [(source, target, "co_occurs"), (target, source, "co_occurs")]:
                    self._remove_edge(key)
        key = (source, target, relation.value)
        if key in self.links:
            self._weigh(key, max(self.links[key], weight))
            return
        self._put_edge(
            ItemEdge(
                source=source,
                target=target,
                relation=relation,
                weight=round(weight, 3),
                origin=origin,
                reason=reason,
            )
        )

    def related(self, a: str, b: str) -> bool:
        return any(
            {k[0], k[1]} == {a, b} and k[2] not in ("co_occurs", "co_used")
            for k in self._at.get(a, {})
        )

    def neighbours(self, item_id: str) -> list[tuple[str, ItemEdge]]:
        out = []
        for key in self._at.get(item_id, {}):
            e = self.edge(key)
            if e is not None:
                out.append((key[1] if key[0] == item_id else key[0], e))
        return out

    # ------------------------------------------------------------------ graduations

    def add_segment(self, segment: Segment) -> Segment:
        self.version += 1
        self._segments[segment.segment_id] = segment
        self._unflushed[("segment", segment.segment_id)] = None
        return segment

    def segment(self, segment_id: str) -> Segment | None:
        got = self.segments(ids=[segment_id])
        return got[0] if got else None

    def segments(
        self, *, session: str | None = None, ids: Iterable[str] | None = None
    ) -> list[Segment]:
        """Graduation records, in the order added: of one session, or the given ones."""
        self.flush()
        if ids is not None:
            want = list(dict.fromkeys(ids))
            rows = self.backend.bodies("segment", want)
            return [Segment.model_validate_json(rows[s]) for s in want if s in rows]
        if session is not None:
            return [
                Segment.model_validate_json(b)
                for b in self.backend.scan("segment", "session = ?", [session])
            ]
        return [Segment.model_validate_json(b) for b in self.backend.scan("segment")]

    def segments_since(self, when: datetime) -> list[Segment]:
        """Graduation records made at `when` or later, in the order added."""
        self.flush()
        day = (when - timedelta(days=1)).date().isoformat()  # a coarse cut in SQL first
        segs = (
            Segment.model_validate_json(b)
            for b in self.backend.scan("segment", "json_extract(body, '$.created_at') >= ?", [day])
        )
        return [g for g in segs if g.created_at >= when]

    def graduated_sessions(self) -> set[str]:
        """The sessions any graduation record is of."""
        self.flush()
        return self.backend.segment_sessions()

    def scan_entities(self) -> Iterator[Entity]:
        """Every entity, read from memory.db in the order added."""
        self.flush()
        for body in self.backend.scan("entity"):
            e = ReadEntity.model_validate_json(body)
            obj = self._entities.get(e.entity_id, _MISSING)
            yield e if obj is _MISSING else obj

    def scan_edges(self) -> Iterator[ItemEdge]:
        """Every edge, read from memory.db in the order added."""
        self.flush()
        for body in self.backend.scan("edge"):
            yield ItemEdge.model_validate_json(body)

    def texts(self) -> Iterator[tuple[str, str, str, list[str]]]:
        """Every item's id, statement, reason and entity ids, in the order added: what a
        search reads, without building the items."""
        self.flush()
        for i, stmt, why, ents in self.backend.texts():
            yield i, stmt or "", why or "", json.loads(ents or "[]")

    def fields(self, kind: str, names: Sequence[str]) -> Iterator[tuple[Any, ...]]:
        """For every row of a kind (item, entity, edge, segment), in the order added: its
        key, then the named fields, read in SQL without building the objects. For pages
        that list everything."""
        self.flush()
        return self.backend.fields(kind, names)

    def disputed(self) -> list[str]:
        """The valid items a reply contradicted since they were last confirmed."""
        self.flush()
        return [
            str(i)
            for i in self.backend.keys("item", "valid = 1 AND json_extract(body, '$.disputes') > 0")
        ]

    def segment_ids(self) -> list[str]:
        self.flush()
        return [str(k) for k in self.backend.keys("segment")]

    # ------------------------------------------------------------------ views

    def stats(self) -> dict[str, Any]:
        valid = [h for h in self.heads.values() if h.valid]
        self.flush()
        return {
            "entities": len(self.names),
            "items": len(valid),
            "invalidated": len(self.heads) - len(valid),
            "edges": len(self.links),
            "segments": self.backend.count("segment"),
            "by_type": dict(Counter(h.type.value for h in valid)),
            "by_area": dict(Counter(h.area.value for h in valid)),
        }

    def differences(self, other: KnowledgeStore) -> list[str]:
        """How what this store holds in memory differs from another's (one opened from
        disk after this one saved): empty when no change went unsaved."""
        out = []
        if self.heads != other.heads or list(self.heads) != list(other.heads):
            bad = [
                i
                for i in set(self.heads) | set(other.heads)
                if self.heads.get(i) != other.heads.get(i)
            ]
            out.append(f"items: {len(bad)} differ, e.g. {sorted(bad)[0] if bad else 'order'}")
        if self.names != other.names:
            out.append("entities differ")
        if self.links != other.links:
            out.append("edges differ")
        if (self._docs, self._doc_len) != (other._docs, other._doc_len):
            out.append("word index size differs")
        return out

    # ------------------------------------------------------------------ deleting

    def delete(self, item_ids: Iterable[str], reason: str = "") -> dict[str, Any]:
        """Remove items for good (a person's clean-up of noise). Their edges go with them;
        neighbours linked only through them get a bridge edge; entities no item names any
        more go too. The batch is logged in full, so `undelete` can put it back."""
        gone = {i for i in item_ids if i in self.heads}
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
        for it in self.get_items(sorted(gone)):
            batch["items"].append(
                {"item": it.model_dump(mode="json"), "vector": self.vectors.get(it.item_id)}
            )
        dropped = [k for k in self.links if k[0] in gone or k[1] in gone]
        for key in dropped:
            e = self.edge(key)
            if e is not None:
                batch["edges"].append(e.model_dump(mode="json"))
            self._remove_edge(key)
        for i in gone:
            self._items[i] = None
            self._unflushed[("item", i)] = None
            self._refile(i)
            self.vectors.remove(i)
        self._words |= gone
        self.flush()
        marks = ",".join("?" * len(gone))
        for it in self.get_items(
            self.backend.keys(
                "item", f"json_extract(body, '$.invalidated_by') IN ({marks})", sorted(gone)
            )
        ):
            batch["unlinked"].append([it.item_id, it.invalidated_by])
            self.edit(it.item_id).invalidated_by = None
        orphans = [e for e in self.names if not self._about.get(e)]
        for ent in self.get_entities(orphans):
            batch["entities"].append(
                {"entity": ent.model_dump(mode="json"), "vector": self.vectors.get(ent.entity_id)}
            )
        for eid in orphans:
            del self.names[eid]
            self._entities[eid] = None
            self._unflushed[("entity", eid)] = None
            self.vectors.remove(eid)
        if orphans:
            self._reindex_names()
        for b in bridges:
            self._put_edge(b)
            batch["bridges"].append([b.source, b.target, b.relation.value])
        self.changed_everything()
        self.backend.log_deletion(batch)
        return batch

    def deleted(self) -> list[dict[str, Any]]:
        """Logged deletions, oldest first."""
        return self.backend.deletions()

    def undelete(self, batch_id: str) -> dict[str, Any]:
        """Put a deleted batch back as it was: items, entities, edges; its bridges go."""
        batch = self.backend.deletion(batch_id)
        if batch is None:
            raise KeyError(f"no deletion {batch_id!r}")
        for rec in batch["entities"]:
            ent = Entity.model_validate(rec["entity"])
            if ent.entity_id not in self.names:
                self._entities[ent.entity_id] = ent
                self._unflushed[("entity", ent.entity_id)] = None
                self.names[ent.entity_id] = EntityName(
                    ent.name, ent.kind, tuple(ent.aliases), ent.mentions
                )
                if rec.get("vector"):
                    self.vectors.upsert(ent.entity_id, rec["vector"], {"kind": "entity"})
        self._reindex_names()
        for rec in batch["items"]:
            it = Item.model_validate(rec["item"])
            self._items[it.item_id] = it
            self._unflushed[("item", it.item_id)] = None
            self._refile(it.item_id)
            self._words.add(it.item_id)
            if rec.get("vector"):
                self.vectors.upsert(it.item_id, rec["vector"], self.vector_meta("item", it.item_id))
            else:
                self._index(it.item_id, self.item_text(it), "item")
        for src, tgt, rel in batch["bridges"]:
            e = self.edge((src, tgt, rel))
            if e is not None and e.origin == "bridge":
                self._remove_edge((src, tgt, rel))
        for raw in batch["edges"]:
            e = ItemEdge.model_validate(raw)
            if e.source in self.heads and e.target in self.heads:
                self._put_edge(e)
        for item_id, by in batch["unlinked"]:
            got = self.item(item_id)
            if got is not None and got.invalidated_by is None:
                self.edit(item_id).invalidated_by = by
        self.changed_everything()
        self.backend.drop_deletion(batch_id)
        return batch

    def _reindex_names(self) -> None:
        self._by_name = {}
        self._shingle_cache = {}
        for eid, n in self.names.items():
            self._index_names_of(eid, n)

    def _bridges(self, gone: set[str]) -> list[ItemEdge]:
        """Edges that keep neighbours of the deleted items within reach of each other.

        Each group of deleted items linked to each other acts as one node: for each pair
        of valid items linked to the group, not already close (a direct edge, a shared
        specific entity or a common neighbour), a bridge as strong as the product of
        their strongest links into the group. The relation carries over when both links
        chain the same transitive relation (A depends on X, X depends on B), else it is
        same_area. At most BRIDGE_MAX per group, strongest first."""
        links = self.links

        def adj(u: str) -> list[tuple[str, Key]]:
            return [
                (k[1] if k[0] == u else k[0], k)
                for k in self._at.get(u, {})
                if k[2] != Relation.SUPERSEDES.value
            ]

        heads = self.heads
        out: list[ItemEdge] = []
        seen: set[str] = set()
        for start in sorted(gone):
            if start in seen:
                continue
            group, stack = {start}, [start]
            while stack:
                u = stack.pop()
                for v, _ in adj(u):
                    if v in gone and v not in group:
                        group.add(v)
                        stack.append(v)
            seen |= group
            border: dict[str, Key] = {}
            for u in sorted(group, key=lambda g: self._seq.get(g, 0)):
                for v, k in adj(u):
                    if v in gone or v not in heads or not heads[v].valid:
                        continue
                    if v not in border or links[k] > links[border[v]]:
                        border[v] = k
            pairs = []
            names = sorted(border)
            for n, a in enumerate(names):
                for b in names[n + 1 :]:
                    w = links[border[a]] * links[border[b]]
                    if w >= BRIDGE_MIN and not self._close(a, b, gone, adj):
                        pairs.append((w, a, b))
            pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
            through = "; ".join(sorted(g.statement[:60] for g in self.get_items(group)))[:200]
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

    def _close(self, a: str, b: str, gone: set[str], adj: Any) -> bool:
        na = {v for v, _ in adj(a) if v not in gone}
        if b in na:
            return True
        if na & {v for v, _ in adj(b) if v not in gone}:
            return True
        shared = set(self.heads[a].entities) & set(self.heads[b].entities)
        return any(not self.hub(e) for e in shared)


def _compose(a: str, ka: Key, b: str, kb: Key) -> tuple[Relation, str, str]:
    """The relation a bridge between a and b keeps, and its direction."""
    if ka[2] == kb[2] and ka[2] in TRANSITIVE:
        if ka[0] == a and kb[1] == b:  # a → group → b
            return Relation(ka[2]), a, b
        if kb[0] == b and ka[1] == a:  # b → group → a
            return Relation(ka[2]), b, a
    return Relation.SAME_AREA, a, b
