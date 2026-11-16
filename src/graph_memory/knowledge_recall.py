"""Recall from the knowledge graph: no model calls, a few milliseconds.

    prompt ─┬─ entities named in it (exact, word-bounded)  ─┐
            ├─ items close in meaning (vector top-k)        ├─ RRF ─▶ seeds
            └─ items sharing rare words (BM25)             ─┘
    seeds ─▶ Personalized PageRank by forward push over items and entities
          ─▶ valid items in scope, ranked by PageRank mass
          ─▶ MMR for diversity ─▶ token budget

PageRank spreads relevance along weighted item edges and through shared entities, so an
item can be recalled without sharing a word with the prompt (a pitfall about the file the
prompt names, the reason behind a decision it asks about). Entity edges are damped by the
entity's degree, so a hub such as the project does not pull in everything.
Forward push (Andersen, Chung, Lang 2006) only touches nodes with enough residual mass,
so cost depends on the neighbourhood of the seeds, not on the size of the graph.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from . import clock
from .knowledge import (
    ORIGIN_LABEL,
    TOPIC_PREFIX,
    Head,
    Item,
    ItemType,
    KnowledgeStore,
    Relation,
    Volatility,
    normalize_name,
    topic_of,
)
from .projects import project_title
from .words import stem as stem
from .words import tokens as tokens

RRF_K = 60
ALPHA = 0.2  # restart probability
EPSILON = 1e-4
MAX_DEGREE = 20  # strongest edges kept per node
_LINKING = tuple(r.value for r in Relation if r is not Relation.SUPERSEDES)  # supersedes is history
_WORD = re.compile(r"\w+")
_AFTER_NAME = re.compile(r"[\w-]|\.\w")


@dataclass(frozen=True)
class RecallPolicy:
    max_items: int = 6
    max_chars: int = 1800
    vector_k: int = 12
    vector_floor: float = 0.25  # below this a vector hit is not a seed
    # give up when nothing in the prompt clearly points into the graph
    min_vector: float = 0.38
    min_bm25: float = 4.0
    # ... or when the best word match covers this share of the prompt's words (at least
    # two of them): BM25 scores stay low while the store is small, coverage does not
    min_coverage: float = 0.34
    min_word_seed: float = 2.0  # BM25 score for a word match to seed
    standing: int = 3  # preferences shown at the start of a session, see recall()
    mmr_lambda: float = 0.6
    # drop items with less than this share of the top item's mass. At 0.35, items two
    # hops away through a shared entity ("Redis") got in on small graphs (simulation:
    # "run the tests" showed the lock decision at 0.43)
    min_mass: float = 0.5


@dataclass
class Recalled:
    items: list[Item]
    scores: dict[str, float]
    seeds: dict[str, float]
    entities: list[str]  # entity names matched in the prompt
    reason: str
    ms: float = 0.0
    why: dict[str, list[str]] = field(default_factory=dict)  # item id -> how it was reached


class KnowledgeIndex:
    """What recall looks things up in, kept up to date in place: the recallable items by
    scope and by entity, word counts with an inverted index for BM25, entity names by
    their first word, and the weighted links PageRank walks.

    It reads the store's change log (`KnowledgeStore.touch`), so a change costs work for
    the items it touched and their neighbours, not a rebuild of everything. Each node's
    strongest links are worked out when PageRank first asks for them, and again only
    after something near it changed."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self.version = -1
        self._epoch = -1
        self._cursor = 0
        self.live: set[str] = set()  # recallable when last touched
        self._volatile: set[str] = set()  # of those, the ones that stop when overdue
        self.by_scope: dict[str, set[str]] = defaultdict(set)
        self._scope_of: dict[str, str] = {}
        self.by_entity: dict[str, set[str]] = defaultdict(set)
        self._entities_of: dict[str, tuple[str, ...]] = {}
        # item links, valid or not, as the strongest weight between two items
        self._links: dict[str, dict[str, float]] = defaultdict(dict)
        self._adj: dict[str, list[tuple[str, float]]] = {}  # built on demand
        self._wdeg: dict[str, float] = {}
        # entity names: first word (or first character, for a name that starts with
        # one that is not a word character) -> {(name as written, entity)}
        self._names: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self._name_keys: dict[str, list[tuple[str, tuple[str, str]]]] = {}  # entity -> entries

    # ------------------------------------------------------------------ upkeep

    def fresh(self) -> None:
        s = self.store
        s.sync_words()  # the word index lives in memory.db, kept by the store
        if self.version == s.version:
            return
        if self._epoch != s.epoch or self.version < 0:
            self._rebuild()
        else:
            items: set[str] = set()
            pairs: set[tuple[str, str]] = set()
            ents: set[str] = set()
            for entry in s.journal[self._cursor :]:
                if entry[0] == "item":
                    items.add(entry[1])
                elif entry[0] == "pair":
                    pairs.add((entry[1], entry[2]))
                else:
                    ents.add(entry[1])
            for a, b in pairs:
                self._update_pair(a, b)
            for i in items:
                self._update_item(i)
            for e in ents:
                self._update_names(e)
        self._epoch, self._cursor, self.version = s.epoch, len(s.journal), s.version

    def _rebuild(self) -> None:
        s = self.store
        self.__init__(s)  # type: ignore[misc]
        for src, tgt, _ in s.links:
            self._link(src, tgt)
        for i, h in s.heads.items():
            if h.recallable:
                self._add_item(i, h)
        for e in self.by_entity:
            self._update_names(e)

    def _link(self, a: str, b: str) -> None:
        """The strongest edge between two items, supersedes left out (it is history)."""
        links = self.store.links
        w = max(
            (
                w
                for rel in _LINKING
                for w in (links.get((a, b, rel)), links.get((b, a, rel)))
                if w is not None
            ),
            default=0.0,
        )
        if w > 0:
            self._links[a][b] = self._links[b][a] = w
        else:
            self._links[a].pop(b, None)
            self._links[b].pop(a, None)

    def _update_pair(self, a: str, b: str) -> None:
        self._link(a, b)
        self._stale(a, b)

    def _update_item(self, item_id: str) -> None:
        h = self.store.heads.get(item_id)
        before = set(self._entities_of.get(item_id, ()))
        if item_id in self.live:
            self._remove_item(item_id)
        if h is not None and h.recallable:
            self._add_item(item_id, h)
        after = set(self._entities_of.get(item_id, ()))
        # an entity's link weight depends on how many items it has, so its items move too
        for e in before | after:
            self._stale("e:" + e, *self.by_entity.get(e, ()))
            self._update_names(e)
        self._stale(item_id, *self._links.get(item_id, ()))

    def _add_item(self, i: str, h: Head) -> None:
        self.live.add(i)
        if h.volatility is Volatility.VOLATILE:
            self._volatile.add(i)
        self.by_scope[h.scope].add(i)
        self._scope_of[i] = h.scope
        self._entities_of[i] = h.entities
        for e in h.entities:
            self.by_entity[e].add(i)

    def _remove_item(self, i: str) -> None:
        self.live.discard(i)
        self._volatile.discard(i)
        scope = self._scope_of.pop(i)
        self.by_scope[scope].discard(i)
        if not self.by_scope[scope]:
            del self.by_scope[scope]
        for e in self._entities_of.pop(i, ()):
            self.by_entity[e].discard(i)
            if not self.by_entity[e]:
                del self.by_entity[e]

    def _stale(self, *nodes: str) -> None:
        for n in nodes:
            self._adj.pop(n, None)
            self._wdeg.pop(n, None)

    def _update_names(self, entity_id: str) -> None:
        for key, entry in self._name_keys.pop(entity_id, []):
            self._names[key].discard(entry)
            if not self._names[key]:
                del self._names[key]
        ent = self.store.names.get(entity_id)
        if ent is None or not self.by_entity.get(entity_id):
            return
        entries = []
        for n in {ent.name, *ent.aliases}:
            norm = normalize_name(n)
            if len(norm) < 4:
                continue
            forms = [norm]
            base = norm.rsplit("/", 1)[-1]
            if "/" in norm and len(base) >= 5 and "." in base:  # a file named by basename
                forms.append(base)
            for form in forms:
                head = _WORD.match(form)
                key = head.group() if head else form[0]
                entries.append((key, (form, entity_id)))
                self._names[key].add((form, entity_id))
        self._name_keys[entity_id] = entries

    # ------------------------------------------------------------------ lookups

    def visible(self, scopes: Iterable[str]) -> set[str]:
        """The recallable items in these scopes."""
        out: set[str] = set()
        for s in scopes:
            out |= self.by_scope.get(s, set())
        if out & self._volatile:
            heads = self.store.heads
            out -= {i for i in out & self._volatile if not heads[i].recallable}
        return out

    def topics(self) -> set[str]:
        """The topic scopes that have recallable items."""
        return {s for s in self.by_scope if s.startswith(TOPIC_PREFIX)}

    def about(self, entity_id: str) -> set[str]:
        """The recallable items about an entity."""
        return self.by_entity.get(entity_id, set())

    def bm25(
        self, query: Sequence[str], scopes: Sequence[str] | None = None, k: int = 10
    ) -> list[tuple[str, float, int]]:
        """The best word matches among the recallable items of these scopes (all when
        None), as (id, score, how many of the query's words it holds). Scored in SQL over
        the store's word index; word statistics are those of the whole index, so a score
        means the same in every scope."""
        self.fresh()
        heads = self.store.heads
        # in the index but no longer recallable: volatile items that fell due since
        skip = [
            i
            for i in self._volatile
            if (scopes is None or self._scope_of[i] in scopes) and not heads[i].recallable
        ]
        return self.store.word_scores(query, k, scopes, skip)

    def mentioned(self, text: str) -> list[str]:
        """The entities the text names, as whole words: a name counts where it is not
        preceded by a word character, `/`, `.` or `-`, nor followed by a word character,
        `-`, or a dot and a word character ("redis" in "redis-cli" does not count)."""
        low = text.lower()
        found: dict[str, None] = {}

        def at(s: int, key: str) -> None:
            if s and low[s - 1] in "/.-":
                return
            for form, eid in self._names.get(key, ()):
                if low.startswith(form, s) and not _AFTER_NAME.match(low, s + len(form)):
                    found.setdefault(eid)

        for m in _WORD.finditer(low):
            at(m.start(), m.group())
        for s, ch in enumerate(low):
            if ch in self._names and not _WORD.match(ch) and not (s and _WORD.match(low[s - 1])):
                at(s, ch)
        return list(found)

    def differences(self) -> list[str]:
        """How this index differs from one built from scratch on the same store: empty
        when the change log missed nothing. For tests and the simulator."""
        self.fresh()
        ref = KnowledgeIndex(self.store)
        ref.fresh()

        def parts(ix: KnowledgeIndex) -> dict[str, object]:
            return {
                "live": ix.live,
                "volatile": ix._volatile,
                "by_scope": {k: v for k, v in ix.by_scope.items() if v},
                "by_entity": {k: v for k, v in ix.by_entity.items() if v},
                "names": {k: v for k, v in ix._names.items() if v},
                "links": {k: v for k, v in ix._links.items() if v},
            }

        mine, theirs = parts(self), parts(ref)
        out = [k for k in mine if mine[k] != theirs[k]]
        nodes = ref.live | {"e:" + e for e in ref.by_entity}
        out += [f"links of {n}" for n in sorted(nodes) if self.neighbours(n) != ref.neighbours(n)]
        return out + self.store.word_differences()

    # ------------------------------------------------------------------ graph

    def neighbours(self, node: str) -> list[tuple[str, float]]:
        """A node's strongest links (at most MAX_DEGREE): an item's links to recallable
        items and to its entities, an entity's links to its recallable items. An entity
        link weighs 1 / log2(2 + the entity's item count), so a hub passes on little."""
        got = self._adj.get(node)
        if got is not None:
            return got
        nb: list[tuple[str, float]]
        if node.startswith("e:"):
            items = self.by_entity.get(node[2:], set())
            w = 1.0 / math.log2(2 + len(items))
            nb = [(i, w) for i in items]
        elif node in self.live:
            nb = [(v, w) for v, w in self._links.get(node, {}).items() if v in self.live]
            nb += [
                ("e:" + e, 1.0 / math.log2(2 + len(self.by_entity[e])))
                for e in dict.fromkeys(self._entities_of[node])
            ]
        else:
            nb = []
        got = sorted(nb, key=lambda t: (-t[1], t[0]))[:MAX_DEGREE]
        self._adj[node] = got
        self._wdeg[node] = sum(w for _, w in got)
        return got

    def degree(self, node: str) -> float:
        if node not in self._wdeg:
            self.neighbours(node)
        return self._wdeg[node]

    def ppr(
        self, seeds: dict[str, float], alpha: float = ALPHA, eps: float = EPSILON
    ) -> dict[str, float]:
        """Forward push. Returns approximate PageRank mass per node.

        A seed with no edges (an item with no entities and no links) keeps the share a
        linked seed keeps for itself, `alpha` of its mass; the rest has nowhere to go.
        Dropping it instead would make such an item impossible to recall, and keeping all
        of it would rank it above every linked item."""
        total = sum(seeds.values())
        if total <= 0:
            return {}
        r: dict[str, float] = {u: w / total for u, w in seeds.items() if w > 0}
        p: dict[str, float] = defaultdict(float)
        queue = list(r)
        pushes = 0
        while queue and pushes < 50000:
            u = queue.pop()
            ru = r.get(u, 0.0)
            deg = self.degree(u)
            if deg <= 0:
                p[u] += alpha * ru
                r[u] = 0.0
                continue
            if ru < eps * deg:
                continue
            pushes += 1
            p[u] += alpha * ru
            spread = (1 - alpha) * ru
            r[u] = 0.0
            for v, w in self.neighbours(u):
                before = r.get(v, 0.0)
                after = before + spread * w / deg
                r[v] = after
                dv = self.degree(v)
                if before < eps * dv <= after:
                    queue.append(v)
        return dict(p)


def _cos(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False))  # index vectors are unit length


@dataclass(frozen=True)
class Here:
    """Where a prompt is asked: its project, and the topics the session is about. An item
    is recalled here when it is global, of this project, or about one of these topics."""

    project: str | None = None
    topics: frozenset[str] = frozenset()

    def sees(self, scope: str) -> bool:
        return scope == "global" or scope == self.project or scope in self.topics

    def scopes(self) -> list[str]:
        return ["global", *([self.project] if self.project else []), *self.topics]


def recall(
    store: KnowledgeStore,
    index: KnowledgeIndex,
    prompt: str,
    *,
    project: str | None = None,
    exclude: Iterable[str] = (),
    policy: RecallPolicy | None = None,
    vector: list[float] | None = None,
    standing: bool = False,
    topics: Iterable[str] = (),
) -> Recalled:
    """Items for this prompt. With `standing` (the first prompt of a session, or the first
    after a compaction) the user's preferences that apply here come first whatever the
    prompt says: "keep answers short" matters for every prompt but shares no words with
    them."""
    pol = policy or RecallPolicy()
    skip = set(exclude)
    here = Here(project, frozenset(topics))
    got = _ranked(store, index, prompt, here, skip, pol, vector)
    _add_open_items(store, prompt, here, skip, got, pol)
    if not standing or pol.standing <= 0:
        return got
    have = {i.item_id for i in got.items}
    now = clock.now()
    heads = store.heads
    prefs = store.get_items(
        sorted(
            (
                i
                for i in index.visible(here.scopes())
                if heads[i].type is ItemType.PREFERENCE and i not in skip and i not in have
            ),
            key=lambda i: -heads[i].vitality(now),
        )[: pol.standing]
    )
    if prefs:
        got.items = prefs + got.items
        for p in prefs:
            got.scores[p.item_id] = 1.0
            got.why[p.item_id] = ["standing preference"]
        if got.reason != "ok":
            got.reason = "standing preferences only"
    return got


# Prompts about what is left to do: they share no words with the open items they need.
_REMAINING = re.compile(
    r"\b(what'?s|what is|what are|anything|is there anything)\b[^?.!]{0,30}?"
    r"\b(left|next|remaining|pending|outstanding|unfinished|open|to ?do|todo)\b"
    r"|\bwhere (were|are|did) we\b|\bwhere we left off\b"
    r"|\b(remaining|outstanding|unfinished|pending|open) (work|items?|tasks?|to-?dos?|questions?)\b"
    r"|^\s*(continue|resume|keep going|carry on|pick up where)\b",
    re.I,
)
OPEN_ITEMS = 3


def asks_about_remaining_work(prompt: str) -> bool:
    return bool(_REMAINING.search(prompt))


def _name_forms(name: str) -> set[str]:
    base = name.lower()
    return {base, base.replace("_", "-"), base.replace("-", "_"), re.sub(r"[-_]", " ", base)}


def project_named(store: KnowledgeStore, prompt: str) -> str | None:
    """The project a prompt names, by its directory name ("graph_memory", "graph memory",
    "trip-planner"), among the projects memory has items for."""
    text = prompt.lower()
    keys = {h.scope for h in store.heads.values() if h.valid and h.scope.startswith("repo:")}
    for key in sorted(keys, key=len, reverse=True):
        name = key.removeprefix("repo:").rstrip("/").rsplit("/", 1)[-1]
        if len(name) >= 3 and any(
            re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", text) for f in _name_forms(name)
        ):
            return key
    return None


def _add_open_items(
    store: KnowledgeStore,
    prompt: str,
    here: Here,
    skip: set[str],
    got: Recalled,
    pol: RecallPolicy,
) -> None:
    """A prompt asking what is left ("what's next?", "where were we?") gets the newest open
    items of the project it names, else of the session's project, else of any project.
    No word matching needed: those prompts name no work."""
    if not asks_about_remaining_work(prompt):
        return
    target = project_named(store, prompt) or here.project
    heads = store.heads
    opens = store.get_items(
        sorted(
            (
                i
                for i, h in heads.items()
                if h.valid
                and h.type is ItemType.OPEN
                and i not in skip
                and (target is None or h.scope in (target, "global") or h.scope in here.topics)
            ),
            key=lambda i: heads[i].created_at,
            reverse=True,
        )[:OPEN_ITEMS]
    )
    if not opens:
        return
    first = {i.item_id for i in opens}  # they lead, even when ranking found them too
    rest = [i for i in got.items if i.item_id not in first]
    got.items = opens + rest[: max(0, pol.max_items - len(opens))]
    for i in opens:
        got.scores[i.item_id] = 1.0
        got.why[i.item_id] = ["open item: the prompt asks what is left"]
    if got.reason != "ok":
        got.reason = "open items only"


def _ranked(
    store: KnowledgeStore,
    index: KnowledgeIndex,
    prompt: str,
    here: Here,
    skip: set[str],
    pol: RecallPolicy,
    vector: list[float] | None,
) -> Recalled:
    started = time.perf_counter()
    index.fresh()

    heads = store.heads

    # only what this session may see can seed: a strong match out of scope (a SQLite
    # item, in a session that is not about SQLite) must neither open the gate below nor
    # push its mass along shared entities to unrelated items in scope
    visible = index.visible(here.scopes())
    vec = vector if vector is not None else store.embed(prompt)
    by_vector = store.similar(
        vec, "item", pol.vector_k, pol.vector_floor, among=visible, scopes=here.scopes()
    )
    raw = index.bm25(tokens(prompt), here.scopes())
    raw_words = [(i, s) for i, s, _ in raw]
    held = {i: n for i, _, n in raw}  # how many of the prompt's words each holds
    q = set(tokens(prompt))

    def covers(i: str) -> bool:
        n = held[i]
        return n >= 2 and n >= pol.min_coverage * len(q)

    # a hit on one common word is not a seed; ranking (RRF) and the mass floor below
    # already favour the best hits, so no cut relative to the best one
    # ... unless it covers much of the prompt (scores stay low while the store is small)
    by_words = [(i, s) for i, s in raw_words if s >= pol.min_word_seed or covers(i)]
    entities = [e for e in index.mentioned(prompt) if index.about(e) & visible]

    best_vec = by_vector[0][1] if by_vector else 0.0
    best_bm = by_words[0][1] if by_words else 0.0
    by_coverage = any(covers(i) for i, _ in raw_words[:3])
    if not entities and best_vec < pol.min_vector and best_bm < pol.min_bm25 and not by_coverage:
        return Recalled(
            [],
            {},
            {},
            [],
            "nothing in the prompt points into the graph",
            round((time.perf_counter() - started) * 1000, 2),
        )

    seeds: dict[str, float] = defaultdict(float)
    why: dict[str, list[str]] = defaultdict(list)
    for rank, (i, s) in enumerate(by_vector):
        seeds[i] += 1 / (RRF_K + rank)
        why[i].append(f"meaning {s:.2f}")
    for rank, (i, s) in enumerate(by_words):
        seeds[i] += 1 / (RRF_K + rank)
        why[i].append(f"words {s:.1f}")
    for e in entities:
        seeds["e:" + e] += 1 / RRF_K
    mass = index.ppr(dict(seeds))

    # a gentle prior from the item's life: a fresh survivor counts 0.6-0.8, an item
    # used or seen again often counts up to 1.0
    now = clock.now()
    scored = [
        (u, m * (0.6 + 0.4 * min(1.0, heads[u].vitality(now) / 3)))
        for u, m in mass.items()
        if u in heads and heads[u].recallable and here.sees(heads[u].scope)
    ]
    # the floor is relative to the best item in scope, shown before or not: excluding
    # the best one must not promote its weak neighbours
    top = max((m for _, m in scored), default=0.0)
    cands = [(i, m) for i, m in scored if i not in skip]
    if not cands:
        return Recalled(
            [],
            {},
            dict(seeds),
            [store.names[e].name for e in entities],
            "seeds lead to nothing in scope",
            round((time.perf_counter() - started) * 1000, 2),
        )
    cands.sort(key=lambda t: -t[1])
    cands = [(i, m / top) for i, m in cands if m >= pol.min_mass * top][: pol.max_items * 4]

    # MMR: relevance from PageRank, redundancy from the item vectors
    vecs = {i: store.vectors.get(i) for i, _ in cands}
    chosen: list[tuple[str, float]] = []
    pool = list(cands)
    used = 0
    while pool and len(chosen) < pol.max_items:

        def mmr(t: tuple[str, float]) -> float:
            sim = max((_cos(vecs[t[0]], vecs[c]) for c, _ in chosen), default=0.0)
            return pol.mmr_lambda * t[1] - (1 - pol.mmr_lambda) * sim

        best = max(pool, key=mmr)
        pool.remove(best)
        size = heads[best[0]].chars + 20
        if chosen and used + size > pol.max_chars:
            continue
        chosen.append(best)
        used += size
    for i, _ in chosen:
        if not why.get(i):
            why[i].append("linked")
    return Recalled(
        store.get_items(i for i, _ in chosen),
        {i: round(s, 3) for i, s in chosen},
        dict(seeds),
        [store.names[e].name for e in entities],
        "ok",
        round((time.perf_counter() - started) * 1000, 2),
        {i: why[i] for i, _ in chosen},
    )


def note(item: Item, here: str | None = None, now: datetime | None = None) -> str:
    """One recalled item as the agent sees it: what kind, when it was last confirmed,
    where it came from and what it applies to, then the statement and reason, the project
    when it is another one, and a warning when it is disputed or due for review.

        [learned, 2026-11-07, from docs, SQLite 3] SQLite foreign keys are off … (why: …)"""
    tags = [item.type.value, item.checked.date().isoformat()]
    if item.origin in ORIGIN_LABEL:
        tags.append(ORIGIN_LABEL[item.origin])
    topic = topic_of(item.scope)
    if topic and topic.lower() not in item.applies_to.lower():
        tags.append(topic)
    if item.applies_to:
        tags.append(item.applies_to)
    line = f"[{', '.join(tags)}] {item.statement}"
    if item.why:
        line += f" (why: {item.why})"
    if item.scope not in ("global", here) and not topic:
        line += f" [project: {project_title(item.scope)}]"
    if item.disputes and item.disputed_at is not None:
        reason = f": {item.dispute_reason}" if item.dispute_reason else ""
        line += (
            f" [disputed {item.disputed_at.date().isoformat()}{reason}; "
            "check it before relying on it]"
        )
    elif item.overdue(now):
        line += (
            f" [not re-checked since {item.checked.date().isoformat()}; "
            "verify it before relying on it]"
        )
    return line


# The reply used an item when it repeats enough of the item's distinctive words: half of
# them (at least two), or at least three that make up 30%. Measured on scenario replies:
# a reply that applies an item names 3 of its 8-9 distinctive words ("the SETNX lock in
# acquire()"); one that ignores it names 0-1.
# Four hits count whatever the item's length (a long statement from a real model has ~15
# distinctive words; "the SETNX lock ... every worker ... EXPIRE" names 4 of them).
USE_COVERAGE = 0.5
USE_MIN_HITS = 3
USE_PARTIAL = 0.3
USE_SURE_HITS = 4
# or: reply and item this close in meaning. Only near-restatements: with MiniLM a reply
# that ignores an item but works on the same file scored 0.56, one that used it 0.50.
USE_SIMILARITY = 0.7


def used_by(
    index: KnowledgeIndex,
    item: Item,
    reply_tokens: set[str],
    reply_vector: Sequence[float] | None,
    item_vector: Sequence[float] | None,
    df: Mapping[str, int] | None = None,
) -> bool:
    """Did the reply use this recalled item? Cheap evidence, no model: most of the item's
    distinctive words appear in the reply, or the reply is close to it in meaning. `df`:
    how many notes hold each word, when the caller read it for several items at once."""
    index.fresh()
    store = index.store
    n = max(1, store.doc_count)
    stmt = tokens(item.statement)
    if df is None:
        df = store.word_df(stmt)
    words = {t for t in stmt if df.get(t, 0) <= max(2, n // 4)}
    hits = len(words & reply_tokens)
    share = hits / len(words) if words else 0.0
    if (
        (hits >= 2 and share >= USE_COVERAGE)
        or (hits >= USE_MIN_HITS and share >= USE_PARTIAL)
        or hits >= USE_SURE_HITS
    ):
        return True
    if reply_vector is not None and item_vector is not None:
        return _cos(_unit(reply_vector), item_vector) >= USE_SIMILARITY
    return False


def _unit(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]
