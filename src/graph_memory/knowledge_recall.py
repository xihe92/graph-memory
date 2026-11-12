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
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from . import clock
from .knowledge import (
    ORIGIN_LABEL,
    Item,
    ItemType,
    KnowledgeStore,
    Relation,
    normalize_name,
    topic_of,
)
from .projects import project_title

RRF_K = 60
ALPHA = 0.2  # restart probability
EPSILON = 1e-4
MAX_DEGREE = 20  # strongest edges kept per node
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-/]*[a-z0-9]|[a-z0-9]")
_STOP = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its "
    "me my no not of on or our so that the their them then there these this to was we "
    "what when where which who why will with you your should would could let lets just "
    "now also any all some more most other into than too very use used using get make "
    "want need like see show tell give go ok yes please after before between instead "
    "up down out over under again still only same such each both one two per re".split()
)


_KEEP_DOUBLE = ("ll", "ss", "zz")


def stem(word: str) -> str:
    """Light suffix stripping for plain words, so "tests" meets "test", "running" meets
    "run" and "raced" meets "race" (both "rac": a final e is dropped from every stem).
    Only consistency matters: the same function reads prompts, items and replies. Names
    with digits or punctuation (files, symbols) are left alone."""
    if len(word) < 4 or not word.isalpha():
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("ing", "ed")) and len(w) - (3 if w.endswith("ing") else 2) >= 3:
        w = w[:-3] if w.endswith("ing") else w[:-2]
        if len(w) > 3 and w[-1] == w[-2] and w[-2:] not in _KEEP_DOUBLE:
            w = w[:-1]
    elif w.endswith("s") and len(w) > 4 and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) >= 4 and w.endswith("e") and not w.endswith("ee"):
        w = w[:-1]
    return w


def tokens(text: str) -> list[str]:
    """Stemmed words without stop words. A hyphenated word also yields its parts, so
    "per-tenant" matches "tenant"; paths and dotted names stay whole."""
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t not in _STOP and len(t) > 1:
            out.append(stem(t))
        if "-" in t and "/" not in t and "." not in t:
            out.extend(stem(p) for p in t.split("-") if p not in _STOP and len(p) > 1)
    return out


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
    """Adjacency and BM25 statistics, rebuilt when the store changes."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self.version = -1
        self.adj: dict[str, list[tuple[str, float]]] = {}
        self.wdeg: dict[str, float] = {}
        self.docs: dict[str, Counter[str]] = {}
        self.df: Counter[str] = Counter()
        self.avg_len = 1.0
        self.names: list[tuple[re.Pattern[str], str]] = []
        self.by_entity: dict[str, set[str]] = defaultdict(set)

    def fresh(self) -> None:
        if self.version == self.store.version:
            return
        self.version = self.store.version
        s = self.store
        valid = {i.item_id: i for i in s.valid_items() if i.recallable}
        edges: dict[str, dict[str, float]] = defaultdict(dict)

        def link(a: str, b: str, w: float) -> None:
            if w <= 0:
                return
            edges[a][b] = max(edges[a].get(b, 0.0), w)
            edges[b][a] = max(edges[b].get(a, 0.0), w)

        degree = Counter(e for i in valid.values() for e in i.entities)
        self.by_entity = defaultdict(set)
        for i in valid.values():
            for e in i.entities:
                self.by_entity[e].add(i.item_id)
        for i in valid.values():
            for e in i.entities:
                link(i.item_id, "e:" + e, 1.0 / math.log2(2 + degree[e]))
        for edge in s.edges.values():
            if edge.relation is Relation.SUPERSEDES:
                continue
            if edge.source in valid and edge.target in valid:
                link(edge.source, edge.target, edge.weight)
        self.adj = {
            u: sorted(nb.items(), key=lambda t: -t[1])[:MAX_DEGREE] for u, nb in edges.items()
        }
        self.wdeg = {u: sum(w for _, w in nb) for u, nb in self.adj.items()}

        ents = s.entities
        self.docs = {}
        self.df = Counter()
        for i in valid.values():
            names = " ".join(ents[e].name for e in i.entities if e in ents)
            tf = Counter(tokens(f"{i.statement} {i.why} {names}"))
            self.docs[i.item_id] = tf
            self.df.update(tf.keys())
        lens = [sum(tf.values()) for tf in self.docs.values()]
        self.avg_len = (sum(lens) / len(lens)) if lens else 1.0

        self.names = []
        for ent in ents.values():
            if degree.get(ent.entity_id, 0) == 0:
                continue
            eid = ent.entity_id
            for n in {ent.name, *ent.aliases}:
                norm = normalize_name(n)
                if len(norm) < 4:
                    continue
                pat = re.compile(rf"(?<![\w/.-]){re.escape(norm)}(?![\w-]|\.\w)")
                self.names.append((pat, eid))
                base = norm.rsplit("/", 1)[-1]
                if "/" in norm and len(base) >= 5 and "." in base:  # a file named by basename
                    self.names.append(
                        (re.compile(rf"(?<![\w/.-]){re.escape(base)}(?![\w-]|\.\w)"), eid)
                    )

    # ------------------------------------------------------------------ seeds

    def about(self, entity_id: str) -> set[str]:
        """The recallable items about an entity."""
        return self.by_entity.get(entity_id, set())

    def bm25(
        self, query: Sequence[str], k: int = 10, among: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """The best word matches, among the given items when a set is given. Word
        statistics stay those of the whole store, so a score means the same in every
        scope."""
        n = len(self.docs)
        if not n:
            return []
        q = Counter(query)
        out = []
        for doc_id, tf in self.docs.items():
            if among is not None and doc_id not in among:
                continue
            dl = sum(tf.values())
            score = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                score += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / self.avg_len))
            if score > 0:
                out.append((doc_id, score))
        out.sort(key=lambda t: -t[1])
        return out[:k]

    def mentioned(self, text: str) -> list[str]:
        low = text.lower()
        return list(dict.fromkeys(eid for pat, eid in self.names if pat.search(low)))

    # ------------------------------------------------------------------ ppr

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
            deg = self.wdeg.get(u, 0.0)
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
            for v, w in self.adj[u]:
                before = r.get(v, 0.0)
                after = before + spread * w / deg
                r[v] = after
                dv = self.wdeg.get(v, 0.0)
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

    def sees(self, item: Item) -> bool:
        return item.scope == "global" or item.scope == self.project or item.scope in self.topics


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
    prefs = sorted(
        (
            i
            for i in store.valid_items()
            if i.type is ItemType.PREFERENCE
            and i.recallable
            and i.item_id not in skip
            and i.item_id not in have
            and here.sees(i)
        ),
        key=lambda i: -i.vitality(now),
    )[: pol.standing]
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
    keys = {i.scope for i in store.valid_items() if i.scope.startswith("repo:")}
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
    opens = sorted(
        (
            i
            for i in store.valid_items()
            if i.type is ItemType.OPEN
            and i.item_id not in skip
            and (target is None or i.scope in (target, "global") or i.scope in here.topics)
        ),
        key=lambda i: i.created_at,
        reverse=True,
    )[:OPEN_ITEMS]
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

    def in_scope(i: Item, skipped: bool = False) -> bool:
        return i.recallable and (skipped or i.item_id not in skip) and here.sees(i)

    # only what this session may see can seed: a strong match out of scope (a SQLite
    # item, in a session that is not about SQLite) must neither open the gate below nor
    # push its mass along shared entities to unrelated items in scope
    visible = {i.item_id for i in store.valid_items() if i.recallable and here.sees(i)}
    vec = vector if vector is not None else store.embed(prompt)
    by_vector = store.similar(vec, "item", pol.vector_k, pol.vector_floor, among=visible)
    raw_words = index.bm25(tokens(prompt), among=visible)
    q = set(tokens(prompt))

    def covers(i: str) -> bool:
        n = len(q & index.docs[i].keys())
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
        (store.items[u], m * (0.6 + 0.4 * min(1.0, store.items[u].vitality(now) / 3)))
        for u, m in mass.items()
        if u in store.items and in_scope(store.items[u], skipped=True)
    ]
    # the floor is relative to the best item in scope, shown before or not: excluding
    # the best one must not promote its weak neighbours
    top = max((m for _, m in scored), default=0.0)
    cands = [(i, m) for i, m in scored if i.item_id not in skip]
    if not cands:
        return Recalled(
            [],
            {},
            dict(seeds),
            [store.entities[e].name for e in entities],
            "seeds lead to nothing in scope",
            round((time.perf_counter() - started) * 1000, 2),
        )
    cands.sort(key=lambda t: -t[1])
    cands = [(i, m / top) for i, m in cands if m >= pol.min_mass * top][: pol.max_items * 4]

    # MMR: relevance from PageRank, redundancy from the item vectors
    vecs = {i.item_id: store.vectors.get(i.item_id) for i, _ in cands}
    chosen: list[tuple[Item, float]] = []
    pool = list(cands)
    used = 0
    while pool and len(chosen) < pol.max_items:

        def mmr(t: tuple[Item, float]) -> float:
            sim = max((_cos(vecs[t[0].item_id], vecs[c.item_id]) for c, _ in chosen), default=0.0)
            return pol.mmr_lambda * t[1] - (1 - pol.mmr_lambda) * sim

        best = max(pool, key=mmr)
        pool.remove(best)
        size = len(best[0].statement) + len(best[0].why) + 20
        if chosen and used + size > pol.max_chars:
            continue
        chosen.append(best)
        used += size
    for it, _ in chosen:
        if not why.get(it.item_id):
            why[it.item_id].append("linked")
    return Recalled(
        [i for i, _ in chosen],
        {i.item_id: round(s, 3) for i, s in chosen},
        dict(seeds),
        [store.entities[e].name for e in entities],
        "ok",
        round((time.perf_counter() - started) * 1000, 2),
        {i.item_id: why[i.item_id] for i, _ in chosen},
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
) -> bool:
    """Did the reply use this recalled item? Cheap evidence, no model: most of the item's
    distinctive words appear in the reply, or the reply is close to it in meaning."""
    index.fresh()
    n = max(1, len(index.docs))
    words = {t for t in tokens(item.statement) if index.df.get(t, 0) <= max(2, n // 4)}
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
