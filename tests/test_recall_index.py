"""The recall index updates itself from the store's change log instead of rebuilding.
After any sequence of changes it must equal an index built from scratch."""

from __future__ import annotations

import random
import re
from datetime import timedelta
from pathlib import Path

import pytest

from graph_memory import clock
from graph_memory.knowledge import (
    Entity,
    EntityKind,
    Item,
    ItemType,
    KnowledgeStore,
    Relation,
    Volatility,
    normalize_name,
)
from graph_memory.knowledge_recall import KnowledgeIndex, tokens

from .conftest import topic_embed

WORDS = (
    "redis lock retry lua script limiter cache tags recipe sqlite join index tests deploy".split()
)
SCOPES = ["global", "repo:/w/a", "repo:/w/b", "topic:SQLite", "topic:Redis"]


@pytest.mark.parametrize("seed", [4, 11, 29])
def test_any_sequence_of_changes_leaves_the_index_as_a_rebuild_would(
    tmp_path: Path, seed: int
) -> None:
    rng = random.Random(seed)
    k = KnowledgeStore(tmp_path, topic_embed)
    ix = KnowledgeIndex(k)
    ents = [
        k.add_entity(Entity(name=n, kind=EntityKind.OTHER))
        for n in ("svc/limiter.py", "Redis", "recipe_tag", "Claude Code", ".env.local")
    ]
    items: list[Item] = []

    def add() -> None:
        it = Item(
            type=rng.choice(list(ItemType)),
            statement=" ".join(rng.choices(WORDS, k=6)),
            why=" ".join(rng.choices(WORDS, k=3)),
            scope=rng.choice(SCOPES),
            entities=[e.entity_id for e in rng.sample(ents, rng.randint(0, 2))],
            volatility=rng.choice(list(Volatility)),
        )
        items.append(k.add_item(it))

    for _ in range(12):
        add()
    ix.fresh()
    try:
        for step in range(400):
            pick = rng.choice(items)
            a, b = rng.sample(items, 2)
            op = rng.randrange(13)
            if op == 0:
                add()
            elif op == 1:
                k.add_edge(a.item_id, b.item_id, rng.choice(list(Relation)), rng.random())
            elif op == 2:
                k.mark_used([a.item_id, b.item_id])
            elif op == 3:
                k.retract(pick.item_id, "test")
            elif op == 4:
                k.restore(pick.item_id)
            elif op == 5:
                k.dispute(pick.item_id, "test")
            elif op == 6:
                k.reinforce(pick.item_id, "seg")
            elif op == 7:
                clock.advance(days=rng.choice([1, 20, 40]))
                k.sweep()
            elif op == 8:
                k.confirm(pick.item_id)
            elif op == 9:
                pick.scope = rng.choice(SCOPES)  # as rescope and `gm knowledge scope` do
                k.touch(pick.item_id)
            elif op == 10:
                k.invalidate(a.item_id, b, "replaced")
            elif op == 11:  # a deletion may have taken an entity no item names any more
                alive = [e for e in ents if e.entity_id in k.entities]
                if alive:
                    k.merge_entity(rng.choice(alive).entity_id, f"alias {step}")
            elif op == 12 and len(items) > 6 and step % 40 == 0:
                gone = k.delete([pick.item_id], "noise")
                items.remove(pick)
                if rng.random() < 0.5:
                    k.undelete(gone["batch"])
                    items.append(k.items[pick.item_id])
            if step % 3 == 0:  # the index reads the log now and then, not after each change
                assert ix.differences() == [], f"after step {step}, op {op}"
        assert ix.differences() == []
    finally:
        clock.reset()


def test_only_the_changed_items_are_redone(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a = k.add_item(Item(type=ItemType.FACT, statement="redis lua script"))
    b = k.add_item(Item(type=ItemType.FACT, statement="sqlite join table"))
    ix = KnowledgeIndex(k)
    ix.fresh()
    doc_b = ix.docs[b.item_id]
    k.mark_used([a.item_id])
    ix.fresh()
    assert ix.docs[b.item_id] is doc_b  # untouched: not rebuilt
    k.retract(a.item_id)
    ix.fresh()
    assert a.item_id not in ix.live and a.item_id not in ix.docs
    assert ix.bm25(tokens("redis")) == [] and ix.bm25(tokens("sqlite"))[0][0] == b.item_id


def test_entity_names_match_as_whole_words(tmp_path: Path) -> None:
    """The name lookup gives what one regex per name gave before it."""
    k = KnowledgeStore(tmp_path, topic_embed)
    names = ["svc/limiter.py", "Redis", "recipe_tag", "Claude Code", ".env.local", "pg"]
    ents = {n: k.add_entity(Entity(name=n, kind=EntityKind.OTHER)) for n in names}
    k.add_item(
        Item(type=ItemType.FACT, statement="x", entities=[e.entity_id for e in ents.values()])
    )
    ix = KnowledgeIndex(k)
    ix.fresh()

    def by_regex(text: str) -> set[str]:
        out = set()
        for n, e in ents.items():
            norm = normalize_name(n)
            forms = [norm] + ([norm.rsplit("/", 1)[-1]] if n == "svc/limiter.py" else [])
            for f in forms:
                if len(norm) >= 4 and re.search(
                    rf"(?<![\w/.-]){re.escape(f)}(?![\w-]|\.\w)", text.lower()
                ):
                    out.add(e.entity_id)
        return out

    for text in (
        "redis is down",
        "redis-cli is down",  # not Redis: a longer name
        "use redis.",  # sentence end counts
        "the redis.conf file",  # not Redis: redis.conf
        "fix svc/limiter.py now",
        "fix limiter.py now",  # by the file's own name
        "fix src/svc/limiter.py",  # not: part of a longer path
        "set it in .env.local please",
        "set it in prod.env.local",
        "claude code hooks",
        "claude  code",  # two spaces: not the name
        "recipe_tag and recipe_tags",
        "pg is too short to count",
    ):
        assert set(ix.mentioned(text)) == by_regex(text), text


def test_a_volatile_item_drops_out_when_it_falls_due(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    v = k.add_item(
        Item(type=ItemType.FACT, statement="latest is 8.2", volatility=Volatility.VOLATILE)
    )
    ix = KnowledgeIndex(k)
    ix.fresh()
    assert ix.visible(["global"]) == {v.item_id}
    try:
        clock.advance(timedelta(days=31))
        assert ix.visible(["global"]) == set()  # with no change logged
    finally:
        clock.reset()
