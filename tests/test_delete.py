"""Deleting knowledge by hand: edges go with the items, neighbours linked only through
them get a weaker bridge, entities no item names go too, and a deletion can be undone."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_memory.knowledge import (
    BRIDGE_MAX,
    Entity,
    EntityKind,
    Item,
    ItemType,
    KnowledgeStore,
    Relation,
)

from .conftest import topic_embed


def _store(tmp_path: Path) -> tuple[KnowledgeStore, dict[str, str]]:
    """a → x → b by depends_on, x — c same_area; a and c share a specific entity."""
    k = KnowledgeStore(tmp_path, topic_embed)
    ents = {
        n: k.add_entity(Entity(name=n, kind=EntityKind.TOOL)).entity_id
        for n in ("api", "queue", "db", "cache", "lonely")
    }
    ids = {}
    for name, about in (
        ("a", ["api", "cache"]),
        ("x", ["queue", "lonely"]),
        ("b", ["db"]),
        ("c", ["cache"]),
    ):
        ids[name] = k.add_item(
            Item(
                type=ItemType.FACT,
                statement=f"item {name} statement",
                entities=[ents[e] for e in about],
            )
        ).item_id
    k.add_edge(ids["a"], ids["x"], Relation.DEPENDS_ON, 0.9)
    k.add_edge(ids["x"], ids["b"], Relation.DEPENDS_ON, 0.6)
    k.add_edge(ids["x"], ids["c"], Relation.SAME_AREA, 0.3)
    return k, ids


def _edge(k: KnowledgeStore, a: str, b: str) -> list[tuple[str, str, str, float, str]]:
    """Edges between a and b; same_area has no direction, so it reads as (a, b)."""
    return [
        (
            *((a, b) if e.relation is Relation.SAME_AREA else (e.source, e.target)),
            e.relation.value,
            e.weight,
            e.origin,
        )
        for e in k.scan_edges()
        if {e.source, e.target} == {a, b}
    ]


def test_delete_bridges_the_neighbours_and_undelete_puts_it_back(tmp_path: Path) -> None:
    k, ids = _store(tmp_path)
    a, x, b, c = ids["a"], ids["x"], ids["b"], ids["c"]
    before_edges = dict(k.links)
    batch = k.delete([x], reason="noise")

    assert not k.has(x) and k.vectors.get(x) is None
    assert not any(x in (e.source, e.target) for e in k.scan_edges())
    # a depends on x, x depends on b: a depends on b, as strong as both links together
    assert _edge(k, a, b) == [(a, b, "depends_on", 0.54, "bridge")]
    # b and c: a weak same_area bridge (0.6 x 0.3)
    assert _edge(k, b, c) == [(b, c, "same_area", 0.18, "bridge")]
    # a and c share the specific entity "cache": already close, no bridge
    assert _edge(k, a, c) == []
    # the entities only x named are gone, the others stay
    names = {e.name for e in k.scan_entities()}
    assert "queue" not in names and "lonely" not in names and "cache" in names
    assert k.entity_by_name("queue") is None
    assert len(k.deleted()) == 1 and batch["reason"] == "noise"

    k.save()
    again = KnowledgeStore(tmp_path, topic_embed)
    again.undelete(batch["batch"])
    assert again.has(x) and again.vectors.get(x) is not None
    assert dict(again.links) == before_edges  # bridges gone
    assert again.entity_by_name("queue") is not None
    assert again.deleted() == []
    with pytest.raises(KeyError):
        again.undelete(batch["batch"])


def test_a_deleted_chain_is_bridged_end_to_end(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path)
    a, x, y, b = (
        k.add_item(Item(type=ItemType.FACT, statement=f"chain item {n}")).item_id for n in "axyb"
    )
    k.add_edge(a, x, Relation.CAUSES, 0.9)
    k.add_edge(x, y, Relation.CAUSES, 0.9)
    k.add_edge(y, b, Relation.CAUSES, 0.9)
    k.delete([x, y])
    assert _edge(k, a, b) == [(a, b, "causes", 0.81, "bridge")]


def test_mixed_or_reversed_relations_bridge_as_same_area(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path)
    a, x, b = (
        k.add_item(Item(type=ItemType.FACT, statement=f"mixed item {n}")).item_id for n in "axb"
    )
    k.add_edge(a, x, Relation.DEPENDS_ON, 0.9)
    k.add_edge(b, x, Relation.DEPENDS_ON, 0.9)  # both depend on x: no chain from a to b
    k.delete([x])
    assert _edge(k, a, b) == [(a, b, "same_area", 0.81, "bridge")]


def test_bridges_are_capped_and_skip_weak_pairs(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path)
    hub = k.add_item(Item(type=ItemType.FACT, statement="a well connected item")).item_id
    spokes = [
        k.add_item(Item(type=ItemType.FACT, statement=f"spoke number {n}")).item_id
        for n in range(10)
    ]
    for n, s in enumerate(spokes):
        k.add_edge(hub, s, Relation.SAME_AREA, 0.9 if n < 8 else 0.1)
    k.delete([hub])
    bridges = [e for e in k.scan_edges() if e.origin == "bridge"]
    assert len(bridges) == BRIDGE_MAX  # not 45 pairs
    assert all(e.weight == 0.81 for e in bridges)  # the strongest first


def test_no_bridge_through_superseded_items_and_replacement_links_survive(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path)
    old = k.add_item(Item(type=ItemType.DECISION, statement="use a SETNX lock for the limiter"))
    new = k.add_item(Item(type=ItemType.DECISION, statement="use a Lua script for the limiter"))
    k.invalidate(old.item_id, new, "replaced")
    batch = k.delete([new.item_id])
    assert batch["bridges"] == [] and k.item(old.item_id).invalidated_by is None
    k.undelete(batch["batch"])
    assert k.item(old.item_id).invalidated_by == new.item_id
    assert (new.item_id, old.item_id, "supersedes") in k.links


def test_deleting_nothing_known_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        KnowledgeStore(tmp_path).delete(["nope"])
