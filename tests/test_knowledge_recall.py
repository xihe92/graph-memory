from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from graph_memory.client import call
from graph_memory.config import load_config
from graph_memory.knowledge import Entity, EntityKind, Item, ItemType, KnowledgeStore, Relation
from graph_memory.knowledge_recall import KnowledgeIndex, RecallPolicy, recall, tokens

from .conftest import topic_embed, turn
from .test_daemon import _start, _wait
from .test_graduation import LOCK, Script


def build(tmp_path: Path) -> tuple[KnowledgeStore, dict[str, str]]:
    k = KnowledgeStore(tmp_path / "k", topic_embed)
    limiter = k.add_entity(Entity(name="svc/limiter.py", kind=EntityKind.FILE))
    proj = k.add_entity(Entity(name="limiter", kind=EntityKind.PROJECT))
    ids = {}

    def add(key: str, type_: ItemType, text: str, ents: list[str], scope: str = "global") -> None:
        ids[key] = k.add_item(Item(type=type_, statement=text, entities=ents, scope=scope)).item_id

    add(
        "lua",
        ItemType.DECISION,
        "The rate limiter uses a Redis Lua script for acquire",
        [limiter.entity_id, proj.entity_id],
    )
    # shares no word with the prompts below; reachable only through the file entity
    add(
        "tz",
        ItemType.PITFALL,
        "Clock skew between hosts breaks window maths in svc/limiter.py",
        [limiter.entity_id],
    )
    add("eslint", ItemType.FACT, "The monorepo shares one eslint config with prettier", [])
    add(
        "other",
        ItemType.FACT,
        "Redis lock retries use backoff in another repo",
        [],
        scope="repo:/elsewhere",
    )
    k.add_edge(ids["lua"], ids["tz"], Relation.CAUSES, 0.6)
    return k, ids


def test_pagerank_reaches_items_through_entities(tmp_path: Path) -> None:
    k, ids = build(tmp_path)
    got = recall(k, KnowledgeIndex(k), "redis lua acquire is slow", project="repo:/w/limiter")
    found = [i.item_id for i in got.items]
    assert found[0] == ids["lua"] and ids["tz"] in found
    assert ids["eslint"] not in found and ids["other"] not in found  # unrelated; other scope


def test_naming_a_file_seeds_its_entity(tmp_path: Path) -> None:
    k, ids = build(tmp_path)
    got = recall(k, KnowledgeIndex(k), "open limiter.py please")  # basename only
    assert got.entities == ["svc/limiter.py"]
    assert {ids["lua"], ids["tz"]} <= {i.item_id for i in got.items}


def test_nothing_for_unrelated_prompts_and_exclusions(tmp_path: Path) -> None:
    k, ids = build(tmp_path)
    ix = KnowledgeIndex(k)
    none = recall(k, ix, "who won the football match")
    assert none.items == [] and "nothing" in none.reason
    got = recall(k, ix, "redis lua acquire", exclude=[ids["lua"]])
    assert ids["lua"] not in [i.item_id for i in got.items]


def test_index_follows_changes_and_invalid_items_drop(tmp_path: Path) -> None:
    k, ids = build(tmp_path)
    ix = KnowledgeIndex(k)
    recall(k, ix, "redis lua")
    new = k.add_item(
        Item(type=ItemType.DECISION, statement="The limiter uses a token bucket in Redis")
    )
    k.invalidate(ids["lua"], new)
    got = recall(k, ix, "redis lua acquire")
    assert ids["lua"] not in [i.item_id for i in got.items]


def test_excluding_the_best_item_does_not_promote_weak_neighbours(tmp_path: Path) -> None:
    k, ids = build(tmp_path)
    ix = KnowledgeIndex(k)
    first = recall(k, ix, "redis lua acquire", project="repo:/w/limiter")
    assert first.items[0].item_id == ids["lua"]
    weak = {i.item_id for i in first.items} - {ids["lua"]}
    again = recall(k, ix, "redis lua acquire", project="repo:/w/limiter", exclude=[ids["lua"]])
    # the floor stays relative to the excluded best item: nothing new gets in
    assert {i.item_id for i in again.items} <= weak


def test_budget_and_tokens(tmp_path: Path) -> None:
    k, _ = build(tmp_path)
    got = recall(k, KnowledgeIndex(k), "redis lua limiter.py", policy=RecallPolicy(max_items=1))
    assert len(got.items) == 1
    assert tokens("Fix the svc/limiter.py race!") == ["fix", "svc/limiter.py", "rac"]


def test_daemon_logs_knowledge_recall_in_shadow_and_shows_it_when_on(short_dir: Path) -> None:
    cfg = replace(load_config(short_dir), save_interval=0.05, knowledge_recall="on")
    daemon, t = _start(cfg, Script([LOCK]))
    try:
        for p in ("add a lock", "add retries"):
            call(cfg, "remember", {"session": "a", "turn": turn(p, "done")}, autostart=False)
        call(cfg, "graduate", {"session": "a"}, autostart=False)
        _wait(lambda: daemon.knowledge.stats()["segments"] == 1)
        # session a learned these itself: not shown back to it
        r = call(cfg, "recall", {"session": "a", "prompt": "redis lock acquire"}, autostart=False)
        assert not r["context"].get("knowledge")
        r = call(cfg, "recall", {"session": "b", "prompt": "redis lock acquire"}, autostart=False)
        assert any("SETNX" in line for line in r["context"]["knowledge"])
        again = call(
            cfg, "recall", {"session": "b", "prompt": "redis lock acquire"}, autostart=False
        )
        assert not again["context"].get("knowledge")  # already shown in this session
        events = [json.loads(x) for x in (cfg.home / "events.jsonl").read_text().splitlines()]
        logged = [e for e in events if e["kind"] == "knowledge_recall"]
        assert len(logged) == 3 and logged[1]["items"] and logged[1]["mode"] == "on"
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)


def test_an_item_with_no_entities_or_links_can_be_recalled(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    lone = k.add_item(Item(type=ItemType.FACT, statement="the limiter uses a redis lua script"))
    got = recall(k, KnowledgeIndex(k), "why does the redis lua script in the limiter fail?")
    assert [i.item_id for i in got.items] == [lone.item_id] and got.reason == "ok"


def test_pagerank_gives_an_unlinked_seed_only_its_own_share(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    redis = k.add_entity(Entity(name="Redis", kind=EntityKind.TOOL))
    linked = k.add_item(Item(type=ItemType.FACT, statement="redis lua", entities=[redis.entity_id]))
    k.add_item(
        Item(type=ItemType.FACT, statement="redis runs in docker", entities=[redis.entity_id])
    )
    lone = k.add_item(Item(type=ItemType.FACT, statement="redis lua too"))
    ix = KnowledgeIndex(k)
    ix.fresh()
    mass = ix.ppr({linked.item_id: 1.0, lone.item_id: 1.0})
    # alpha (0.2) of its half: no more than the linked seed keeps for itself, before the
    # mass that flows back to it through its entity
    assert abs(mass[lone.item_id] - 0.1) < 1e-9
    assert mass[linked.item_id] > mass[lone.item_id]
    assert ix.ppr({lone.item_id: 0.0}) == {}
