from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from graph_memory.client import call
from graph_memory.config import load_config
from graph_memory.knowledge import Area, Item, ItemType, KnowledgeStore, Relation
from graph_memory.knowledge_recall import KnowledgeIndex, recall, tokens, used_by

from .conftest import topic_embed, turn
from .test_daemon import _start, _wait
from .test_graduation import LOCK, Script

NOW = datetime(2026, 11, 4, tzinfo=UTC)


def item(k: KnowledgeStore, text: str, **kw: object) -> Item:
    return k.add_item(Item(type=ItemType.FACT, statement=text, **kw))  # type: ignore[arg-type]


def test_vitality_decays_by_area_and_promotion(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a = item(k, "redis lua script for acquire", last_seen=NOW)
    assert a.area is Area.SURVIVOR and a.vitality(NOW) == 1.0
    assert abs(a.vitality(NOW + timedelta(days=14)) - 0.5) < 1e-9
    k.reinforce(a.item_id, "seg")  # seen again in a later session
    assert a.area is Area.OLD and a.strength == 2.0
    a.last_seen = NOW
    assert abs(a.vitality(NOW + timedelta(days=90)) - 1.0) < 1e-9  # old: 90-day half-life


def test_unused_survivors_go_dormant_and_come_back(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a = item(k, "redis lua script for acquire", last_seen=NOW)
    assert k.sweep(NOW + timedelta(days=20)) == []
    assert k.sweep(NOW + timedelta(days=30)) == [a.item_id]
    assert a.area is Area.DORMANT and not a.recallable
    assert recall(k, KnowledgeIndex(k), "redis lua acquire").items == []
    k.reinforce(a.item_id, "seg2")
    assert a.area is Area.OLD and a.recallable


def test_use_strengthens_and_links_items_used_together(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a = item(k, "redis lua script for acquire")
    b = item(k, "acquire retries five times with backoff")
    k.mark_used([a.item_id, b.item_id])
    assert a.uses == 1 and a.strength == 1.5 and a.area is Area.SURVIVOR
    k.mark_used([a.item_id, b.item_id])
    assert a.area is Area.OLD
    (edge,) = k.scan_edges()
    assert edge.relation is Relation.CO_USED and edge.weight == 0.3 and edge.origin == "co_use"


def test_used_by_words_or_meaning(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    a = item(k, "the limiter keeps counters in a Redis hash keyed by tenant")
    item(k, "the weather in Seoul is rainy")
    ix = KnowledgeIndex(k)
    reply = set(tokens("I updated the Redis hash so counters are keyed by tenant id"))
    assert used_by(ix, a, reply, None, None)
    assert not used_by(ix, a, set(tokens("Arsenal won 2-1")), None, None)
    vec = topic_embed("redis lua limiter acquire")
    assert used_by(ix, a, set(), vec, k.vectors.get(a.item_id))


def test_daemon_counts_use_after_the_reply(short_dir: Path) -> None:
    cfg = replace(load_config(short_dir), save_interval=0.05, knowledge_recall="shadow")
    daemon, t = _start(cfg, Script([LOCK]))
    try:
        for p in ("add a lock", "add retries"):
            call(cfg, "remember", {"session": "a", "turn": turn(p, "done")}, autostart=False)
        call(cfg, "graduate", {"session": "a"}, autostart=False)
        _wait(lambda: daemon.knowledge.stats()["segments"] == 1)
        call(cfg, "recall", {"session": "b", "prompt": "redis SETNX lock acquire"}, autostart=False)
        call(
            cfg,
            "remember",
            {
                "session": "b",
                "turn": turn(
                    "redis SETNX lock acquire",
                    "acquire() takes the Redis SETNX lock with 30s expiry",
                ),
            },
            autostart=False,
        )
        events = cfg.home / "events.jsonl"
        _wait(lambda: '"knowledge_use"' in events.read_text())
        use = next(json.loads(x) for x in events.read_text().splitlines() if '"knowledge_use"' in x)
        assert use["used"] and use["mode"] == "shadow"
        lock = daemon.knowledge.item(use["used"][0])
        assert lock.uses == 1 and "SETNX" in lock.statement
    finally:
        call(cfg, "shutdown", autostart=False)
        t.join(5)
