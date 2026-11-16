"""Topic scopes: knowledge about one technology is recalled when a session is about it,
in any project, and not otherwise. Extraction picks the scope; `gm knowledge rescope`
has the model fix the scope of what was learned before; `gm knowledge scope` by hand."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from graph_memory import cli
from graph_memory.client import RemoteError, call
from graph_memory.config import Config
from graph_memory.daemon import scope_names
from graph_memory.graduation import SegmentInput, build_extract_prompt, graduate
from graph_memory.knowledge import (
    Item,
    ItemType,
    KnowledgeStore,
    named_topics,
    topic_of,
    topic_scope,
)
from graph_memory.knowledge_recall import KnowledgeIndex, note, recall

from .conftest import topic_embed
from .test_daemon_buffer import _remember, cfg, daemon_with  # noqa: F401
from .test_graduation import T0, Script, seg

REPO = "repo:/w/limiter"
ONEWAY = "SQLite enforces foreign keys only after PRAGMA foreign_keys = ON: they are off by default"


def _no_project(*pairs: tuple[str, str]) -> SegmentInput:
    s = seg(*pairs)
    return replace(s, project=None, project_name=None)


def _extract(*items: dict[str, Any]) -> dict[str, Any]:
    return {
        "entities": [{"name": "SQLite", "kind": "tool", "summary": "embedded SQL database"}],
        "items": [
            {"type": "learned", "entities": ["SQLite"], "evidence": ["X1"], **it} for it in items
        ],
    }


# --------------------------------------------------------------------------- names


def test_topic_names_and_matching() -> None:
    assert topic_scope(" SQLite ") == "topic:SQLite" and topic_scope("") is None
    assert topic_scope("topic:Redis") == "topic:Redis"
    assert topic_of("topic:SQLite") == "SQLite" and topic_of(REPO) is None
    scopes = ["topic:SQLite", "topic:Claude Code", "topic:Go", REPO, "global"]
    assert named_topics(scopes, "how does sqlite do foreign keys?") == {"topic:SQLite"}
    assert named_topics(scopes, "a claude-code hook") == {"topic:Claude Code"}
    assert named_topics(scopes, "ClaudeCode settings") == {"topic:Claude Code"}
    assert named_topics(scopes, "pysqlite bindings") == set()  # whole words only
    assert named_topics(scopes, "written in Go") == {"topic:Go"}


def test_the_store_reuses_a_topic_spelling(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    k.add_item(Item(type=ItemType.FACT, statement=ONEWAY, scope="topic:SQLite"))
    assert k.topics() == {"sqlite": "topic:SQLite"}
    assert k.topic_key("SQLITE") == "topic:SQLite" and k.topic_key("Redis") == "topic:Redis"


# --------------------------------------------------------------------------- extraction


def test_extraction_scopes_items_to_a_topic(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    s = _no_project(("does sqlite check foreign keys?", "No, they are off by default."))
    prompt = build_extract_prompt(s, list(enumerate(s.exchanges)), [], topics=["SQLite"])
    assert '"topic"' in prompt and "Known topics: SQLite" in prompt
    assert "these exchanges are in no project" in prompt
    with_project = build_extract_prompt(seg(("q", "a")), [(0, seg(("q", "a")).exchanges[0])], [])
    assert "only matters for the project limiter" in with_project
    ask = Script(
        [
            _extract(
                {"statement": ONEWAY, "scope": "topic", "topic": "SQLite"},
                # "project" with no project: the topic it named
                {
                    "statement": "SQLite keeps a whole database in one file",
                    "scope": "project",
                    "topic": "sqlite",
                },  # fmt: skip
                # a topic with no name: global
                {"statement": "The user reads docs before trusting answers", "scope": "topic"},
            )
        ]
    )
    graduate(k, s, ask, topic_embed)
    by = {i.statement: i.scope for i in k.scan()}
    assert by[ONEWAY] == "topic:SQLite"
    assert by["SQLite keeps a whole database in one file"] == "topic:SQLite"  # one spelling
    assert by["The user reads docs before trusting answers"] == "global"


def test_a_project_item_stays_in_its_project_and_a_global_one_narrows(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    old = k.add_item(Item(type=ItemType.LEARNED, statement=ONEWAY, scope="global"))
    ask = Script(
        [
            _extract(
                {"statement": ONEWAY, "scope": "topic", "topic": "SQLite"},
                {"statement": "limiter stores counters in SQLite", "scope": "project"},
            )
        ],
        merges=[{"items": [{"id": "N1", "duplicate_of": "K1"}]}],
    )
    graduate(k, seg(("sqlite foreign keys?", "off by default")), ask, topic_embed)
    assert old.scope == "topic:SQLite"  # restated as about SQLite: recalled only there now
    new = next(i for i in k.scan() if i.statement.startswith("limiter"))
    assert new.scope == REPO


# --------------------------------------------------------------------------- recall


def test_a_topic_item_is_recalled_only_when_the_session_is_about_it(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path, topic_embed)
    it = k.add_item(Item(type=ItemType.LEARNED, statement=ONEWAY, scope="topic:SQLite"))
    index = KnowledgeIndex(k)
    q = "does a fresh connection check foreign keys?"
    assert it not in recall(k, index, q, project=REPO).items
    got = recall(k, index, q, project=REPO, topics={"topic:SQLite"})
    assert it in got.items
    assert note(it, REPO).startswith("[learned, ") and ", SQLite] " in note(it, REPO)
    assert "[project:" not in note(it, REPO)


def test_a_match_out_of_scope_neither_seeds_nor_opens_the_gate(tmp_path: Path) -> None:
    """A SQLite item matches the prompt, in a session that is not about SQLite: it must not
    be shown, and must not pass its relevance to an item in scope that shares its entity."""
    from graph_memory.knowledge import Entity, EntityKind

    k = KnowledgeStore(tmp_path, topic_embed)
    doc = k.add_entity(Entity(name="tag index", kind=EntityKind.OTHER))
    hidden = k.add_item(
        Item(
            type=ItemType.LEARNED, statement=ONEWAY, scope="topic:SQLite", entities=[doc.entity_id]
        )
    )
    bystander = k.add_item(
        Item(
            type=ItemType.PREFERENCE,
            statement="limiter answers are written in plain terms",
            scope=REPO,
            entities=[doc.entity_id],
        )
    )
    index = KnowledgeIndex(k)
    q = "does a sqlite connection check foreign keys by default?"
    got = recall(k, index, q, project=REPO)
    assert got.items == [] and got.reason == "nothing in the prompt points into the graph"
    about = recall(k, index, q, project=REPO, topics={"topic:SQLite"})
    assert hidden in about.items
    # the entity itself is named: it seeds only through items the session may see
    named = recall(k, index, "what is the tag index?", project=REPO)
    assert named.items == [bystander]
    elsewhere = recall(k, index, "what is the tag index?", project="repo:/w/other")
    assert elsewhere.items == []


def test_the_daemon_finds_the_session_topics(cfg: Config) -> None:  # noqa: F811
    with daemon_with(cfg, None) as d:
        d.knowledge.add_item(Item(type=ItemType.LEARNED, statement=ONEWAY, scope="topic:SQLite"))
        d.kindex.fresh()
        q = "does a fresh connection check foreign keys?"
        res = call(cfg, "recall", {"session": "s1", "prompt": q}, autostart=False)
        assert not res["context"].get("knowledge")
        # the session's last reply named SQLite: it is about SQLite now
        _remember(cfg, "s1", "which database for the cache?", "SQLite, for its single file.")
        res = call(cfg, "recall", {"session": "s1", "prompt": q}, autostart=False)
        assert any("PRAGMA" in ln for ln in res["context"]["knowledge"])
        ev = [
            json.loads(x)
            for x in (cfg.home / "events.jsonl").read_text().splitlines()
            if '"knowledge_recall"' in x
        ]
        assert ev[-1]["topics"] == ["SQLite"] and ev[0]["topics"] == []


# --------------------------------------------------------------------------- rescope


def test_rescope_proposes_then_applies(
    cfg: Config,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GM_HOME", str(cfg.home))
    answer = {
        "items": [
            {"id": "K1", "scope": "topic", "topic": "SQLite"},
            {"id": "K2", "scope": "project", "project": "P1"},
            {"id": "K3", "scope": "global"},
            {"id": "K9", "scope": "global"},  # no such item
        ]
    }
    with daemon_with(cfg, Script([], merges=[answer])) as d:
        k = d.knowledge
        a = k.add_item(Item(type=ItemType.LEARNED, statement=ONEWAY))
        b = k.add_item(Item(type=ItemType.FACT, statement="limiter keeps counters in Redis"))
        c = k.add_item(Item(type=ItemType.PREFERENCE, statement="answer briefly", scope=REPO))
        for n, i in enumerate((a, b, c)):
            k.edit(i.item_id).created_at = T0.replace(minute=n)
            k.touch(i.item_id)
        assert cli.main(["knowledge", "rescope", "--wait"]) == 0
        out = capsys.readouterr().out
        assert "-> topic SQLite (1)" in out and "-> project limiter (1)" in out
        assert "-> global (1)" in out and "3 items would move" in out
        assert k.heads[a.item_id].scope == "global"  # nothing changed yet
        k.edit(b.item_id).scope = "topic:Redis"  # changed since the preview: kept
        k.touch(b.item_id)
        assert cli.main(["knowledge", "rescope", "--apply"]) == 0
        out = capsys.readouterr().out
        assert "rescoped 2 items; 1 changed since the preview kept theirs" in out
        scopes = tuple(k.heads[i.item_id].scope for i in (a, b, c))
        assert scopes == ("topic:SQLite", "topic:Redis", "global")
        assert not (cfg.home / "rescope.json").exists()
        assert cli.main(["knowledge", "rescope", "--apply"]) == 1


def test_scope_by_hand(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setenv("GM_HOME", str(cfg.home))
    with daemon_with(cfg, None) as d:
        k = d.knowledge
        a = k.add_item(Item(type=ItemType.LEARNED, statement=ONEWAY))
        k.add_item(Item(type=ItemType.FACT, statement="limiter uses redis", scope=REPO))
        k.add_item(Item(type=ItemType.FACT, statement="sqlite is small", scope="topic:SQLite"))
        assert cli.main(["knowledge", "scope", a.item_id[:8], "--to", "topic:sqlite"]) == 0
        assert k.heads[a.item_id].scope == "topic:SQLite"
        assert cli.main(["knowledge", "scope", a.item_id[:8], "--to", "limiter"]) == 0
        assert k.heads[a.item_id].scope == REPO
        assert cli.main(["knowledge", "scope", a.item_id[:8], "--to", "global"]) == 0
        assert k.heads[a.item_id].scope == "global"
        assert cli.main(["knowledge", "scope", a.item_id[:8], "--to", "nowhere"]) == 1
        assert cli.main(["knowledge", "scope", a.item_id[:8]]) == 2
        with pytest.raises(RemoteError):
            call(cfg, "knowledge_scope", {"ids": [], "to": "global"}, autostart=False)


def test_scope_names_list_topics_last() -> None:
    names = scope_names(["topic:sqlite", REPO, "global", "topic:Redis", "topic:sqlite"])
    assert [(x["key"], x["name"], x["items"]) for x in names] == [
        ("global", "global", 1),
        (REPO, "limiter", 1),
        ("topic:Redis", "Redis", 1),
        ("topic:sqlite", "sqlite", 2),
    ]
    assert names[-1]["topic"] and "topic" not in names[1]
