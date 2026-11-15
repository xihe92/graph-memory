"""Slow: the real embedder (all-MiniLM-L6-v2) and the real graduation model through the
claude CLI. Run with `pytest -m slow`."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from graph_memory.graduation import Exchange, SegmentInput, graduate
from graph_memory.knowledge import Entity, Item, ItemType, KnowledgeStore
from graph_memory.knowledge_recall import KnowledgeIndex, recall
from graph_memory.models import claude_code_model

pytestmark = pytest.mark.slow
T0 = datetime(2026, 11, 1, 10, tzinfo=UTC)


@pytest.fixture(scope="module")
def embedder():  # type: ignore[no-untyped-def]
    from graph_memory.embedding import default_embedder

    return default_embedder()


def test_recall_by_meaning_with_the_real_embedder(tmp_path: Path, embedder) -> None:  # type: ignore[no-untyped-def]
    k = KnowledgeStore(tmp_path, embedder)
    redis = k.add_entity(Entity(name="Redis", kind="tool"))
    lock = k.add_item(
        Item(
            type=ItemType.DECISION,
            statement="svc/limiter.py guards acquire() with a Redis SETNX lock with 30s expiry",
            why="workers raced on the counter",
            entities=[redis.entity_id],
        )
    )
    k.add_item(Item(type=ItemType.FACT, statement="the user lives in Seoul and likes football"))
    got = recall(k, KnowledgeIndex(k), "two workers can both take the rate limit slot at once")
    assert lock.item_id in [i.item_id for i in got.items]
    none = recall(k, KnowledgeIndex(k), "write a haiku about autumn leaves")
    assert lock.item_id not in [i.item_id for i in none.items]


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI")
def test_real_graduation_learns_the_decision(tmp_path: Path, embedder) -> None:  # type: ignore[no-untyped-def]
    k = KnowledgeStore(tmp_path, embedder)
    xs = (
        Exchange(
            "x1",
            T0,
            "two workers both acquire the limiter slot, can you fix the race in svc/limiter.py?",
            "Fixed: acquire() now takes a Redis SETNX lock with a 30s expiry before touching "
            "the counter, so only one worker updates it at a time.",
            ("svc/limiter.py",),
        ),
        Exchange(
            "x2",
            T0 + timedelta(minutes=3),
            "great. from now on always run the limiter tests with -x so they stop early",
            "Noted, I'll run `pytest tests/limiter -x` from now on.",
        ),
    )
    seg = SegmentInput("s1", xs, "session_end", "repo:/w/limiter", "limiter")
    graduate(k, seg, claude_code_model("haiku", timeout=300).complete, embedder)
    statements = " ".join(i.statement.lower() for i in k.items.values())
    assert "setnx" in statements
    assert any(i.type is ItemType.PREFERENCE for i in k.items.values())
    assert all(i.sources and i.sources[0].session == "s1" for i in k.items.values())


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI")
def test_real_judge_tells_a_useful_note_from_an_unrelated_one() -> None:
    from graph_memory.judge import build_judge_prompt, parse_judgement

    lock = Item(
        type=ItemType.DECISION,
        statement="svc/limiter.py guards acquire() with a Redis SETNX lock with 30s expiry",
    )
    seoul = Item(type=ItemType.FACT, statement="the user lives in Seoul and likes football")
    prompt = build_judge_prompt(
        "the limiter double counts again under load, why?",
        "The SETNX lock around acquire() expires after 30s; a slow worker holds it longer, "
        "so a second worker gets in. I raised the expiry and made the release check the token.",
        [lock, seoul],
    )
    labels = parse_judgement(
        claude_code_model("haiku", timeout=300).complete(prompt), [lock, seoul]
    )
    assert labels[lock.item_id] == "useful"
    assert labels[seoul.item_id] in ("neutral", "misleading")


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI")
def test_real_graduation_keeps_what_the_user_learned_by_asking(tmp_path: Path, embedder) -> None:  # type: ignore[no-untyped-def]
    """A session of questions, no code: what the user learned is kept, with where it came
    from, and the graduation writes a summary of where the session stands."""
    from graph_memory.graduation import apply, extract_step, merge_step, plan_merge
    from graph_memory.knowledge import Origin

    k = KnowledgeStore(tmp_path, embedder)
    xs = (
        Exchange(
            "x1",
            T0,
            "how does SQLite handle a many-to-many between my recipes and tags?",
            "SQLite has no array column type, so a column can't hold a list of tag ids. For "
            "a many-to-many between recipes and tags, add a recipe_tag join table with one "
            "row per recipe and tag (option A), or keep the tag names as JSON text on the "
            "recipe (option B).",
        ),
        Exchange(
            "x2",
            T0 + timedelta(minutes=40),
            "let's go with option A. why doesn't SQLite stop a tag row pointing at a "
            "missing recipe?",
            "I read https://www.sqlite.org/foreignkeys.html: SQLite only enforces foreign "
            "keys after PRAGMA foreign_keys = ON, and every connection must set it, so turn "
            "it on when you open the database.",
        ),
    )
    seg = SegmentInput("s1", xs, "idle", "repo:/w/recipes", "recipes")
    ask = claude_code_model("haiku", timeout=300).complete
    ex = extract_step(seg, ask, embedder, [])
    plan = plan_merge(k, seg, ex)
    apply(k, plan, merge_step(plan, ask), 0.0)
    items = list(k.items.values())
    text = " | ".join(f"{i.type.value}/{i.origin.value}: {i.statement}" for i in items)
    assert any("join" in i.statement.lower() or "array" in i.statement.lower() for i in items), text
    keys = [i for i in items if "foreign" in i.statement.lower()]
    assert keys and keys[0].origin is Origin.DOCS, text
    assert "sqlite" in ex.summary.lower() and len(ex.summary) > 40, ex.summary


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI")
def test_real_judge_flags_a_note_the_reply_contradicts() -> None:
    from graph_memory.judge import build_judge_prompt, verdicts

    on = Item(
        type=ItemType.LEARNED,
        statement="SQLite enforces foreign keys by default, with no setup needed",
    )
    prompt = build_judge_prompt(
        "do I need to do anything for SQLite to check my foreign keys?",
        "Yes. I checked https://www.sqlite.org/foreignkeys.html: foreign keys are off by "
        "default, and each connection must run PRAGMA foreign_keys = ON to enforce them.",
        [on],
    )
    found = verdicts(claude_code_model("haiku", timeout=300).complete(prompt), [on])
    label, why = found[on.item_id]
    assert label == "contradicted" and why, found


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI")
def test_real_merge_replaces_a_disputed_item(tmp_path: Path, embedder) -> None:  # type: ignore[no-untyped-def]
    """A note the judge disputed, then the docs say what is true: the merge retires it."""
    from graph_memory.graduation import apply, extract_step, merge_step, plan_merge
    from graph_memory.knowledge import Entity, EntityKind, Origin, Volatility

    k = KnowledgeStore(tmp_path, embedder)
    ents = [
        k.add_entity(Entity(name=n, kind=EntityKind.COMPONENT)).entity_id
        for n in ("recipe", "ALTER TABLE")
    ]
    old = k.add_item(
        Item(
            type=ItemType.FACT,
            statement="SQLite's ALTER TABLE cannot add a column to the recipe table, so adding "
            "one means rebuilding the table and copying every row.",
            entities=ents,
            origin=Origin.ASSISTANT,
            volatility=Volatility.VERSIONED,
        )
    )
    k.dispute(
        old.item_id,
        "The reply says the SQLite docs show ALTER TABLE ADD COLUMN adds the column in "
        "place without rebuilding the table.",
    )
    xs = (
        Exchange(
            "x1",
            T0,
            "can sqlite really not add a column to the recipe table without a rebuild?",
            "I checked the SQLite docs: ALTER TABLE ADD COLUMN adds the column in place; the "
            "recipe table is not rebuilt and no rows are copied.",
            web=("https://www.sqlite.org/lang_altertable.html",),
        ),
    )
    seg = SegmentInput("s2", xs, "session_end", "repo:/w/recipes", "recipes")
    ask = claude_code_model("haiku", timeout=300).complete
    ex = extract_step(seg, ask, embedder, [e.name for e in k.entities.values()])
    plan = plan_merge(k, seg, ex)
    assert plan.prompt is not None and "DISPUTED" in plan.prompt
    apply(k, plan, merge_step(plan, ask), 0.0)
    assert not old.valid, [i.statement for i in k.items.values()]
    replacement = k.items[old.invalidated_by or ""]
    said = replacement.statement.lower()
    assert (
        "place" in said or "add column" in said or "without" in said
    ) and replacement.origin is Origin.DOCS, said


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI")
def test_real_summary_cites_the_notes_it_was_given(tmp_path: Path) -> None:
    """The real model writes a brief of a few notes: every point cites one of them, and
    the open item lands among the open items."""
    from graph_memory.summarize import select, summarize

    k = KnowledgeStore(tmp_path)
    notes = [
        k.add_item(Item(type=kind, statement=text, why=why, scope="repo:/w/recipes"))
        for kind, text, why in (
            (
                ItemType.DECISION,
                "recipe_tag's primary key is (recipe_id, tag_id)",
                "adding the same tag to a recipe twice becomes a no-op under INSERT OR IGNORE",
            ),
            (
                ItemType.PITFALL,
                "Without an index on recipe_tag.tag_id, deleting a tag scans the whole join table",
                "",
            ),
            (
                ItemType.OPEN,
                "The import from the old JSON export has not been tested on real data",
                "",
            ),
        )
    ]
    items, left_out = select(notes)
    ask = claude_code_model("haiku", timeout=300).complete
    brief = summarize("the project recipes", items, ask)
    ids = {i.item_id for i in notes}
    points = [(s["key"], p) for s in brief["sections"] for p in s["points"]]
    assert left_out == 0 and brief["overview"] and points, brief
    assert all(p["notes"] and set(p["notes"]) <= ids for _, p in points), brief
    opened = [p for key, p in points if key == "open"]
    assert any(notes[2].item_id in p["notes"] for p in opened), brief
