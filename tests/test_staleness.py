"""Stale knowledge: where an item came from, how likely it is to change, review dates, and
disputes from the live judge."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from graph_memory import clock
from graph_memory.knowledge import (
    DISPUTE_LIMIT,
    Item,
    ItemType,
    KnowledgeStore,
    Origin,
    Source,
    Volatility,
)


@pytest.fixture
def at() -> Iterator[Callable[[float], None]]:
    """Move the simulated clock: at(days) is that many days after the test started."""
    moved = {"days": 0.0}

    def move(days: float) -> None:
        clock.advance(days=days - moved["days"])
        moved["days"] = days

    try:
        yield move
    finally:
        clock.reset()


def test_old_items_load_as_stable_with_no_origin(tmp_path: Path) -> None:
    import json

    (tmp_path / "knowledge.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "entities": [],
                "items": [{"item_id": "a", "type": "fact", "statement": "x runs on y"}],
                "edges": [],
                "segments": [],
            }
        )
    )
    from graph_memory import migrate

    migrate.sqlite_apply(migrate.sqlite_plan(tmp_path))
    it = KnowledgeStore(tmp_path).item("a")
    assert it is not None and it.origin is Origin.UNKNOWN and it.volatility is Volatility.STABLE
    assert it.review_due() is None and not it.overdue() and it.recallable


def test_review_dates_follow_volatility(tmp_path: Path, at: Callable[[float], None]) -> None:
    k = KnowledgeStore(tmp_path)
    stable = k.add_item(Item(type=ItemType.LEARNED, statement="a join table has one row per pair"))
    versioned = k.add_item(
        Item(
            type=ItemType.LEARNED,
            statement="sqlite 3.35 can drop a column with alter table",
            volatility=Volatility.VERSIONED,
            applies_to="SQLite 3.35",
            origin=Origin.DOCS,
        )
    )
    volatile = k.add_item(
        Item(
            type=ItemType.FACT,
            statement="the latest sqlite release is 3.47",
            volatility=Volatility.VOLATILE,
        )
    )
    assert versioned.review_due() == versioned.created_at + timedelta(days=180)
    at(31)
    assert volatile.overdue() and not volatile.recallable  # a volatile item stops
    assert not versioned.overdue() and versioned.recallable and stable.recallable
    at(181)
    assert versioned.overdue() and versioned.recallable  # shown, with a caveat
    k.reinforce(versioned.item_id, "seg", Source(session="s", until=clock.now()))
    assert not versioned.overdue() and clock.now() - versioned.checked < timedelta(seconds=1)
    k.confirm(volatile.item_id)
    assert volatile.recallable


def test_disputed_twice_is_retracted_and_a_restatement_clears_it(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path)
    it = k.add_item(Item(type=ItemType.LEARNED, statement="foreign keys are on by default"))
    k.dispute(it.item_id, "the reply showed they are off by default")
    assert it.disputes == 1 and it.valid and it.recallable and it.disputed_at is not None
    k.reinforce(it.item_id, "seg")  # a later session states it again: confirmed
    assert it.disputes == 0 and it.dispute_reason == ""
    for n in range(DISPUTE_LIMIT):
        k.dispute(it.item_id, f"contradicted {n}")
    assert not it.valid and "contradicted 2 times" in it.invalid_reason
    assert it.invalidated_by is None  # retracted, not replaced
    k.dispute(it.item_id, "again")  # nothing more happens to a retracted item
    assert it.disputes == DISPUTE_LIMIT


def test_provenance_survives_a_save(tmp_path: Path) -> None:
    k = KnowledgeStore(tmp_path)
    it = k.add_item(
        Item(
            type=ItemType.LEARNED,
            statement="sqlite sets foreign key enforcement per connection",
            origin=Origin.DOCS,
            ref="https://www.sqlite.org/foreignkeys.html",
            applies_to="SQLite 3",
            volatility=Volatility.VERSIONED,
        )
    )
    k.dispute(it.item_id, "maybe not")
    k.save()
    again = KnowledgeStore(tmp_path).item(it.item_id)
    assert (again.origin, again.ref, again.applies_to, again.volatility) == (
        Origin.DOCS,
        "https://www.sqlite.org/foreignkeys.html",
        "SQLite 3",
        Volatility.VERSIONED,
    )
    assert again.disputes == 1 and again.dispute_reason == "maybe not"


def test_a_dormant_item_stays_dormant_when_memory_loads(tmp_path: Path) -> None:
    """Loading used to run the "stated again" promotion, which revived every dormant
    item until the next maintenance pass put it back to sleep."""
    from graph_memory.knowledge import Area

    k = KnowledgeStore(tmp_path)
    asleep = k.add_item(Item(type=ItemType.FACT, statement="faded long ago", strength=3.0))
    asleep.area = Area.DORMANT
    k.touch(asleep.item_id)
    k.save()
    k.close()
    assert KnowledgeStore(tmp_path).item(asleep.item_id).area is Area.DORMANT
