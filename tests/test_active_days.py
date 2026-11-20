"""Notes age by the days you worked, not calendar days: active_days.py, and the store's
list of them."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from graph_memory.active_days import ALL, ActiveDays, list_of, local_date

T = datetime(2026, 11, 4, 12, tzinfo=UTC)


@pytest.fixture
def pacific(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The machine's time zone is US Pacific (UTC-8 in November)."""
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def days(*offsets: int, project: str | None = None) -> ActiveDays:
    d = ActiveDays()
    for n in offsets:
        d.add(T + timedelta(days=n), project)
    return d


def test_age_counts_only_the_days_you_worked() -> None:
    d = days(1, 2, 30, 31)  # two days, a month away, two days
    assert d.age(T, ALL, T + timedelta(days=31)) == 4
    assert d.age(T, ALL, T + timedelta(days=29)) == 2  # the time away added nothing
    assert d.age(T + timedelta(days=2), ALL, T + timedelta(days=31)) == 2
    assert d.age(T, ALL, T) == 0 and d.age(T + timedelta(days=5), ALL, T) == 0


def test_today_counts_only_once_you_have_worked_today() -> None:
    d = days(1)
    today = T + timedelta(days=2)
    assert d.age(T, ALL, today) == 1  # the daemon running today ages nothing yet
    assert d.add(today) == [(local_date(today).isoformat(), ALL)]
    assert d.age(T, ALL, today) == 2


def test_a_day_is_added_once_per_list() -> None:
    d = ActiveDays()
    assert d.add(T, "repo:/w/a") == [
        (local_date(T).isoformat(), ALL),
        (local_date(T).isoformat(), "repo:/w/a"),
    ]
    assert d.add(T + timedelta(hours=3), "repo:/w/a") == []  # later the same day
    assert d.add(T, "repo:/w/b") == [(local_date(T).isoformat(), "repo:/w/b")]
    assert d.add(T, "topic:SQLite") == []  # not a project: the overall list only
    assert d.count() == 1 and d.count("repo:/w/a") == 1
    assert ActiveDays(d.rows()) == d


def test_project_notes_age_by_their_project_days_only() -> None:
    d = days(1, 2, 3, project="repo:/w/b")
    d.add(T + timedelta(days=4), "repo:/w/a")
    end = T + timedelta(days=4)
    assert d.age(T, "repo:/w/a", end) == 1  # three days on project b do not count
    assert d.age(T, "repo:/w/b", end) == 3
    assert d.age(T, "global", end) == 4 and d.age(T, "topic:SQLite", end) == 4
    assert d.age(T, "repo:/w/never", end) == 0
    assert list_of("topic:SQLite") == ALL and list_of("repo:/w/a") == "repo:/w/a"


def test_days_are_the_users_local_dates(pacific: None) -> None:
    late = datetime(2026, 11, 5, 3, tzinfo=UTC)  # 7 pm on Nov 4 in California
    d = ActiveDays()
    assert d.add(late) == [("2026-11-04", ALL)]
    assert d.add(datetime(2026, 11, 4, 17, tzinfo=UTC)) == []  # 9 am the same local day
    assert d.add(datetime(2026, 11, 5, 8, 30, tzinfo=UTC)) == [("2026-11-05", ALL)]  # 00:30
    # learned at 7 pm, worked again the next local morning: one day old
    assert d.age(late, ALL, datetime(2026, 11, 5, 17, tzinfo=UTC)) == 1
