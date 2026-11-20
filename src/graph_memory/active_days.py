"""The days you worked: the local dates on which a prompt you typed reached memory, in
all and per project. A note's age, which its vitality decays by, is counted in these days
rather than calendar days, so time away from the agent ages nothing.

    "global"       every active day; global and topic notes age by these
    "repo:/path"   the days you worked in that project; its notes age by these only,
                   so weeks spent on another project do not fade them

Today is an active day only once a prompt has arrived today. A note last seen or used on
an active day is 0 days old that day, and one day older on each later active day.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from collections.abc import Iterable
from datetime import date, datetime

from . import clock
from .projects import REPO_PREFIX

ALL = "global"  # the scope key of the list of every active day


def local_date(when: datetime) -> date:
    """The date `when` falls on where the user is (the machine's time zone)."""
    return when.astimezone().date()


def list_of(scope: str) -> str:
    """Which list a note of this scope ages by: its project's, else every active day."""
    return scope if scope.startswith(REPO_PREFIX) else ALL


class ActiveDays:
    """Sorted lists of active dates, by scope. A few hundred dates a year per list."""

    def __init__(self, rows: Iterable[tuple[str, str]] = ()) -> None:
        self._days: dict[str, list[date]] = {}
        for day, scope in rows:
            self._put(date.fromisoformat(day), scope)

    def _put(self, day: date, scope: str) -> bool:
        got = self._days.setdefault(scope, [])
        at = bisect_right(got, day)
        if at and got[at - 1] == day:
            return False
        insort(got, day, lo=at)
        return True

    def add(self, when: datetime, project: str | None = None) -> list[tuple[str, str]]:
        """Record that you worked at `when` (in `project`, if any). Returns the rows that
        are new, as (date, scope), for the store to write; empty when the day was known."""
        day = local_date(when)
        scopes = [ALL, project] if project and project.startswith(REPO_PREFIX) else [ALL]
        return [(day.isoformat(), s) for s in scopes if self._put(day, s)]

    def age(self, since: datetime, scope: str, now: datetime | None = None) -> int:
        """Active days after the date of `since`, up to and including today, in the list
        a note of `scope` ages by."""
        got = self._days.get(list_of(scope))
        if not got:
            return 0
        first, last = local_date(since), local_date(now or clock.now())
        if last <= first:
            return 0
        return bisect_right(got, last) - bisect_right(got, first)

    def rows(self) -> list[tuple[str, str]]:
        """Every (date, scope), oldest first."""
        return sorted((d.isoformat(), s) for s, days in self._days.items() for d in days)

    def count(self, scope: str = ALL) -> int:
        return len(self._days.get(scope, ()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ActiveDays) and self._days == other._days

    __hash__ = None  # type: ignore[assignment]
