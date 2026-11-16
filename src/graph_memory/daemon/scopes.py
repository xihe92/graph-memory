"""Scopes as a person reads them: global, each project by a short name, each topic."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from ..knowledge import TOPIC_PREFIX


def scope_names(scopes: Iterable[str]) -> list[dict[str, Any]]:
    """Each scope with a short display name and its item count: global, then projects,
    then topics. A project is named by its directory, with parent directories added
    until the names are unique; a topic by its name."""
    counts = Counter(scopes)
    topics = sorted((s for s in counts if s.startswith(TOPIC_PREFIX)), key=str.lower)
    projects = sorted(s for s in counts if s != "global" and s not in topics)
    parts = {s: [x for x in s.removeprefix("repo:").split("/") if x] or [s] for s in projects}
    depth = dict.fromkeys(projects, 1)
    while True:
        names = {s: "/".join(parts[s][-depth[s] :]) for s in projects}
        clash = [s for s in projects if list(names.values()).count(names[s]) > 1]
        grow = [s for s in clash if depth[s] < len(parts[s])]
        if not grow:
            break
        for s in grow:
            depth[s] += 1
    out = (
        [{"key": "global", "name": "global", "items": counts["global"]}]
        if "global" in counts
        else []
    )
    return (
        out
        + [{"key": s, "name": names[s], "items": counts[s]} for s in projects]
        + [
            {"key": s, "name": s[len(TOPIC_PREFIX) :], "items": counts[s], "topic": True}
            for s in topics
        ]
    )
