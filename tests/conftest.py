from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

TOPICS: dict[str, set[str]] = {
    "redis": {
        "redis",
        "race",
        "acquire",
        "lua",
        "limiter",
        "ratelimiter",
        "retry",
        "retries",
        "workers",
        "lock",
        "redlock",
        "backoff",
        "incr",
        "expire",
        "concurrency",
    },
    "football": {
        "football",
        "arsenal",
        "chelsea",
        "goal",
        "goals",
        "match",
        "league",
        "scored",
        "saka",
        "striker",
        "premier",
        "liverpool",
        "table",
        "points",
    },
    "weather": {
        "weather",
        "rain",
        "sunny",
        "forecast",
        "seoul",
        "temperature",
        "weekend",
        "umbrella",
        "cloudy",
        "wind",
    },
    "eslint": {"eslint", "typescript", "monorepo", "lint", "prettier", "config"},
}
_WORD = re.compile(r"[a-z]+")


def topic_embed(text: str) -> list[float]:
    """Deterministic stand-in for a real model: one dimension per topic, plus a small
    constant so unrelated text is near-orthogonal rather than a zero vector."""
    words = _WORD.findall(text.lower())
    vec = [sum(1.0 for w in words if w in vocab) for vocab in TOPICS.values()]
    return [*vec, 0.15]


@pytest.fixture(autouse=True)
def _interactive_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as the user's own session even when pytest is started from `claude -p`, where
    the hooks would skip everything as headless."""
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
    # daemons started by tests serve no web page unless a test asks for one
    monkeypatch.setenv("GM_UI_PORT", "0")


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """Unix socket paths must stay under ~100 chars; pytest's tmp_path can be longer."""
    d = Path(tempfile.mkdtemp(prefix="gm"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def turn(prompt: str, reply: str, **kw: object) -> dict[str, object]:
    return {"prompt": prompt, "reply": reply, **kw}
