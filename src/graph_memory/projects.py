"""Which project an exchange belongs to: the git repository of the files it touched, or
else of the session's working directory. Deterministic, no model involved.

A project is identified by its repository root ("repo:/abs/path"). Paths outside any
repository (scratch files, ~/.config) say nothing about the project and are ignored.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from .anchors import Access, CodeAnchor

REPO_PREFIX = "repo:"


@lru_cache(maxsize=4096)
def repo_root(directory: str) -> str | None:
    """The nearest enclosing directory with a .git entry, or None."""
    cur = Path(directory)
    for p in (cur, *cur.parents):
        if (p / ".git").exists():
            return str(p)
    return None


def _absolute(path: str, cwd: str | None) -> str | None:
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        if not cwd:
            return None
        p = os.path.join(cwd, p)
    return os.path.normpath(p)


def project_of(files: Iterable[CodeAnchor], cwd: str | None, *, use_cwd: bool = True) -> str | None:
    """Project key for an exchange: the repo most of its written files are in, else the
    repo most of its other files are in, else (with use_cwd) the repo of the working
    directory."""
    written: Counter[str] = Counter()
    other: Counter[str] = Counter()
    for a in files:
        full = _absolute(a.path, cwd)
        if full is None:
            continue
        root = repo_root(os.path.dirname(full))
        if root is None:
            continue
        (written if a.access is Access.WRITE else other)[root] += 1
    for counts in (written, other):
        if counts:
            return REPO_PREFIX + counts.most_common(1)[0][0]
    root = repo_root(cwd) if cwd and use_cwd else None
    return REPO_PREFIX + root if root else None


def project_title(key: str) -> str:
    return os.path.basename(key.removeprefix(REPO_PREFIX).rstrip("/")) or key
