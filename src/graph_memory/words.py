"""Words as recall counts them: lower case, stop words dropped, light stemming. The same
function reads prompts, notes and replies, so only consistency matters."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-/]*[a-z0-9]|[a-z0-9]")
_STOP = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its "
    "me my no not of on or our so that the their them then there these this to was we "
    "what when where which who why will with you your should would could let lets just "
    "now also any all some more most other into than too very use used using get make "
    "want need like see show tell give go ok yes please after before between instead "
    "up down out over under again still only same such each both one two per re".split()
)


_KEEP_DOUBLE = ("ll", "ss", "zz")


def stem(word: str) -> str:
    """Light suffix stripping for plain words, so "tests" meets "test", "running" meets
    "run" and "raced" meets "race" (both "rac": a final e is dropped from every stem).
    Only consistency matters: the same function reads prompts, items and replies. Names
    with digits or punctuation (files, symbols) are left alone."""
    if len(word) < 4 or not word.isalpha():
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("ing", "ed")) and len(w) - (3 if w.endswith("ing") else 2) >= 3:
        w = w[:-3] if w.endswith("ing") else w[:-2]
        if len(w) > 3 and w[-1] == w[-2] and w[-2:] not in _KEEP_DOUBLE:
            w = w[:-1]
    elif w.endswith("s") and len(w) > 4 and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) >= 4 and w.endswith("e") and not w.endswith("ee"):
        w = w[:-1]
    return w


def tokens(text: str) -> list[str]:
    """Stemmed words without stop words. A hyphenated word also yields its parts, so
    "per-tenant" matches "tenant"; paths and dotted names stay whole."""
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t not in _STOP and len(t) > 1:
            out.append(stem(t))
        if "-" in t and "/" not in t and "." not in t:
            out.extend(stem(p) for p in t.split("-") if p not in _STOP and len(p) > 1)
    return out
