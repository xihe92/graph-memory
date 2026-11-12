"""A deterministic, dependency-free embedder for simulations.

Bag of word stems hashed into a fixed number of signed dimensions, L2-normalised. Two
texts are close when they share distinctive words, and unrelated text is near-orthogonal.
Letter trigrams make spelling variants close ("Postgres", "PostgreSQL"). It knows nothing
of synonyms, so it tests the pipeline, not semantic quality: run the simulator with the
real embedder (`--embedder real`) for that.
"""

from __future__ import annotations

import hashlib
import math
import re

DIMS = 512
_WORD = re.compile(r"[a-z0-9_]+")
STOP = frozenset(
    "a an the and or but if then so to of in on at by for with from as is are was were be "
    "been it its this that these those i you we they he she me my our your their do does "
    "did done not no yes can could should would will just also there here what which who "
    "how why when where all any some more most very into out up down over than too about "
    "let lets let's ok okay please now use used uses using make made get got".split()
)


def stem(word: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


def words(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


def _slot(token: str) -> tuple[int, float]:
    h = hashlib.blake2b(token.encode(), digest_size=8).digest()
    n = int.from_bytes(h, "big")
    return n % DIMS, 1.0 if (n >> 32) & 1 else -1.0


def hash_embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    ws = words(text)
    for w in ws:
        i, sign = _slot(w)
        vec[i] += sign
    for a, b in zip(ws, ws[1:], strict=False):  # word pairs: a little word order
        i, sign = _slot(a + " " + b)
        vec[i] += 0.5 * sign
    # letter trigrams: "Postgres" is near "PostgreSQL", as with real models. They carry
    # most of the weight for names (a few words) and little for sentences.
    weight = 3.0 if len(ws) <= 3 else 0.3
    for w in ws:
        grams = [f"#{w}#"[k : k + 3] for k in range(len(w))] if len(w) >= 5 else []
        for g in grams:
            i, sign = _slot("3:" + g)
            vec[i] += weight * sign / math.sqrt(len(grams))
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        vec[0] = 1.0
        return vec
    return [x / norm for x in vec]
