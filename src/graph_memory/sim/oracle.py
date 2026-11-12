"""A fake model that knows the scenario's answers.

It reads the real extraction, merge and judge prompts, recognises the exchanges and items
in them, and answers from the scenario's facts: an exchange yields the facts its turn
`learn`s; a new item is a duplicate of an earlier item stating the same fact,
invalidates items of the facts it `supersedes`, and relates to its `related` facts; a
note shown for a turn is contradicted when the turn `contradicts` its fact, useful when
the turn `used` it. Everything around the model (prompt building, parsing, entity
resolution, applying merges, edges, disputes) runs for real.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..graduation import PROMPT_CHARS
from .scenario import Compact, Fact, Scenario, Turn

_X = re.compile(r"^X(\d+) \(\d{4}-\d\d-\d\d\)$", re.M)
_N = re.compile(r"^N(\d+) \[(\w+)[^\]]*\] (.*)$", re.M)  # [type, from …]
# [type, date, from …] statement (DISPUTED: …)
_K = re.compile(r"^K(\d+) \[(\w+), [\d-]+[^\]]*\] (.*?)(?: \(DISPUTED: .*\))?$", re.M)
_USER = re.compile(r"^USER: (.*)$", re.M)
_A = re.compile(r"^A(\d+) \[(\w+), [\d-]+\] (.*)$", re.M)  # a note to judge


def kind_of(prompt: str) -> str:
    """Which of the model's jobs a prompt is."""
    if "distil" in prompt:
        return "extract"
    if "You judge memory notes" in prompt:
        return "judge"
    return "merge"


_Q = re.compile(r"^Q(\d+): (.*?) \((\w+)\): .*? — (.*)$", re.M)


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def _strip_why(line: str) -> str:
    cut = line.rfind(" (why: ")
    return line[:cut] if cut >= 0 and line.endswith(")") else line


def guess_kind(name: str) -> str:
    if "/" in name or re.search(r"\.[a-z]{1,5}$", name):
        return "file"
    if name.endswith("()"):
        return "symbol"
    return "tool"


@dataclass
class OracleModel:
    scenario: Scenario
    fail_extract: int = 0
    fail_merge: int = 0
    garbage: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        s = self.scenario
        self.statements: dict[str, str] = {}  # wording -> fact key
        for f in s.facts.values():
            for text in [f.statement, *f.variants]:
                self.statements[norm(text)] = f.key
        self.turns: list[tuple[str, tuple[tuple[str, str], ...]]] = []  # (marker, learn)
        self.summaries: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        self.judged: dict[str, Turn] = {}  # prompt -> its turn, for judging
        seen: dict[str, tuple[tuple[str, str], ...]] = {}
        for st in s.steps:
            if isinstance(st, Turn):
                self.judged[st.prompt[:3000].strip()] = st
            if isinstance(st, Turn) and st.learn:
                marker = f"USER: {st.prompt[:PROMPT_CHARS]}\nASSISTANT: {st.reply[:200]}"
                if marker in seen and seen[marker] != st.learn:
                    # the oracle recognises exchanges by their text: two identical
                    # exchanges teaching different facts would be credited to both
                    raise ValueError(
                        f"{s.name}: identical exchanges learn different facts: {marker[:120]!r}"
                    )
                seen[marker] = st.learn
                self.turns.append((marker, st.learn))
            elif isinstance(st, Compact) and st.learn:
                self.summaries.append((st.summary[:300], st.learn))

    # ------------------------------------------------------------------ entry point

    def __call__(self, prompt: str) -> str:
        kind = kind_of(prompt)
        started = time.perf_counter()
        try:
            if self.garbage:
                self.garbage -= 1
                return "I'm sorry, I can't produce JSON for this."
            if kind == "extract" and self.fail_extract:
                self.fail_extract -= 1
                raise RuntimeError("oracle: injected extraction failure")
            if kind == "merge" and self.fail_merge:
                self.fail_merge -= 1
                raise RuntimeError("oracle: injected merge failure")
            if kind == "judge":
                return json.dumps(self.judge(prompt))
            out = self.extract(prompt) if kind == "extract" else self.merge(prompt)
            return json.dumps(out)
        finally:
            self.calls.append(
                {"kind": kind, "ms": round((time.perf_counter() - started) * 1000, 1)}
            )

    def key_of(self, statement: str) -> str | None:
        return self.statements.get(norm(_strip_why(statement)))

    def _item(self, f: Fact, text: str, evidence: list[str]) -> dict[str, Any]:
        return {
            "type": f.type,
            "statement": text,
            "why": f.why,
            "entities": list(f.about),
            "scope": f.scope,
            "topic": f.topic,
            "evidence": evidence,
            "origin": f.origin,
            "ref": f.ref,
            "applies_to": f.applies_to,
            "volatility": f.volatility,
        }

    # ------------------------------------------------------------------ extraction

    def extract(self, prompt: str) -> dict[str, Any]:
        xs = [(m.start(), m.group(1)) for m in _X.finditer(prompt)]
        items: list[dict[str, Any]] = []
        names: dict[str, None] = {}
        for marker, learn in self.turns:
            at = prompt.find(marker)
            if at < 0:
                continue
            x = next((n for pos, n in reversed(xs) if pos < at), None)
            for key, text in learn:
                f = self.scenario.facts[key]
                items.append(self._item(f, text, [f"X{x}"] if x else []))
                names.update(dict.fromkeys(f.about))
        if "<summary>" in prompt:
            for head, learn in self.summaries:
                if head and head in prompt:
                    for key, text in learn:
                        f = self.scenario.facts[key]
                        items.append(self._item(f, text, []))
                        names.update(dict.fromkeys(f.about))
        entities = [
            {
                "name": n,
                "kind": self.scenario.kinds.get(n, guess_kind(n)),
                "summary": self.scenario.summaries.get(n, ""),
            }
            for n in names
        ]
        return {"entities": entities, "items": items, "summary": self._summary(prompt)}

    @staticmethod
    def _summary(prompt: str) -> str:
        """A summary of the session so far a test can trace: the earlier summary, then the
        start of each prompt in these exchanges."""
        block = prompt.split("<exchanges>", 1)[1] if "<exchanges>" in prompt else prompt
        heads = "; ".join(m.group(1)[:50] for m in _USER.finditer(block))
        if "<earlier>" in prompt:
            earlier = prompt.split("<earlier>", 1)[1].split("</earlier>", 1)[0].strip()
            return f"{earlier} Then: {heads}"
        return f"Covered: {heads}"

    # ------------------------------------------------------------------ judge

    def judge(self, prompt: str) -> dict[str, Any]:
        asked = prompt.split("<prompt>", 1)[1].split("</prompt>", 1)[0].strip()
        notes = prompt.split("<notes>", 1)[1].split("</notes>", 1)[0]
        turn = self.judged.get(asked)
        out: dict[str, Any] = {}
        for m in _A.finditer(notes):
            key = self.key_of(m.group(3))
            used = turn is not None and key in turn.used
            verdict: dict[str, Any] = {"label": "useful" if used else "neutral", "wrong": False}
            if turn is not None and key in turn.contradicts:
                verdict = {
                    "label": "misleading",
                    "wrong": True,
                    "why": f"the reply shows {key} wrong",
                }
            out[f"A{m.group(1)}"] = verdict
        return out

    # ------------------------------------------------------------------ merge

    def merge(self, prompt: str) -> dict[str, Any]:
        new_block = prompt.split("<new_items>", 1)[1].split("</new_items>", 1)[0]
        old_block = prompt.split("<existing_items>", 1)[1].split("</existing_items>", 1)[0]
        new = [(f"N{m.group(1)}", self.key_of(m.group(3))) for m in _N.finditer(new_block)]
        old = [(f"K{m.group(1)}", self.key_of(m.group(3))) for m in _K.finditer(old_block)]
        out: list[dict[str, Any]] = []
        for idx, (nid, key) in enumerate(new):
            if key is None:
                out.append({"id": nid, "duplicate_of": None, "invalidates": [], "related": []})
                continue
            f = self.scenario.facts[key]
            dup = next((k for k, kk in old if kk == key), None) or next(
                (n for n, nk in new[:idx] if nk == key), None
            )
            invalid = [
                {"id": k, "reason": f"{key} replaces {kk}"} for k, kk in old if kk in f.supersedes
            ]
            related = []
            for rkey, rel, strength in f.related:
                for ref, kk in [*old, *new[:idx]]:
                    if kk == rkey:
                        related.append({"id": ref, "relation": rel, "strength": strength})
            out.append(
                {"id": nid, "duplicate_of": dup, "invalidates": invalid, "related": related[:4]}
            )
        result: dict[str, Any] = {"items": out}
        if "<entity_questions>" in prompt:
            result["entities"] = self._entities(prompt)
        return result

    def _same(self, a: str, b: str) -> bool:
        a, b = norm(a), norm(b)
        return any(
            a in {norm(x) for x in g} and b in {norm(x) for x in g} for g in self.scenario.aliases
        )

    def _entities(self, prompt: str) -> list[dict[str, Any]]:
        block = prompt.split("<entity_questions>", 1)[1].split("</entity_questions>", 1)[0]
        answers = []
        for m in _Q.finditer(block):
            name = m.group(2)
            same = None
            for opt in re.split(r";\s*(?=C\d+: )", m.group(4)):
                cid, _, rest = opt.partition(": ")
                cand = rest.split(" (")[0].strip()
                if self._same(name, cand):
                    same = cid.strip()
                    break
            answers.append({"q": f"Q{m.group(1)}", "same_as": same})
        return answers
