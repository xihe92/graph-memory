"""Scenarios: a scripted user working with a coding agent over days and weeks.

A scenario is a list of steps the simulator replays through the real hooks:

    s = Scenario("supersede", "a decision reversed in a later session")
    lim = s.project("limiter")
    lock = s.fact("lock", "decision", "svc/limiter.py guards acquire() with a Redis SETNX lock",
                  about=["svc/limiter.py", "Redis"], match=["setnx"])
    a = s.session("a", lim)
    a.turn("the limiter double counts under load", "Added a SETNX lock ...",
           files=["svc/limiter.py"], learn=["lock"])
    a.end()
    s.advance(days=2)
    b = s.session("b", lim)
    b.turn("acquire() is slow, look at the redis lock", "...", has=["lock"])

Facts are the knowledge the turns contain. In oracle mode the fake model extracts exactly
the facts a turn `learn`s; with a real model the facts are what we hope it extracts, and
`match` keywords recognise them in whatever words the model chose.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .harness import SimState

ITEM_TYPES = ("decision", "fact", "preference", "pitfall", "procedure", "open", "learned")
RELATIONS = ("depends_on", "causes", "refines", "alternative", "same_area")
_WORD = re.compile(r"[a-z0-9_./()-]+")
_STOP = frozenset(
    "the and with for from into that this when then than uses use used each every only "
    "never always must should after before over under about which their there".split()
)


@dataclass
class Fact:
    key: str
    type: str
    statement: str
    about: tuple[str, ...]
    why: str = ""
    scope: str = "project"  # "project", "topic" (see `topic`) or "global"
    # regexes that all appear when some wording states it; "!regex" must not appear
    match: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()  # facts this one makes false
    related: tuple[tuple[str, str, str], ...] = ()  # (fact key, relation, strength)
    variants: list[str] = field(default_factory=list)  # other wordings, see Turn.learn
    origin: str = "user"  # where the oracle says it came from (see knowledge.Origin)
    ref: str = ""
    applies_to: str = ""
    volatility: str = "stable"
    topic: str = ""  # the subject a "topic" fact is about

    def matches(self, text: str) -> bool:
        low = text.lower()
        if self.match:
            return all(
                not re.search(m[1:], low) if m.startswith("!") else re.search(m, low)
                for m in self.match
            )
        want = [w for w in _WORD.findall(self.statement.lower()) if len(w) > 4 and w not in _STOP]
        if not want:
            return self.statement.lower() in low
        have = set(_WORD.findall(low))
        return sum(w in have for w in want) / len(want) >= 0.6


@dataclass
class Expect:
    """What recall should show for a prompt. `has`: facts that must be shown. `lacks`:
    facts that must not be. `allow`: facts that are fine either way (not counted as
    noise). `nothing`: no knowledge except what `allow` permits."""

    has: tuple[str, ...] = ()
    lacks: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()
    nothing: bool = False

    @property
    def active(self) -> bool:
        return bool(self.has or self.lacks or self.nothing)


@dataclass
class Turn:
    session: str
    prompt: str
    reply: str
    files: tuple[str, ...] = ()
    learn: tuple[tuple[str, str], ...] = ()  # (fact key, wording)
    expect: Expect = field(default_factory=Expect)
    used: tuple[str, ...] = ()  # recalled facts the reply must count as used
    unused: tuple[str, ...] = ()  # recalled facts the reply must not count as used
    label: str = ""
    contradicts: tuple[str, ...] = ()  # recalled facts the reply shows wrong (oracle judge)
    web: tuple[str, ...] = ()  # pages the agent reads before replying (WebFetch calls)


@dataclass
class Compact:
    session: str
    summary: str
    learn: tuple[tuple[str, str], ...] = ()  # facts only the summary carries
    label: str = ""
    fails: bool = False  # graduation is expected to fail (see FailNext)


@dataclass
class End:
    session: str
    label: str = ""
    fails: bool = False


@dataclass
class Advance:
    seconds: float
    label: str = ""


@dataclass
class Restart:
    downtime: float = 0.0  # simulated seconds the daemon was down
    label: str = ""


@dataclass
class Probe:
    """A recall from a fresh session: what a new conversation would be shown."""

    prompt: str
    project: str | None
    expect: Expect
    label: str = ""


@dataclass
class Check:
    """State of the items stating a fact."""

    key: str
    area: str | None = None
    valid: bool | None = None
    count: int | None = None  # how many items (valid or not) state it
    min_count: int | None = None  # at least this many (a real model may add related ones)
    min_strength: float | None = None
    max_strength: float | None = None
    min_uses: int | None = None
    label: str = ""


@dataclass
class Assert:
    """Any other check: fn returns None when fine, else what is wrong."""

    name: str
    fn: Callable[[SimState], str | None]
    label: str = ""


@dataclass
class FailNext:
    """Make the next model calls fail (oracle mode only)."""

    extract: int = 0
    merge: int = 0
    garbage: int = 0
    label: str = ""


@dataclass
class Retract:
    """A person marks the item of a fact no longer true (`gm knowledge retract`), or with
    `restore` makes it true again (`gm knowledge restore`)."""

    key: str
    reason: str = ""
    restore: bool = False
    label: str = ""


Step = Turn | Compact | End | Advance | Restart | Probe | Check | Assert | FailNext | Retract


def _learn(
    spec: Iterable[str | tuple[str, str]], facts: dict[str, Fact]
) -> tuple[tuple[str, str], ...]:
    out = []
    for x in spec:
        key, text = (x, None) if isinstance(x, str) else x
        if key not in facts:
            raise KeyError(f"unknown fact {key!r}")
        f = facts[key]
        if text is None:
            text = f.statement
        elif text != f.statement and text not in f.variants:
            f.variants.append(text)
        out.append((key, text))
    return tuple(out)


def _expect(
    has: Sequence[str], lacks: Sequence[str], allow: Sequence[str], nothing: bool
) -> Expect:
    return Expect(tuple(has), tuple(lacks), tuple(allow), nothing)


class SessionBuilder:
    def __init__(self, scenario: Scenario, name: str, project: str | None) -> None:
        self.s = scenario
        self.name = name
        self.project = project

    def turn(
        self,
        prompt: str,
        reply: str,
        *,
        files: Sequence[str] = (),
        learn: Sequence[str | tuple[str, str]] = (),
        has: Sequence[str] = (),
        lacks: Sequence[str] = (),
        allow: Sequence[str] = (),
        nothing: bool = False,
        used: Sequence[str] = (),
        unused: Sequence[str] = (),
        label: str = "",
        contradicts: Sequence[str] = (),
        web: Sequence[str] = (),
    ) -> SessionBuilder:
        for k in [*has, *lacks, *allow, *used, *unused, *contradicts]:
            if k not in self.s.facts:
                raise KeyError(f"unknown fact {k!r}")
        self.s.steps.append(
            Turn(
                self.name,
                prompt,
                reply,
                tuple(files),
                _learn(learn, self.s.facts),
                _expect(has, lacks, allow, nothing),
                tuple(used),
                tuple(unused),
                label,
                tuple(contradicts),
                tuple(web),
            )
        )
        return self

    def compact(
        self,
        summary: str,
        learn: Sequence[str | tuple[str, str]] = (),
        label: str = "",
        fails: bool = False,
    ) -> SessionBuilder:
        self.s.steps.append(Compact(self.name, summary, _learn(learn, self.s.facts), label, fails))
        return self

    def end(self, label: str = "", fails: bool = False) -> None:
        self.s.steps.append(End(self.name, label, fails))


class Scenario:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.facts: dict[str, Fact] = {}
        self.steps: list[Step] = []
        self.projects: list[str] = []
        self.sessions: dict[str, str | None] = {}  # session name -> project
        self.aliases: list[set[str]] = []
        self.kinds: dict[str, str] = {}
        self.summaries: dict[str, str] = {}
        self.turn_gap = 60.0  # simulated seconds between a session's turns
        self.oracle_only = False  # needs the oracle (injected failures, exact merges)
        self.expensive = False  # too many model calls to run with a real model by default

    # ------------------------------------------------------------------ world

    def project(self, name: str) -> str:
        """A git repository the user works in."""
        if name not in self.projects:
            self.projects.append(name)
        return name

    def entity(self, name: str, kind: str, summary: str = "") -> None:
        """How the oracle describes an entity (a real model writes a short summary)."""
        self.kinds[name] = kind
        if summary:
            self.summaries[name] = summary

    def alias(self, *names: str) -> None:
        """Names that are the same real thing (the oracle answers entity questions so)."""
        self.aliases.append(set(names))

    def fact(
        self,
        key: str,
        type: str,
        statement: str,
        *,
        about: Sequence[str],
        why: str = "",
        scope: str = "project",
        match: Sequence[str] = (),
        supersedes: Sequence[str] = (),
        related: Sequence[tuple[str, str, str]] = (),
        origin: str = "user",
        ref: str = "",
        applies_to: str = "",
        volatility: str = "stable",
        topic: str = "",
    ) -> Fact:
        if type not in ITEM_TYPES:
            raise ValueError(f"fact {key}: unknown type {type!r}")
        for k in supersedes:
            if k not in self.facts:
                raise KeyError(f"fact {key}: supersedes unknown {k!r}")
        for k, rel, _ in related:
            if k not in self.facts or rel not in RELATIONS:
                raise KeyError(f"fact {key}: bad relation {(k, rel)}")
        f = Fact(
            key,
            type,
            statement,
            tuple(about),
            why,
            scope,
            tuple(m.lower() for m in match),
            tuple(supersedes),
            tuple(related),
            origin=origin,
            ref=ref,
            applies_to=applies_to,
            volatility=volatility,
            topic=topic,
        )
        self.facts[key] = f
        return f

    # ------------------------------------------------------------------ steps

    def session(self, name: str, project: str | None = None) -> SessionBuilder:
        """A Claude Code session; `project` None means a directory outside any repo.
        Using the same name again resumes the session."""
        if project is not None and project not in self.projects:
            raise KeyError(f"unknown project {project!r}")
        self.sessions.setdefault(name, project)
        return SessionBuilder(self, name, project)

    def advance(
        self, *, days: float = 0, hours: float = 0, minutes: float = 0, label: str = ""
    ) -> None:
        """Let time pass. The daemon's maintenance runs afterwards: idle sessions graduate
        and faded knowledge goes dormant."""
        self.steps.append(Advance(days * 86400 + hours * 3600 + minutes * 60, label))

    def restart(self, *, hours: float = 0, label: str = "") -> None:
        """Stop the daemon and start a new one from what is on disk, optionally after
        `hours` of downtime (no maintenance runs while it is down)."""
        self.steps.append(Restart(hours * 3600, label))

    def probe(
        self,
        prompt: str,
        project: str | None = None,
        *,
        has: Sequence[str] = (),
        lacks: Sequence[str] = (),
        allow: Sequence[str] = (),
        nothing: bool = False,
        label: str = "",
    ) -> None:
        for k in [*has, *lacks, *allow]:
            if k not in self.facts:
                raise KeyError(f"unknown fact {k!r}")
        self.steps.append(Probe(prompt, project, _expect(has, lacks, allow, nothing), label))

    def check(self, key: str, **kw: Any) -> None:
        if key not in self.facts:
            raise KeyError(f"unknown fact {key!r}")
        self.steps.append(Check(key, **kw))

    def retract(self, key: str, reason: str = "", *, label: str = "") -> None:
        if key not in self.facts:
            raise KeyError(f"unknown fact {key!r}")
        self.steps.append(Retract(key, reason, label=label))

    def restore(self, key: str, *, label: str = "") -> None:
        if key not in self.facts:
            raise KeyError(f"unknown fact {key!r}")
        self.steps.append(Retract(key, restore=True, label=label))

    def check_fn(self, name: str, fn: Callable[[SimState], str | None]) -> None:
        self.steps.append(Assert(name, fn))

    def fail_next(self, *, extract: int = 0, merge: int = 0, garbage: int = 0) -> None:
        self.steps.append(FailNext(extract, merge, garbage))
