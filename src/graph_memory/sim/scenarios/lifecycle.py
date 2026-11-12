"""Open items: a to-do that a later session finishes is closed (no longer true), and a
person can close or reopen an item by hand when no session says so."""

from __future__ import annotations

from collections.abc import Callable

from ..harness import Simulator
from ..scenario import Scenario
from .core import limiter_facts, teach_limiter


def resolve_open() -> Scenario:
    s = Scenario(
        "resolve_open",
        "a to-do restated until it is old, then done in a later session: the open item "
        "becomes no longer true and stops being recalled; an open item nobody closes is "
        "retracted by hand, survives a restart that way, and can be restored",
    )
    lim = limiter_facts(s)
    s.fact(
        "tenant_todo",
        "open",
        "Per-tenant rate limits are still to do: svc/limiter.py applies one global rate "
        "to every tenant",
        about=["svc/limiter.py", "per-tenant rate limits"],
        match=["tenant", r"to do|todo|not yet|still|one global|single global|missing"],
    )
    s.fact(
        "tenant_done",
        "fact",
        "svc/limiter.py now applies per-tenant rate limits read from config/tenants.yaml",
        about=["svc/limiter.py", "config/tenants.yaml", "per-tenant rate limits"],
        match=["tenant", r"!to do|todo|not yet|still need|missing"],
        supersedes=["tenant_todo"],
    )
    s.fact(
        "flaky",
        "open",
        "test_burst in tests/test_limiter.py is flaky under parallel pytest runs and still "
        "needs a fix",
        about=["tests/test_limiter.py", "test_burst"],
        match=["test_burst"],
    )
    teach_limiter(s, lim)
    a = s.session("todo", lim)
    a.turn(
        "can every tenant get its own limit?",
        "Not yet: svc/limiter.py applies one global rate to every tenant. Per-tenant limits "
        "are still to do; I left a TODO in acquire().",
        files=["svc/limiter.py"],
        learn=["tenant_todo"],
        allow=["lock", "skew", "compose", "terse"],
    )
    a.turn(
        "test_burst failed again on CI",
        "test_burst in tests/test_limiter.py is flaky when pytest runs in parallel; it still "
        "needs a proper fix, I re-ran it for now.",
        files=["tests/test_limiter.py"],
        learn=["flaky"],
        allow=["compose", "lock", "terse"],
    )
    a.end()
    s.advance(days=1)
    b = s.session("again", lim)
    b.turn(
        "a big tenant complained about being throttled by a small one",
        "That is the missing per-tenant limits: svc/limiter.py still applies one global rate "
        "to every tenant. Still to do.",
        learn=["tenant_todo"],
        # known limit: only "tenant" is shared and MiniLM puts the prompt at 0.36, under
        # the 0.38 vector gate, so the to-do may stay silent for this indirect phrasing
        allow=["tenant_todo", "lock", "skew", "compose", "terse", "flaky"],
        label="restated by a prompt that names it only indirectly",
    )
    b.end()
    s.check("tenant_todo", count=1, area="old", label="restated: the to-do is old now")
    s.advance(days=2)
    s.probe(
        "what is still to do on the rate limiter per tenant?",
        lim,
        has=["tenant_todo"],
        allow=["lock", "skew", "compose", "terse", "flaky"],
    )
    s.probe(
        "where were we?",
        lim,
        has=["tenant_todo"],
        allow=["flaky", "lock", "skew", "compose", "terse"],
        label="a prompt that names no work gets the project's open items",
    )
    c = s.session("done", lim)
    c.turn(
        "let's implement per-tenant limits now",
        "Done: svc/limiter.py now applies per-tenant rate limits read from "
        "config/tenants.yaml, with the old global rate as the default.",
        files=["svc/limiter.py", "config/tenants.yaml"],
        learn=["tenant_done"],
        has=["tenant_todo"],
        allow=["lock", "skew", "compose", "terse", "flaky"],
        used=["tenant_todo"],
        label="the to-do is recalled when the work starts",
    )
    c.end()
    s.check("tenant_todo", valid=False, count=1, label="the finished to-do is closed")
    s.check("tenant_done", valid=True, count=1)
    s.check_fn("the done item supersedes the to-do", _closed_by("tenant_done", "tenant_todo"))
    s.advance(days=1)
    s.probe(
        "what is still to do on the rate limiter per tenant?",
        lim,
        lacks=["tenant_todo"],
        allow=["tenant_done", "lock", "skew", "compose", "terse", "flaky"],
        label="a closed to-do is never recalled",
    )
    s.probe(
        "what's left?",
        lim,
        lacks=["tenant_todo"],
        allow=["flaky", "tenant_done", "lock", "skew", "compose", "terse"],
        label="what's left: not the closed to-do",
    )
    s.probe(
        "is test_burst still flaky?",
        lim,
        has=["flaky"],
        allow=["compose", "lock", "terse"],
    )
    s.retract("flaky", "fixed outside these sessions", label="closed by hand")
    s.check("flaky", valid=False, count=1)
    s.restart(label="a hand retraction persists")
    s.check("flaky", valid=False, count=1, label="still closed after a restart")
    s.probe(
        "is test_burst still flaky?",
        lim,
        lacks=["flaky"],
        allow=["compose", "lock", "terse"],
        label="a retracted item is never recalled",
    )
    s.restore("flaky", label="reopened by hand")
    s.check("flaky", valid=True, count=1)
    s.probe(
        "is test_burst still flaky?",
        lim,
        has=["flaky"],
        allow=["compose", "lock", "terse"],
        label="a restored item is recalled again",
    )
    return s


def _closed_by(new_key: str, old_key: str) -> Callable[[Simulator], str | None]:
    def check(sim: Simulator) -> str | None:
        new = sim.items(new_key, valid_only=True)
        old = sim.items(old_key)
        if not new or not old:
            return f"missing items: {new_key}={len(new)}, {old_key}={len(old)}"
        if old[-1].invalidated_by != new[0].item_id:
            return f"{old_key} invalidated by {old[-1].invalidated_by}, not {new[0].item_id}"
        return None

    return check
