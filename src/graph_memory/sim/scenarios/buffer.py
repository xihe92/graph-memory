"""The session buffer: exchanges wait until they graduate, then leave."""

from __future__ import annotations

from collections.abc import Callable

from ..harness import Simulator
from ..scenario import Scenario
from .core import limiter_facts


def _buffer_of(name: str, want: str) -> Callable[[Simulator], str | None]:
    """Check one session's buffer: `want` is its record kinds in order, e.g. "S E" for a
    summary then an exchange, or "" for no folder at all."""

    def check(sim: Simulator) -> str | None:
        assert sim.daemon is not None
        key = f"claude-code:{sim.session(name).sid}"
        buf = sim.daemon.buffer
        got = " ".join("S" if r.kind == "summary" else "E" for r in buf.records(key))
        if key not in buf.sessions() and want == "":
            return None
        return None if got == want else f"buffer of {name}: {got or 'gone'!r}, want {want!r}"

    return check


def buffer_lifecycle() -> Scenario:
    s = Scenario(
        "buffer_lifecycle",
        "a compaction turns the exchanges it covered into one summary; exchanges after it "
        "wait; the session's folder goes when it ends; a graduation that keeps failing gives "
        "up after three tries, so the buffer never grows without limit",
    )
    s.oracle_only = True
    lim = limiter_facts(s)
    a = s.session("a", lim)
    a.turn(
        "workers double count requests in the limiter",
        "I wrapped acquire() in svc/limiter.py in a Redis SETNX lock with a 30 second "
        "expiry; counts are exact now.",
        files=["svc/limiter.py"],
        learn=["lock"],
    )
    a.turn("and the tests?", "Re-ran them: green.")
    a.compact("Fixed the limiter double counting with a Redis SETNX lock around acquire().")
    s.check_fn("compaction: one summary, no exchanges", _buffer_of("a", "S"))
    s.check("lock", count=1)
    a.turn(
        "the integration tests fail with connection refused",
        "Start Redis with `docker compose up redis` before pytest; all pass now.",
        learn=["compose"],
    )
    s.check_fn("the next exchange waits after the summary", _buffer_of("a", "S E"))
    a.end()
    s.check_fn("the ended session's folder is gone", _buffer_of("a", ""))
    s.check("compose", count=1)

    b = s.session("b", lim)
    b.turn(
        "clock skew between hosts breaks the window maths",
        "The limiter now reads time from Redis TIME instead of the host clock.",
        files=["svc/limiter.py"],
        learn=["skew"],
    )
    s.fail_next(extract=3)
    s.advance(minutes=25, label="idle graduation fails (1)")
    s.check_fn("still waiting after one failure", _buffer_of("b", "E"))
    s.advance(minutes=25, label="fails again (2)")
    s.advance(minutes=25, label="fails a third time: given up")
    s.check_fn("given up after three tries", _buffer_of("b", ""))
    s.check_fn("the give-up is logged", _dropped_logged)
    s.check("skew", count=0, label="what failed three times is not learned")
    return s


def _dropped_logged(sim: Simulator) -> str | None:
    dropped = sim.events_of("buffer_dropped")
    return None if len(dropped) == 1 and dropped[0]["exchanges"] == 1 else f"{dropped}"
