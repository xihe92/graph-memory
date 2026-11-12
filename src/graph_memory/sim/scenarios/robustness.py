"""Things going wrong: sessions that never end, daemon restarts, failing models, chatter
that should leave no trace, prompts that should stay quiet."""

from __future__ import annotations

from collections.abc import Callable

from ..harness import Simulator
from ..scenario import Scenario
from .core import limiter_facts


def crash_and_restart() -> Scenario:
    s = Scenario(
        "crash_and_restart",
        "a session that never sends SessionEnd graduates after 20 idle minutes; one left "
        "over while the daemon was down graduates when it starts; everything persists "
        "across restarts; a resumed session graduates only what is new",
    )
    lim = limiter_facts(s)
    a = s.session("a", lim)
    a.turn(
        "the rate limiter double counts under load, workers race on the counter",
        "I wrapped acquire() in svc/limiter.py in a Redis SETNX lock with a 30 second "
        "expiry; counts are exact now.",
        files=["svc/limiter.py"],
        learn=["lock"],
    )
    s.advance(minutes=10, label="not idle long enough")
    s.check("lock", count=0, label="still young after 10 quiet minutes")
    s.advance(minutes=15, label="idle")
    s.check("lock", count=1, label="graduated by idleness")
    a.turn(
        "the tests fail with connection refused",
        "Start Redis with `docker compose up redis`, then run pytest: all pass.",
        learn=["compose"],
        label="the same session resumes after its idle graduation",
    )
    a.end()
    s.check("compose", count=1)
    s.check("lock", count=1, max_strength=1.0, label="not graduated twice")
    b = s.session("b", lim)
    b.turn(
        "the second worker host shows wrong windows",
        "The hosts' clocks are two seconds apart; svc/limiter.py now reads Redis TIME so "
        "every worker shares a clock.",
        files=["svc/limiter.py"],
        learn=["skew"],
    )
    s.restart(hours=2, label="daemon down for two hours while b was open")
    s.check("skew", min_count=1, label="graduated by the startup sweep")
    c = s.session("c", lim)
    c.turn(
        "is the redis lock in acquire() still needed after the TIME change?",
        "Yes: Redis TIME fixes clock skew between hosts, but the SETNX lock in acquire() "
        "is what stops workers racing on the counter.",
        has=["lock"],
        allow=["skew", "compose", "terse"],
        used=["lock"],
    )
    c.end()
    s.restart(label="restart: knowledge, use counts and edges come back from disk")
    s.check("lock", min_uses=1, min_strength=1.5)
    s.check_fn("graduation triggers recorded", _triggers({"idle", "session_end", "sweep"}))
    s.probe(
        "is the redis lock in acquire() still needed?",
        lim,
        has=["lock"],
        allow=["skew", "compose", "terse"],
    )
    return s


def _triggers(want: set[str]) -> Callable[[Simulator], str | None]:
    def check(sim: Simulator) -> str | None:
        got = {sg.trigger for sg in sim.store.segments.values()}
        return None if want <= got else f"triggers {sorted(got)}, want {sorted(want)}"

    return check


def model_failures() -> Scenario:
    s = Scenario(
        "model_failures",
        "a failed or garbled extraction loses nothing: the exchanges stay young and "
        "graduate on the next trigger; a failed merge keeps the items without dedup",
    )
    s.oracle_only = True
    lim = limiter_facts(s)
    a = s.session("a", lim)
    a.turn(
        "the rate limiter double counts under load, workers race on the counter",
        "I wrapped acquire() in svc/limiter.py in a Redis SETNX lock with a 30 second "
        "expiry; counts are exact now.",
        files=["svc/limiter.py"],
        learn=["lock"],
    )
    s.fail_next(extract=1)
    a.end(fails=True, label="extraction call fails")
    s.check("lock", count=0, label="nothing applied")
    s.advance(minutes=30, label="the idle sweep retries")
    s.check("lock", count=1)
    b = s.session("b", lim)
    b.turn(
        "the tests fail with connection refused",
        "Start Redis with `docker compose up redis`, then run pytest: all pass.",
        learn=["compose"],
    )
    s.fail_next(garbage=2)
    b.end(fails=True, label="the model answers prose instead of JSON, twice")
    s.restart(hours=1, label="the startup sweep retries")
    s.check("compose", count=1)
    c = s.session("c", lim)
    c.turn(
        "CI can't reach redis in the integration job",
        "Same as locally: bring Redis up with docker compose before pytest. Added that "
        "step to the CI workflow.",
        learn=[
            (
                "compose",
                "Redis must be running via `docker compose up redis` before the limiter's "
                "pytest integration tests can pass",
            )
        ],
        allow=["compose", "lock", "terse"],
    )
    s.fail_next(merge=2)
    c.end(label="merge call fails twice: graduation still succeeds")
    s.check(
        "compose",
        count=2,
        label="known degradation: without the merge call a reworded duplicate is kept",
    )
    s.check("lock", count=1, valid=True)
    return s


def noise() -> Scenario:
    s = Scenario(
        "noise",
        "chit-chat and routine commands leave no knowledge and get none; a long pasted log "
        "gets a bounded block",
    )
    lim = limiter_facts(s)
    from .core import teach_limiter

    teach_limiter(s, lim)
    chat = s.session("chat")
    chat.turn(
        "who won the premier league last season?",
        "Manchester City won the 2023-24 Premier League.",
        nothing=True,
        allow=["terse"],
    )
    chat.turn(
        "will it rain in Seoul this weekend?",
        "I can't check live weather; a forecast site will have it.",
        nothing=True,
    )
    chat.end()
    s.check_fn("chit-chat learned nothing", _items_at_most(4))
    b = s.session("b", lim)
    for prompt in ("run the tests", "commit this", "fix the typo in the README", "yes, go ahead"):
        b.turn(
            prompt,
            "Done.",
            nothing=True,
            allow=["terse", "compose"] if "test" in prompt else ["terse"],
            label=f"routine: {prompt!r}",
        )
    log = "\n".join(
        f"2026-11-03T10:{m:02d}:00Z worker-{m % 3} limiter acquire redis timeout "
        f"key=rl:{m} setnx lock held window={m * 7} clock host-{m % 2} pytest docker"
        for m in range(60)
    )
    b.turn(
        "what is going on here?\n" + log,
        "Every worker waits on the SETNX lock in acquire() and times out.",
        allow=["lock", "skew", "compose", "terse"],
        label="a long log full of known words",
    )
    b.end()
    s.check_fn("the log's recall stayed within budget", _bounded_last_recall)
    return s


def _items_at_most(n: int) -> Callable[[Simulator], str | None]:
    def check(sim: Simulator) -> str | None:
        k = len(sim.store.items)
        return None if k <= n else f"{k} items"

    return check


def _bounded_last_recall(sim: Simulator) -> str | None:
    recalls = [e for e in sim.events_of("knowledge_recall") if e.get("items")]
    if not recalls:
        return "nothing recalled for the log"
    items = recalls[-1]["items"]
    chars = sum(len(sim.store.items[i].statement) + len(sim.store.items[i].why) for i in items)
    if len(items) > 6 or chars > 2000:
        return f"{len(items)} items, {chars} chars"
    return None
