"""The memory lifecycle, one behaviour per scenario: learn and recall, compaction,
superseded decisions, reinforcement, use, decay."""

from __future__ import annotations

from collections.abc import Callable

from ..harness import Simulator
from ..scenario import Scenario


def limiter_facts(s: Scenario) -> str:
    """A rate limiter project and what working on it teaches."""
    lim = s.project("limiter")
    s.entity("Redis", "tool")
    s.entity("pytest", "tool")
    s.entity("acquire()", "symbol")
    s.fact(
        "lock",
        "decision",
        "svc/limiter.py guards acquire() with a Redis SETNX lock that expires after 30 seconds",
        about=["svc/limiter.py", "acquire()", "Redis"],
        why="workers raced on the shared counter and double counted requests",
        match=["setnx", "!lua", r"!skew|\bclock|redis time|time value"],
    )
    s.fact(
        "skew",
        "pitfall",
        "Clock skew between worker hosts breaks the window maths in svc/limiter.py, so the "
        "limiter reads time from Redis TIME instead of the host clock",
        about=["svc/limiter.py", "Redis"],
        match=["skew|redis time", r"!setnx|\block\b"],
    )
    s.fact(
        "compose",
        "procedure",
        "The limiter's integration tests need Redis started with `docker compose up redis` "
        "before running pytest",
        about=["Redis", "pytest"],
        match=["docker compose"],
    )
    s.fact(
        "terse",
        "preference",
        "The user wants short answers with no recap of the changes at the end",
        about=["the user"],
        scope="global",
        match=["short|concise|terse|brief"],
    )
    return lim


def teach_limiter(s: Scenario, lim: str, name: str = "a") -> None:
    a = s.session(name, lim)
    a.turn(
        "the rate limiter double counts under load. workers seem to race on the counter in "
        "svc/limiter.py",
        "The race is between the GET and the SET of the counter. I wrapped acquire() in a "
        "Redis SETNX lock with a 30 second expiry, so only one worker updates the counter at "
        "a time. The load test now counts exactly.",
        files=["svc/limiter.py"],
        learn=["lock"],
        nothing=True,
        label="the first prompt: memory is empty",
    )
    a.turn(
        "the tests fail with connection refused",
        "The integration tests talk to a real Redis. Start it with `docker compose up redis` "
        "and then run pytest: all 14 pass now.",
        learn=["compose"],
        nothing=True,
        label="this session's own learning is not shown back to it",
    )
    a.turn(
        "the windows also look wrong on the second worker host",
        "Window boundaries were computed from each host's clock, and the hosts are two "
        "seconds apart. svc/limiter.py now reads the time from Redis TIME, so every worker "
        "shares one clock.",
        files=["svc/limiter.py"],
        learn=["skew"],
    )
    a.turn(
        "please stop recapping everything you changed at the end, just keep answers short",
        "Understood: short answers, no recap.",
        learn=["terse"],
    )
    a.end()


def cross_session_recall() -> Scenario:
    s = Scenario(
        "cross_session_recall",
        "a session ends and graduates; the next session gets what applies, and nothing for "
        "unrelated prompts",
    )
    lim = limiter_facts(s)
    teach_limiter(s, lim)
    s.check("lock", area="survivor", valid=True, count=1)
    s.check("terse", valid=True, count=1)
    s.advance(hours=20)
    b = s.session("b", lim)
    b.turn(
        "acquire() is slow under load, can we make it faster?",
        "Most of the time goes into waiting for the SETNX lock: every worker serialises on "
        "it. A Lua script doing INCR and EXPIRE atomically would remove the lock entirely.",
        has=["lock"],
        allow=["skew", "compose", "terse"],
        used=["lock"],
        label="a related prompt recalls the decision",
    )
    b.turn(
        "open limiter.py and add a burst allowance of 20 requests",
        "Added a burst allowance of 20 on top of the steady rate in svc/limiter.py.",
        files=["svc/limiter.py"],
        allow=["lock", "skew", "compose", "terse"],
        unused=["skew", "compose"],
        label="what the session was already shown is not shown again",
    )
    s.probe(
        "open limiter.py and add a burst allowance of 20 requests",
        lim,
        has=["skew"],
        allow=["lock", "compose", "terse"],
        label="naming the file reaches knowledge that shares no words with the prompt",
    )
    # a new session starts with the standing preferences, whatever the prompt
    s.probe(
        "what is the weather in Seoul this weekend?",
        nothing=True,
        allow=["terse"],
        label="off-topic",
    )
    s.probe(
        "who scored for Arsenal last night",
        lim,
        nothing=True,
        allow=["terse"],
        label="off-topic in the project",
    )
    s.probe(
        "how do I run the integration tests?",
        lim,
        has=["compose"],
        allow=["lock", "skew", "terse"],
        label="a procedure",
    )
    web = s.project("webapp")
    s.probe(
        "add a dark mode toggle to the settings page",
        web,
        has=["terse"],
        lacks=["lock", "skew", "compose"],
        label="a global preference applies in another project; project knowledge does not",
    )
    return s


def compaction() -> Scenario:
    s = Scenario(
        "compaction",
        "a long session compacts twice and then ends: each part graduates once, the compact "
        "summary is read, and nothing is learned twice",
    )
    lim = limiter_facts(s)
    s.fact(
        "metrics",
        "decision",
        "Ops pages on-call when the Prometheus counter limiter_rejected_total rises above "
        "100 per minute",
        about=["limiter_rejected_total", "Prometheus"],
        why="sustained rejections mean a tenant is being throttled by mistake",
        match=["limiter_rejected_total"],
    )
    s.fact(
        "summary_only",
        "open",
        "Per-tenant limits are still to do: the limiter applies one global rate to all tenants",
        about=["svc/limiter.py"],
        match=["tenant"],
    )
    a = s.session("a", lim)
    a.turn(
        "the rate limiter double counts under load, workers race on the counter",
        "I wrapped acquire() in svc/limiter.py in a Redis SETNX lock with a 30 second expiry; "
        "counts are exact now.",
        files=["svc/limiter.py"],
        learn=["lock"],
    )
    a.turn(
        "tests fail with connection refused",
        "They need Redis: run `docker compose up redis`, then pytest. All pass.",
        learn=["compose"],
    )
    a.compact(
        "We fixed double counting in svc/limiter.py with a SETNX lock and got the integration "
        "tests running against Redis from docker compose. Per-tenant limits are still to do: "
        "the limiter applies one global rate to all tenants.",
        learn=["summary_only"],
    )
    s.check("lock", count=1, area="survivor")
    s.check("summary_only", valid=True, label="learned from the compact summary alone")
    a.turn(
        "ops want to be paged when we reject too many requests",
        "Agreed with ops: they page on-call when limiter_rejected_total rises above 100 per "
        "minute, since sustained rejections mean a tenant is throttled by mistake. I added "
        "the alert rule to their Prometheus config.",
        files=["svc/limiter.py"],
        learn=["metrics"],
    )
    a.turn(
        "the second worker host shows wrong windows",
        "The hosts' clocks are two seconds apart; svc/limiter.py now uses Redis TIME so all "
        "workers share a clock.",
        files=["svc/limiter.py"],
        learn=["skew", "lock"],
        label="restates the lock decision: reinforced, not duplicated",
    )
    a.turn(
        "what did we decide about the redis lock in acquire() earlier?",
        "We guard acquire() with a Redis SETNX lock with a 30 second expiry.",
        has=["lock"],
        allow=["compose", "summary_only", "terse"],
        label="after compaction the session no longer has its early decisions in context",
    )
    a.compact(
        "Added limiter_rejected_total and moved the window clock to Redis TIME. The SETNX lock "
        "from earlier stays."
    )
    a.turn("thanks, that's all for today", "Done.")
    a.end()
    s.check("lock", count=1, min_strength=2.0, area="old")
    s.check("metrics", count=1)
    s.check_fn(
        "three segments, each exchange graduated once, nothing left in the buffer",
        _segments_cover_once,
    )
    return s


def _segments_cover_once(sim: Simulator) -> str | None:
    segs = sim.store.segments()
    triggers = [sg.trigger for sg in segs]
    if triggers != ["compact", "compact", "session_end"]:
        return f"segments by trigger {triggers}"
    graduated = sum(sg.exchanges for sg in segs)
    total = len(sim.exchanges())
    if graduated != total:
        return f"{graduated} exchanges graduated for {total} stored"
    if sim.buffered():
        return f"left in the buffer after the session ended: {sim.buffered()}"
    return None


def supersede() -> Scenario:
    s = Scenario(
        "supersede",
        "a decision reversed in a later session: the old item becomes no longer true, stays "
        "in history, and is never recalled again",
    )
    lim = limiter_facts(s)
    s.fact(
        "lua",
        "decision",
        "svc/limiter.py replaces the SETNX lock with an atomic Redis Lua script doing INCR "
        "and EXPIRE",
        about=["svc/limiter.py", "Redis"],
        why="the lock serialised every worker and deadlocked when a worker died holding it",
        match=["lua", "!noscript|script load"],
        supersedes=["lock"],
    )
    s.fact(
        "lua_reload",
        "pitfall",
        "After a Redis failover the limiter's Lua script must be loaded again with SCRIPT "
        "LOAD, or EVALSHA fails with NOSCRIPT",
        about=["svc/limiter.py", "Redis"],
        match=["noscript|script load"],
        related=[("lua", "depends_on", "strong")],
    )
    teach_limiter(s, lim)
    s.advance(days=3)
    b = s.session("b", lim)
    b.turn(
        "workers hang forever sometimes, I think the lock in acquire() is stuck",
        "A worker died while holding the SETNX lock. Rather than tune the expiry I replaced "
        "the lock with an atomic Lua script doing INCR and EXPIRE in one call, so there is no "
        "lock to get stuck.",
        files=["svc/limiter.py"],
        learn=["lua"],
        has=["lock"],
        allow=["skew", "compose", "terse"],
        used=["lock"],
        label="the old decision is recalled when the problem shows up",
    )
    b.turn(
        "after the redis failover yesterday every request errors",
        "EVALSHA fails with NOSCRIPT: the new primary does not have the script. The limiter "
        "now calls SCRIPT LOAD again when it sees NOSCRIPT.",
        files=["svc/limiter.py"],
        learn=["lua_reload"],
    )
    b.end()
    s.check("lock", valid=False, count=1, label="the lock decision is history now")
    s.check("lua", valid=True, count=1)
    s.check_fn("a supersedes edge from lua to the lock", _supersedes_edge)
    s.advance(days=1)
    s.probe(
        "how does acquire() count requests in redis?",
        lim,
        has=["lua"],
        lacks=["lock"],
        allow=["skew", "compose", "lua_reload", "terse"],
        label="only the current decision is recalled",
    )
    s.probe(
        "requests fail with NOSCRIPT errors",
        lim,
        has=["lua_reload"],
        lacks=["lock"],
        allow=["lua", "skew", "compose", "terse"],
    )
    return s


def _supersedes_edge(sim: Simulator) -> str | None:
    (lua,) = sim.items("lua")
    (lock,) = sim.items("lock")
    for e in sim.store.scan_edges():
        if (
            e.relation.value == "supersedes"
            and e.source == lua.item_id
            and e.target == lock.item_id
        ):
            return None
    return "no supersedes edge"


def reinforce() -> Scenario:
    s = Scenario(
        "reinforce",
        "the same knowledge stated again in later sessions, in the same or other words, "
        "makes one item stronger instead of making copies; at strength 2 it becomes old",
    )
    lim = limiter_facts(s)
    teach_limiter(s, lim)
    s.check("compose", count=1, area="survivor", max_strength=1.0)
    s.advance(days=2)
    b = s.session("b", lim)
    b.turn(
        "tests can't connect to redis again",
        "Same as before: `docker compose up redis` first, then pytest.",
        learn=["compose"],
        allow=["compose", "lock", "skew", "terse"],
        label="stated again in the same words",
    )
    b.end()
    s.check("compose", count=1, area="old", min_strength=2.0)
    s.advance(days=2)
    c = s.session("c", lim)
    c.turn(
        "the integration tests can't reach redis again",
        "Same cause as before: Redis was not running. Bring it up with docker compose "
        "before pytest; they pass now.",
        learn=[
            (
                "compose",
                "Redis must be running via `docker compose up redis` before the limiter's "
                "pytest integration tests can pass",
            )
        ],
        allow=["compose", "lock", "skew", "terse"],
        label="stated again in other words",
    )
    c.end()
    s.check("compose", count=1, min_strength=3.0, label="the reworded copy merged as a duplicate")
    return s


def use_and_hebbian() -> Scenario:
    s = Scenario(
        "use_and_hebbian",
        "recalled items the reply uses get stronger and get linked to each other (co_used); "
        "items the reply ignores do not",
    )
    lim = limiter_facts(s)
    teach_limiter(s, lim)
    s.advance(days=1)
    b = s.session("b", lim)
    b.turn(
        "the counts are off again on the second host, is the redis lock in acquire() broken?",
        "The SETNX lock in acquire() is fine; the second host's windows are off because of "
        "clock skew. Redis TIME is used for the window maths in svc/limiter.py, but the new "
        "burst code read the host clock. Fixed to use Redis TIME too.",
        files=["svc/limiter.py"],
        has=["lock", "skew"],
        allow=["compose", "terse"],
        used=["lock", "skew"],
        label="both recalled and both used",
    )
    b.end()
    s.check("lock", min_uses=1, min_strength=1.5)
    s.check("skew", min_uses=1)
    s.check_fn("lock and skew linked by co_used", _co_used(0.15))
    s.advance(days=1)
    c = s.session("c", lim)
    c.turn(
        "the redis lock and the window clock in acquire() again: requests at the window edge "
        "get double counted",
        "At the window edge the SETNX lock is released before the Redis TIME read for the "
        "next window, so clock skew between hosts is not the cause this time. I moved the "
        "TIME read inside the lock in acquire().",
        files=["svc/limiter.py"],
        has=["lock", "skew"],
        allow=["compose", "terse"],
        used=["lock", "skew"],
        label="used together a second time",
    )
    c.end()
    s.check("lock", min_uses=2, area="old", label="two uses promote it")
    s.check_fn("co_used edge grew", _co_used(0.3))
    s.advance(days=1)
    d = s.session("d", lim)
    d.turn(
        "the redis lock in acquire(): should we shorten its expiry?",
        "Yes: 5 seconds is plenty since a single acquire() takes under a millisecond. Changed.",
        has=["lock"],
        allow=["skew", "compose", "terse"],
        unused=["compose", "terse"],
        label="recalled but irrelevant items are not counted as used",
    )
    return s


def _co_used(at_least: float) -> Callable[[Simulator], str | None]:
    def check(sim: Simulator) -> str | None:
        # the first item stating each fact; a real model may learn related ones later
        lock = sim.items("lock", valid_only=True)[0]
        skew = sim.items("skew", valid_only=True)[0]
        for e in sim.store.scan_edges():
            if e.relation.value == "co_used" and {e.source, e.target} == {
                lock.item_id,
                skew.item_id,
            }:
                return None if e.weight >= at_least - 1e-9 else f"weight {e.weight} < {at_least}"
        return "no co_used edge"

    return check


def decay() -> Scenario:
    s = Scenario(
        "decay",
        "unused survivors fade to dormant in about four weeks and stop being recalled; old "
        "items last much longer; a dormant item stated again comes back",
    )
    lim = limiter_facts(s)
    teach_limiter(s, lim)
    b = s.session("b", lim)
    b.turn(
        "tests can't connect to redis",
        "As before: `docker compose up redis`, then pytest.",
        learn=["compose"],
        allow=["compose", "lock", "skew", "terse"],
    )
    b.end()
    s.check("compose", area="old")
    s.check("lock", area="survivor")
    s.advance(days=20)
    s.check("lock", area="survivor", label="after 20 days a survivor is fading but recallable")
    s.probe("is acquire() protected by a redis lock?", lim, has=["lock"], allow=["skew", "compose"])
    s.advance(days=12)
    s.check("lock", area="dormant", label="after 32 days unused it is dormant")
    s.check("compose", area="old", label="an old item outlives it")
    s.probe(
        "is acquire() protected by a redis lock?",
        lim,
        lacks=["lock"],
        allow=["skew", "compose", "terse"],
        label="dormant items are not recalled",
    )
    s.probe(
        "how do I run the limiter integration tests?",
        lim,
        has=["compose"],
        allow=["lock", "skew", "terse"],
    )
    c = s.session("c", lim)
    c.turn(
        "workers race on the counter in svc/limiter.py again after the refactor",
        "The refactor dropped the lock. I put back the Redis SETNX lock with a 30 second "
        "expiry around acquire().",
        files=["svc/limiter.py"],
        learn=["lock"],
        allow=["skew", "compose", "terse"],
    )
    c.end()
    s.check("lock", count=1, area="old", label="stated again: revived and promoted")
    s.advance(days=1)
    s.probe(
        "is acquire() protected by a redis lock?",
        lim,
        has=["lock"],
        allow=["skew", "compose", "terse"],
    )
    s.advance(days=300)
    s.check("compose", area="dormant", label="even old items fade after months unused")
    return s
