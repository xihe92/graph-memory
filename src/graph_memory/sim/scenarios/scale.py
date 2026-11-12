"""Many projects, sessions and items, generated: recall stays correct and fast as memory
grows, and projects that share file and setting names stay apart."""

from __future__ import annotations

import random
from collections.abc import Callable

from ..harness import Simulator
from ..scenario import Scenario

COMPONENTS = [
    "auth",
    "billing",
    "search",
    "upload",
    "mailer",
    "scheduler",
    "gateway",
    "reports",
    "payments",
    "profile",
    "inventory",
    "audit",
]
REASONS = [
    "the upstream provider is slow at peak hours",
    "large tenants send bursts after midnight",
    "retries were piling up behind a slow dependency",
    "p99 latency doubled after the last migration",
]
TOOLS = ["Celery", "Kafka", "S3", "Elasticsearch", "Stripe", "SendGrid", "Redis", "nginx"]


def scale(projects: int = 5, components: int = 12, seed: int = 7) -> Scenario:
    rnd = random.Random(seed)
    s = Scenario(
        "scale",
        f"{projects} projects x {components} components x 3 facts, one session per "
        "component; then probes for specific facts in each project",
    )
    s.expensive = True  # one graduation per component
    names = [f"proj{p}" for p in range(projects)]
    for n in names:
        s.project(n)
    keys: dict[tuple[str, str, str], str] = {}
    for proj in names:
        for c in COMPONENTS[:components]:
            up = c.upper()
            val = 5 * rnd.randint(2, 24)
            tool = rnd.choice(TOOLS)
            reason = rnd.choice(REASONS)
            file = f"{c}/{c}_service.py"
            k_timeout = f"{proj}-{c}-timeout"
            k_tool = f"{proj}-{c}-tool"
            k_pitfall = f"{proj}-{c}-pitfall"
            s.fact(
                k_timeout,
                "decision",
                f"In {proj}, {file} sets {up}_TIMEOUT to {val} seconds",
                about=[file, f"{up}_TIMEOUT"],
                why=reason,
                match=[f"in {proj}", f"{c}_timeout", f"{val} seconds"],
            )
            s.fact(
                k_tool,
                "fact",
                f"In {proj}, the {c} component hands background work to {tool} through "
                f"{c}/{c}_tasks.py",
                about=[f"{c}/{c}_tasks.py", tool],
                match=[f"in {proj}", f"{c}_tasks"],
            )
            s.fact(
                k_pitfall,
                "pitfall",
                f"In {proj}, {c} tests hang unless {up}_SANDBOX=1 is exported, because "
                f"the {c} client otherwise calls the live {tool} API",
                about=[f"{up}_SANDBOX", file],
                match=[f"in {proj}", f"{c}_sandbox"],
            )
            keys[(proj, c, "timeout")] = k_timeout
            keys[(proj, c, "tool")] = k_tool
            keys[(proj, c, "pitfall")] = k_pitfall
            ses = s.session(f"{proj}-{c}", proj)
            ses.turn(
                f"{c} requests time out under load",
                f"Because {reason}, I raised {up}_TIMEOUT in {file} of {proj} to {val} seconds.",
                files=[file],
                learn=[k_timeout],
            )
            ses.turn(
                f"where does {c} run its background jobs?",
                f"In {proj}, {c}/{c}_tasks.py hands them to {tool}.",
                files=[f"{c}/{c}_tasks.py"],
                learn=[k_tool],
            )
            ses.turn(
                f"the {c} tests hang forever",
                f"They call the live {tool} API unless {up}_SANDBOX=1 is exported; I added "
                f"it to the test environment of {proj}.",
                learn=[k_pitfall],
            )
            ses.end()
        s.advance(days=1)
    for _ in range(3 * projects):
        proj = rnd.choice(names)
        c = rnd.choice(COMPONENTS[:components])
        kind = rnd.choice(["timeout", "tool", "pitfall"])
        prompt = {
            "timeout": f"what is {c.upper()}_TIMEOUT set to?",
            "tool": f"which queue does {c}/{c}_tasks.py use?",
            "pitfall": f"{c} tests hang on CI, why?",
        }[kind]
        same_component = [keys[(proj, c, k)] for k in ("timeout", "tool", "pitfall")]
        elsewhere = [keys[(q, c, kind)] for q in names if q != proj]
        s.probe(
            prompt,
            proj,
            has=[keys[(proj, c, kind)]],
            lacks=elsewhere,
            allow=same_component,
            label=f"{proj} {c} {kind}",
        )
    s.check_fn("recall p95 under 100 ms", _latency(100.0))
    return s


def _latency(limit_ms: float) -> Callable[[Simulator], str | None]:
    def check(sim: Simulator) -> str | None:
        ms = sorted(e["ms"] for e in sim.events_of("knowledge_recall") if "ms" in e)
        if not ms:
            return "no recalls"
        p95 = ms[min(len(ms) - 1, int(0.95 * len(ms)))]
        return None if p95 <= limit_ms else f"p95 {p95} ms over {limit_ms} ms"

    return check
