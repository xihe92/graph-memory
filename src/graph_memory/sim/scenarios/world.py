"""Several projects, names that vary, sessions outside any repository."""

from __future__ import annotations

from collections.abc import Callable

from ..harness import Simulator
from ..scenario import Scenario
from .core import limiter_facts, teach_limiter


def project_isolation() -> Scenario:
    s = Scenario(
        "project_isolation",
        "two projects that both use Redis: project knowledge stays in its project, global "
        "knowledge goes everywhere, a session outside any repo sees only global knowledge",
    )
    lim = limiter_facts(s)
    bill = s.project("billing")
    s.fact(
        "invoice_cache",
        "decision",
        "billing/cache.py caches rendered invoices in Redis with a one hour TTL keyed by "
        "invoice id",
        about=["billing/cache.py", "Redis"],
        why="rendering a PDF invoice takes two seconds",
        match=["invoice"],
    )
    s.fact(
        "brew",
        "fact",
        "The user's laptop is an Apple Silicon Mac; system packages such as Redis are "
        "installed with Homebrew",
        about=["the user's laptop", "Homebrew"],
        scope="global",
        match=["homebrew|brew"],
    )
    teach_limiter(s, lim)
    b = s.session("b", bill)
    b.turn(
        "invoices take forever to render",
        "Rendering a PDF takes two seconds, so billing/cache.py now caches rendered invoices "
        "in Redis for an hour, keyed by invoice id.",
        files=["billing/cache.py"],
        learn=["invoice_cache"],
        lacks=["lock", "skew", "compose"],
        allow=["terse"],
        label="the other project's Redis knowledge stays out",
    )
    b.turn(
        "redis isn't installed on my laptop",
        "Installed it with Homebrew: `brew install redis`, then `brew services start redis`.",
        learn=["brew"],
        lacks=["lock", "skew", "compose"],
        allow=["terse"],
    )
    b.end()
    s.advance(days=1)
    s.probe(
        "the invoice cache in redis returns stale invoices after an edit",
        bill,
        has=["invoice_cache"],
        lacks=["lock", "skew", "compose"],
        allow=["brew", "terse"],
    )
    s.probe(
        "is the redis lock in acquire() still needed?",
        bill,
        lacks=["lock", "skew", "compose"],
        allow=["invoice_cache", "brew", "terse"],
        label="naming another project's symbol does not pull its knowledge in",
    )
    s.probe(
        "is the redis lock in acquire() still needed?",
        lim,
        has=["lock"],
        lacks=["invoice_cache"],
        allow=["skew", "compose", "brew", "terse"],
    )
    s.probe(
        "how do I install postgres on my laptop?",
        None,
        has=["brew"],
        lacks=["lock", "skew", "compose", "invoice_cache"],
        allow=["terse"],
        label="outside any repo: global knowledge only",
    )
    scratch = s.session("notes")
    scratch.turn(
        "summarise this article about redis cluster failover for me",
        "Redis Cluster promotes a replica when a majority of primaries agree a primary is "
        "down; writes acknowledged just before can be lost.",
        lacks=["lock", "skew", "compose", "invoice_cache"],
        allow=["brew", "terse"],
        label="a chat outside any repo learns nothing and sees no project knowledge",
    )
    scratch.end()
    s.check_fn("no item scoped to the scratch directory", _no_scratch_scope)
    return s


def _no_scratch_scope(sim: Simulator) -> str | None:
    bad = [i.statement for i in sim.store.items.values() if "scratch" in i.scope]
    return f"scoped to scratch: {bad}" if bad else None


def entity_names() -> Scenario:
    s = Scenario(
        "entity_names",
        "the same thing under different names is one entity: Postgres and PostgreSQL, a "
        "file by its path and by its basename; prompts reach items through any of them",
    )
    api = s.project("api")
    s.entity("PostgreSQL", "tool", "the API's relational database")
    s.entity("Postgres", "tool", "the relational database behind the API")
    s.alias("PostgreSQL", "Postgres")
    s.fact(
        "pool",
        "decision",
        "api/db.py caps the PostgreSQL connection pool at 20 connections per pod",
        about=["api/db.py", "PostgreSQL"],
        why="the database allows 200 connections and production runs 8 pods",
        match=["pool"],
    )
    s.fact(
        "orders_index",
        "fact",
        "Postgres has a btree index on orders.customer_id; order lookups by customer must "
        "filter on it",
        about=["Postgres", "orders.customer_id"],
        match=["customer_id"],
    )
    s.fact(
        "migrations",
        "procedure",
        "Schema changes go through alembic migrations in api/migrations; never edit tables by hand",
        about=["alembic", "db.py"],
        match=["alembic"],
    )
    a = s.session("a", api)
    a.turn(
        "we hit too many connections in production",
        "The database allows 200 connections and we run 8 pods, so api/db.py now caps the "
        "PostgreSQL pool at 20 connections per pod.",
        files=["api/db.py"],
        learn=["pool"],
    )
    a.turn(
        "how should the new table be created?",
        "Through an alembic migration in api/migrations, like every schema change; I added "
        "one for the table.",
        files=["api/migrations/0042_add_refunds.py"],
        learn=["migrations"],
    )
    a.end()
    s.check_fn("db.py merged into api/db.py", _one_entity("api/db.py", "db.py"))
    s.advance(days=1)
    b = s.session("b", api)
    b.turn(
        "listing a customer's orders is slow",
        "The query filtered on email instead of customer_id. Postgres has a btree index on "
        "orders.customer_id, so I switched the filter to it.",
        learn=["orders_index"],
    )
    b.end()
    s.check_fn("Postgres resolved to PostgreSQL", _one_entity("PostgreSQL", "Postgres"))
    s.advance(days=1)
    s.probe(
        "postgres is refusing connections again",
        api,
        has=["pool"],
        allow=["orders_index", "migrations"],
        label="the alias reaches what was learned under the other name",
    )
    s.probe(
        "change the default timeout in db.py",
        api,
        has=["pool"],
        allow=["orders_index", "migrations"],
        label="the basename reaches the file's knowledge",
    )
    return s


def _one_entity(*names: str) -> Callable[[Simulator], str | None]:
    def check(sim: Simulator) -> str | None:
        hits = {
            e.entity_id
            for e in sim.store.entities.values()
            for n in names
            if e.name.lower() == n.lower() or n.lower() in (a.lower() for a in e.aliases)
        }
        return None if len(hits) == 1 else f"{len(hits)} entities for {names}"

    return check
