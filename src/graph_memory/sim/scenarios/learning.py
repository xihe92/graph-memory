"""Learning by asking, knowledge that goes stale, and cleaning up by hand."""

from __future__ import annotations

from collections.abc import Callable

from ..harness import Simulator, note_statement
from ..scenario import Scenario

Check = Callable[[Simulator], str | None]


def _recall_lines(sim: Simulator, prompt: str, project: str | None) -> list[str]:
    """What a brand-new session would be shown for `prompt` (not a scenario step)."""
    cwd = str(sim.repos[project]) if project else str(sim.scratch)
    res = sim._call(
        "recall", {"session": f"look-{len(sim.outcomes)}", "prompt": prompt, "cwd": cwd}
    )
    return list((res.get("context") or {}).get("knowledge") or [])


def _oracle_only(check: Check) -> Check:
    """A check of exactly what the oracle answers (its wording, its tags); a real model
    words and tags things its own way, so the check is skipped for it."""

    def run(sim: Simulator) -> str | None:
        return check(sim) if sim.oracle is not None else None

    return run


def _summary(name: str, graduations: int, *mentions: str) -> Check:
    """The session keeps one summary, folded from `graduations` graduations; with the
    oracle, its text carries `mentions` (the rolling summary reads the earlier one)."""

    def check(sim: Simulator) -> str | None:
        assert sim.daemon is not None
        key = f"claude-code:{sim.session(name).sid}"
        sums = sim.daemon.buffer.summaries(key)
        if len(sums) != 1 or sums[0].graduations != graduations:
            return f"summaries: {[(x.graduations, x.text[:60]) for x in sums]}"
        missing = [m for m in mentions if sim.oracle is not None and m not in sums[0].text]
        return f"summary {sums[0].text!r} lacks {missing}" if missing else None

    return check


def _disputes(key: str, n: int) -> Check:
    def check(sim: Simulator) -> str | None:
        items = sim.items(key)
        if not items:
            return f"no item states {key}"
        got = max(i.disputes for i in items)
        return None if got == n else f"{key}: {got} disputes, want {n}"

    return check


def _shown_with(prompt: str, project: str | None, key: str, *texts: str) -> Check:
    """A new session asking `prompt` is shown `key`'s note, and the note carries `texts`."""

    def check(sim: Simulator) -> str | None:
        lines = [
            ln
            for ln in _recall_lines(sim, prompt, project)
            if key in sim.keys_of(note_statement(ln))
        ]
        if not lines:
            return f"{key} not shown for {prompt!r}"
        if any(all(t in ln for t in texts) for ln in lines):
            return None
        return f"{key} shown as {lines}, none with all of {list(texts)}"

    return check


# --------------------------------------------------------------------------- Q&A


def recipe_facts(s: Scenario) -> str:
    recipes = s.project("recipes")
    s.entity("SQLite", "tool", "embedded SQL database")
    s.entity("recipe_tag", "component", "the join table of recipes and tags")
    s.entity("tag", "component", "the tag table")
    s.fact(
        "refs",
        "learned",
        "SQLite has no array column type, so a recipe's tags go in a recipe_tag join table "
        "with one row per recipe and tag",
        about=["SQLite", "recipe_tag", "tag"],
        match=["recipe_tag|join table", "one row|array"],
        origin="assistant",
        applies_to="SQLite 3",
        volatility="versioned",
    )
    s.fact(
        "lean",
        "open",
        "the user is weighing option A (a recipe_tag join table, tags fetched with a join) "
        "for recipes and tags in SQLite, leaning to it",
        about=["SQLite", "recipe_tag", "tag"],
        match=["option a"],
        origin="user",
    )
    s.fact(
        "fk",
        "learned",
        "SQLite enforces foreign keys only after PRAGMA foreign_keys = ON, which every "
        "connection must set",
        about=["SQLite", "recipe_tag"],
        match=["foreign key"],
        origin="docs",
        ref="https://www.sqlite.org/foreignkeys.html",
    )
    return recipes


def qa_learning() -> Scenario:
    s = Scenario(
        "qa_learning",
        "a slow session of questions about a library: what the user learns is kept with "
        "where it came from; each pause graduates into one rolling summary of the "
        "session, and the next day 'where were we' finds the thread",
    )
    recipes = recipe_facts(s)
    a = s.session("a", recipes)
    a.turn(
        "how does SQLite handle a many-to-many between recipes and tags?",
        "SQLite has no array column type. For a many-to-many, add a recipe_tag join table "
        "with one row per recipe and tag, and look the tags up through it.",
        learn=["refs"],
    )
    s.advance(minutes=30, label="a pause: the first question graduates")
    a.turn(
        "can a recipe row hold several tag ids in one column?",
        "Only as text such as a JSON list, which no foreign key can check and no index "
        "searches well; that is why the join table works better here.",
    )
    s.advance(minutes=30, label="another pause: it folds into the same summary")
    s.check_fn(
        "one summary covers both questions",
        _summary("a", 2, "many-to-many", "several tag ids"),
    )
    a.turn(
        "if we go with option A, I need a join to get recipes and their tags?",
        "Yes: select the recipes, join recipe_tag on recipe_id, then join tag. Going with "
        "option A keeps tags editable without touching the recipes.",
        learn=["lean"],
    )
    s.advance(minutes=30)
    a.turn(
        "why doesn't SQLite stop me from tagging a recipe that doesn't exist?",
        "I checked the SQLite docs on foreign keys: they are only enforced after "
        "PRAGMA foreign_keys = ON, and every connection must set it.",
        web=["https://www.sqlite.org/foreignkeys.html"],
        learn=["fk"],
    )
    s.advance(minutes=30)
    s.check_fn(
        "still one summary, four graduations in",
        _summary("a", 4, "many-to-many", "option A", "stop me"),
    )
    for key in ("refs", "lean", "fk"):  # a real model may split one into two
        s.check(key, min_count=1)
    s.advance(days=1)
    s.check_fn(
        "a note says when and where it came from",
        _shown_with(
            "how do sqlite join tables model recipes and tags?",
            recipes,
            "refs",
            "[learned, ",
            "from the assistant",
        ),
    )
    s.check_fn(
        "and what it applies to",
        _oracle_only(
            _shown_with(
                "how do sqlite join tables model recipes and tags?", recipes, "refs", ", SQLite 3"
            )
        ),
    )
    s.check_fn(
        "an answer read in the docs says so",
        _oracle_only(_shown_with("foreign keys enforced in sqlite", recipes, "fk", "from docs")),
    )
    s.check_fn("the docs page was recorded as read", _read_in_docs("fk"))
    s.probe(
        "where were we on the sqlite schema for recipes and tags?",
        recipes,
        has=["refs", "lean"],
        allow=["fk"],
        label="the next day, the design thread comes back",
    )
    s.probe("write a haiku about autumn leaves", None, nothing=True)
    return s


def _read_in_docs(key: str) -> Check:
    """Some item stating `key` came from docs: the page read during the turn was the
    evidence the graduation saw."""

    def check(sim: Simulator) -> str | None:
        got = [i.origin.value for i in sim.items(key)]
        return None if "docs" in got else f"{key}: origins {got}"

    return check


# --------------------------------------------------------------------------- topics


def _topic_scoped(topic: str, *keys: str) -> Check:
    """The items stating `keys` are scoped to `topic`, whatever the model spelled it."""

    def check(sim: Simulator) -> str | None:
        bad = [
            (k, i.scope)
            for k in keys
            for i in sim.items(k)
            if i.valid and i.scope.lower() != f"topic:{topic.lower()}"
        ]
        return f"not scoped to topic {topic}: {bad}" if bad else None

    return check


def topic_scope() -> Scenario:
    s = Scenario(
        "topic_scope",
        "what a research session outside any project learns about one technology is "
        "scoped to that topic: recalled in any project once the session is about it, "
        "not in a project session about something else; a general preference stays global",
    )
    planner = s.project("planner")
    s.entity("SQLite", "tool", "embedded SQL database")
    s.entity("recipe_tag", "component", "the join table of recipes and tags")
    s.fact(
        "fkoff",
        "learned",
        "SQLite enforces foreign keys only after PRAGMA foreign_keys = ON, which every "
        "connection must set",
        about=["SQLite"],
        match=["foreign key", "pragma|off|connection"],
        scope="topic",
        topic="SQLite",
        origin="docs",
    )
    s.fact(
        "pk",
        "decision",
        "recipe_tag's primary key is (recipe_id, tag_id), so adding the same tag to a recipe "
        "twice is a no-op under INSERT OR IGNORE",
        about=["SQLite", "recipe_tag"],
        match=["primary key"],
        scope="topic",
        topic="SQLite",
    )
    s.fact(
        "plain",
        "preference",
        "the user wants design choices explained in plain terms with their reasons",
        about=["the user"],
        match=["plain terms"],
        scope="global",
    )
    r = s.session("research")  # outside any project
    r.turn(
        "does SQLite check foreign keys by default?",
        "No. SQLite only enforces foreign keys after PRAGMA foreign_keys = ON, and every "
        "connection has to set it.",
        learn=["fkoff"],
    )
    r.turn(
        "then for recipe_tag, how do I stop the same tag being added to a recipe twice?",
        "Make (recipe_id, tag_id) the primary key: with INSERT OR IGNORE, adding a tag that "
        "is already there does nothing, so retries don't create duplicates.",
        learn=["pk"],
    )
    r.turn(
        "from now on, whatever we work on, explain design choices to me in plain terms "
        "with their reasons",
        "Will do: plain terms, and the reason behind each choice.",
        learn=["plain"],
    )
    s.advance(minutes=30, label="the research session graduates")
    s.check_fn(
        "SQLite items are scoped to the topic SQLite", _topic_scoped("SQLite", "fkoff", "pk")
    )
    s.check("plain", min_count=1)
    s.advance(days=1)
    s.probe(
        "how should the planner stop the same tag being added to a trip twice, primary key?",
        planner,
        lacks=["fkoff", "pk"],
        allow=["plain"],
        label="a project session that never mentions SQLite: no SQLite notes",
    )
    b = s.session("move", planner)
    b.turn(
        "which database should the planner use for its local cache?",
        "SQLite: it is one file with no server, and trips and tags become two of its tables.",
        lacks=["fkoff", "pk"],
        allow=["plain"],
    )
    b.turn(
        "how should we stop the same tag being added to a trip twice, primary key?",
        "As decided before for recipe_tag: make the two ids the primary key, so adding the "
        "same tag twice is a no-op under INSERT OR IGNORE.",
        has=["pk"],
        allow=["fkoff", "plain"],
        used=["pk"],
        label="the reply before named SQLite: the session is about it now",
    )
    s.probe(
        "does SQLite check foreign keys by default?",
        planner,
        has=["fkoff"],
        allow=["pk", "plain"],
        label="a prompt naming SQLite recalls it in any project",
    )
    return s


# --------------------------------------------------------------------------- stale


def stale_knowledge() -> Scenario:
    s = Scenario(
        "stale_knowledge",
        "an explanation that turns out wrong: the reply that contradicts it disputes it at "
        "once, its replacement from the docs supersedes it at graduation; one contradicted "
        "twice with no replacement is retracted; neither comes back",
    )
    recipes = s.project("recipes")
    s.entity("SQLite", "tool", "embedded SQL database")
    s.fact(
        "rebuild",
        "learned",
        "SQLite's ALTER TABLE cannot add a column to an existing table, so adding one means "
        "rebuilding the table and copying its rows",
        about=["SQLite"],
        match=[
            "alter table|add column|adding a column",
            "rebuild|copy|cannot|can't",
            "!in place|in-place|without rebuild|weigh|unresolved|undecided",
        ],
        origin="assistant",
        applies_to="SQLite 3",
        volatility="versioned",
    )
    s.fact(
        "inplace",
        "learned",
        "SQLite's ALTER TABLE ADD COLUMN adds the column in place without rebuilding the table",
        about=["SQLite"],
        match=["add column|adding a column", "in place|in-place|without rebuild"],
        origin="docs",
        applies_to="SQLite 3",
        volatility="versioned",
        supersedes=["rebuild"],
    )
    s.fact(
        "rename",
        "learned",
        "renaming a column in SQLite means recreating the table and copying every row",
        about=["SQLite"],
        match=["renam", "recreat|copy", "!weigh|unresolved|undecided|not yet decided"],
        origin="assistant",
        applies_to="SQLite 3",
        volatility="versioned",
    )
    a = s.session("a", recipes)
    a.turn(
        "how expensive is it to add a column to the recipe table later?",
        "SQLite's ALTER TABLE can't add a column to an existing table, so you rebuild the "
        "table and copy its rows.",
        learn=["rebuild"],
    )
    a.turn(
        "and when I rename a column?",
        "Then you recreate the table with the new name and copy every row.",
        learn=["rename"],
    )
    a.end()
    s.advance(days=1)

    b = s.session("b", recipes)
    b.turn(
        "can SQLite's ALTER TABLE really not add a column without rebuilding the table?",
        "I checked the SQLite docs: ALTER TABLE ADD COLUMN adds the column in place; the "
        "table is not rebuilt.",
        has=["rebuild"],
        allow=["rename"],
        contradicts=["rebuild"],
        web=["https://www.sqlite.org/lang_altertable.html"],
        learn=["inplace"],
        label="the docs say otherwise",
    )
    s.check_fn("contradicted once: disputed, not yet gone", _disputes("rebuild", 1))
    s.check("rebuild", valid=True)
    b.end()
    s.check("rebuild", valid=False, label="the replacement from the docs supersedes it")
    s.check("inplace", valid=True)

    c = s.session("c", recipes)
    c.turn(
        "does renaming a column in sqlite mean recreating the table?",
        "No: ALTER TABLE RENAME COLUMN changes the name directly, with no copy of the rows.",
        has=["rename"],
        allow=["inplace"],
        contradicts=["rename"],
    )
    s.check_fn("disputed", _disputes("rename", 1))
    s.check_fn(
        "a disputed note is shown with a warning",
        _shown_with("sqlite rename column recreate table", recipes, "rename", "[disputed "),
    )
    d = s.session("d", recipes)
    d.turn(
        "renaming a column and recreating the table in sqlite again",
        "There is no recreate: RENAME COLUMN changes the schema directly.",
        has=["rename"],
        allow=["inplace"],
        contradicts=["rename"],
    )
    s.check("rename", valid=False, label="contradicted twice: retracted")
    s.probe(
        "how do adding a column and renaming a column work in sqlite?",
        recipes,
        has=["inplace"],
        lacks=["rebuild", "rename"],
    )
    return s


# --------------------------------------------------------------------------- summary


def _summarized(project: str, has: list[str], lacks: list[str]) -> Check:
    """Have the model write `project`'s summary, then read it back: every point cites a
    note that is true and in that project; none cites `lacks`; with the oracle, which
    writes one point per note, `has` are all cited."""

    def check(sim: Simulator) -> str | None:
        res = sim._call("knowledge_summarize", {"scope": project})
        if not res["queued"]:
            return f"not queued: {res}"
        sim.settle()
        got = sim._call("knowledge_summary", {"scope": project})
        brief = got["summary"]
        if got["error"] or not brief or got["changed"]:
            return f"no current summary: {got['error']}, {got['changed']} changed"
        cited = {n for sec in brief["sections"] for pt in sec["points"] for n in pt["notes"]}
        if not cited:
            return "the summary cites nothing"
        scope = brief["scope"]
        heads = sim.store.heads
        stray = [n for n in cited if n not in heads or not heads[n].valid]
        stray += [n for n in cited if n in heads and heads[n].scope != scope]
        if stray:
            return f"cites notes that are not true or not in {project}: {stray}"
        keys = {k for it in sim.store.get_items(cited) for k in sim.keys_of(it.statement)}
        bad = [k for k in lacks if k in keys]
        if sim.oracle is not None:
            bad += [f"missing {k}" for k in has if k not in keys]
        return f"cited {sorted(keys)}: {bad}" if bad else None

    return check


def _summary_changed(project: str, at_least: int) -> Check:
    def check(sim: Simulator) -> str | None:
        got = sim._call("knowledge_summary", {"scope": project})
        ok = got["summary"] is not None and got["changed"] >= at_least
        return None if ok else f"{got['changed']} changed, want {at_least} or more"

    return check


def _item_count(n: int) -> Check:
    def check(sim: Simulator) -> str | None:
        got = len(sim.store.heads)
        return None if got == n else f"{got} items in memory, want {n}"

    return check


def summarize_scope() -> Scenario:
    s = Scenario(
        "summarize_scope",
        "a person asks for a summary of one project: the model writes it from that "
        "project's true notes only and every point cites one; it is a view, so memory "
        "holds no more than before; a replaced note marks it out of date, and writing it "
        "again leaves that note out",
    )
    s.oracle_only = True  # counts the items exactly; a real model may split one in two
    api = s.project("api")
    web = s.project("web")
    s.fact(
        "breaker",
        "decision",
        "the circuit breaker opens after five upstream failures within a minute",
        about=["circuit breaker"],
        why="a slow upstream took the whole api down",
    )
    s.fact(
        "helper",
        "fact",
        "the retry helper wraps every upstream call",
        about=["retry helper"],
    )
    s.fact(
        "theme",
        "preference",
        "the web client uses the dark theme by default",
        about=["web client"],
    )
    s.fact(
        "breaker10",
        "decision",
        "the circuit breaker opens after ten upstream failures within a minute, not five",
        about=["circuit breaker"],
        why="five tripped on ordinary deploys",
        match=["ten upstream"],
        supersedes=["breaker"],
    )
    a = s.session("a", api)
    a.turn(
        "add a circuit breaker",
        "The breaker opens after five failures a minute.",
        learn=["breaker"],
    )
    a.turn("wrap upstream calls", "A retry helper now wraps every upstream call.", learn=["helper"])
    a.end()
    w = s.session("w", web)
    w.turn("default theme?", "The web client now starts in the dark theme.", learn=["theme"])
    w.end()
    s.check_fn(
        "the summary cites the project's own notes",
        _summarized(api, has=["breaker", "helper"], lacks=["theme"]),
    )
    s.check_fn("a summary adds nothing to memory", _item_count(3))
    s.advance(days=1)
    b = s.session("b", api)
    b.turn(
        "the breaker trips on every deploy",
        "Raised the threshold: the breaker now opens after ten upstream failures a minute.",
        learn=["breaker10"],
        allow=["breaker", "helper"],
    )
    b.end()
    s.check("breaker", valid=False)
    s.check_fn("the summary is marked out of date", _summary_changed(api, 2))
    s.check_fn(
        "written again, it leaves the replaced note out",
        _summarized(api, has=["breaker10", "helper"], lacks=["breaker", "theme"]),
    )
    s.probe(
        "when does the circuit breaker open?",
        api,
        has=["breaker10"],
        lacks=["breaker", "theme"],
        allow=["helper"],
        label="recall never shows a summary, only notes",
    )
    return s


# --------------------------------------------------------------------------- delete


def delete_and_bridge() -> Scenario:
    s = Scenario(
        "delete_and_bridge",
        "a person deletes a noisy item in the middle of a chain: its neighbours get a "
        "bridge, recall still connects them, and the deletion can be undone",
    )
    s.oracle_only = True
    api = s.project("api")
    s.fact(
        "breaker",
        "decision",
        "the circuit breaker opens after five upstream failures within a minute",
        about=["circuit breaker"],
    )
    s.fact(
        "helper",
        "fact",
        "the retry helper wraps every upstream call",
        about=["retry helper"],
        related=[("breaker", "depends_on", "strong")],
    )
    s.fact(
        "gateway",
        "decision",
        "the gateway retries a failed request three times",
        about=["gateway"],
        related=[("helper", "depends_on", "strong")],
    )
    a = s.session("a", api)
    a.turn(
        "add a circuit breaker",
        "The breaker opens after five failures a minute.",
        learn=["breaker"],
    )
    a.turn("wrap upstream calls", "A retry helper now wraps every upstream call.", learn=["helper"])
    a.turn(
        "gateway retries?", "The gateway retries a failed request three times.", learn=["gateway"]
    )
    a.end()
    s.check_fn("delete the helper", _delete("helper"))
    s.check("helper", count=0)
    s.check_fn("the gateway and the breaker are bridged", _bridged("gateway", "breaker"))
    s.probe(
        "the gateway retries",
        api,
        has=["gateway", "breaker"],
        lacks=["helper"],
        label="recall still reaches the breaker from the gateway",
    )
    s.check_fn("undo the deletion", _undelete())
    s.check("helper", count=1)
    s.check_fn("the bridge is gone again", _not_bridged("gateway", "breaker"))
    return s


def _delete(key: str) -> Check:
    def check(sim: Simulator) -> str | None:
        ids = [i.item_id for i in sim.items(key)]
        if not ids:
            return f"no item states {key}"
        res = sim._call("knowledge_delete", {"ids": ids, "reason": "noise"})
        sim.batches.append(res["batch"])
        return None if res["bridges"] else f"no bridges: {res}"

    return check


def _undelete() -> Check:
    def check(sim: Simulator) -> str | None:
        res = sim._call("knowledge_undelete", {"batch": sim.batches.pop()})
        return None if res["restored"] else f"{res}"

    return check


def _edges(sim: Simulator, a: str, b: str) -> list[str]:
    ia = {i.item_id for i in sim.items(a)}
    ib = {i.item_id for i in sim.items(b)}
    return [
        e.origin
        for e in sim.store.scan_edges()
        if (e.source in ia and e.target in ib) or (e.source in ib and e.target in ia)
    ]


def _bridged(a: str, b: str) -> Check:
    def check(sim: Simulator) -> str | None:
        got = _edges(sim, a, b)
        return None if got == ["bridge"] else f"edges between {a} and {b}: {got}"

    return check


def _not_bridged(a: str, b: str) -> Check:
    def check(sim: Simulator) -> str | None:
        got = _edges(sim, a, b)
        return None if not got else f"edges between {a} and {b}: {got}"

    return check
