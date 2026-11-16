"""`gm`: hook client and admin commands.

Hook commands (start, recall, remember, graduate) never fail the agent: on any error they
log to ~/.graph-memory/gm.log and exit 0, so the agent carries on without memory for that
turn. They do nothing when GM_DISABLE is set (model subprocesses run with it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import clock
from .adapters import load_adapter
from .client import DaemonUnavailable, RemoteError, call, knowledge_items
from .config import Config, load_config
from .events import EventLog, events_path, read_events
from .models import DISABLE_ENV

log = logging.getLogger("graph_memory.cli")


def render_context(result: dict[str, Any]) -> str:
    """The text block handed to the agent. Empty when there is nothing worth adding."""
    knowledge = (result.get("context") or {}).get("knowledge") or []
    if not knowledge:
        return ""
    lines = ["<graph-memory>", "Knowledge from earlier sessions that may apply here:"]
    lines.extend(f"- {k}" for k in knowledge)
    lines.append(
        "Each note shows when it was last confirmed and where it came from. Notes can be "
        "out of date: what you see now in the code, docs or tool output wins, and say so "
        "when it contradicts a note."
    )
    lines.append("</graph-memory>")
    return "\n".join(lines) + "\n"


def _read_payload() -> dict[str, Any]:
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    raw = sys.stdin.read()
    return json.loads(raw) if raw.strip() else {}


def _hook(cfg: Config, args: argparse.Namespace) -> int:
    if os.environ.get(DISABLE_ENV):
        return 0
    started = time.perf_counter()
    error: str | None = None
    try:
        payload = _read_payload()
        if args.cmd == "start":
            call(cfg, "ping", timeout=cfg.recall_timeout)
            return 0
        adapter = load_adapter(args.agent)
        if not cfg.remember_headless and adapter.is_headless(payload):
            return 0
        if args.cmd == "recall":
            ev = adapter.parse_prompt_event(payload)
            if ev.system:
                return 0
            res = call(
                cfg,
                "recall",
                {
                    "session": adapter.session_key(ev.session_id),
                    "prompt": ev.prompt,
                    "cwd": ev.cwd,
                },
                timeout=cfg.recall_timeout,
            )
            adapter.emit_context(render_context(res))
        elif args.cmd == "graduate":
            # SessionEnd, or SessionStart after a compaction: graduate what the session
            # has added since its last graduation, with the agent's own summary if any
            sid = payload.get("session_id")
            if not sid:
                return 0
            compacted = payload.get("source") == "compact" or "trigger" in payload
            summary = adapter.compact_summary(payload.get("transcript_path")) if compacted else ""
            call(
                cfg,
                "graduate",
                {
                    "session": adapter.session_key(str(sid)),
                    "trigger": "compact" if compacted else "session_end",
                    "compact_summary": summary,
                },
                timeout=cfg.recall_timeout,
            )
        elif args.cmd == "remember":
            stop = adapter.parse_stop_event(payload)
            turn = adapter.read_turn(stop)
            if turn is None:
                raise RuntimeError(f"could not read the turn from {stop.transcript_path}")
            call(
                cfg,
                "remember",
                {
                    "session": adapter.session_key(stop.session_id),
                    "turn": turn.model_dump(mode="json"),
                },
                timeout=cfg.recall_timeout,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.exception("hook %s failed", args.cmd)
    EventLog(events_path(cfg.home), rotate=False, keep_recent=0).emit(
        "hook",
        hook=args.cmd,
        ms=round((time.perf_counter() - started) * 1000, 1),
        ok=error is None,
        error=error[:300] if error else None,
    )
    return 0


def _stats(cfg: Config, args: argparse.Namespace) -> int:
    from .events import events_path, parse_since, read_events
    from .stats import compute_stats, format_help, format_stats, memory_help

    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    events = list(read_events(events_path(cfg.home), since=since))
    st = compute_stats(events)
    help_ = memory_help(events)
    try:
        status = call(cfg, "status", autostart=False)
    except DaemonUnavailable:
        status = None
    if args.json:
        print(json.dumps({**st, "status": status, "memory_help": help_}, indent=2))
    else:
        print(format_stats(st, status))
        if st["window"]["first"]:
            print(format_help(help_, knowledge_items(cfg)))
    return 0


def _ui(cfg: Config, args: argparse.Namespace) -> int:
    """Open the page the daemon serves, starting the daemon if needed. With --port, or
    when the daemon serves no page (ui_port = 0, or its port was taken), run a server
    here in the foreground instead."""
    from .webui import serve_ui

    if args.port is None:
        try:
            url = call(cfg, "status").get("ui")
        except (DaemonUnavailable, RemoteError) as exc:
            print(f"daemon unavailable: {exc}", file=sys.stderr)
            return 1
        if url:
            print(f"graph-memory UI on {url}  (up while the daemon runs)")
            if not args.no_open:
                import webbrowser

                webbrowser.open(url)
            return 0
        print(
            f"the daemon serves no page (ui_port = {cfg.ui_port}; see {cfg.log_path}); "
            "running one here",
            file=sys.stderr,
        )
    port = args.port if args.port is not None else (cfg.ui_port or 7477)
    return serve_ui(cfg, port=port, open_browser=not args.no_open)


def _migrate(cfg: Config, args: argparse.Namespace) -> int:
    """The moves memory still needs, in order: off the conversation graph, then from the
    JSON files into memory.db. Each runs once; with --dry-run, only the plans."""
    from . import migrate

    old = migrate.plan(cfg.home)
    print(old.text())
    if old.needed and args.dry_run:
        print("Then: the knowledge moves from the JSON files into memory.db (SQLite).")
        return 0
    to_db = migrate.sqlite_plan(cfg.home) if not old.needed else None
    if to_db is not None:
        print(to_db.text())
    if args.dry_run or not (old.needed or (to_db is not None and to_db.needed)):
        return 0
    try:
        call(cfg, "ping", autostart=False, timeout=2)
    except DaemonUnavailable:
        pass
    else:
        print(
            "The daemon is running: stop it first with `gm stop` (it starts again on the "
            "next prompt).",
            file=sys.stderr,
        )
        return 1
    if old.needed:
        res = migrate.apply(old)
        print(
            f"Done: {res['buffered']} exchanges moved to the buffer; archive "
            f"{res['archive']} ({res['archive_bytes'] / 1e6:.1f} MB); "
            f"removed {', '.join(res['removed'])}"
        )
        to_db = migrate.sqlite_plan(cfg.home)
        print(to_db.text())
    if to_db is not None and to_db.needed:
        try:
            res = migrate.sqlite_apply(to_db)
        except RuntimeError as exc:
            print(f"Not moved, the JSON files are as they were: {exc}", file=sys.stderr)
            return 1
        print(
            f"Done: {res['items']} items, {res['entities']} entities, {res['edges']} edges, "
            f"{res['vectors']} vectors now in {cfg.home / 'memory.db'}, checked against the "
            f"JSON files; those are in {res['archive']} "
            f"({res['archive_bytes'] / 1e6:.1f} MB). `gm export` writes them out again."
        )
    return 0


def _export(cfg: Config, args: argparse.Namespace) -> int:
    """Memory written out in the JSON format, into a folder: readable in any editor, and
    what a version from before SQLite reads."""
    from .knowledge import KnowledgeStore
    from .migrate import export_json

    dest = (
        Path(args.dest).expanduser()
        if args.dest
        else (Path.home() / f"graph-memory-export-{clock.now().astimezone().date().isoformat()}")
    )
    if dest.resolve() == cfg.home.resolve():
        print("export into another folder than the memory itself", file=sys.stderr)
        return 2
    store = KnowledgeStore(cfg.home)  # reading only: fine while the daemon runs
    try:
        counts = export_json(store, dest)
    finally:
        store.close()
    print(
        f"Wrote {counts['items']} items, {counts['entities']} entities, {counts['edges']} "
        f"edges, {counts['vectors']} vectors, {counts['deletions']} deletions to {dest}"
    )
    return 0


def _knowledge(cfg: Config, args: argparse.Namespace) -> int:
    """`gm knowledge [action]`: with no action, or `list`, every item by type."""
    actions = {
        "backfill": _backfill,
        "review": lambda cfg, args: _review(cfg),
        "graduate": _graduate_session,
        "regraduate": _graduate_session,
        "resummarize": _resummarize,
        "rescope": _rescope,
        "scope": _scope,
        "summarize": _summarize,
        **dict.fromkeys(("show", "retract", "restore", "confirm", "delete", "undelete"), _by_id),
    }
    return actions.get(args.action, _list_knowledge)(cfg, args)


def _backfill(cfg: Config, args: argparse.Namespace) -> int:
    res = call(cfg, "graduate", {"all": True})
    print(f"queued {res['queued']} sessions; watch with `gm status` or `gm knowledge`")
    return 0


def _by_id(cfg: Config, args: argparse.Namespace) -> int:
    """The actions on one item (or one deletion, for `undelete`)."""
    if not args.id:
        what = "a batch id" if args.action == "undelete" else "an id"
        print(f"knowledge {args.action} needs {what}", file=sys.stderr)
        return 2
    if args.action == "show":
        print(json.dumps(call(cfg, "knowledge_item", {"id": args.id[0]}), indent=2))
        return 0
    try:
        if args.action == "delete":
            return _delete(cfg, args)
        if args.action == "undelete":
            res = call(cfg, "knowledge_undelete", {"batch": args.id[0]})
            print(f"put back {res['restored']} items (deletion {res['batch']})")
            return 0
        res = call(cfg, f"knowledge_{args.action}", {"id": args.id[0], "reason": args.reason})
    except RemoteError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.action == "confirm":
        print(f"{res['id'][:12]} confirmed, its review starts over: {res['statement']}")
        return 0
    state = "true again" if res["valid"] else "no longer true"
    print(f"{res['id'][:12]} is now {state}: {res['statement']}")
    return 0


def _list_knowledge(cfg: Config, args: argparse.Namespace) -> int:
    view = call(cfg, "knowledge")
    ents = {e["id"]: e["name"] for e in view["entities"]}
    items = [i for i in view["items"] if args.all or i["valid"]]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for i in items:
        by_type.setdefault(i["type"], []).append(i)
    for t, group in sorted(by_type.items()):
        print(f"\n{t} ({len(group)})")
        for i in sorted(group, key=lambda i: i["created_at"]):
            mark = "" if i["valid"] else "  [no longer true]"
            if i["valid"] and i.get("area") != "survivor":
                mark = f"  [{i.get('area')}]"
            if i.get("uses"):
                mark += f"  used {i['uses']}x"
            about = ", ".join(ents.get(e, "?") for e in i["entities"])
            print(f"  {i['id'][:8]}  {i['statement']}{mark}")
            if about:
                print(f"            about: {about}")
    st = view["stats"]
    print(
        f"\n{st['items']} items ({st['invalidated']} no longer true), {st['entities']} "
        f"entities, {st['edges']} edges, from {st['segments']} segments"
    )
    return 0


def _delete(cfg: Config, args: argparse.Namespace) -> int:
    view = call(cfg, "knowledge")
    hits = [i for i in view["items"] if any(i["id"].startswith(r) for r in args.id)]
    if not args.yes:
        for i in hits:
            print(f"  {i['id'][:8]}  {i['statement']}")
        if sys.stdin is None or not sys.stdin.isatty():
            print("delete these? add --yes to confirm", file=sys.stderr)
            return 1
        if input(f"delete {len(hits)} items? [y/N] ").strip().lower() not in ("y", "yes"):
            print("nothing deleted")
            return 1
    res = call(cfg, "knowledge_delete", {"ids": args.id, "reason": args.reason})
    print(
        f"deleted {len(res['deleted'])} items, {len(res['entities'])} entities no item names "
        f"any more; {res['bridges']} bridge edges keep their neighbours linked"
    )
    print(f"undo with: gm knowledge undelete {res['batch']}")
    return 0


def _graduate_session(cfg: Config, args: argparse.Namespace) -> int:
    """`graduate`: what a session has waiting, now. `regraduate`: the whole session again
    from its transcript, under the current rules; `--preview` on a copy of memory first,
    `--apply` to keep a preview."""
    if not args.id:
        print(f"knowledge {args.action} needs a session id (or a prefix of it)", file=sys.stderr)
        return 2
    if args.action == "regraduate" and args.apply:
        try:
            res = call(cfg, "apply_rebuild", {"session": args.id[0]})
        except RemoteError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(
            f"applied the rebuild of {res['session']}: {len(res['new'])} new items, "
            f"{len(res['retired'])} no longer true, {res['replaced']} summaries replaced by one"
        )
        return 0
    method = "graduate_now" if args.action == "graduate" else "regraduate"
    params: dict[str, Any] = {"session": args.id[0]}
    if args.action == "regraduate":
        params["preview"] = args.preview
        params["force"] = args.force
    try:
        res = call(cfg, method, params)
    except RemoteError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.action == "graduate" and not res["queued"]:
        print(f"{res['session']}: nothing waiting to graduate")
        return 0
    what = f"{res['waiting']} exchanges" if args.action == "graduate" else "its transcript"
    print(f"graduating {res['session']} ({what})")
    if args.action == "regraduate" and not args.preview and not args.force:
        print(
            f"a session in more than {res['long_parts']} parts is previewed first and "
            "changes nothing until you add --apply (--force skips the preview)"
        )
    if not args.wait:
        print("it runs in the background; see `gm status`, or add --wait")
        return 0
    deadline = time.monotonic() + 1800
    while not call(cfg, "settled")["settled"]:
        if time.monotonic() > deadline:
            print("still running after 30 minutes", file=sys.stderr)
            return 1
        time.sleep(1)
    done = [
        e
        for e in read_events(events_path(cfg.home), kinds=["graduate"])
        if e.get("session") == res["session"]
    ]
    last = done[-1] if done else {}
    if not last.get("ok"):
        print(f"failed: {last.get('error', 'no result')}", file=sys.stderr)
        return 1
    if last.get("preview"):
        if last.get("long"):
            print(f"long session ({last.get('parts')} parts): previewed, nothing changed yet\n")
        return _show_preview(cfg, res["session"])
    print(
        f"done: {last.get('exchanges', 0)} exchanges -> {last.get('items', 0)} items "
        f"({last.get('new', 0)} new)"
        + (f"; {last['replaced']} old summaries replaced by one" if last.get("replaced") else "")
    )
    return 0


def _resummarize(cfg: Config, args: argparse.Namespace) -> int:
    if not args.id:
        print("knowledge resummarize needs a session id (or a prefix of it)", file=sys.stderr)
        return 2
    try:
        res = call(cfg, "resummarize", {"session": args.id[0]})
    except RemoteError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not res["queued"]:
        print(f"{res['session']}: {res['summaries']} summary, nothing to fold")
        return 0
    deadline = time.monotonic() + 600
    while not call(cfg, "settled")["settled"] and time.monotonic() < deadline:
        time.sleep(0.5)
    done = [
        e
        for e in read_events(events_path(cfg.home), kinds=["summaries_folded"])
        if e.get("session") == res["session"]
    ]
    if not done or not done[-1].get("ok"):
        print(f"failed: {done[-1].get('error') if done else 'no result'}", file=sys.stderr)
        return 1
    print(f"{res['session']}: {done[-1]['replaced']} summaries folded into one")
    return 0


def _rescope(cfg: Config, args: argparse.Namespace) -> int:
    """The model reads every item and proposes where it should be recalled; `--apply`
    keeps the proposal."""
    try:
        if args.apply:
            res = call(cfg, "apply_rescope", {})
            print(f"rescoped {res['changed']} items", end="")
            print(
                f"; {res['skipped']} changed since the preview kept theirs"
                if res["skipped"]
                else ""
            )
            return 0
        res = call(cfg, "rescope", {})
    except RemoteError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"the model is reading {res['items']} items for their scope; nothing changes yet")
    if not args.wait:
        print("it runs in the background; see it with `gm knowledge rescope --wait`")
        return 0
    deadline = time.monotonic() + 1800
    while not call(cfg, "settled")["settled"]:
        if time.monotonic() > deadline:
            print("still running after 30 minutes", file=sys.stderr)
            return 1
        time.sleep(1)
    runs = list(read_events(events_path(cfg.home), kinds=["rescope"]))
    last: dict[str, Any] = runs[-1] if runs else {}
    if not last.get("ok"):
        print(f"failed: {last.get('error', 'no result')}", file=sys.stderr)
        return 1
    return _show_rescope(cfg)


def _summarize(cfg: Config, args: argparse.Namespace) -> int:
    """Print the brief of one project or topic; the model writes it again when notes
    changed since the last one (or with --force)."""
    scope = " ".join(args.id).strip()
    if not scope:
        print(
            "knowledge summarize needs a scope: global, topic:<name>, or a project's name",
            file=sys.stderr,
        )
        return 2
    try:
        res = call(cfg, "knowledge_summarize", {"scope": scope, "force": args.force})
        if res["running"]:
            print(f"the model is reading {res['items']} notes…", file=sys.stderr)
            deadline = time.monotonic() + 900
            while call(cfg, "knowledge_summary", {"scope": scope})["running"]:
                if time.monotonic() > deadline:
                    print("still running after 15 minutes", file=sys.stderr)
                    return 1
                time.sleep(1)
        got = call(cfg, "knowledge_summary", {"scope": scope})
    except RemoteError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not got["summary"] or (res["running"] and got["error"]):
        print(f"failed: {got['error'] or 'no summary was written'}", file=sys.stderr)
        return 1
    s = got["summary"]
    kind = (
        "global"
        if s["scope"] == "global"
        else "topic"
        if s["scope"].startswith("topic:")
        else "project"
    )
    head = "global notes" if kind == "global" else f"{kind} {s['name']}"
    left = f", {s['left_out']} left out" if s["left_out"] else ""
    notes = f"{s['items']} note{'' if s['items'] == 1 else 's'}{left}"
    print(f"Summary of {head}: {notes}, written {s['at'][:16].replace('T', ' ')}")
    if s["overview"]:
        print(f"\n{s['overview']}")
    for sec in s["sections"]:
        print(f"\n{sec['title'].upper()}")
        for pt in sec["points"]:
            print(f"- {pt['text']} [{', '.join(n[:8] for n in pt['notes'])}]")
    print("\nthe ids in brackets are the notes behind each point: gm knowledge show <id>")
    return 0


def _show_rescope(cfg: Config) -> int:
    plan = call(cfg, "rescope_preview", {})
    names = {x["key"]: x["name"] for x in plan["scopes"]}

    def label(scope: str) -> str:
        if scope == "global":
            return "global"
        kind = "topic" if scope.startswith("topic:") else "project"
        return f"{kind} {names.get(scope, scope)}"

    changes = plan["changes"]
    if not changes:
        print("every item keeps its scope")
        return 0
    by_to: dict[str, list[dict[str, Any]]] = {}
    for c in changes:
        by_to.setdefault(c["to"], []).append(c)
    for to, group in sorted(by_to.items(), key=lambda t: (-len(t[1]), t[0])):
        print(f"\n-> {label(to)} ({len(group)})")
        for c in group:
            print(f"  {c['id'][:8]}  {c['statement'][:110]}  [was {label(c['from'])}]")
    print(f"\n{len(changes)} items would move. keep it: gm knowledge rescope --apply")
    print("or fix one by hand: gm knowledge scope <id> --to global|topic:<name>|<project>")
    return 0


def _scope(cfg: Config, args: argparse.Namespace) -> int:
    if not args.id or not args.to:
        print(
            "knowledge scope needs item ids and --to global|topic:<name>|<project>", file=sys.stderr
        )
        return 2
    try:
        res = call(cfg, "knowledge_scope", {"ids": args.id, "to": args.to})
    except RemoteError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{len(res['ids'])} items now scoped {res['scope']}")
    return 0


def _show_preview(cfg: Config, session: str) -> int:
    p = call(cfg, "rebuild_preview", {"session": session})
    print(
        f"preview: {p['exchanges']} exchanges in {p['parts']} parts would add "
        f"{len(p['new'])} items, match {len(p['matched'])} known ones, and retire "
        f"{len(p['retired'])}"
    )
    print(f"\nsummary: {p['summary']}")
    for title, key in (("new", "new"), ("no longer true", "retired")):
        if p[key]:
            print(f"\n{title}:")
            for i in p[key]:
                gone = "" if i["valid"] else "  [no longer true]"
                print(f"  [{i['type']}] {i['statement']}{gone}")
    print(f"\nkeep it: gm knowledge regraduate {session.split(':', 1)[-1][:8]} --apply")
    return 0


def _review(cfg: Config) -> int:
    res = call(cfg, "knowledge_review")
    if not res["items"]:
        print("nothing to review")
    for i in res["items"]:
        state = "" if i["recalled"] else "  [not recalled]"
        print(f"  {i['id'][:8]}  {i['statement']}\n            {i['why']}{state}")
    if res["items"]:
        print(
            "\nconfirm what still holds (gm knowledge confirm <id>), retract what doesn't "
            "(gm knowledge retract <id>)"
        )
    if res["deleted"]:
        print("\ndeletions you can undo (gm knowledge undelete <batch>):")
        for b in res["deleted"][:10]:
            print(
                f"  {b['batch']}  {b['at'][:16]}  {len(b['statements'])} items: "
                f"{b['statements'][0][:60]}"
            )
    return 0


def _knowledge_actions(actions: Any) -> None:
    """`gm knowledge <action>`: each action with only the options it takes. A missing id
    is reported by the action itself, with what it needs."""

    def action(name: str, about: str, what: str = "", help_: str = "") -> argparse.ArgumentParser:
        p: argparse.ArgumentParser = actions.add_parser(name, help=about, description=about)
        if what:
            p.add_argument("id", nargs="*", metavar=what, help=help_)
        return p

    p = action("list", "every item by type, with its entities (the default)")
    # SUPPRESS: `gm knowledge --all` keeps its value when `list` is named after it
    p.add_argument(
        "--all", action="store_true", default=argparse.SUPPRESS, help="include items no longer true"
    )
    action(
        "show",
        "one item or entity, its neighbours and the graduations it came from",
        "ID",
        "an item or entity id (a prefix will do)",
    )
    action("backfill", "graduate every session waiting in the buffer now")
    p = action(
        "retract",
        "mark an item no longer true: kept for history, never recalled",
        "ID",
        "an item id (a prefix will do)",
    )
    p.add_argument("--reason", default="", help="why it is no longer true")
    for name, about in (
        ("restore", "make an item true again (undoes a retraction or the model's)"),
        ("confirm", "it still holds: clears a dispute and starts its review period over"),
    ):
        action(name, about, "ID", "an item id (a prefix will do)").set_defaults(reason="")
    action("review", "disputed items, items due for review, and deletions you can undo")
    p = action(
        "delete",
        "remove noise for good; neighbours linked through it stay linked",
        "ID",
        "one or more item ids (prefixes will do)",
    )
    p.add_argument("--reason", default="", help="why, kept with the deletion")
    p.add_argument("--yes", action="store_true", help="delete without asking")
    action(
        "undelete",
        "put a deletion back as it was",
        "BATCH",
        "the deletion's batch id (`gm knowledge review` lists them)",
    )
    session = "a session id (a prefix will do)"
    p = action(
        "graduate",
        "graduate a session's waiting exchanges now; the session goes on",
        "SESSION",
        session,
    )
    p.add_argument("--wait", action="store_true", help="wait until it is done")
    p = action(
        "regraduate",
        "graduate a whole session again from its transcript, under the current rules",
        "SESSION",
        session,
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="run it on a copy of memory and show what would change",
    )
    p.add_argument("--apply", action="store_true", help="keep the last preview")
    p.add_argument(
        "--force",
        action="store_true",
        help="change memory directly even for a long session (no preview first)",
    )
    p.add_argument("--wait", action="store_true", help="wait until it is done")
    action(
        "resummarize",
        "fold a session's summaries into one; knowledge is not touched",
        "SESSION",
        session,
    )
    p = action(
        "rescope",
        "the model proposes where each item should be recalled; nothing changes until --apply",
    )
    p.add_argument("--apply", action="store_true", help="keep the last proposal")
    p.add_argument("--wait", action="store_true", help="wait for the proposal and show it")
    p = action("scope", "set items' scope by hand", "ID", "one or more item ids")
    p.add_argument("--to", default="", help="global, topic:<name>, or a project's name")
    p = action(
        "summarize",
        "a brief of what memory knows about one project or topic",
        "SCOPE",
        "global, topic:<name>, or a project's name",
    )
    p.add_argument("--force", action="store_true", help="write it again even when no note changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gm", description="Graph memory for coding agents")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon", help="run the daemon in the foreground")
    sub.add_parser("start", help="hook: make sure the daemon is running")
    for name, text in (
        ("recall", "hook: user submitted a prompt"),
        ("remember", "hook: the agent finished replying"),
        ("graduate", "hook: session ended or compacted; distil it into knowledge"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument(
            "--agent", default="claude-code", help="adapter: built-in name or package.module:Class"
        )
    sub.add_parser("status", help="show daemon, buffer and knowledge status")
    kp = sub.add_parser(
        "knowledge",
        help="list what graduated into the knowledge graph, and correct it",
        description="With no action, lists every item (see `list`). "
        "`gm knowledge <action> --help` shows an action's options.",
    )
    kp.add_argument("--all", action="store_true", help="list: include items no longer true")
    _knowledge_actions(kp.add_subparsers(dest="action", metavar="ACTION", title="actions"))
    kp.set_defaults(action="list")  # after the actions, so it is their default too
    sp = sub.add_parser("stats", help="health and quality numbers from the event log")
    sp.add_argument("--since", default="7d", help="window: 30m, 24h, 7d, 2w or all")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    up = sub.add_parser("ui", help="open the web page the daemon serves (starts the daemon)")
    up.add_argument(
        "--port", type=int, default=None, help="run a separate server on this port instead"
    )
    up.add_argument("--no-open", action="store_true", help="don't open a browser")
    sub.add_parser("stop", help="stop the daemon")
    mp = sub.add_parser(
        "migrate",
        help="move memory to the current storage: off the conversation graph, then from the "
        "JSON files into memory.db (each once)",
    )
    mp.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    ep = sub.add_parser(
        "export", help="write memory out as JSON files (readable; what older versions read)"
    )
    ep.add_argument("dest", nargs="?", help="folder (default ~/graph-memory-export-<date>)")
    args = parser.parse_args(argv)

    is_hook = args.cmd in ("start", "recall", "remember", "graduate")
    try:
        cfg = load_config()
    except Exception as exc:
        if is_hook:  # a hook must never fail the agent, even with a broken config
            print(f"graph-memory: config error: {exc}", file=sys.stderr)
            return 0
        raise
    logging.basicConfig(
        filename=str(cfg.log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "daemon":
        from .daemon import run_daemon

        return run_daemon(cfg)
    if is_hook:
        return _hook(cfg, args)
    if args.cmd == "ui":
        return _ui(cfg, args)
    if args.cmd == "stats":
        return _stats(cfg, args)
    if args.cmd == "migrate":
        return _migrate(cfg, args)
    if args.cmd == "export":
        return _export(cfg, args)

    try:
        if args.cmd == "status":
            print(json.dumps(call(cfg, "status", autostart=False), indent=2))
        elif args.cmd == "knowledge":
            return _knowledge(cfg, args)
        elif args.cmd == "stop":
            call(cfg, "shutdown", autostart=False)
            # wait until it is gone: a command right after must not reach a daemon that
            # is still shutting down (its page already closed)
            deadline = time.monotonic() + cfg.start_timeout
            while cfg.socket_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            if cfg.socket_path.exists():
                print("the daemon is still shutting down", file=sys.stderr)
                return 1
            print("stopped")
    except DaemonUnavailable:
        print("Daemon is not running.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
