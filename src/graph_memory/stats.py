"""`gm stats`: health and quality numbers computed from the event log."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, tzinfo
from typing import Any


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _timing(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_ms": _pct(values, 0.5),
        "p95_ms": _pct(values, 0.95),
        "max_ms": max(values) if values else None,
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def compute_stats(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    hooks: dict[str, list[float]] = defaultdict(list)
    hook_errors: Counter[str] = Counter()
    requests: dict[str, list[float]] = defaultdict(list)
    request_errors: Counter[str] = Counter()
    grad_ms: list[float] = []
    grad_ok = grad_failed = grad_given_up = 0
    grad_by_trigger: Counter[str] = Counter()
    grad_errors: Counter[str] = Counter()
    exchanges = items = new_items = calls = 0
    remembered = rewinds = continued = forgotten = dropped = cleared = 0
    daemon_starts = 0
    first = last = None

    for ev in events:
        kind = ev.get("kind")
        ts = ev.get("ts")
        first = first or ts
        last = ts or last
        if kind == "hook":
            hooks[str(ev.get("hook"))].append(float(ev.get("ms") or 0))
            if not ev.get("ok", True):
                hook_errors[str(ev.get("hook"))] += 1
        elif kind == "request":
            requests[str(ev.get("method"))].append(float(ev.get("ms") or 0))
            if not ev.get("ok", True):
                request_errors[str(ev.get("method"))] += 1
        elif kind == "graduate":
            if ev.get("ok"):
                grad_ok += 1
                grad_ms.append(float(ev.get("ms") or 0))
                grad_by_trigger[str(ev.get("trigger"))] += 1
                exchanges += int(ev.get("exchanges") or 0)
                items += int(ev.get("items") or 0)
                new_items += int(ev.get("new") or 0)
                calls += int(ev.get("calls") or 0)
            else:
                grad_failed += 1
                grad_given_up += 1 if ev.get("given_up") else 0
                grad_errors[str(ev.get("error") or "?").split(":")[0]] += 1
        elif kind == "remember":
            remembered += 1
            rewinds += 1 if ev.get("rewound") else 0
            continued += 1 if ev.get("continued") else 0
        elif kind == "rewind":
            pass  # counted on the remember event
        elif kind == "forget":
            forgotten += 1
        elif kind == "buffer_dropped":
            dropped += int(ev.get("exchanges") or 0)
        elif kind == "session_cleared":
            cleared += 1
        elif kind == "daemon" and ev.get("op") == "start":
            daemon_starts += 1

    return {
        "window": {"first": first, "last": last},
        "health": {
            "hooks": {
                h: {**_timing(v), "errors": hook_errors[h]} for h, v in sorted(hooks.items())
            },
            "daemon": {
                m: {**_timing(v), "errors": request_errors[m]} for m, v in sorted(requests.items())
            },
            "daemon_starts": daemon_starts,
        },
        "buffer": {
            "remembered": remembered,
            "continued": continued,
            "rewinds": rewinds,
            "forgotten": forgotten,
            # exchanges given up on after graduation kept failing
            "dropped": dropped,
        },
        "graduation": {
            **_timing(grad_ms),
            "ok": grad_ok,
            "failed": grad_failed,
            "given_up": grad_given_up,
            "by_trigger": dict(grad_by_trigger),
            "errors": dict(grad_errors),
            "exchanges": exchanges,
            "items": items,
            "new_items": new_items,
            "model_calls": calls,
            "sessions_cleared": cleared,
        },
    }


def _ms(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"


def _pc(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _counts(c: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) or "-"


def format_stats(st: dict[str, Any], status: dict[str, Any] | None = None) -> str:
    h, b, g = st["health"], st["buffer"], st["graduation"]
    out: list[str] = []
    w = st["window"]
    if not w["first"]:
        return "No events recorded in this window yet."
    out.append(
        f"Events {w['first'][:16].replace('T', ' ')} → {w['last'][:16].replace('T', ' ')} UTC"
    )
    if status:
        k, buf = status.get("knowledge") or {}, status.get("buffer") or {}
        out.append(
            f"Now     {k.get('items', 0)} knowledge items ({k.get('invalidated', 0)} no longer "
            f"true), {k.get('entities', 0)} entities · buffer: {buf.get('exchanges', 0)} "
            f"exchanges waiting in {buf.get('sessions', 0)} sessions"
        )

    out.append("\nHEALTH")
    for name, t in h["hooks"].items():
        out.append(
            f"  hook {name:<9} {t['count']:>5}×  p50 {_ms(t['p50_ms']):>6}  "
            f"p95 {_ms(t['p95_ms']):>6}  max {_ms(t['max_ms']):>6}  errors {t['errors']}"
        )
    for name, t in h["daemon"].items():
        out.append(
            f"  daemon {name:<7} {t['count']:>5}×  p50 {_ms(t['p50_ms']):>6}  "
            f"p95 {_ms(t['p95_ms']):>6}  max {_ms(t['max_ms']):>6}  errors {t['errors']}"
        )
    out.append(f"  daemon starts  {h['daemon_starts']}")

    out.append("\nGRADUATION")
    out.append(
        f"  graduations    {g['ok']} ok, {g['failed']} failed ({g['given_up']} given up)  "
        f"p50 {_ms(g['p50_ms'])}  p95 {_ms(g['p95_ms'])}"
    )
    if g["by_trigger"]:
        out.append(f"    by trigger   {_counts(g['by_trigger'])}")
    out.append(
        f"  learned        {g['new_items']} new items, {g['items'] - g['new_items']} seen "
        f"again, from {g['exchanges']} exchanges ({g['model_calls']} model calls)"
    )
    if g["errors"]:
        out.append(f"  failures       {_counts(g['errors'])}")
    out.append(
        f"  buffer         {b['remembered']} exchanges stored, {b['rewinds']} rewinds, "
        f"{b['continued']} continued, {b['forgotten']} forgotten, {b['dropped']} given up; "
        f"{g['sessions_cleared']} sessions cleared"
    )
    return "\n".join(out)


# --------------------------------------------------------------------------- memory help


def _day(ts: str, tz: tzinfo | None) -> str:
    try:
        when = datetime.fromisoformat(ts)
    except ValueError:
        return ts[:10]
    return when.astimezone(tz).date().isoformat()


def _blank_day() -> dict[str, int]:
    return dict.fromkeys(
        ("prompts", "with_notes", "notes", "checked", "helped", "notes_checked", "used"), 0
    )


# Recall found matches, but the best were already shown in this session; the rest fell
# below the relevance floor.
_NO_NOTES = {"ok": "best matches already shown", "None": "unknown", "": "unknown"}


def memory_help(
    events: Iterable[dict[str, Any]], *, tz: tzinfo | None = None, max_turns: int = 500
) -> dict[str, Any]:
    """How often knowledge recall gave Claude notes, and how often the reply used one.

    Each prompt logs a `knowledge_recall` event (the notes shown, or why there were none);
    after the reply a `knowledge_use` event says which of them the reply used. Only
    prompts with recall "on" count: in "shadow" mode nothing was shown. A turn whose notes
    were never checked (the reply was not stored) counts as shown but not checked.
    Days are in `tz` (default: this machine's local time)."""
    turns: list[dict[str, Any]] = []
    waiting: dict[str, dict[str, Any]] = {}  # session -> its turn awaiting the use check
    shadow = 0
    judged_turns = judge_failed = 0
    label_count: Counter[str] = Counter()
    misleading: Counter[str] = Counter()
    useful: Counter[str] = Counter()
    for ev in events:
        kind = ev.get("kind")
        session = str(ev.get("session"))
        if kind == "knowledge_recall":
            if ev.get("mode") != "on":
                shadow += 1
                waiting.pop(session, None)
                continue
            shown = [str(i) for i in ev.get("items") or []]
            turn: dict[str, Any] = {
                "ts": str(ev.get("ts", "")),
                "day": _day(str(ev.get("ts", "")), tz),
                "session": session,
                "prompt": str(ev.get("prompt") or ""),
                "shown": shown,
                "used": None,
                "node": None,
                "reason": "" if shown else _NO_NOTES.get(str(ev.get("reason")), ev.get("reason")),
                "judged": None,
            }
            turns.append(turn)
            if shown:
                waiting[session] = turn
            else:
                waiting.pop(session, None)
        elif kind == "knowledge_use" and ev.get("mode") == "on":
            done = waiting.pop(session, None)
            if done is not None and [str(i) for i in ev.get("recalled") or []] == done["shown"]:
                done["used"] = [str(i) for i in ev.get("used") or []]
                done["node"] = ev.get("node")
        elif kind == "knowledge_judge":
            if not ev.get("ok"):
                judge_failed += 1
                continue
            labels = {str(k): str(v) for k, v in (ev.get("labels") or {}).items()}
            items = [str(i) for i in ev.get("items") or []]
            judged_turns += 1
            label_count.update(labels.values())
            misleading.update(
                i for i, lab in labels.items() if lab in ("misleading", "contradicted")
            )
            useful.update(i for i, lab in labels.items() if lab == "useful")
            for t in reversed(turns):  # the turn it judged: same session, same notes
                if t["session"] == session and t["shown"] == items:
                    t["judged"] = labels
                    break

    days: dict[str, dict[str, int]] = defaultdict(_blank_day)
    total = _blank_day()
    used_count: Counter[str] = Counter()
    shown_count: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for t in turns:
        for bucket in (days[t["day"]], total):
            bucket["prompts"] += 1
            if t["shown"]:
                bucket["with_notes"] += 1
                bucket["notes"] += len(t["shown"])
            if t["used"] is not None:
                bucket["checked"] += 1
                bucket["notes_checked"] += len(t["shown"])
                bucket["used"] += len(t["used"])
                bucket["helped"] += 1 if t["used"] else 0
        if not t["shown"]:
            reasons[str(t["reason"])] += 1
        if t["used"] is not None:
            shown_count.update(t["shown"])
            used_count.update(t["used"])

    def rates(b: dict[str, int]) -> dict[str, Any]:
        return {
            **b,
            "unchecked": b["with_notes"] - b["checked"],
            "notes_rate": _rate(b["with_notes"], b["prompts"]),
            "help_rate": _rate(b["helped"], b["checked"]),
            "use_rate": _rate(b["used"], b["notes_checked"]),
        }

    never_used = [(i, n) for i, n in shown_count.most_common() if n >= 2 and not used_count[i]]
    return {
        "totals": rates(total),
        "days": [{"day": d, **rates(b)} for d, b in sorted(days.items())],
        "most_used": [
            {"item": i, "used": n, "shown": shown_count[i]} for i, n in used_count.most_common(8)
        ],
        "never_used": [{"item": i, "shown": n} for i, n in never_used[:8]],
        "nothing_shown": dict(reasons.most_common()),
        "shadow": shadow,
        # a model's verdict on a sample of turns (judge_rate): did each note help?
        "judged": {
            "turns": judged_turns,
            "failed": judge_failed,
            "notes": sum(label_count.values()),
            **{lab: label_count[lab] for lab in ("useful", "neutral", "misleading")},
            "contradicted": label_count["contradicted"],
            "useful_rate": _rate(label_count["useful"], sum(label_count.values())),
            # misleading or contradicted: a note that did harm
            "misleading_rate": _rate(
                label_count["misleading"] + label_count["contradicted"],
                sum(label_count.values()),
            ),
            "most_misleading": [{"item": i, "times": n} for i, n in misleading.most_common(5)],
            "most_useful": [{"item": i, "times": n} for i, n in useful.most_common(5)],
        },
        "turns": list(reversed(turns))[:max_turns],
    }


def _short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def format_help(h: dict[str, Any], items: dict[str, dict[str, Any]] | None = None) -> str:
    """The MEMORY HELP section of `gm stats`. `items`: item id -> {statement, type}."""
    t = h["totals"]
    out = ["\nMEMORY HELP  (knowledge notes shown to Claude before it replied)"]
    if not t["prompts"]:
        note = f" ({h['shadow']} computed in shadow mode, not shown)" if h["shadow"] else ""
        out.append(f"  no prompts with knowledge recall on in this window{note}")
        return "\n".join(out)

    def item(i: str) -> str:
        it = (items or {}).get(i)
        if not it:
            return i[:8]
        return f"[{it.get('type', '?')}] {_short(str(it.get('statement', '')), 64)}  ({i[:8]})"

    out.append(
        f"  prompts        {t['prompts']}   got notes {t['with_notes']} "
        f"({_pc(t['notes_rate'])}), {t['notes']} notes"
    )
    out.append(
        f"  replies        checked {t['checked']}: used a note {t['helped']} "
        f"({_pc(t['help_rate'])}); notes used {t['used']} of {t['notes_checked']} "
        f"({_pc(t['use_rate'])})"
    )
    if t["unchecked"]:
        out.append(
            f"  not checked    {t['unchecked']} (no reply stored, or the daemon restarted first)"
        )
    if len(h["days"]) > 1:
        out.append("  by day         prompts  notes  used a note")
        for d in h["days"]:
            out.append(
                f"    {d['day']}  {d['prompts']:>7}  {d['with_notes']:>5}  {d['helped']:>11}"
            )
    if h["most_used"]:
        out.append("  most used")
        out.extend(
            f"    {m['used']}× of {m['shown']}  {item(m['item'])}" for m in h["most_used"][:5]
        )
    if h["never_used"]:
        out.append("  shown, never used")
        out.extend(f"    {m['shown']}×  {item(m['item'])}" for m in h["never_used"][:5])
    if h["nothing_shown"]:
        out.append(f"  no notes       {_counts(h['nothing_shown'])}")
    j = h["judged"]
    if j["turns"]:
        out.append(
            f"  judged         {j['turns']} turns by a model: {j['useful']} of {j['notes']} notes "
            f"useful ({_pc(j['useful_rate'])}), {j['neutral']} neutral, "
            f"{j['misleading']} misleading, {j['contradicted']} contradicted "
            f"({_pc(j['misleading_rate'])})"
        )
        out.extend(f"    wrong {m['times']}×  {item(m['item'])}" for m in j["most_misleading"][:3])
    if h["shadow"]:
        out.append(f"  shadow mode    {h['shadow']} prompts (computed, not shown)")
    return "\n".join(out)
