"""`gm setup`: add graph-memory's hooks to Claude Code's settings, or take them out.

The hooks go into `~/.claude/settings.json` (under CLAUDE_CONFIG_DIR when that is set),
merged with what is there: other hooks and settings stay as they are, graph-memory's
own hooks are replaced, and running it twice changes nothing. The file is backed up
before it is changed. Each hook calls `gm` by its full path, so a shell alias or another
program named gm (GraphicsMagick) cannot get in the way.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .files import atomic_write

OURS = frozenset({"start", "recall", "remember", "graduate"})  # the gm commands hooks run

# event -> (matcher or None, gm command, timeout in seconds), in the order they are added
HOOKS: list[tuple[str, str | None, str, int]] = [
    ("SessionStart", None, "start", 30),  # the daemon is up before the first prompt
    ("SessionStart", "compact", "graduate", 15),  # after a compaction: graduate what it hid
    ("UserPromptSubmit", None, "recall", 15),
    ("Stop", None, "remember", 15),
    ("SessionEnd", None, "graduate", 15),
]


def settings_path() -> Path:
    """Claude Code's user settings file."""
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base).expanduser() if base else Path.home() / ".claude") / "settings.json"


def gm_path() -> str:
    """The full path of the `gm` running now (the one just installed)."""
    me = Path(sys.argv[0])
    if me.name == "gm" and me.exists():
        return str(me.absolute())
    found = shutil.which("gm")
    return str(Path(found).absolute()) if found else "gm"


def is_ours(command: str) -> bool:
    """A hook command graph-memory added: `<path>/gm start|recall|remember|graduate ...`."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return len(parts) >= 2 and Path(parts[0]).name == "gm" and parts[1] in OURS


PLUGIN_ID = "graph-memory@graph-memory"  # the Claude Code plugin (integrations/claude-code)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def has_ours(path: Path | None = None) -> bool:
    """Whether Claude Code's settings already run graph-memory's hooks (`gm setup`)."""
    settings = _load(path or settings_path())
    hooks = settings.get("hooks") if isinstance(settings, dict) else None
    if not isinstance(hooks, dict):
        return False
    return any(
        isinstance(h, dict) and is_ours(str(h.get("command", "")))
        for groups in hooks.values()
        if isinstance(groups, list)
        for group in groups
        if isinstance(group, dict) and isinstance(group.get("hooks"), list)
        for h in group["hooks"]
    )


def plugin_installed(path: Path | None = None) -> bool:
    """Whether graph-memory's Claude Code plugin is installed (it brings its own hooks)."""
    found = _load(path or settings_path().parent / "plugins" / "installed_plugins.json")
    plugins = found.get("plugins") if isinstance(found, dict) else None
    return isinstance(plugins, dict) and bool(plugins.get(PLUGIN_ID))


def without_ours(settings: dict[str, Any]) -> dict[str, Any]:
    """The settings with graph-memory's hooks taken out; groups and events left empty by
    that go too, everything else stays."""
    out = dict(settings)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return out
    kept: dict[str, Any] = {}
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            kept[event] = groups
            continue
        new_groups = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                new_groups.append(group)
                continue
            rest = [
                h
                for h in group["hooks"]
                if not (isinstance(h, dict) and is_ours(str(h.get("command", ""))))
            ]
            if rest:
                new_groups.append({**group, "hooks": rest})
        if new_groups:
            kept[event] = new_groups
    if kept:
        out["hooks"] = kept
    else:
        out.pop("hooks", None)
    return out


def with_ours(settings: dict[str, Any], gm: str) -> dict[str, Any]:
    """The settings with graph-memory's hooks, each calling `gm` at this path, after any
    other hooks of the same event."""
    out = without_ours(settings)
    hooks: dict[str, Any] = dict(out.get("hooks") or {})
    for event, matcher, command, timeout in HOOKS:
        group: dict[str, Any] = {}
        if matcher is not None:
            group["matcher"] = matcher
        group["hooks"] = [
            {"type": "command", "command": f"{shlex.quote(gm)} {command}", "timeout": timeout}
        ]
        hooks[event] = [*hooks.get(event, []), group]
    out["hooks"] = hooks
    return out


@dataclass
class Result:
    path: Path
    changed: bool
    backup: Path | None = None
    notes: list[str] = field(default_factory=list)


def setup(
    path: Path | None = None, *, remove: bool = False, dry_run: bool = False, gm: str | None = None
) -> Result:
    """Add (or with `remove`, take out) graph-memory's hooks in Claude Code's settings."""
    path = path or settings_path()
    current: dict[str, Any] = {}
    if path.exists():
        text = path.read_text(encoding="utf-8")
        try:
            loaded = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON ({exc}); fix it, then run again") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} does not hold a JSON object")
        current = loaded
    wanted = without_ours(current) if remove else with_ours(current, gm or gm_path())
    result = Result(path, changed=wanted != current)
    if not remove:
        if plugin_installed(path.parent / "plugins" / "installed_plugins.json"):
            result.notes.append(
                "graph-memory's Claude Code plugin is installed too and brings the same hooks; "
                "while these are in settings.json the plugin's stand aside. Use one: "
                "`gm setup --remove`, or `/plugin uninstall graph-memory@graph-memory`"
            )
        if shutil.which("claude") is None:
            result.notes.append(
                "Claude Code's `claude` command is not on PATH: graduation calls it to "
                "turn sessions into notes"
            )
        from .storage import sqlite_problem

        problem = sqlite_problem()
        if problem:
            result.notes.append(f"the daemon will not start: {problem}")
    if not result.changed or dry_run:
        return result
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result.backup = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, result.backup)
    atomic_write(path, json.dumps(wanted, indent=2, ensure_ascii=False) + "\n")
    os.chmod(path, mode)
    return result
