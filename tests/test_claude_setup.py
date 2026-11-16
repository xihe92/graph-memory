"""`gm setup`: graph-memory's hooks merged into Claude Code's settings, and taken out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_memory import cli
from graph_memory.claude_setup import is_ours, setup

GM = "/home/u/.local/bin/gm"
THEIRS = {
    "type": "command",
    "command": "bash '/home/u/.claude/hooks/notify.sh' session",
    "timeout": 10,
}
SETTINGS = {
    "model": "opus",
    "hooks": {
        "SessionStart": [{"matcher": "^(startup|resume)$", "hooks": [THEIRS]}],
        "Stop": [{"hooks": [THEIRS, {"type": "command", "command": "gm remember"}]}],
    },
    "theme": "dark",
}


def _write(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def test_hooks_merge_with_what_is_there(tmp_path: Path) -> None:
    path = _write(tmp_path / "settings.json", SETTINGS)
    res = setup(path, gm=GM)
    assert res.changed and res.backup is not None
    assert json.loads(res.backup.read_text()) == SETTINGS  # the file as it was
    got = json.loads(path.read_text())
    assert got["model"] == "opus" and got["theme"] == "dark"
    hooks = got["hooks"]
    # theirs first and untouched; the old `gm remember` replaced by the full path
    assert hooks["SessionStart"][0] == SETTINGS["hooks"]["SessionStart"][0]  # type: ignore[index]
    assert hooks["SessionStart"][1:] == [
        {"hooks": [{"type": "command", "command": f"{GM} start", "timeout": 30}]},
        {
            "matcher": "compact",
            "hooks": [{"type": "command", "command": f"{GM} graduate", "timeout": 15}],
        },
    ]
    assert hooks["Stop"] == [
        {"hooks": [THEIRS]},
        {"hooks": [{"type": "command", "command": f"{GM} remember", "timeout": 15}]},
    ]
    assert [h["hooks"][0]["command"] for h in hooks["UserPromptSubmit"]] == [f"{GM} recall"]
    assert [h["hooks"][0]["command"] for h in hooks["SessionEnd"]] == [f"{GM} graduate"]


def test_running_it_again_changes_nothing(tmp_path: Path) -> None:
    path = _write(tmp_path / "settings.json", SETTINGS)
    setup(path, gm=GM)
    before = path.read_text()
    backups = sorted(tmp_path.glob("settings.json.bak-*"))
    res = setup(path, gm=GM)
    assert not res.changed and res.backup is None
    assert path.read_text() == before and sorted(tmp_path.glob("settings.json.bak-*")) == backups
    # a new place for gm: its hooks move there, nothing doubles
    setup(path, gm="/opt/gm/bin/gm")
    commands = [
        h["command"]
        for groups in json.loads(path.read_text())["hooks"].values()
        for g in groups
        for h in g["hooks"]
    ]
    assert sum(c.startswith("/opt/gm/bin/gm ") for c in commands) == 5
    assert not any(c.startswith(GM) for c in commands)


def test_remove_takes_out_only_graph_memorys_hooks(tmp_path: Path) -> None:
    path = _write(tmp_path / "settings.json", SETTINGS)
    setup(path, gm=GM)
    res = setup(path, remove=True)
    assert res.changed
    assert json.loads(path.read_text()) == {
        "model": "opus",
        "hooks": {
            "SessionStart": [{"matcher": "^(startup|resume)$", "hooks": [THEIRS]}],
            "Stop": [{"hooks": [THEIRS]}],
        },
        "theme": "dark",
    }
    assert not setup(path, remove=True).changed
    alone = _write(
        tmp_path / "alone.json", {"hooks": {"Stop": [{"hooks": [{"command": "gm remember"}]}]}}
    )
    setup(alone, remove=True)
    assert json.loads(alone.read_text()) == {}  # no hooks left: no "hooks" key either


def test_a_new_file_is_made_and_a_broken_one_left_alone(tmp_path: Path) -> None:
    path = tmp_path / "fresh" / "settings.json"
    res = setup(path, gm=GM)
    assert res.changed and res.backup is None
    assert set(json.loads(path.read_text())["hooks"]) == {
        "SessionStart",
        "UserPromptSubmit",
        "Stop",
        "SessionEnd",
    }
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        setup(broken, gm=GM)
    assert broken.read_text() == "{ not json"


def test_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    path = _write(tmp_path / "settings.json", SETTINGS)
    res = setup(path, gm=GM, dry_run=True)
    assert res.changed and res.backup is None
    assert json.loads(path.read_text()) == SETTINGS and not list(tmp_path.glob("*.bak-*"))


def test_which_commands_are_graph_memorys() -> None:
    assert is_ours("gm recall") and is_ours("/a b/gm remember") is False
    assert is_ours("'/a b/gm' remember") and is_ours("/x/gm graduate --agent m:C")
    assert not is_ours("gm status") and not is_ours("/x/gmx recall") and not is_ours("gm")
    assert not is_ours("bash hooks.sh 'unclosed")


def test_gm_setup_uses_claude_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("GM_HOME", str(tmp_path / "gm-home"))
    assert cli.main(["setup", "--dry-run"]) == 0
    assert "would be added" in capsys.readouterr().out
    assert not (tmp_path / "claude").exists()
    assert cli.main(["setup"]) == 0
    assert "hooks added to" in capsys.readouterr().out
    assert (tmp_path / "claude" / "settings.json").exists()
    assert not (tmp_path / "gm-home").exists()  # setup touches only Claude Code's settings
    assert cli.main(["setup"]) == 0
    assert "already in" in capsys.readouterr().out
    assert cli.main(["setup", "--remove"]) == 0
    assert "taken out of" in capsys.readouterr().out
    (tmp_path / "claude" / "settings.json").write_text("[]")
    assert cli.main(["setup"]) == 1
    assert "JSON object" in capsys.readouterr().err
