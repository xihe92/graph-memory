"""The Claude Code plugin (.claude-plugin/, integrations/claude-code/): its manifests, and
the gm-hook wrapper, run against fake `uv` and `gm` programs in a temporary folder."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from graph_memory import cli
from graph_memory.claude_setup import HOOKS, has_ours, plugin_installed, setup

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / ".claude-plugin" / "plugin.json"
MARKET = REPO / ".claude-plugin" / "marketplace.json"
HOOK = REPO / "integrations" / "claude-code" / "gm-hook"


def _json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


# ------------------------------------------------------------------------- the manifests


def test_the_plugin_version_is_the_package_version() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert _json(PLUGIN)["version"] == pyproject["project"]["version"]


def test_the_marketplace_offers_the_plugin_from_the_repo_root() -> None:
    market = _json(MARKET)
    assert market["name"] == "graph-memory"
    (entry,) = market["plugins"]  # type: ignore[misc]
    assert entry["name"] == _json(PLUGIN)["name"] == "graph-memory" and entry["source"] == "./"


def test_the_plugin_hooks_are_the_ones_gm_setup_writes() -> None:
    hooks_file = REPO / str(_json(PLUGIN)["hooks"])
    hooks = _json(hooks_file)["hooks"]
    assert isinstance(hooks, dict)
    got = [
        (event, group.get("matcher"), h["command"].split()[-1], h["timeout"])
        for event, groups in hooks.items()
        for group in groups
        for h in group["hooks"]
    ]
    assert sorted(got, key=str) == sorted(HOOKS, key=str)
    for event, groups in hooks.items():
        for group in groups:
            for h in group["hooks"]:
                path = h["command"].split('"')[1].replace("${CLAUDE_PLUGIN_ROOT}", str(REPO))
                assert Path(path) == HOOK, event
    assert os.access(HOOK, os.X_OK)


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs Claude Code")
def test_claude_code_accepts_the_manifests() -> None:
    res = subprocess.run(
        ["claude", "plugin", "validate", str(REPO)], capture_output=True, text=True, timeout=60
    )
    assert res.returncode == 0, res.stdout + res.stderr


# --------------------------------------------------------------------- the gm-hook wrapper


class Box:
    """A plugin copy, a home and a PATH with fake `uv` and `gm`, for one wrapper run."""

    def __init__(self, tmp: Path, *, uv: str | None = "ok", gm: bool = False, stamp: str = ""):
        self.tmp = tmp
        self.root = tmp / "plugin"
        (self.root / ".claude-plugin").mkdir(parents=True)
        (self.root / ".claude-plugin" / "plugin.json").write_text(
            json.dumps({"name": "graph-memory", "version": "9.9.0"}, indent=2)
        )
        self.data = tmp / "data"
        self.bin = tmp / "tools"  # uv's tool folder, where gm goes
        self.fakes = tmp / "fakes"  # what is on PATH
        self.calls = tmp / "calls.log"
        for d in (self.data, self.bin, self.fakes, tmp / "home"):
            d.mkdir(exist_ok=True)
        gm_script = (
            f'#!/bin/sh\necho "gm $* source=$GM_HOOK_SOURCE" >>"{self.calls}"\n'
            "cat >/dev/null\n"  # the payload
        )
        self.gm_script = gm_script
        if gm:
            self._put(self.bin / "gm", gm_script)
        if stamp:
            (self.data / "installed-version").write_text(stamp + "\n")
        if uv is not None:
            act = {
                "ok": f'cat >"{self.bin}/gm" <<\'GM\'\n{gm_script}GM\nchmod +x "{self.bin}/gm"\n',
                "fail": "exit 1\n",
                "slow": "sleep 30\n",
            }[uv]
            self._put(
                self.fakes / "uv",
                f'#!/bin/sh\necho "uv $*" >>"{self.calls}"\n'
                f'[ "$1 $2" = "tool dir" ] && {{ echo "{self.bin}"; exit 0; }}\n' + act,
            )

    @staticmethod
    def _put(path: Path, text: str) -> None:
        path.write_text(text)
        path.chmod(0o755)

    def run(self, cmd: str) -> str:
        env = {
            "PATH": f"{self.fakes}:/usr/bin:/bin",
            "HOME": str(self.tmp / "home"),
            "CLAUDE_PLUGIN_ROOT": str(self.root),
            "CLAUDE_PLUGIN_DATA": str(self.data),
            "UV_TOOL_BIN_DIR": str(self.bin),
        }
        res = subprocess.run(
            ["sh", str(HOOK), cmd], input="{}", capture_output=True, text=True, env=env, timeout=20
        )
        assert res.returncode == 0, res.stderr
        return res.stdout

    def log(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def settle(self) -> None:
        """Wait for a background install to finish."""
        deadline = time.monotonic() + 10
        while (self.data / "install.lock").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)


def _message(out: str) -> str:
    return str(json.loads(out)["systemMessage"])


def test_an_installed_gm_gets_the_hook(tmp_path: Path) -> None:
    box = Box(tmp_path, gm=True, stamp="9.9.0")
    assert box.run("recall") == ""
    assert box.log() == ["gm recall source=plugin"]  # uv is never asked


def test_the_first_session_installs_gm_in_the_background(tmp_path: Path) -> None:
    box = Box(tmp_path)
    assert "installing in the background" in _message(box.run("start"))
    box.settle()
    assert (box.data / "installed-version").read_text().strip() == "9.9.0"
    install = [c for c in box.log() if c.startswith("uv tool install")]
    assert install == [f"uv tool install --managed-python --python 3.12 --reinstall {box.root}"]
    assert box.run("recall") == "" and box.log()[-1] == "gm recall source=plugin"


def test_hooks_wait_quietly_while_gm_installs(tmp_path: Path) -> None:
    box = Box(tmp_path, uv="slow")
    box.run("start")
    assert box.run("recall") == ""  # no message outside session start, no second install
    assert "still installing" in _message(box.run("start"))
    deadline = time.monotonic() + 10
    while not any(c.startswith("uv tool install") for c in box.log()):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert len([c for c in box.log() if c.startswith("uv tool install")]) == 1
    pid = (box.data / "install.lock" / "pid").read_text().strip()
    os.kill(int(pid), 9)


def test_a_failed_install_is_reported_and_tried_again(tmp_path: Path) -> None:
    box = Box(tmp_path, uv="fail")
    box.run("start")
    box.settle()
    assert (box.data / "install.failed").exists()
    msg = _message(box.run("start"))
    assert "could not install" in msg and str(box.data / "install.log") in msg
    box.settle()
    assert len([c for c in box.log() if c.startswith("uv tool install")]) == 2


def test_without_uv_it_says_how_and_installs_nothing(tmp_path: Path) -> None:
    box = Box(tmp_path, uv=None)
    assert "needs uv" in _message(box.run("start"))
    assert box.run("recall") == "" and box.log() == []


def test_a_plugin_update_reinstalls_and_restarts_the_daemon(tmp_path: Path) -> None:
    box = Box(tmp_path, gm=True, stamp="9.8.0")
    assert box.run("start") == ""  # this session goes on with the gm it has
    box.settle()
    assert (box.data / "installed-version").read_text().strip() == "9.9.0"
    log = box.log()
    assert "gm start source=plugin" in log and log[-1] == "gm stop source="
    assert any(c.startswith("uv tool install") for c in log)


def test_gm_is_looked_for_only_where_uv_puts_it(tmp_path: Path) -> None:
    box = Box(tmp_path)
    elsewhere = tmp_path / "home" / ".local" / "bin"
    elsewhere.mkdir(parents=True)
    Box._put(elsewhere / "gm", box.gm_script)  # uv's tool folder is set to another place
    assert "installing in the background" in _message(box.run("start"))
    box.settle()
    assert (box.bin / "gm").exists() and not any(c.startswith("gm ") for c in box.log())


def test_a_gm_from_the_one_liner_is_used_as_it_is(tmp_path: Path) -> None:
    box = Box(tmp_path, gm=True)  # no stamp: not installed by the plugin
    box.run("start")
    assert box.log() == ["gm start source=plugin"]


# ------------------------------------------------------------- plugin and gm setup together


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    return tmp_path / "claude" / "settings.json"


def test_plugin_hooks_stand_aside_for_gm_setups(
    tmp_path: Path, short_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _settings(tmp_path, monkeypatch)
    monkeypatch.setenv("GM_HOME", str(short_dir))
    monkeypatch.setenv("GM_HOOK_SOURCE", "plugin")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"session_id": "s", "prompt": "hi"})))
    calls: list[str] = []
    monkeypatch.setattr(cli, "call", lambda cfg, method, *a, **k: calls.append(method) or {})
    assert not has_ours()
    assert cli.main(["recall"]) == 0 and calls == ["recall"]  # alone, the plugin's hook works
    setup(path, gm="/x/gm")
    assert has_ours()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"session_id": "s", "prompt": "hi"})))
    assert cli.main(["recall"]) == 0 and calls == ["recall"]  # beside gm setup's, it does nothing
    monkeypatch.delenv("GM_HOOK_SOURCE")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"session_id": "s", "prompt": "hi"})))
    assert cli.main(["recall"]) == 0 and calls == ["recall", "recall"]  # gm setup's own


def test_gm_setup_warns_when_the_plugin_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _settings(tmp_path, monkeypatch)
    assert not plugin_installed()
    assert not any("plugin" in n for n in setup(path, gm="/x/gm", dry_run=True).notes)
    listed = tmp_path / "claude" / "plugins" / "installed_plugins.json"
    listed.parent.mkdir(parents=True)
    listed.write_text(json.dumps({"version": 2, "plugins": {"graph-memory@graph-memory": [{}]}}))
    assert plugin_installed()
    assert any("plugin is installed too" in n for n in setup(path, gm="/x/gm", dry_run=True).notes)
