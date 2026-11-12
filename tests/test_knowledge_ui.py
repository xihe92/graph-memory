"""The knowledge page's filter logic (ui/kview.js), run under Node, and a syntax check of
every inline script in the UI pages. Skipped when Node is not installed."""

from __future__ import annotations

import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")
UI = resources.files("graph_memory.ui")


def _node(*args: str) -> str:
    res = subprocess.run([NODE or "node", *args], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr or res.stdout
    return res.stdout


def test_filter_logic() -> None:
    tests = Path(__file__).with_name("kview.test.js")
    assert _node(str(tests), str(UI.joinpath("kview.js"))).strip() == "ok"


def test_graph_stops_drawing_while_nothing_moves() -> None:
    tests = Path(__file__).with_name("idle.test.js")
    assert _node(str(tests), str(UI.joinpath("idle.js"))).strip() == "ok"


@pytest.mark.parametrize("page", ["knowledge.html", "index.html"])
def test_graph_pages_use_the_idle_pause(page: str) -> None:
    html = UI.joinpath(page).read_text(encoding="utf-8")
    assert '<script src="/idle.js"></script>' in html and "GraphIdle.quietWhenStill(G)" in html
    assert ".cooldownTicks(200)" in html  # a bounded layout, so drawing stops within seconds


@pytest.mark.parametrize("page", ["knowledge.html", "index.html", "help.html"])
def test_inline_scripts_parse(page: str, tmp_path: Path) -> None:
    html = UI.joinpath(page).read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts
    for n, body in enumerate(scripts):
        f = tmp_path / f"{n}.js"
        f.write_text(body, encoding="utf-8")
        _node("--check", str(f))
