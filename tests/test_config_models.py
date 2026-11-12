"""Settings files from before the redesign still load; the model runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from graph_memory.config import load_config
from graph_memory.daemon import build_model
from graph_memory.models import DISABLE_ENV, CommandModel, ModelError, claude_code_model


def test_old_settings_are_renamed_or_ignored(short_dir: Path) -> None:
    (short_dir / "config.toml").write_text(
        'summarizer = "command"\nsummarizer_model = "sonnet"\n'
        'summarizer_command = ["my-model", "-p"]\nsummary_workers = 3\n'
        'vector_backend = "chroma"\n[placement]\ncontinue_threshold = 0.3\n'
    )
    cfg = load_config(short_dir)
    assert (cfg.model_backend, cfg.model, cfg.model_command) == (
        "command",
        "sonnet",
        ("my-model", "-p"),
    )
    (short_dir / "config.toml").write_text('summarizer = "heuristic"\n')
    assert load_config(short_dir).model_backend == "none"
    (short_dir / "config.toml").write_text("not_a_setting = 1\n")
    with pytest.raises(ValueError, match="unknown settings"):
        load_config(short_dir)


def test_defaults(short_dir: Path) -> None:
    cfg = load_config(short_dir)
    assert cfg.knowledge_recall == "on" and cfg.judge_rate == 0.2
    assert (cfg.graduation_attempts, cfg.buffer_max_days, cfg.buffer_abandon_days) == (3, 3, 7)


def test_build_model(short_dir: Path) -> None:
    cfg = load_config(short_dir)
    assert build_model(cfg) is not None  # claude-code by default
    from dataclasses import replace

    assert build_model(replace(cfg, graduation=False)) is None
    assert build_model(replace(cfg, model_backend="none")) is None
    assert build_model(replace(cfg, model_backend="command")) is None  # no command given
    assert build_model(replace(cfg, model_backend="command", model_command=("cat",))) is not None


def test_command_model_runs_with_hooks_disabled() -> None:
    echo = CommandModel(
        [
            sys.executable,
            "-c",
            f"import os,sys; print(os.environ['{DISABLE_ENV}'], sys.stdin.read())",
        ]
    )
    assert echo.complete("hello").strip() == "1 hello"
    with pytest.raises(ModelError, match="exit 3"):
        CommandModel([sys.executable, "-c", "import sys; sys.exit(3)"]).complete("x")
    with pytest.raises(ModelError):
        CommandModel(["/no/such/model"]).complete("x")
    with pytest.raises(ModelError):
        CommandModel([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2).complete(
            "x"
        )
    m = claude_code_model("haiku", 10, "/tmp")
    assert m.argv[:4] == ["claude", "-p", "--model", "haiku"] and m.cwd == "/tmp"
