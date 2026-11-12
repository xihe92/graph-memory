"""The model graduation and note judging call: any headless agent or model CLI that reads
a prompt on stdin and prints the answer. Claude Code in print mode is the built-in one.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

DISABLE_ENV = "GM_DISABLE"  # set for model subprocesses, so their own hooks record nothing

Ask = Callable[[str], str]


class ModelError(RuntimeError):
    pass


@dataclass
class CommandModel:
    """Runs `argv` with the prompt on stdin and returns stdout, e.g.
    CommandModel(["claude", "-p", "--model", "haiku"])
    CommandModel(["codex", "exec", "-"])            # check your product's flags
    """

    argv: Sequence[str]
    name: str = "command"
    timeout: float = 300.0
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None

    def complete(self, prompt: str) -> str:
        env = {**os.environ, **self.env, DISABLE_ENV: "1"}
        try:
            proc = subprocess.run(
                list(self.argv),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=self.cwd,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ModelError(f"{self.name}: {exc}") from exc
        if proc.returncode != 0:
            raise ModelError(f"{self.name}: exit {proc.returncode}: {proc.stderr.strip()[:300]}")
        return proc.stdout


def claude_code_model(
    model: str = "haiku", timeout: float = 300.0, cwd: str | None = None
) -> CommandModel:
    """Claude Code in print mode with a cheap model, using the user's existing login."""
    return CommandModel(
        ["claude", "-p", "--model", model, "--output-format", "text"],
        name="claude-code",
        timeout=timeout,
        cwd=cwd,
    )
