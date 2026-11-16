#!/bin/sh
# Install graph-memory and add its hooks to Claude Code:
#
#   curl -LsSf https://raw.githubusercontent.com/xihe92/graph-memory/main/install.sh | sh
#
# It installs uv if it is missing (https://docs.astral.sh/uv/), installs graph-memory
# with it on uv's own Python, then runs `gm setup`, which merges the hooks into
# ~/.claude/settings.json (backed up first). Running it again updates graph-memory.
# GM_SOURCE installs from somewhere else: another git URL, or a local folder.
set -eu

SOURCE="${GM_SOURCE:-git+https://github.com/xihe92/graph-memory}"

say() { printf 'graph-memory: %s\n' "$*"; }
fail() { say "$*" >&2; exit 1; }

case "$(uname -s)" in
  Darwin | Linux) ;;
  *) fail "runs on macOS and Linux only" ;;
esac

case "$SOURCE" in
  git+*) command -v git > /dev/null 2>&1 || fail "git is needed to fetch graph-memory; install it first" ;;
esac

if ! command -v uv > /dev/null 2>&1; then
  say "installing uv"
  command -v curl > /dev/null 2>&1 || fail "curl is needed to install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.cargo/bin:$PATH"
  export PATH
  command -v uv > /dev/null 2>&1 || fail "uv was installed but is not on PATH; open a new terminal and run this again"
fi

say "installing graph-memory from $SOURCE"
uv tool install --managed-python --reinstall "$SOURCE"

GM="$(uv tool dir --bin)/gm"
[ -x "$GM" ] || fail "the install did not leave $GM"
"$GM" setup

if ! command -v claude > /dev/null 2>&1; then
  say "Claude Code's 'claude' command was not found; install Claude Code to use graph-memory"
fi
case ":$PATH:" in
  *":$(dirname "$GM"):"*) ;;
  *) say "add $(dirname "$GM") to your PATH to type 'gm' (uv tool update-shell does it)" ;;
esac
say "done. Start a Claude Code session; '$GM ui' opens the memory page"
