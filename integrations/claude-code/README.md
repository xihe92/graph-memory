# graph-memory for Claude Code

The Claude Code plugin. Install it inside Claude Code:

```
/plugin marketplace add xihe92/graph-memory
/plugin install graph-memory@graph-memory
```

- `hooks.json`: the hooks the plugin adds. Recall before each prompt, remember after
  each reply, graduate when a session ends or compacts. They are the same hooks
  `gm setup` writes to `~/.claude/settings.json`.
- `gm-hook`: what each hook runs. It hands the hook to `gm`. When `gm` is missing, or
  older than the plugin, it first installs it with uv in the background, from the
  plugin's own copy of this repo. Its files (the installed version, a lock, the install
  log) are in the plugin's data folder (`$CLAUDE_PLUGIN_DATA`).

The plugin's manifests are in [`.claude-plugin/`](../../.claude-plugin/) at the repo
root, where Claude Code looks for them. The one-line installer in the main
[README](../../README.md#install) is the other way to install. Use one, not both: when
`gm setup`'s hooks are in settings.json, the plugin's step aside.
