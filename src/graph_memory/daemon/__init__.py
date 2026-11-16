"""The graph memory daemon: one long-running local process per memory folder.

Protocol: one JSON request per connection on a Unix socket; the methods are in
routes.py.

What it keeps: the knowledge graph, for good; each live session's exchanges in the
session buffer, until they graduate; the event log. Recall reads only knowledge. The
event loop thread owns all three; model calls and embeddings run in worker threads.

    state.py       the state every part shares, seeded from the event log on start
    turns.py       recall for a prompt; remember the exchange; check and judge notes
    graduating.py  queue quiet, compacted and ended sessions; graduate them
    rebuild.py     graduate a session again from its transcript; fold its summaries
    admin.py       retract, restore, confirm, delete, scope and rescope items by hand
    summaries.py   a brief of one scope's notes
    views.py       what the web page and the CLI read
    core.py        Daemon: the parts put together; the socket and background tasks
    routes.py      each request method and what it calls
    process.py     building the daemon, its lock, `gm daemon`
"""

from .core import Daemon
from .process import _answers, acquire_lock, build_embedder, build_model, run_daemon
from .scopes import scope_names

__all__ = [
    "Daemon",
    "_answers",
    "acquire_lock",
    "build_embedder",
    "build_model",
    "run_daemon",
    "scope_names",
]
