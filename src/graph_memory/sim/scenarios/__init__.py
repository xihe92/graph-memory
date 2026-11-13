"""The scenario library. Each entry builds a fresh Scenario."""

from __future__ import annotations

from collections.abc import Callable

from ..scenario import Scenario
from . import buffer, core, learning, lifecycle, robustness, scale, world

SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "cross_session_recall": core.cross_session_recall,
    "compaction": core.compaction,
    "supersede": core.supersede,
    "reinforce": core.reinforce,
    "use_and_hebbian": core.use_and_hebbian,
    "decay": core.decay,
    "resolve_open": lifecycle.resolve_open,
    "buffer_lifecycle": buffer.buffer_lifecycle,
    "qa_learning": learning.qa_learning,
    "stale_knowledge": learning.stale_knowledge,
    "delete_and_bridge": learning.delete_and_bridge,
    "topic_scope": learning.topic_scope,
    "summarize_scope": learning.summarize_scope,
    "project_isolation": world.project_isolation,
    "entity_names": world.entity_names,
    "crash_and_restart": robustness.crash_and_restart,
    "model_failures": robustness.model_failures,
    "noise": robustness.noise,
    "scale": scale.scale,
}
