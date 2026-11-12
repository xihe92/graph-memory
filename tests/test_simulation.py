"""Simulation tests: every scenario in the library, replayed through the real hooks
against a real daemon with simulated time, the oracle model and the hash embedder.

A failing scenario prints each failed step. To see a scenario step by step:
    python -m graph_memory.sim <name> -v
"""

from __future__ import annotations

import pytest

from graph_memory import clock
from graph_memory.knowledge_recall import stem, tokens
from graph_memory.sim import Scenario, Simulator
from graph_memory.sim.oracle import OracleModel
from graph_memory.sim.scenarios import SCENARIOS
from graph_memory.sim.scenarios.core import limiter_facts, teach_limiter


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(name: str) -> None:
    report = Simulator(SCENARIOS[name]()).run()
    assert report.ok, report.text()


def test_shadow_mode_shows_nothing_but_judges_what_it_would_show() -> None:
    s = SCENARIOS["cross_session_recall"]()
    sim = Simulator(s, recall_mode="shadow")
    report = sim.run()
    assert report.ok, report.text()
    assert report.metrics["recall"]["required_shown"] == "4/4"
    # the hook printed no knowledge: every recall was logged in shadow mode
    modes = {e["mode"] for e in sim.events if e["kind"] == "knowledge_recall"}
    assert modes == {"shadow"}


def test_the_harness_catches_wrong_expectations() -> None:
    """A green run means something only if a wrong expectation turns it red."""
    s = Scenario("wrong", "expectations that cannot hold")
    lim = limiter_facts(s)
    teach_limiter(s, lim)
    s.probe("what's the weather tomorrow?", lim, has=["lock"], label="unrelated prompt")
    s.probe("is acquire() protected by a redis lock?", lim, lacks=["lock"], label="related")
    s.check("lock", area="old", label="never reinforced")
    s.check("compose", count=2, label="learned once")
    report = Simulator(s).run()
    failed = {o.label for o in report.failures}
    assert failed == {"unrelated prompt", "related", "never reinforced", "learned once"}
    assert report.metrics["recall"]["forbidden_shown"] == 1


def test_the_harness_reports_hook_and_graduation_failures() -> None:
    s = Scenario("broken", "the model fails and nobody expected it")
    lim = limiter_facts(s)
    a = s.session("a", lim)
    a.turn("add a lock", "Added a SETNX lock to acquire().", learn=["lock"])
    s.fail_next(extract=1)
    a.end(label="unexpected failure")
    report = Simulator(s).run()
    assert [o.label for o in report.failures] == ["unexpected failure"]
    assert "injected extraction failure" in report.failures[0].detail


def test_oracle_refuses_ambiguous_scenarios() -> None:
    s = Scenario("ambiguous")
    lim = limiter_facts(s)
    s.session("a", lim).turn("same", "same", learn=["lock"])
    s.session("b", lim).turn("same", "same", learn=["skew"])
    with pytest.raises(ValueError, match="identical exchanges"):
        OracleModel(s)


def test_clock_is_restored_after_a_run() -> None:
    Simulator(SCENARIOS["decay"]()).run()
    assert clock.offset().total_seconds() == 0


def test_stemming_matches_word_forms_but_leaves_names_alone() -> None:
    assert stem("tests") == "test" and stem("running") == "run" and stem("queries") == "query"
    assert stem("passed") == "pass" and stem("class") == "class" and stem("status") == "status"
    assert stem("raced") == stem("race") == stem("racing") and stem("caused") == stem("cause")
    assert tokens("Running svc/limiter.py tests") == ["run", "svc/limiter.py", "test"]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["cross_session_recall", "supersede", "project_isolation"])
def test_scenario_with_the_real_embedder(name: str) -> None:
    report = Simulator(SCENARIOS[name](), embedder="real").run()
    assert report.ok, report.text()
