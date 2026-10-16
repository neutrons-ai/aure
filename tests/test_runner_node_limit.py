"""The runner's node-execution cap must leave room for the refinement budget.

It was a flat 20, which stopped any run with ``-m`` above 6 after six
evaluations and said nothing: a doubled budget (``-m 10``) ran no further than
the default.
"""

from __future__ import annotations

import logging

import pytest

from aure.workflow import runner


def test_the_limit_leaves_room_for_every_pass():
    for passes in range(1, 31):
        # intake and analysis, then modeling, fitting and evaluation per pass
        assert runner._node_limit(passes) >= 2 + 3 * passes


def test_the_default_budget_keeps_the_old_floor():
    assert runner._node_limit(5) >= 20
    assert runner._node_limit(None) == runner._node_limit(5)


@pytest.fixture
def stub_nodes(monkeypatch):
    """Every node a stub; evaluation advances the pass count, as the real one does."""
    calls = []

    def stub(name):
        def node(state):
            calls.append(name)
            if name == "evaluation":
                return {"iteration": state.get("iteration", 0) + 1}
            if name == "finalize":
                return {"finalized": True}
            return {}

        return node

    for name in list(runner.NODE_FUNCTIONS):
        monkeypatch.setitem(runner.NODE_FUNCTIONS, name, stub(name))
    routes = {
        "intake": lambda s: "analysis",
        "analysis": lambda s: "modeling",
        "modeling": lambda s: "fitting",
        "fitting": lambda s: "evaluation",
        "evaluation": lambda s: "complete" if s["iteration"] >= s["max_iterations"] else "modeling",
    }
    for name, route in routes.items():
        monkeypatch.setitem(runner.ROUTING_FUNCTIONS, name, route)
    return calls


def test_a_ten_pass_budget_runs_ten_passes(stub_nodes):
    runner.run_workflow_with_checkpoints({"max_iterations": 10, "iteration": 0})
    assert stub_nodes.count("evaluation") == 10


def test_hitting_the_limit_is_reported(stub_nodes, monkeypatch, caplog):
    monkeypatch.setattr(runner, "_node_limit", lambda max_iterations: 8)
    with caplog.at_level(logging.WARNING, logger=runner.logger.name):
        runner.run_workflow_with_checkpoints({"max_iterations": 10, "iteration": 0})
    assert stub_nodes.count("evaluation") == 2
    assert "safety limit" in caplog.text
