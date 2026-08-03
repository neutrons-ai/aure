"""The MCP `run_fit` and `evaluate_fit` tools must actually be callable.

Both were dead on every invocation and neither had any coverage, so nothing caught
it: `run_fit` passed `model_script` / `data_file` / `max_iterations`, none of which
are parameters of `run_refl1d_fit` (the names date from when models were Python
scripts), and `evaluate_fit` imported `analyze_fit_quality`, which exists nowhere
in the package. A bare `except` turned both into a returned `{"error": ...}`, so
they looked like runtime failures rather than wiring bugs.

These tests call the tools for real — the point is that no error comes back at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import aure.mcp_server as mcp_server

DATA_FILE = str(Path(__file__).parent / "data" / "REFL_218386_combined_data_auto.txt")


def _tool(name: str):
    """The plain function behind a FastMCP-decorated tool."""
    tool = getattr(mcp_server, name)
    return getattr(tool, "fn", tool)


def _definition() -> dict:
    return {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "polystyrene",
                "sld": 1.41,
                "sld_min": -1.0,
                "sld_max": 4.0,
                "thickness": 100.0,
                "thickness_min": 50.0,
                "thickness_max": 200.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            }
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "constraints": [],
        "back_reflection": False,
        "data_file": DATA_FILE,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
    }


@pytest.fixture
def session():
    """A session carrying a fittable model, cleaned up afterwards."""
    sid = "test-session"
    mcp_server._sessions[sid] = {"current_model": _definition(), "iteration": 0}
    yield sid
    mcp_server._sessions.pop(sid, None)


def test_run_fit_completes(session):
    out = _tool("run_fit")(session, method="lm", steps=10)

    assert "error" not in out, out["error"]
    assert out["chi_squared"] > 0
    assert out["parameters"]
    # `success` was never a FitResult field; `converged` is.
    assert "converged" in out
    assert "success" not in out


def test_evaluate_fit_returns_an_assessment(session):
    _tool("run_fit")(session, method="lm", steps=10)

    out = _tool("evaluate_fit")(session)

    assert "error" not in out, out["error"]
    assert out["chi_squared_quality"] in {"good", "poor"}
    assert isinstance(out["issues"], list)
    # No acceptance verdict: `_simple_evaluation` reports and does not decide, so
    # the workflow's χ² clamp stays the single acceptance point. Deriving one here
    # would be a second rule, blind to the SLD-profile check this tool never runs.
    assert "acceptable" not in out
    assert "workflow" in out["acceptance"]


def test_both_tools_still_report_a_missing_session():
    for name in ("run_fit", "evaluate_fit"):
        assert "error" in _tool(name)("no-such-session")
