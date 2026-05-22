"""Smoke tests for the MCP `co_refine_states` tool (Ticket 10)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

import aure.mcp_server as mcp_module


def _write_minimal_data(path: Path) -> None:
    path.write_text("# Q R dR\n0.01 1.0 0.01\n0.02 0.5 0.01\n", encoding="utf-8")


def test_co_refine_states_tool_registered() -> None:
    """The tool must be discoverable by name in the MCP server module
    and registered with FastMCP via the ``@mcp.tool()`` decorator."""
    assert hasattr(mcp_module, "co_refine_states")
    assert callable(mcp_module.co_refine_states)
    # FastMCP 3.x annotates the function with `__fastmcp__` ToolMeta
    # instead of wrapping it. Verify the decorator ran.
    assert getattr(mcp_module.co_refine_states, "__fastmcp__", None) is not None


def test_co_refine_states_rejects_single_state(tmp_path: Path) -> None:
    data = tmp_path / "REFL_1_combined_data_auto.txt"
    _write_minimal_data(data)
    cfg = {
        "sample_description": "Cu on Si",
        "states": [
            {"name": "only", "data_files": [{"file": str(data)}]},
        ],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = mcp_module.co_refine_states(str(cfg_path))
    assert "error" in out
    assert "two" in out["error"].lower() or "states" in out["error"].lower()


def test_co_refine_states_dispatches_to_run_analysis(tmp_path: Path) -> None:
    d2o = tmp_path / "REFL_1_combined_data_auto.txt"
    h2o = tmp_path / "REFL_2_combined_data_auto.txt"
    _write_minimal_data(d2o)
    _write_minimal_data(h2o)

    cfg = {
        "sample_description": "Cu on Si",
        "states": [
            {"name": "D2O", "data_files": [{"file": str(d2o)}]},
            {"name": "H2O", "data_files": [{"file": str(h2o)}]},
        ],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    fake_result = {
        "fit_results": [
            {
                "iteration": 0,
                "chi_squared": 1.2,
                "per_file_results": [
                    {"state": "D2O", "label": "a", "chi_squared": 1.1},
                    {"state": "H2O", "label": "b", "chi_squared": 1.3},
                ],
            }
        ],
        "error": None,
    }

    with patch.object(mcp_module, "run_analysis", return_value=fake_result) as mock_ra:
        out = mcp_module.co_refine_states(
            str(cfg_path), output_dir=str(tmp_path / "out"), max_iterations=1
        )

    assert mock_ra.called
    kwargs = mock_ra.call_args.kwargs
    assert kwargs["max_iterations"] == 1
    assert kwargs.get("user_config") is not None
    assert kwargs.get("states") and len(kwargs["states"]) == 2

    assert out["success"] is True
    assert out["n_states"] == 2
    assert sorted(out["states"]) == ["D2O", "H2O"]
    assert out["aggregate_chi_squared"] == 1.2
    assert {pf["state"] for pf in out["per_file_chi_squared"]} == {"D2O", "H2O"}
