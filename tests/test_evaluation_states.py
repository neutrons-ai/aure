"""Tests for evaluation node multi-state plumbing (Ticket 08)."""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Boundary hits
# ----------------------------------------------------------------------


def test_boundary_hits_carry_state_prefix_in_name():
    """Untied parameters carry state prefix; boundary hits surface it."""
    from aure.nodes.evaluation import _check_boundary_hits

    fit_result = {
        "parameters": {
            # Tied params: no state prefix (Ticket 04)
            "Cu thickness": 500.0,
            # Untied params: state-prefixed
            "D2O Cu interface": 5.0,  # at upper bound
            "H2O ambient rho": 6.4,
        },
        "bounds": {
            "Cu thickness": [250.0, 750.0],
            "D2O Cu interface": [1.0, 5.0],
            "H2O ambient rho": [5.0, 6.6],
        },
    }
    hits = _check_boundary_hits(fit_result)
    names = {h["name"] for h in hits}
    assert "D2O Cu interface" in names
    # Tied parameter (Cu thickness, mid-range) is not a hit
    assert "Cu thickness" not in names


# ----------------------------------------------------------------------
# Per-state residual fringe analysis
# ----------------------------------------------------------------------


def test_per_state_residual_analysis_invokes_per_state(monkeypatch):
    """When per_file_results has state names, residual analysis runs per
    state and stores results under ``per_state_residual_analysis``."""
    from aure.nodes import evaluation

    captured_calls: list[tuple[int, int]] = []

    def fake_analyze(Q, ratio):
        captured_calls.append((len(Q), len(ratio)))
        # Make D2O the one with fringes; H2O without
        if len(Q) > 0 and float(Q[0]) > 0.05:
            return {
                "has_residual_fringes": True,
                "unmodeled_thicknesses": [
                    {"thickness": 120.0, "confidence": "high"},
                ],
            }
        return {"has_residual_fringes": False, "unmodeled_thicknesses": []}

    # Patch via the module path the evaluation node actually imports
    import aure.tools.feature_tools as ft

    monkeypatch.setattr(ft, "analyze_residual_fringes", fake_analyze)

    # Force LLM-required path to short-circuit so we only exercise the residual block
    monkeypatch.setattr(evaluation, "llm_available", lambda: False)
    monkeypatch.setenv("CHI2_MAX", "1.0")  # ensure chi2 > chi2_max triggers analysis

    state = {
        "iteration": 0,
        "fit_results": [
            {
                "chi_squared": 50.0,
                "method": "lm",
                "parameters": {},
                "uncertainties": None,
                "bounds": None,
                "Q_fit": [],
                "R_fit": [],
                "residual_ratio": [],
                "per_file_results": [
                    {
                        "file": "/tmp/d2o.dat",
                        "label": "D2O",
                        "state": "D2O",
                        "Q_fit": list(np.linspace(0.06, 0.10, 30)),
                        "residual_ratio": list(
                            1.0 + 0.1 * np.cos(np.linspace(0, 8, 30))
                        ),
                    },
                    {
                        "file": "/tmp/h2o.dat",
                        "label": "H2O",
                        "state": "H2O",
                        "Q_fit": list(np.linspace(0.01, 0.05, 30)),
                        "residual_ratio": list(np.ones(30)),
                    },
                ],
            }
        ],
        "current_model": {
            "layers": [{"name": "Cu"}],
            "states": [{"name": "D2O"}, {"name": "H2O"}],
        },
        "Q": [],
        "messages": [],
        "active_skills": [],
        "user_config": {},
    }

    out = evaluation.evaluation_node(state)

    # LLM path errors out (because we set llm_available False) — that's fine;
    # we only need to check that residual analysis ran per-state first.
    latest = state["fit_results"][-1]
    assert "per_state_residual_analysis" in latest
    psa = latest["per_state_residual_analysis"]
    assert set(psa.keys()) == {"D2O", "H2O"}
    assert psa["D2O"]["has_residual_fringes"] is True
    assert psa["H2O"]["has_residual_fringes"] is False
    # analyze called twice (once per state)
    assert len(captured_calls) == 2
    # llm-required short-circuits cleanly
    assert "error" in out


# ----------------------------------------------------------------------
# Regression revert preserves multi-state structure
# ----------------------------------------------------------------------


def test_chi2_revert_restores_full_multi_state_model(monkeypatch):
    """When χ² regresses, evaluation reverts to best_model — including its
    states/shared_parameters fields (full state restoration)."""
    from aure.nodes import evaluation

    # Stub the LLM call — return a "fit acceptable=False" analysis so we go
    # into the regression-guardrail branch.
    def fake_llm_analyze(**kwargs):
        return {
            "fit_acceptable": False,
            "acceptable": False,
            "issues": [],
            "suggestions": [],
            "next_action": "refine",
            "_used_fallback": False,
        }

    monkeypatch.setattr(evaluation, "llm_available", lambda: True)
    monkeypatch.setattr(evaluation, "analyze_fit_quality_with_llm", fake_llm_analyze)

    best_model = {
        "substrate": {"name": "silicon", "sld": 2.07},
        "layers": [{"name": "Cu", "thickness": 500.0}],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "constraints": [],
        "data_file": "/tmp/d2o.dat",
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": True},
        "dq_is_fwhm": True,
        "states": [
            {"name": "D2O", "data_files": [{"file": "/tmp/d2o.dat"}]},
            {"name": "H2O", "data_files": [{"file": "/tmp/h2o.dat"}]},
        ],
        "shared_parameters": ["Cu.thickness"],
        "unshared_parameters": [],
    }

    current_model = {**best_model, "layers": [{"name": "Cu", "thickness": 700.0}]}

    state = {
        "iteration": 1,
        "fit_results": [
            {
                "chi_squared": 50.0,
                "method": "lm",
                "parameters": {},
                "uncertainties": None,
                "bounds": None,
                "Q_fit": [],
                "R_fit": [],
                "residual_ratio": [],
                "per_file_results": [],
            }
        ],
        "current_model": current_model,
        "best_chi2": 5.0,
        "best_model": best_model,
        "best_bic": 1000.0,
        "best_bic_model": best_model,
        "Q": list(range(10)),
        "messages": [],
        "active_skills": [],
        "user_config": {},
    }

    out = evaluation.evaluation_node(state)

    # current_model has been reverted to a snapshot of best_model.
    # Identity must be *independent* so the next refine iteration's
    # in-place edits of current_model cannot corrupt best_model.
    assert "current_model" in out
    reverted = out["current_model"]
    assert reverted is not best_model
    assert reverted["states"][0]["name"] == "D2O"
    assert reverted["shared_parameters"] == ["Cu.thickness"]
    # Mutating the reverted current_model must not reach best_model.
    reverted["states"][0]["name"] = "MUTATED"
    assert best_model["states"][0]["name"] == "D2O"
