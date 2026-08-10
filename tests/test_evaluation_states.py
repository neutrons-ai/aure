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


def test_boundary_hits_flag_an_uncertainty_that_reaches_the_bound():
    """A point-estimate test alone misses a posterior pressed against a range.

    Real case from a REF_L oxide fit: ``CuOx interface`` lands at 5.11 in
    ``[5, 11]`` -- 1.8% of the span from its floor, so the 1% test passes it --
    while its uncertainty runs down through the floor. The range is
    constraining the answer and the evaluator never heard about it.
    """
    from aure.nodes.evaluation import _check_boundary_hits

    # The real numbers from that fit, so this also pins the calibration of
    # `max_width_fraction`: dx=0.906 on a span of 6 means the interval covers
    # 60% of the range, and a guard tighter than that rejects a genuine pin.
    fit_result = {
        "parameters": {"CuOx interface": 5.1104},
        "bounds": {"CuOx interface": [5.0, 11.0]},
        "uncertainties": {"CuOx interface": 0.9059},
    }

    hits = _check_boundary_hits(fit_result)

    assert [h["name"] for h in hits] == ["CuOx interface"]
    assert hits[0]["bound_hit"] == "lower"
    assert hits[0]["detected_by"] == "uncertainty"
    assert hits[0]["uncertainty"] == 0.9059

    # And the point-estimate test alone -- what AuRE did before -- sees nothing.
    without = {k: v for k, v in fit_result.items() if k != "uncertainties"}
    assert _check_boundary_hits(without) == []


def test_boundary_hits_ignore_an_unconstrained_parameter():
    """A posterior spanning most of the range is not pinned against an edge --
    it is unconstrained, and widening its bounds makes the next fit worse.

    Without this guard the workflow expands the range every iteration, and
    each expansion makes the parameter less constrained than before.
    """
    from aure.nodes.evaluation import _check_boundary_hits

    fit_result = {
        "parameters": {"Ti rho": 0.0},
        "bounds": {"Ti rho": [-2.0, 1.0]},
        "uncertainties": {"Ti rho": 1.2},  # +/- 2 sigma covers 4.8 of a span of 3
    }

    assert _check_boundary_hits(fit_result) == []


def test_boundary_hits_are_unchanged_without_uncertainties():
    """Optimizers that do not estimate dx must behave exactly as before."""
    from aure.nodes.evaluation import _check_boundary_hits

    fit_result = {
        "parameters": {"Cu thickness": 500.0, "D2O Cu interface": 5.0},
        "bounds": {"Cu thickness": [250.0, 750.0], "D2O Cu interface": [1.0, 5.0]},
    }

    hits = _check_boundary_hits(fit_result)

    assert [h["name"] for h in hits] == ["D2O Cu interface"]
    assert hits[0]["detected_by"] == "value"


def test_a_value_at_the_bound_is_reported_once_not_twice():
    """A railed parameter satisfies both tests; reporting it twice would
    double-expand its range."""
    from aure.nodes.evaluation import _check_boundary_hits

    fit_result = {
        "parameters": {"Cu thickness": 750.0},
        "bounds": {"Cu thickness": [250.0, 750.0]},
        "uncertainties": {"Cu thickness": 5.0},
    }

    hits = _check_boundary_hits(fit_result)

    assert len(hits) == 1
    assert hits[0]["detected_by"] == "value"


def test_a_comfortable_parameter_with_a_tight_posterior_is_not_flagged():
    """The guard has to leave normal fits alone."""
    from aure.nodes.evaluation import _check_boundary_hits

    fit_result = {
        "parameters": {"Cu thickness": 500.0},
        "bounds": {"Cu thickness": [250.0, 750.0]},
        "uncertainties": {"Cu thickness": 5.0},
    }

    assert _check_boundary_hits(fit_result) == []


def test_a_nonsense_uncertainty_is_ignored():
    """dx arrives from a fitter, not from us."""
    import numpy as np

    from aure.nodes.evaluation import _check_boundary_hits

    bounds = {"a": [0.0, 10.0]}
    for bad in (0.0, -1.0, np.nan, None, "wide"):
        fit_result = {
            "parameters": {"a": 5.0},
            "bounds": bounds,
            "uncertainties": {"a": bad},
        }
        assert _check_boundary_hits(fit_result) == [], bad


def test_an_uncertainty_hit_is_described_as_one_in_the_prompt():
    """The value looks comfortable, so a reader told only that it "hit its
    bound" would look for a number that is not there."""
    from aure.nodes.prompts import _format_boundary_hits

    text = _format_boundary_hits(
        [
            {
                "name": "CuOx interface",
                "value": 5.11,
                "bound_hit": "lower",
                "bound_value": 5.0,
                "detected_by": "uncertainty",
                "uncertainty": 0.4,
            }
        ]
    )

    assert "not itself at a bound" in text
    assert "uncertainty reaches the lower bound" in text
    assert "5.1100" in text


def test_a_value_hit_keeps_its_original_wording():
    from aure.nodes.prompts import _format_boundary_hits

    text = _format_boundary_hits(
        [
            {
                "name": "Cu thickness",
                "value": 750.0,
                "bound_hit": "upper",
                "bound_value": 750.0,
                "detected_by": "value",
            }
        ]
    )

    assert "value 750.0000 hit upper bound (750.0000)" in text


def test_an_uncertainty_hit_expands_the_range_like_any_other():
    """The prompt tells the model the range was auto-expanded, so it has to
    have been -- otherwise the next iteration reports the same pin again."""
    from aure.nodes.evaluation import _expand_model_bounds

    model = {
        "layers": [
            {
                "name": "CuOx",
                "roughness_min": 5.0,
                "roughness_max": 11.0,
            }
        ]
    }

    expanded = _expand_model_bounds(
        model,
        [
            {
                "name": "CuOx interface",
                "value": 5.11,
                "bound_hit": "lower",
                "bound_value": 5.0,
                "detected_by": "uncertainty",
            }
        ],
    )

    assert expanded["layers"][0]["roughness_min"] == 2.0
    assert model["layers"][0]["roughness_min"] == 5.0, "the input is not mutated"


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
