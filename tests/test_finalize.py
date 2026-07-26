"""Tests for the terminal `finalize` node and the max-iterations exit path.

Before this node existed, a run that exhausted `max_iterations` reported
whatever model happened to be current when the loop broke — `best_model` was
tracked but never consumed at the end. These tests pin the selection rule, the
routing that reaches it, and the idempotency the runner relies on.
"""

import copy

import pytest

from aure.nodes.finalize import _apply_fitted_values, finalize_node
from aure.nodes.routing import route_after_evaluation

# ======================================================================
# Fixtures
# ======================================================================


def _model(thickness=100.0, sld=2.0, roughness=5.0, extra_layers=()):
    layers = [
        {
            "name": "Cu",
            "thickness": thickness,
            "sld": sld,
            "roughness": roughness,
        }
    ]
    for name in extra_layers:
        layers.append({"name": name, "thickness": 20.0, "sld": 1.0, "roughness": 3.0})
    return {
        "layers": layers,
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {"value": 1.0},
    }


def _fit(iteration, chi2, params=None, bic=None):
    return {
        "iteration": iteration,
        "method": "lm",
        "chi_squared": chi2,
        "bic": bic,
        "parameters": params or {},
        "uncertainties": None,
    }


def _state(fit_results, model_history=None, **extra):
    state = {
        "fit_results": fit_results,
        "model_history": model_history or [],
        "current_model": _model(),
        "iteration": len(fit_results),
        "max_iterations": len(fit_results),
    }
    state.update(extra)
    return state


# ======================================================================
# Selection
# ======================================================================


def test_selects_best_chi2_not_last_iteration():
    """The core regression: loop wandered, last fit is worse than iteration 1."""
    history = [
        {"iteration": 0, "definition": _model(thickness=111.0)},
        {"iteration": 1, "definition": _model(thickness=222.0)},
    ]
    fits = [_fit(0, 1.20), _fit(1, 4.80)]

    updates = finalize_node(_state(fits, history, best_chi2=1.20))

    sel = updates["final_selection"]
    assert sel["selected"] is True
    assert sel["iteration"] == 0
    assert sel["index"] == 0
    assert sel["chi_squared"] == pytest.approx(1.20)
    assert sel["superseded_last_iteration"] is True
    assert sel["last_iteration_chi2"] == pytest.approx(4.80)
    # current_chi2 must follow the selection, or every downstream consumer of
    # the "current" keys reports a mismatched number.
    assert updates["current_chi2"] == pytest.approx(1.20)
    assert updates["current_model"]["layers"][0]["thickness"] == pytest.approx(111.0)
    assert updates["finalized"] is True


def test_keeps_last_iteration_when_it_is_the_best():
    history = [
        {"iteration": 0, "definition": _model(thickness=111.0)},
        {"iteration": 1, "definition": _model(thickness=222.0)},
    ]
    fits = [_fit(0, 4.80), _fit(1, 1.20)]

    updates = finalize_node(_state(fits, history, best_chi2=1.20))

    sel = updates["final_selection"]
    assert sel["iteration"] == 1
    assert sel["superseded_last_iteration"] is False
    assert updates["current_model"]["layers"][0]["thickness"] == pytest.approx(222.0)


def test_writes_fitted_values_into_selected_definition():
    """model_history holds pre-fit seeds; the answer must carry fitted values.

    Parameter names are the ones refl1d actually produces — note that the beam
    intensity is spelled ``"intensity <probe label>"``, never bare
    ``"intensity"`` (model_builder names it from the probe).
    """
    history = [{"iteration": 0, "definition": _model(thickness=100.0, sld=2.0)}]
    fits = [
        _fit(
            0,
            1.5,
            params={
                "Cu thickness": 137.5,
                "Cu rho": 6.4,
                "Cu interface": 9.1,
                "Si interface": 4.2,
                "intensity REFL_218386_combined_data_auto": 0.97,
            },
        )
    ]

    updates = finalize_node(_state(fits, history))

    layer = updates["current_model"]["layers"][0]
    assert layer["thickness"] == pytest.approx(137.5)
    assert layer["sld"] == pytest.approx(6.4)
    assert layer["roughness"] == pytest.approx(9.1)
    assert updates["current_model"]["substrate"]["roughness"] == pytest.approx(4.2)
    assert updates["current_model"]["intensity"]["value"] == pytest.approx(0.97)
    assert updates["final_selection"]["values_applied"] == 5
    assert updates["final_selection"]["values_unapplied"] == []


def test_multi_file_per_probe_intensities_are_reported_not_guessed():
    """Two probes, two fitted intensities, one definition field — don't pick one."""
    history = [{"iteration": 0, "definition": _model()}]
    fits = [
        _fit(
            0,
            1.5,
            params={"intensity lowQ": 0.95, "intensity highQ": 1.04},
        )
    ]

    updates = finalize_node(_state(fits, history))

    assert updates["current_model"]["intensity"]["value"] == pytest.approx(1.0)
    assert updates["final_selection"]["values_unapplied"] == [
        "intensity highQ",
        "intensity lowQ",
    ]


def test_tied_roughness_fraction_is_converted_to_a_numeric_sigma():
    """`roughness_tie` replaces "<layer> interface" with "<layer> rough_frac"."""
    defn = _model(thickness=80.0, roughness=6.0)
    defn["layers"][0]["roughness_tie"] = {"fraction_init": 0.3}
    history = [{"iteration": 0, "definition": defn}]
    fits = [_fit(0, 1.5, params={"Cu thickness": 80.0, "Cu rough_frac": 0.25})]

    updates = finalize_node(_state(fits, history))

    # sigma = fraction x thickness, matching model_builder.extract_definition
    assert updates["current_model"]["layers"][0]["roughness"] == pytest.approx(20.0)
    assert updates["final_selection"]["values_unapplied"] == []


def test_back_reflection_ambient_interface_is_not_written_to_a_dead_field():
    """In back reflection the builder seeds sample[0].interface from
    layers[-1]["roughness"], so ambient["roughness"] is never read back."""
    defn = _model()
    defn["back_reflection"] = True
    defn["ambient"] = {"name": "D2O", "sld": 6.3}
    history = [{"iteration": 0, "definition": defn}]
    fits = [_fit(0, 1.5, params={"D2O interface": 42.0, "Cu interface": 9.0})]

    updates = finalize_node(_state(fits, history))

    assert "roughness" not in updates["current_model"]["ambient"]
    assert updates["current_model"]["layers"][0]["roughness"] == pytest.approx(9.0)
    # Surfaced rather than silently swallowed.
    assert "D2O interface" in updates["final_selection"]["values_unapplied"]


def test_back_reflection_substrate_interface_is_not_written_either():
    """Back reflection puts the substrate at the far end with no free interface."""
    defn = _model()
    defn["back_reflection"] = True
    history = [{"iteration": 0, "definition": defn}]
    fits = [_fit(0, 1.5, params={"Si interface": 11.0})]

    updates = finalize_node(_state(fits, history))

    assert updates["current_model"]["substrate"]["roughness"] == pytest.approx(3.0)
    assert "Si interface" in updates["final_selection"]["values_unapplied"]


def test_bounds_only_refit_resolves_to_the_preceding_structure():
    """The bounds-only route re-fits without going through modeling, so that
    iteration has no model_history entry — it must not inherit a later model."""
    history = [
        {"iteration": 0, "definition": _model(thickness=111.0)},
        # iteration 1 was a bounds-only re-fit: no history entry
        {"iteration": 2, "definition": _model(thickness=333.0, extra_layers=("Ox",))},
    ]
    fits = [_fit(0, 4.00), _fit(1, 1.01), _fit(2, 1.00)]

    updates = finalize_node(_state(fits, history, best_chi2=1.00))

    sel = updates["final_selection"]
    assert sel["iteration"] == 1
    # Iteration 1 re-fit iteration 0's structure — one layer, not two.
    assert [layer["name"] for layer in updates["current_model"]["layers"]] == ["Cu"]
    assert updates["current_model"]["layers"][0]["thickness"] == pytest.approx(111.0)


def test_duplicate_history_iteration_takes_the_live_entry():
    """The interactive rewind replays an iteration; the newest entry wins."""
    history = [
        {"iteration": 0, "definition": _model(thickness=111.0)},
        {"iteration": 1, "definition": _model(thickness=222.0, extra_layers=("Ox",))},
        # …user rewound and re-ran iteration 1 with a different structure
        {"iteration": 1, "definition": _model(thickness=555.0)},
    ]
    fits = [_fit(0, 3.0), _fit(1, 1.0, params={"Cu thickness": 175.0})]

    updates = finalize_node(_state(fits, history, best_chi2=1.0))

    assert [layer["name"] for layer in updates["current_model"]["layers"]] == ["Cu"]
    assert updates["current_model"]["layers"][0]["thickness"] == pytest.approx(175.0)


def test_reported_free_params_match_the_promoted_model():
    """The ranking and the recorded count must use the definition promoted."""
    fits = [_fit(0, 1.000), _fit(1, 1.005)]
    state = _state(
        fits,
        [],
        best_chi2=1.000,
        best_model=_model(extra_layers=("Ox", "C")),
        current_model=_model(),
    )

    updates = finalize_node(state)
    sel = updates["final_selection"]

    promoted_layers = len(updates["current_model"]["layers"])
    # 3 free params per layer + intensity (ambient is air, substrate has no
    # roughness_max) — the count must describe what was actually promoted.
    assert sel["n_free_params"] == 3 * promoted_layers + 1


def test_does_not_mutate_model_history_entry():
    defn = _model(thickness=100.0)
    history = [{"iteration": 0, "definition": defn}]
    snapshot = copy.deepcopy(defn)
    fits = [_fit(0, 1.5, params={"Cu thickness": 137.5})]

    finalize_node(_state(fits, history))

    assert defn == snapshot


def test_parsimony_tiebreak_prefers_simpler_model_within_tolerance():
    """A 3-layer fit that is 1% better than a 1-layer fit isn't worth it."""
    history = [
        {"iteration": 0, "definition": _model()},
        {"iteration": 1, "definition": _model(extra_layers=("Ox", "C"))},
    ]
    fits = [_fit(0, 1.010), _fit(1, 1.000)]

    updates = finalize_node(_state(fits, history))

    sel = updates["final_selection"]
    assert sel["iteration"] == 0
    assert sel["parsimony_tiebreak"] is True
    assert sel["lowest_chi2"] == pytest.approx(1.000)
    assert sel["candidates_in_band"] == 2


def test_parsimony_tiebreak_does_not_apply_outside_tolerance():
    """A genuinely better complex model wins."""
    history = [
        {"iteration": 0, "definition": _model()},
        {"iteration": 1, "definition": _model(extra_layers=("Ox", "C"))},
    ]
    fits = [_fit(0, 2.00), _fit(1, 1.00)]

    updates = finalize_node(_state(fits, history))

    sel = updates["final_selection"]
    assert sel["iteration"] == 1
    assert sel["parsimony_tiebreak"] is False
    assert sel["candidates_in_band"] == 1


def test_tolerance_is_configurable(monkeypatch):
    history = [
        {"iteration": 0, "definition": _model()},
        {"iteration": 1, "definition": _model(extra_layers=("Ox",))},
    ]
    fits = [_fit(0, 1.10), _fit(1, 1.00)]

    monkeypatch.setenv("FINAL_SELECTION_TOL", "0.0")
    assert finalize_node(_state(fits, history))["final_selection"]["iteration"] == 1

    monkeypatch.setenv("FINAL_SELECTION_TOL", "0.2")
    assert finalize_node(_state(fits, history))["final_selection"]["iteration"] == 0


def test_prefers_reported_free_param_count_over_definition_count():
    """Multi-state fits report the true count; don't recount the template."""
    history = [
        {"iteration": 0, "definition": _model()},
        {"iteration": 1, "definition": _model()},
    ]
    fits = [_fit(0, 1.005), _fit(1, 1.000)]
    fits[0]["_n_free_params"] = 12
    fits[1]["_n_free_params"] = 4

    updates = finalize_node(_state(fits, history))

    assert updates["final_selection"]["iteration"] == 1
    assert updates["final_selection"]["n_free_params"] == 4


# ======================================================================
# Degenerate inputs
# ======================================================================


def test_no_fit_results_is_a_noop_but_still_finalizes():
    updates = finalize_node(_state([]))

    assert updates["finalized"] is True
    assert updates["final_selection"]["selected"] is False
    assert "current_model" not in updates
    assert "current_chi2" not in updates


def test_non_finite_chi2_is_ignored():
    history = [
        {"iteration": 0, "definition": _model(thickness=111.0)},
        {"iteration": 1, "definition": _model(thickness=222.0)},
    ]
    fits = [_fit(0, 1.5), _fit(1, float("inf"))]

    updates = finalize_node(_state(fits, history))

    sel = updates["final_selection"]
    assert sel["iteration"] == 0
    assert sel["candidates_considered"] == 1
    # A diverged final fit still counts as "the last iteration" — the user
    # needs to be told the answer came from an earlier one.
    assert sel["superseded_last_iteration"] is True
    assert sel["last_iteration_chi2"] == float("inf")


def test_nonpositive_chi2_is_ignored_rather_than_crashing():
    """chi2 <= 0 makes the parsimony band meaningless; skip such entries."""
    history = [{"iteration": 0, "definition": _model()}]
    fits = [_fit(0, 0.0), _fit(1, -1.0)]

    updates = finalize_node(_state(fits, history))

    assert updates["final_selection"]["selected"] is False
    assert updates["finalized"] is True


def test_garbage_tolerance_falls_back_to_the_default(monkeypatch):
    history = [
        {"iteration": 0, "definition": _model()},
        {"iteration": 1, "definition": _model(extra_layers=("Ox",))},
    ]
    fits = [_fit(0, 1.10), _fit(1, 1.00)]

    for bad in ("not-a-number", "inf", "-0.5", "nan"):
        monkeypatch.setenv("FINAL_SELECTION_TOL", bad)
        sel = finalize_node(_state(fits, history))["final_selection"]
        assert sel["tolerance"] == pytest.approx(0.02), bad
        # 10% apart, so the default 2% band excludes the simpler model.
        assert sel["iteration"] == 1, bad


def test_is_idempotent():
    """The runner calls it defensively after the loop; it must not run twice."""
    fits = [_fit(0, 1.5)]
    state = _state(fits, [{"iteration": 0, "definition": _model()}])

    first = finalize_node(state)
    assert first["finalized"] is True

    state.update(first)
    assert finalize_node(state) == {}


def test_falls_back_to_best_model_when_history_has_no_entry():
    """A legacy/imported state may carry best_model but no model_history."""
    fits = [_fit(0, 1.2), _fit(1, 3.4)]
    state = _state(fits, [], best_chi2=1.2, best_model=_model(thickness=999.0))

    updates = finalize_node(state)

    assert updates["current_model"]["layers"][0]["thickness"] == pytest.approx(999.0)


def test_never_writes_the_regression_baseline():
    """best_* is the loop's baseline and `aure resume`'s comparison point."""
    fits = [_fit(0, 1.2), _fit(1, 3.4)]
    updates = finalize_node(
        _state(fits, [{"iteration": 0, "definition": _model()}], best_chi2=1.2)
    )

    for key in ("best_chi2", "best_model", "best_bic", "best_bic_model"):
        assert key not in updates


def test_legacy_script_string_model_is_passed_through():
    fits = [_fit(0, 1.2, params={"Cu thickness": 50.0})]
    state = _state(fits, [{"iteration": 0, "script": "# refl1d script"}])

    updates = finalize_node(state)

    assert updates["current_model"] == "# refl1d script"
    assert updates["final_selection"]["values_applied"] == 0


# ======================================================================
# Fitted-value write-back, multi-state
# ======================================================================


def test_apply_fitted_values_multi_state_tied_and_untied():
    defn = {
        "layers": [{"name": "Cu", "thickness": 100.0, "sld": 2.0, "roughness": 5.0}],
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "states": [
            {"name": "D2O", "ambient": {"name": "D2O", "sld": 6.3}},
            {"name": "H2O", "ambient": {"name": "H2O", "sld": -0.56}},
        ],
    }
    fitted = {
        "Cu thickness": 142.0,  # tied — keeps the default spelling
        "D2O Cu rho": 6.9,  # untied — state-prefixed
        "H2O Cu rho": 6.1,
        "D2O D2O rho": 6.35,
        "H2O H2O rho": -0.50,
    }

    n_applied, unapplied = _apply_fitted_values(defn, fitted)

    assert n_applied == 5
    assert unapplied == []
    assert defn["layers"][0]["thickness"] == pytest.approx(142.0)
    # Inherited layers were materialized per state so untied values have a home
    # (tying is driven by shared_parameters, so this does not change re-fits).
    assert defn["states"][0]["layers"][0]["sld"] == pytest.approx(6.9)
    assert defn["states"][1]["layers"][0]["sld"] == pytest.approx(6.1)
    assert defn["states"][0]["ambient"]["sld"] == pytest.approx(6.35)
    assert defn["states"][1]["ambient"]["sld"] == pytest.approx(-0.50)


def test_inherited_state_substrate_and_intensity_are_materialized():
    """Untied per-state substrate roughness / intensity have no home until the
    inherited block is copied onto the state."""
    defn = {
        "layers": [{"name": "Cu", "thickness": 100.0, "sld": 6.0, "roughness": 5.0}],
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "intensity": {"value": 1.0},
        "states": [{"name": "D2O"}, {"name": "H2O"}],
    }
    fitted = {
        "D2O Si interface": 11.0,
        "H2O Si interface": 16.0,
        "D2O intensity": 0.93,
        "H2O intensity": 1.02,
    }

    n_applied, unapplied = _apply_fitted_values(defn, fitted)

    assert n_applied == 4
    assert unapplied == []
    assert defn["states"][0]["substrate"]["roughness"] == pytest.approx(11.0)
    assert defn["states"][1]["substrate"]["roughness"] == pytest.approx(16.0)
    assert defn["states"][0]["intensity"]["value"] == pytest.approx(0.93)
    assert defn["states"][1]["intensity"]["value"] == pytest.approx(1.02)
    # The shared template is untouched.
    assert defn["substrate"]["roughness"] == pytest.approx(3.0)
    assert defn["intensity"]["value"] == pytest.approx(1.0)


def test_state_intensity_does_not_leak_into_the_model_level_block():
    """The model-level scope has an empty prefix, so it must not claim
    "<state> intensity"."""
    defn = {
        "layers": [{"name": "Cu", "thickness": 100.0, "sld": 2.0, "roughness": 5.0}],
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "intensity": {"value": 1.0},
        "states": [
            {"name": "D2O", "intensity": {"value": 1.0}},
        ],
    }

    n_applied, unapplied = _apply_fitted_values(defn, {"D2O intensity": 0.93})

    assert n_applied == 1
    assert unapplied == []
    assert defn["states"][0]["intensity"]["value"] == pytest.approx(0.93)
    assert defn["intensity"]["value"] == pytest.approx(1.0)


def test_apply_fitted_values_reports_unplaceable_names():
    defn = _model()
    n_applied, unapplied = _apply_fitted_values(
        defn, {"Cu thickness": 120.0, "theta_offset": 0.01, "background": 1e-7}
    )

    assert n_applied == 1
    assert unapplied == ["background", "theta_offset"]


# ======================================================================
# Routing into the node
# ======================================================================


@pytest.mark.parametrize("iteration,max_iter", [(5, 5), (6, 5), (1, 1)])
def test_route_after_evaluation_completes_at_max_iterations(iteration, max_iter):
    state = {
        "iteration": iteration,
        "max_iterations": max_iter,
        "fit_results": [_fit(iteration, 9.9)],
    }
    assert route_after_evaluation(state) == "complete"


def test_route_after_evaluation_refines_below_max_iterations():
    state = {
        "iteration": 2,
        "max_iterations": 5,
        "fit_results": [{**_fit(2, 9.9), "issues": ["bad"]}],
    }
    assert route_after_evaluation(state) == "modeling"


def test_graph_routes_every_fitting_exit_through_finalize():
    from aure.workflow.graph import create_workflow

    graph = create_workflow(include_fitting=True)
    nodes = graph.get_graph().nodes
    assert "finalize" in nodes

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("evaluation", "finalize") in edges
    assert ("fitting", "finalize") in edges
    assert ("modeling", "finalize") in edges
    assert ("evaluation", "__end__") not in edges


def test_finalize_is_registered_with_the_runner_but_is_terminal():
    from aure.workflow.runner import (
        NODE_FUNCTIONS,
        NODE_ORDER,
        ROUTING_FUNCTIONS,
    )

    assert "finalize" in NODE_ORDER
    assert NODE_FUNCTIONS["finalize"] is finalize_node
    # No router — that is what makes the runner's loop break after it.
    assert "finalize" not in ROUTING_FUNCTIONS


# ======================================================================
# End-to-end through the runner (the engine `aure analyze` actually uses)
# ======================================================================


def _fake_nodes(chi2_schedule, acceptable_at=None):
    """Node stand-ins that walk the refinement loop with a scripted χ² series.

    Mirrors the real nodes' iteration bookkeeping: modeling and fitting both
    read ``state["iteration"]``; evaluation is what increments it.
    """

    def intake(state):
        return {
            "current_node": "intake",
            "Q": [0.01, 0.02],
            "R": [1.0, 0.5],
            "dR": [0.1, 0.1],
        }

    def analysis(state):
        return {"current_node": "analysis"}

    def modeling(state):
        it = state.get("iteration", 0)
        defn = _model(thickness=100.0 + it)
        return {
            "current_node": "modeling",
            "current_model": defn,
            "model_history": [{"iteration": it, "definition": defn}],
        }

    def fitting(state):
        it = state.get("iteration", 0)
        chi2 = chi2_schedule[it]
        updates = {
            "current_node": "fitting",
            "fit_results": [_fit(it, chi2, params={"Cu thickness": 500.0 + it})],
            "current_chi2": chi2,
        }
        best = state.get("best_chi2")
        if best is None or chi2 < best:
            updates["best_chi2"] = chi2
            updates["best_model"] = copy.deepcopy(state["current_model"])
        return updates

    def evaluation(state):
        it = state.get("iteration", 0) + 1
        updates = {"current_node": "evaluation", "iteration": it}
        if acceptable_at is not None and it == acceptable_at:
            updates["workflow_complete"] = True
        else:
            state["fit_results"][-1]["issues"] = ["not good enough"]
        return updates

    return {
        "intake": intake,
        "analysis": analysis,
        "modeling": modeling,
        "fitting": fitting,
        "evaluation": evaluation,
        "finalize": finalize_node,
    }


def _run(chi2_schedule, max_iterations, acceptable_at=None):
    from unittest.mock import patch

    from aure.workflow import runner

    state = {
        "current_node": "intake",
        "iteration": 0,
        "max_iterations": max_iterations,
        "messages": [],
        "fit_results": [],
        "model_history": [],
    }
    with patch.object(
        runner, "NODE_FUNCTIONS", _fake_nodes(chi2_schedule, acceptable_at)
    ):
        return runner.run_workflow_with_checkpoints(initial_state=state)


def test_max_iterations_exit_finalizes_on_the_best_iteration():
    """The reported regression: 3 iterations, the middle one is the best."""
    final = _run([4.0, 1.0, 3.0], max_iterations=3)

    assert len(final["fit_results"]) == 3
    assert final["finalized"] is True
    sel = final["final_selection"]
    assert sel["iteration"] == 1
    assert sel["superseded_last_iteration"] is True
    assert final["current_chi2"] == pytest.approx(1.0)
    # …and the promoted model is iteration 1's, carrying its fitted value.
    assert final["current_model"]["layers"][0]["thickness"] == pytest.approx(501.0)


def test_acceptable_exit_also_finalizes():
    """`workflow_complete` breaks the loop before routing runs at all — the
    post-loop call is the only thing that reaches finalize on this path."""
    final = _run([4.0, 1.0, 3.0], max_iterations=9, acceptable_at=3)

    assert final["workflow_complete"] is True
    assert final["finalized"] is True
    assert final["final_selection"]["iteration"] == 1
    assert final["current_chi2"] == pytest.approx(1.0)


def test_finalize_runs_only_once():
    final = _run([4.0, 1.0, 3.0], max_iterations=3)

    selections = [
        m
        for m in final["messages"]
        if str(m.get("content", "")).startswith("**Final model:**")
    ]
    assert len(selections) == 1


def test_prepare_state_for_restart_clears_the_selection():
    from aure.workflow.runner import prepare_state_for_restart

    final = _run([4.0, 1.0, 3.0], max_iterations=3)
    assert final["finalized"] is True

    restarted = prepare_state_for_restart(final, "try a thicker oxide")
    assert restarted["finalized"] is False
