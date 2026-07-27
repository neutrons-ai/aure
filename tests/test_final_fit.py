"""Tests for the optional ``final_fit`` uncertainty-polish step.

Exploration uses a fast optimizer (``FIT_METHOD``, e.g. amoeba) that reports no
usable uncertainties. When ``FIT_METHOD_FINAL`` is set, the runner's terminal
block runs one more fit (typically dream) on the finalize-selected model to
attach uncertainties. These tests pin:

* the self-gating (disabled / same-method / no-model / poor-χ² → no fit),
* the budget/threshold env parsing,
* the adopt-vs-keep decision and the model/χ² write-back,
* that a failed polish is never fatal,
* that ``problem.json`` follows an adopted polish, and
* the runner wiring that makes it fire for both the CLI and the web UI.
"""

import copy

import pytest

from aure.nodes import final_fit as ff
from aure.nodes.final_fit import (
    _final_burn,
    _final_steps,
    _quality_gate,
    final_fit_node,
)
from aure.workflow.checkpoints import CheckpointManager
from aure.workflow.runner import run_workflow_with_checkpoints


# ======================================================================
# Fixtures
# ======================================================================


def _model(thickness=100.0, sld=2.0, roughness=5.0):
    return {
        "layers": [
            {"name": "Cu", "thickness": thickness, "sld": sld, "roughness": roughness}
        ],
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {"value": 1.0},
    }


def _fit_result(chi2, params=None, uncertainties=None, iteration=0, method="dream"):
    return {
        "iteration": iteration,
        "method": method,
        "chi_squared": chi2,
        "converged": True,
        "parameters": params or {},
        "uncertainties": uncertainties,
        "sld_z": None,
        "sld_rho": None,
    }


def _state(**extra):
    state = {
        "current_model": _model(),
        "current_chi2": 1.0,
        "iteration": 2,
        "final_selection": {"selected": True, "iteration": 2, "chi_squared": 1.0},
        "output_dir": None,
        "data_files": [],
    }
    state.update(extra)
    return state


@pytest.fixture
def explore_amoeba_final_dream(monkeypatch):
    """The intended config: explore with amoeba, polish with dream."""
    monkeypatch.setenv("FIT_METHOD", "amoeba")
    monkeypatch.setenv("FIT_METHOD_FINAL", "dream")
    monkeypatch.setenv("CHI2_MAX", "1.5")
    monkeypatch.delenv("FINAL_FIT_CHI2_MAX", raising=False)
    monkeypatch.delenv("FIT_STEPS_FINAL", raising=False)
    monkeypatch.delenv("FIT_BURN_FINAL", raising=False)


# ======================================================================
# Gating — the node must be inert unless explicitly requested
# ======================================================================


def test_disabled_is_inert(monkeypatch):
    monkeypatch.delenv("FIT_METHOD_FINAL", raising=False)
    assert final_fit_node(_state()) == {}


def test_same_as_explore_method_is_inert(monkeypatch):
    monkeypatch.setenv("FIT_METHOD", "dream")
    monkeypatch.setenv("FIT_METHOD_FINAL", "dream")
    # No point re-running the same method exploration already used.
    assert final_fit_node(_state()) == {}


def test_no_dict_model_skips(explore_amoeba_final_dream):
    out = final_fit_node(_state(current_model=None))
    assert out["final_fit"]["ran"] is False
    assert "current_model" not in out


def test_chi2_above_gate_skips_without_fitting(
    explore_amoeba_final_dream, monkeypatch
):
    called = {"n": 0}

    def _boom(**_kwargs):
        called["n"] += 1
        raise AssertionError("run_fit_for_model must not be called above the gate")

    monkeypatch.setattr(ff, "run_fit_for_model", _boom)
    out = final_fit_node(_state(current_chi2=9.0))  # CHI2_MAX=1.5
    assert out["final_fit"]["ran"] is False
    assert called["n"] == 0
    assert "current_model" not in out


def test_nonfinite_chi2_skips(explore_amoeba_final_dream, monkeypatch):
    monkeypatch.setattr(ff, "run_fit_for_model", lambda **_k: _fit_result(1.0))
    out = final_fit_node(_state(current_chi2=float("inf")))
    assert out["final_fit"]["ran"] is False


# ======================================================================
# Budget / threshold parsing
# ======================================================================


def test_default_budget_is_10k(monkeypatch):
    monkeypatch.delenv("FIT_STEPS_FINAL", raising=False)
    monkeypatch.delenv("FIT_BURN_FINAL", raising=False)
    assert _final_steps() == 10000
    assert _final_burn(_final_steps()) == 10000


def test_budget_overrides(monkeypatch):
    monkeypatch.setenv("FIT_STEPS_FINAL", "20000")
    monkeypatch.setenv("FIT_BURN_FINAL", "5000")
    assert _final_steps() == 20000
    assert _final_burn(_final_steps()) == 5000


def test_budget_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("FIT_STEPS_FINAL", "not-a-number")
    monkeypatch.setenv("FIT_BURN_FINAL", "-5")
    assert _final_steps() == 10000
    assert _final_burn(_final_steps()) == 10000  # negative burn → default (=steps)


def test_quality_gate_prefers_dedicated_env(monkeypatch):
    monkeypatch.setenv("CHI2_MAX", "1.5")
    monkeypatch.setenv("FINAL_FIT_CHI2_MAX", "3.0")
    assert _quality_gate() == 3.0


def test_quality_gate_falls_back_to_chi2_max(monkeypatch):
    monkeypatch.setenv("CHI2_MAX", "2.0")
    monkeypatch.delenv("FINAL_FIT_CHI2_MAX", raising=False)
    assert _quality_gate() == 2.0


# ======================================================================
# Adopt vs keep, and the write-back
# ======================================================================


def test_adopts_improved_dream_fit(explore_amoeba_final_dream, monkeypatch):
    # dream holds/improves the fit and reports uncertainties → adopt it.
    result = _fit_result(
        chi2=0.95,
        params={"Cu thickness": 123.0, "Cu rho": 2.5, "Cu interface": 7.0},
        uncertainties={"Cu thickness": 4.0, "Cu rho": 0.1, "Cu interface": 0.5},
        iteration=2,
    )
    monkeypatch.setattr(ff, "run_fit_for_model", lambda **_k: result)

    out = final_fit_node(_state(current_chi2=1.0))

    assert out["final_fit"]["ran"] is True
    assert out["final_fit"]["adopted"] is True
    assert out["final_fit"]["n_uncertainties"] == 3
    # χ² and model values are updated to the dream result.
    assert out["current_chi2"] == 0.95
    assert out["current_model"]["layers"][0]["thickness"] == 123.0
    assert out["current_model"]["layers"][0]["sld"] == 2.5
    # The dream fit is appended to fit_results (uncertainties persisted).
    assert out["fit_results"] == [result]


def test_keeps_selection_when_dream_worse(explore_amoeba_final_dream, monkeypatch):
    # dream comes back materially worse (degenerate) → keep the better selection,
    # but still record the fit + uncertainties.
    result = _fit_result(
        chi2=5.0,
        params={"Cu thickness": 999.0},
        uncertainties={"Cu thickness": 50.0},
        iteration=2,
    )
    monkeypatch.setattr(ff, "run_fit_for_model", lambda **_k: result)

    base = _state(current_chi2=1.0)
    out = final_fit_node(copy.deepcopy(base))

    assert out["final_fit"]["ran"] is True
    assert out["final_fit"]["adopted"] is False
    # Reported model/χ² are NOT overwritten with the worse dream point.
    assert "current_model" not in out
    assert "current_chi2" not in out
    # But the dream fit is still on record so its uncertainties survive.
    assert out["fit_results"] == [result]


def test_fit_failure_is_not_fatal(explore_amoeba_final_dream, monkeypatch):
    def _raise(**_kwargs):
        raise RuntimeError("bumps blew up")

    monkeypatch.setattr(ff, "run_fit_for_model", _raise)
    out = final_fit_node(_state(current_chi2=1.0))

    assert out["final_fit"]["ran"] is False
    assert "fit error" in out["final_fit"]["reason"]
    # Never overwrites the selected model, never raises.
    assert "current_model" not in out
    assert "current_chi2" not in out


# ======================================================================
# problem.json follows an adopted polish
# ======================================================================


def test_problem_json_prefers_adopted_final_export(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    mgr.refl1d_output_dir.mkdir(parents=True, exist_ok=True)
    mgr.output_dir.mkdir(parents=True, exist_ok=True)

    # Exploration export (what the finalize selection points at) and the final
    # dream export both exist; the dream one must win when adopted.
    explore_dir = mgr.refl1d_output_dir / "fit_iter2_amoeba"
    explore_dir.mkdir()
    (explore_dir / "model.json").write_text('{"from": "amoeba"}')

    final_dir = mgr.refl1d_output_dir / "final_dream"
    final_dir.mkdir()
    (final_dir / "model.json").write_text('{"from": "dream"}')

    state = {
        "fit_results": [_fit_result(1.0, iteration=2, method="amoeba")],
        "final_selection": {"selected": True, "iteration": 2},
        "final_fit": {"ran": True, "adopted": True, "export_dir": str(final_dir)},
        "user_config": {},
    }
    mgr._copy_best_problem_json(state)

    copied = (mgr.output_dir / "problem.json").read_text()
    assert '"from": "dream"' in copied


def test_problem_json_ignores_unadopted_final_export(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    mgr.refl1d_output_dir.mkdir(parents=True, exist_ok=True)
    mgr.output_dir.mkdir(parents=True, exist_ok=True)

    explore_dir = mgr.refl1d_output_dir / "fit_iter2_amoeba"
    explore_dir.mkdir()
    (explore_dir / "model.json").write_text('{"from": "amoeba"}')

    final_dir = mgr.refl1d_output_dir / "final_dream"
    final_dir.mkdir()
    (final_dir / "model.json").write_text('{"from": "dream"}')

    state = {
        "fit_results": [_fit_result(1.0, iteration=2, method="amoeba")],
        "final_selection": {"selected": True, "iteration": 2},
        # Not adopted → problem.json must stay on the exploration export.
        "final_fit": {"ran": True, "adopted": False, "export_dir": str(final_dir)},
        "user_config": {},
    }
    mgr._copy_best_problem_json(state)

    copied = (mgr.output_dir / "problem.json").read_text()
    assert '"from": "amoeba"' in copied


# ======================================================================
# Runner wiring — the single path both CLI and web UI take
# ======================================================================


def _runner_state():
    """Minimal state that finalize can select from, starting at 'finalize'."""
    model = _model()
    return {
        "current_model": copy.deepcopy(model),
        "fit_results": [
            _fit_result(
                1.0,
                params={"Cu thickness": 100.0},
                iteration=0,
                method="amoeba",
            )
        ],
        "model_history": [{"iteration": 0, "definition": copy.deepcopy(model)}],
        "best_chi2": 1.0,
        "best_model": copy.deepcopy(model),
        "iteration": 0,
        "max_iterations": 1,
        "finalized": False,
    }


def test_runner_fires_final_fit_callback_when_enabled(
    explore_amoeba_final_dream, monkeypatch
):
    monkeypatch.setattr(
        ff,
        "run_fit_for_model",
        lambda **_k: _fit_result(
            0.9,
            params={"Cu thickness": 111.0},
            uncertainties={"Cu thickness": 3.0},
            iteration=0,
        ),
    )
    seen = []
    final_state = run_workflow_with_checkpoints(
        initial_state=_runner_state(),
        output_dir=None,  # no checkpoint mgr; callback still fires
        checkpoint_callback=lambda s, node: seen.append(node),
        start_node="finalize",
    )

    assert "finalize" in seen
    assert "final_fit" in seen  # the UI's _checkpoint_cb captures fit_results here
    assert final_state["final_fit"]["adopted"] is True
    assert final_state["current_chi2"] == 0.9


def test_runner_no_final_fit_when_disabled(monkeypatch):
    monkeypatch.setenv("FIT_METHOD", "amoeba")
    monkeypatch.delenv("FIT_METHOD_FINAL", raising=False)

    def _boom(**_kwargs):
        raise AssertionError("final fit must not run when disabled")

    monkeypatch.setattr(ff, "run_fit_for_model", _boom)
    seen = []
    final_state = run_workflow_with_checkpoints(
        initial_state=_runner_state(),
        output_dir=None,
        checkpoint_callback=lambda s, node: seen.append(node),
        start_node="finalize",
    )

    assert "finalize" in seen
    assert "final_fit" not in seen  # inert: no extra checkpoint / callback
    assert final_state.get("final_fit") in (None, {})
