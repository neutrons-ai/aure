"""BIC convention and cross-node consistency.

Two defects motivate this file:

* ``fitting`` and ``evaluation`` each derived ``n`` and ``k`` for themselves and
  then compared the resulting BIC values. ``evaluation`` used
  ``len(state["Q"])`` — the *primary* data file only — so on any co-refinement
  the two sides of the regression guardrail were different statistics.
* The formula was the unknown-variance ``n·ln(χ²_red) + k·ln(n)``, which is
  scale-free in χ²: it demanded the same *relative* χ² improvement to justify a
  layer no matter how bad the fit was. The variances here are known (the ``dR``
  column), so the correct form is ``χ²_total + k·ln(n)``.
"""

import math

import pytest

from aure.nodes.evaluation import (
    BIC_FORMULA,
    _compute_bic,
    _n_data_from_state,
    bic_baseline_is_stale,
    bic_inputs_for,
)


# ----------------------------------------------------------------------
# The formula
# ----------------------------------------------------------------------


def test_bic_is_the_known_variance_gaussian_form():
    """BIC = χ²_total + k·ln(n), with the total (un-normalized) χ²."""
    assert _compute_bic(100.0, 1000, 10) == pytest.approx(100.0 + 10 * math.log(1000))
    # k = 0 leaves the likelihood term alone.
    assert _compute_bic(42.0, 500, 0) == pytest.approx(42.0)


@pytest.mark.parametrize(
    "chi2_total, n_data",
    [
        (float("inf"), 1000),  # fit-failed / infeasible sentinel
        (float("nan"), 1000),
        (-1.0, 1000),  # a negative sum of squares is not a thing
        (100.0, 0),  # no data
        (100.0, -5),
    ],
)
def test_degenerate_inputs_give_inf_so_they_never_claim_the_baseline(
    chi2_total, n_data
):
    assert _compute_bic(chi2_total, n_data, 8) == float("inf")


def test_the_penalty_scales_with_fit_quality():
    """The bug the old formula had: a scale-free likelihood term.

    ``n·ln(χ²_red)`` made the minimum *relative* χ² improvement needed to buy a
    parameter independent of χ² — whether a layer was justified did not depend
    on how badly the model fitted. The known-variance form is sensitive to the
    absolute χ² gain, so a 1 % improvement on a poor fit is strong evidence and
    a 1 % improvement on a good one is not.
    """
    n, k_simple, k_complex = 2000, 12, 16

    def justified(chi2_red, rel_gain):
        """Does a `rel_gain` relative χ² improvement pay for 4 more params?"""
        simple = _compute_bic(chi2_red * (n - k_simple), n, k_simple)
        better = chi2_red * (1 - rel_gain)
        complex_ = _compute_bic(better * (n - k_complex), n, k_complex)
        return complex_ < simple

    # A 1 % gain on a badly fitting model is worth four parameters...
    assert justified(5.0, 0.01) is True
    # ...but the same relative gain on an already-good fit is not.
    assert justified(1.0, 0.01) is False


# ----------------------------------------------------------------------
# Input resolution — the cross-node bug
# ----------------------------------------------------------------------


def test_recorded_inputs_are_used_verbatim():
    """What the fit recorded from the bumps problem always wins."""
    fit = {
        "chi_squared": 2.0,
        "_chi2_total": 3970.0,
        "_n_data": 6000,
        "_n_free_params": 21,
    }
    # A state and model that would give completely different answers.
    state = {"Q": list(range(10))}
    model = {"layers": [{"name": "a"}]}

    chi2_total, n_data, n_params = bic_inputs_for(fit, state, model)
    assert (chi2_total, n_data, n_params) == (3970.0, 6000, 21)


def test_fitting_and_evaluation_resolve_identical_inputs():
    """The regression test for the original defect.

    Both nodes now route through ``bic_inputs_for``, so a co-refinement fit
    result scores the same wherever it is scored. Previously ``evaluation``
    reached for ``len(state["Q"])`` and got one file's point count.
    """
    fit = {
        "chi_squared": 1.8,
        "_chi2_total": 10_700.0,
        "_n_data": 6000,
        "_n_free_params": 18,
    }
    # `evaluation` sees the flattened primary-file view...
    evaluation_state = {"Q": list(range(2000))}
    # ...while `fitting` sees the per-state model. Same fit, same answer.
    fitting_state = {"Q": list(range(2000))}
    model = {
        "states": [
            {"name": s, "data_files": [{"Q": list(range(2000))}]}
            for s in ("d2o", "h2o", "cm4")
        ]
    }

    from_eval = _compute_bic(*bic_inputs_for(fit, evaluation_state, model))
    from_fit = _compute_bic(*bic_inputs_for(fit, fitting_state, model))
    assert from_eval == from_fit
    assert from_eval == pytest.approx(10_700.0 + 18 * math.log(6000))


def test_legacy_fit_results_reconstruct_a_total_chi2():
    """A checkpoint predating the recorded inputs still scores in the right units.

    The stored χ² is *reduced*, so it is scaled back up by the degrees of
    freedom rather than fed to the formula as-is.
    """
    fit = {"chi_squared": 2.0}  # no _chi2_total / _n_data / _n_free_params
    state = {"Q": list(range(1000))}
    model = {"layers": [{"name": "a"}, {"name": "b"}]}

    chi2_total, n_data, n_params = bic_inputs_for(fit, state, model)
    assert n_data == 1000
    assert n_params > 0
    # Reduced χ² of 2.0 over dof = n - k.
    assert chi2_total == pytest.approx(2.0 * (1000 - n_params))
    # Emphatically not the reduced value.
    assert chi2_total > 100


# ----------------------------------------------------------------------
# n counting
# ----------------------------------------------------------------------


def test_n_data_sums_every_dataset_of_every_state():
    """The root cause: `state["Q"]` is the primary file, not the whole fit."""
    model = {
        "states": [
            {"name": "d2o", "data_files": [{"Q": list(range(800))}]},
            {
                "name": "h2o",
                "data_files": [{"Q": list(range(500))}, {"Q": list(range(300))}],
            },
        ]
    }
    state = {"Q": list(range(800))}  # would have been the old answer
    assert _n_data_from_state(state, model) == 1600


def test_n_data_falls_back_through_flat_data_files_then_primary_q():
    # Flat multi-file shape (no states block).
    state = {
        "Q": list(range(400)),
        "data_files": [{"Q": list(range(400))}, {"Q": list(range(250))}],
    }
    assert _n_data_from_state(state, {}) == 650

    # Nothing but the primary file.
    assert _n_data_from_state({"Q": list(range(123))}, {}) == 123
    assert _n_data_from_state({}, {}) == 0


# ----------------------------------------------------------------------
# Convention marker
# ----------------------------------------------------------------------


def test_a_baseline_from_the_old_convention_is_stale():
    """An unmarked `best_bic` predates the marker, which is the stale case."""
    assert bic_baseline_is_stale({"best_bic": 1386.0}) is True
    assert (
        bic_baseline_is_stale({"best_bic": 1386.0, "bic_formula": "n_ln_chi2"}) is True
    )


def test_a_current_baseline_is_not_stale():
    assert (
        bic_baseline_is_stale({"best_bic": 3970.0, "bic_formula": BIC_FORMULA}) is False
    )


def test_no_baseline_is_not_stale():
    """Nothing to discard — the next fit establishes it."""
    assert bic_baseline_is_stale({}) is False
    assert bic_baseline_is_stale({"best_bic": None}) is False


# ----------------------------------------------------------------------
# Reading the inputs off a bumps problem
# ----------------------------------------------------------------------


class _FakeProblem:
    """Enough of the bumps ``FitProblem`` surface for the accessors."""

    def __init__(self, pmodel=0.0, failing=(), n_points=1000, n_pars=10):
        self._pmodel = pmodel
        self._failing = failing
        self._n_points = n_points
        self._n_pars = n_pars

    def _nllf_components(self):
        return (0.0, 0.0, self._pmodel, self._failing)

    def model_points(self):
        return self._n_points

    def getp(self):
        return [0.0] * self._n_pars

    def chisq(self, nllf=None, norm=True):  # pragma: no cover - fallback only
        raise AssertionError("should not be reached when _nllf_components works")


def test_total_chi2_is_twice_the_model_nllf():
    """bumps defines pmodel = ½·Σresiduals² for Gaussian independent errors."""
    from aure.nodes.model_builder import data_chisq_total

    assert data_chisq_total(_FakeProblem(pmodel=1985.0)) == pytest.approx(3970.0)


def test_infeasible_parameters_give_inf_not_zero():
    """bumps returns pmodel = 0.0 without evaluating the model when a constraint
    fails; reporting that would read as a perfect fit."""
    from aure.nodes.model_builder import data_chisq_total

    problem = _FakeProblem(pmodel=0.0, failing=["SEI.rho > 0"])
    assert data_chisq_total(problem) == float("inf")


def test_bic_inputs_reads_n_and_k_from_the_problem():
    from aure.nodes.model_builder import bic_inputs

    got = bic_inputs(_FakeProblem(pmodel=50.0, n_points=6000, n_pars=21))
    assert got == {"_chi2_total": 100.0, "_n_data": 6000, "_n_free_params": 21}


def test_total_chi2_falls_back_to_chisq_when_bumps_internals_move():
    """chisq(norm=False) scales the nllf by exactly 2, giving the total."""
    from aure.nodes.model_builder import data_chisq_total

    class Renamed:
        def chisq(self, nllf=None, norm=True):
            assert norm is False
            return 3970.0

    assert data_chisq_total(Renamed()) == pytest.approx(3970.0)


def test_bic_inputs_degrades_to_empty_rather_than_raising():
    """A missing accessor must not take the fit down; BIC is then reconstructed."""
    from aure.nodes.model_builder import bic_inputs

    class Broken:
        def _nllf_components(self):
            return (0.0, 0.0, 1.0, ())

        def model_points(self):
            raise RuntimeError("gone")

    assert bic_inputs(Broken()) == {}
